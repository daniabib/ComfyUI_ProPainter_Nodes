import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from .model.modules.flow_comp_raft import RAFT_bi
from .model.recurrent_flow_completion import RecurrentFlowCompleteNet
from .model.propainter import InpaintGenerator

# For Debbuging only
from .utils.img_util import imwrite 
from icecream import ic

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


def compute_flow(frames: torch.Tensor, raft_model, raft_iter, video_length) -> tuple:
    """
    Compute forward and backward flows.
    Optical Flow Computation: The fix_raft function is called in batches defined by short_clip_len. If the video has more frames than short_clip_len, it processes the frames in chunks to estimate the forward (flows_f) and backward (flows_b) optical flows. These flows are then concatenated to form gt_flows_f and gt_flows_b.
    """
    if frames.size(dim=-1) <= 640: 
        short_clip_len = 12
    elif frames.size(dim=-1) <= 720: 
        short_clip_len = 8
    elif frames.size(dim=-1) <= 1280:
        short_clip_len = 4
    else:
        short_clip_len = 2

    # use fp32 for RAFT
    if frames.size(dim=1) > short_clip_len:
        gt_flows_f_list, gt_flows_b_list = [], []
        for chunck in range(0, video_length, short_clip_len):
            end_f = min(video_length, chunck + short_clip_len)
            if chunck == 0:
                flows_f, flows_b = raft_model(frames[:,chunck:end_f], iters=raft_iter)
            else:
                flows_f, flows_b = raft_model(frames[:,chunck-1:end_f], iters=raft_iter)
            
            gt_flows_f_list.append(flows_f)
            gt_flows_b_list.append(flows_b)
            torch.cuda.empty_cache()
            
        gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
        gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
        gt_flows_bi = (gt_flows_f, gt_flows_b)
    else:
        gt_flows_bi = raft_model(frames, iters=raft_iter)
        torch.cuda.empty_cache()
            
    return gt_flows_bi


def complete_flow(recurrent_flow_model, flows_tuple, flow_masks, subvideo_length):
    """
    Complete Flow Computation: Based on the computed flows and subvideo_length, the flows are further processed to generate predicted flows using a model. This involves adjusting for padding and managing frame boundaries.
    """    
    flow_length = flows_tuple[0].size(dim=1)
    ic(flow_length)
    if flow_length > subvideo_length:
        pred_flows_f, pred_flows_b = [], []
        pad_len = 5
        for f in range(0, flow_length, subvideo_length):
            s_f = max(0, f - pad_len)
            e_f = min(flow_length, f + subvideo_length + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(flow_length, f + subvideo_length)
            pred_flows_bi_sub, _ = recurrent_flow_model.forward_bidirect_flow(
                (flows_tuple[0][:, s_f:e_f], flows_tuple[1][:, s_f:e_f]), 
                flow_masks[:, s_f:e_f+1])
            pred_flows_bi_sub = recurrent_flow_model.combine_flow(
                (flows_tuple[0][:, s_f:e_f], flows_tuple[1][:, s_f:e_f]), 
                pred_flows_bi_sub, 
                flow_masks[:, s_f:e_f+1])

            pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
            pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
            torch.cuda.empty_cache()
            
        pred_flows_f = torch.cat(pred_flows_f, dim=1)
        pred_flows_b = torch.cat(pred_flows_b, dim=1)

        pred_flows_bi = (pred_flows_f, pred_flows_b)

        ic(pred_flows_f.size())
        ic(pred_flows_b.size())
    
    else:
        pred_flows_bi, _ = recurrent_flow_model.forward_bidirect_flow(flows_tuple, flow_masks)
        pred_flows_bi = recurrent_flow_model.combine_flow(flows_tuple, pred_flows_bi, flow_masks)
        
        ic(pred_flows_bi[0].size())
        
        torch.cuda.empty_cache()
    
    return pred_flows_bi


def image_propagation(inpaint_model, 
                      frames, 
                      masks_dilated,
                      prediction_flows,
                      video_length, 
                      subvideo_length,  
                      process_size): 
    """
    The masked frames are computed by blending original frames and propagated images based on the masks. The process is again segmented if the video is longer than a defined threshold (subvideo_length_img_prop).
    """
    process_width, process_height = process_size
    masked_frames = frames * (1 - masks_dilated)
    ic(masked_frames.size())
    subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
    if video_length > subvideo_length_img_prop:
        updated_frames, updated_masks = [], []
        pad_len = 10
        for f in range(0, video_length, subvideo_length_img_prop):
            s_f = max(0, f - pad_len)
            e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
            b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
            pred_flows_bi_sub = (prediction_flows[0][:, s_f:e_f-1], prediction_flows[1][:, s_f:e_f-1])
            prop_imgs_sub, updated_local_masks_sub = inpaint_model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                pred_flows_bi_sub, 
                                                                masks_dilated[:, s_f:e_f], 
                                                                'nearest')
            updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                prop_imgs_sub.view(b, t, 3, process_height, process_width) * masks_dilated[:, s_f:e_f]
            updated_masks_sub = updated_local_masks_sub.view(b, t, 1, process_height, process_width)
            
            updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
            updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
            torch.cuda.empty_cache()
            
        updated_frames = torch.cat(updated_frames, dim=1)
        updated_masks = torch.cat(updated_masks, dim=1)
    else:
        b, t, _, _, _ = masks_dilated.size()
        prop_imgs, updated_local_masks = inpaint_model.img_propagation(masked_frames, prediction_flows, masks_dilated, 'nearest')
        updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, process_height, process_width) * masks_dilated
        updated_masks = updated_local_masks.view(b, t, 1, process_height, process_width)
        torch.cuda.empty_cache()
    ic(updated_frames.size())
    ic(updated_masks.size())
    
    return updated_frames, updated_masks


def feature_propagation(inpaint_model,
                        updated_frames, 
                        updated_masks, 
                        masks_dilated, 
                        prediction_flows,
                        original_frames,
                        video_length,
                        subvideo_length,
                        neighbor_length,
                        ref_stride,
                        process_size):
    """
    Feature Propagation and Transformation: This is done in a loop where features from neighboring frames are propagated using a model. The result is adjusted for color normalization and combined with original frames to produce the final composited frames.
    """
    process_width, process_height = process_size
    
    comp_frames = [None] * video_length

    neighbor_stride = neighbor_length // 2
    if video_length > subvideo_length:
        ref_num = subvideo_length // ref_stride
    else:
        ref_num = -1
    
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (prediction_flows[0][:, neighbor_ids[:-1], :, :, :], prediction_flows[1][:, neighbor_ids[:-1], :, :, :])
        
        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            
            # pred_img = selected_imgs # results of image propagation
            pred_img = inpaint_model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, process_height, process_width)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + original_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        
        torch.cuda.empty_cache()
        
    ic(type(comp_frames[0]))
    ic(comp_frames[0].shape)
    ic(comp_frames[0].dtype)
    
    
    # For Debugging
    for idx in range(video_length):
        f = comp_frames[idx]
        f = cv2.resize(f, process_size, interpolation = cv2.INTER_CUBIC)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img_save_root = os.path.join("custom_nodes/ComfyUI-ProPainter-Nodes/results", "frames", str(idx).zfill(4)+'.png')
        imwrite(f, img_save_root)
        
    return comp_frames