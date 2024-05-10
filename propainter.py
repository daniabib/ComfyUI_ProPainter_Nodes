import os
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
from icecream import ic

import torch
from torchvision.transforms.functional import to_pil_image

from .model.modules.flow_comp_raft import RAFT_bi
from .model.recurrent_flow_completion import RecurrentFlowCompleteNet
from .model.propainter import InpaintGenerator
from .utils.download_util import load_file_from_url
from .core.utils import to_tensors
from .model.misc import get_device

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """
    For debugging.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def resize_images(images: list[Image.Image], 
                  input_size: tuple[int, int], 
                  output_size: tuple[int, int]) -> tuple[list[Image.Image], tuple[int, int]]:
    """
    Resizes each image in the list to a new size divisible by 8.

    Returns:
        A list of resized images with dimensions divisible by 8 and process size.
    """    
    process_size = (output_size[0]-output_size[0]%8, output_size[1]-output_size[1]%8)
    ic(process_size)
    
    if process_size != input_size:
        images = [f.resize(process_size) for f in images]

    return images, process_size


def convert_image_to_frames(images: torch.Tensor) -> list[Image.Image]:
    """
    Convert a batch of PyTorch tensors into a list of PIL Image frames 
    
    Args:
    images (torch.Tensor): A batch of images represented as tensors.

    Returns:
    List[Image]: A list of images converted to PIL 
    """
    frames = []
    for image in images:
        torch_frame = image.detach().cpu()
        np_frame = torch_frame.numpy()
        np_frame = (np_frame * 255).clip(0, 255).astype(np.uint8)
        frame = Image.fromarray(np_frame)
        frames.append(frame)
    
    # For Debbuging
    save_root = "custom_nodes/ComfyUI-ProPainter-Nodes/results"
    for i, mask in enumerate(frames):
        mask.save(os.path.join(save_root, 'test_pil_frames', f"pil_frame_{i}.png"))
    
    return frames

def convert_mask_to_frames(images: torch.Tensor) -> list[Image.Image]:
    """
    Convert a batch of PyTorch tensors into a list of PIL Image frames 
    
    Args:
    images (torch.Tensor): A batch of images represented as tensors.

    Returns:
    List[Image.Image]: A list of images converted to PIL 
    """
    frames = []
    for image in images:        
        image = image.detach().cpu()

        # Adjust the scaling based on the data type
        if image.dtype == torch.float32:
            image = (image * 255).clamp(0, 255).byte()

        frame: Image.Image = to_pil_image(image)
        frames.append(frame)
    
    # For Debugging
    save_root = "custom_nodes/ComfyUI-ProPainter-Nodes/results"
    for i, mask in enumerate(frames):
        mask.save(os.path.join(save_root, 'test_pil_masks', f"pil_mask_{i}.png"))
    
    return frames
    
def binary_mask(mask: np.ndarray, 
                th: float = 0.1) -> np.ndarray:
    mask[mask>th] = 1
    mask[mask<=th] = 0
    
    return mask


def read_masks(masks: torch.Tensor, 
               input_size: tuple[int, int], 
               output_size: tuple[int, int], 
               length, 
               flow_mask_dilates=8, 
               mask_dilates=5) -> tuple[list[Image.Image], list[Image.Image]]:
    """
    TODO: Docstring.
    """
    mask_imgs = convert_mask_to_frames(masks)
    mask_imgs, _ = resize_images(mask_imgs, input_size, output_size)
    masks_dilated = []
    flow_masks = []

    for mask_img in mask_imgs:
        mask_img = np.array(mask_img.convert('L'))
        # ic("Initial mask values:")
        # ic( np.unique(mask_img))  
        # ic(mask_img.shape)

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # ic("Flow mask values after dilation:")
        # ic(np.unique(flow_mask_img))
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(mask_imgs) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    # For Debugging
    save_root = "custom_nodes/ComfyUI-ProPainter-Nodes/results"
    for i, mask in enumerate(flow_masks):
        mask.save(os.path.join(save_root, 'mask_frames', f"flow_mask_{i}.png"))

    return flow_masks, masks_dilated


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


class ProPainter:
    """
    ProPainterInpainter

    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), # --video
                "mask": ("MASK",), # --mask
                "width": ("INT",{
                    "default": 640,
                    "min": 0,
                    "max": 2560}), # --width
                "height": ("INT",{
                    "default": 360,
                    "min": 0,
                    "max": 2560}), # --height
                "mask_dilates": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100}), # --mask_dilates
                "flow_mask_dilates": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 100}), # arg dont exist on original code
                "ref_stride": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100}), # --ref_stride
                "neighbor_length": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 300}), # --neighbor_length
                "subvideo_length": ("INT", {
                    "default": 80,
                    "min": 0,
                    "max": 300}), # --subvideo_length
                "raft_iter": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100}), # --raft_iter
                "fp16": (["enable", "disable"],), #--fp16
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK",)
    RETURN_NAMES = ("IMAGE", "FLOW_MASK", "MASK_DILATE",)

    # FUNCTION = "test"
    FUNCTION = "propainter_inpainting"

    #OUTPUT_NODE = False

    CATEGORY = "ProPainter"

    def propainter_inpainting(self, 
                              image: torch.Tensor, 
                              mask: torch.Tensor, 
                              width: int, 
                              height: int,
                              mask_dilates: int,
                              flow_mask_dilates: int, 
                              ref_stride: int,
                              neighbor_length: int,
                              subvideo_length: int,
                              raft_iter: int,
                              fp16) -> tuple[torch.Tensor]:
        ic({type(image)})
        ic({image.size()})
        ic({type(mask)})
        ic({mask.size()})
        
        ic(width)
        ic(height)
        ic(mask_dilates)
        ic(flow_mask_dilates)
        ic(ref_stride)
        ic(neighbor_length)
        ic(fp16)
        
        device = get_device()
        ic(device)
        
        # Use fp16 precision during inference to reduce running memory cost
        use_half = True if fp16 else False 
        if device == torch.device('cpu'):
            use_half = False

        frames = convert_image_to_frames(image)
        input_size = frames[0].size
        
        ic(type(frames))
        ic(len(frames))
        ic(type(frames[0]))
        ic(frames[0].size)
        ic(frames[0].mode)
        ic(input_size)
        
        output_size = (width, height)
        
        frames, process_size = resize_images(frames, input_size, output_size)   
        ic(frames[0].size)
        
        process_width, process_height = process_size
        
        flow_masks, masks_dilated = read_masks(mask,  input_size, output_size, mask.size(dim=0), flow_mask_dilates, mask_dilates)
        # flow_masks, masks_dilated = read_mask_from_tensor(mask, input_size, output_size, mask.size(dim=0))

        ic(type(flow_masks[0]))
        ic(flow_masks[0].size)
        ic(flow_masks[0].mode) # L
        ic(type(masks_dilated[0]))
        ic(masks_dilated[0].size)
        ic(masks_dilated[0].mode) # L
        
        ori_frames = [np.array(f).astype(np.uint8) for f in frames]
        frames: torch.Tensor = to_tensors()(frames).unsqueeze(0) * 2 - 1    
        flow_masks: torch.Tensor = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated: torch.Tensor = to_tensors()(masks_dilated).unsqueeze(0)
        # flow_masks = torch.stack(flow_masks)
        # masks_dilated = torch.stack(masks_dilated)
        frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)
        
        ic("-------- AFTER to_tensor() transformation --------")
        ic( type(frames))
        ic(frames.size())
        ic(type(flow_masks))
        ic(flow_masks.size())
        ic(type(masks_dilated))
        ic(masks_dilated.size())
        
        ##############################################
        # set up RAFT and flow competition model
        ##############################################
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                        model_dir='weights', progress=True, file_name=None)
        fix_raft = RAFT_bi(ckpt_path, device)
        
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                        model_dir='weights', progress=True, file_name=None)
        fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in fix_flow_complete.parameters():
            p.requires_grad = False
        fix_flow_complete.to(device)
        fix_flow_complete.eval()
    
        ##############################################
        # set up ProPainter model
        ##############################################
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                        model_dir='weights', progress=True, file_name=None)
        model = InpaintGenerator(model_path=ckpt_path).to(device)
        model.eval()    
        
        
        ##############################################
        # ProPainter inference
        ##############################################
        video_length = frames.size(dim=1)
        print(f'\nProcessing  {video_length} frames...')
        with torch.no_grad():
            # ---- compute flow ----
            if frames.size(dim=-1) <= 640: 
                short_clip_len = 12
            elif frames.size(dim=-1) <= 720: 
                short_clip_len = 8
            elif frames.size(dim=-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2
        
            # use fp32 for RAFT
            """
            Optical Flow Computation: The fix_raft function is called in batches defined by short_clip_len. If the video has more frames than short_clip_len, it processes the frames in chunks to estimate the forward (flows_f) and backward (flows_b) optical flows. These flows are then concatenated to form gt_flows_f and gt_flows_b.
            """
            if frames.size(dim=1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for chunck in range(0, video_length, short_clip_len):
                    end_f = min(video_length, chunck + short_clip_len)
                    if chunck == 0:
                        flows_f, flows_b = fix_raft(frames[:,chunck:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = fix_raft(frames[:,chunck-1:end_f], iters=raft_iter)
                    
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                    
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = fix_raft(frames, iters=raft_iter)
                torch.cuda.empty_cache()
                
            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                fix_flow_complete = fix_flow_complete.half()
                model = model.half()
                
            ic(gt_flows_f.size())
            ic(gt_flows_b.size())
            
            # ---- complete flow ----
            """
            Complete Flow Computation: Based on the computed flows and subvideo_length, the flows are further processed to generate predicted flows using a model. This involves adjusting for padding and managing frame boundaries.
            """    
            flow_length = gt_flows_bi[0].size(dim=1)
            ic(flow_length)
            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + subvideo_length)
                    pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        flow_masks[:, s_f:e_f+1])
                    pred_flows_bi_sub = fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
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
                pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                
                ic(pred_flows_bi[0].size())
                
                torch.cuda.empty_cache()
                
            ic("-------- AFTER COMPLETE FLOW --------")
            ic(type(frames))
            ic(frames.size())
            ic(type(flow_masks))
            ic(flow_masks.size())
            ic(type(masks_dilated))
            ic(masks_dilated.size())
                
        
            # ---- image propagation ----
            """
            The masked frames are computed by blending original frames and propagated images based on the masks. The process is again segmented if the video is longer than a defined threshold (subvideo_length_img_prop).
            """
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
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                    prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], 
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
                prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, height, width) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, height, width)
                torch.cuda.empty_cache()
            ic(updated_frames.size())
            ic(updated_masks.size())
            

        
        # ---- feature propagation + transformer ----
        """
        Feature Propagation and Transformation: This is done in a loop where features from neighboring frames are propagated using a model. The result is adjusted for color normalization and combined with original frames to produce the final composited frames.
        """
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
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                
                # pred_img = selected_imgs # results of image propagation
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                
                pred_img = pred_img.view(-1, 3, process_height, process_width)

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])
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
            
            
        # output_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in comp_frames]
        output_frames = [torch.from_numpy(frame.astype(np.float32) / 255.0) for frame in comp_frames]
        ic(output_frames[0].size())
        
        output_frames = torch.stack(output_frames)
        ic(output_frames.size())
        ic(output_frames.dtype)
        
        output_flow_masks = flow_masks.squeeze()
        output_masks_dilated = masks_dilated.squeeze()

        ic(output_flow_masks.size())
        ic(output_masks_dilated.size())
        
        # return (stack_images, stack_flow_masks, stack_dilated_masks)
        return (output_frames, output_flow_masks, output_masks_dilated)
    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ProPainter": ProPainter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPainter": "ProPainter Inpainting"
}
