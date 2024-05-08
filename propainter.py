import os
from typing import List, Tuple
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from .model.modules.flow_comp_raft import RAFT_bi
from .model.recurrent_flow_completion import RecurrentFlowCompleteNet
from .model.propainter import InpaintGenerator
from .utils.download_util import load_file_from_url
from .core.utils import to_tensors
from .model.misc import get_device

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def resize_frames(frames: List[Image.Image], size: Tuple[int, int]) -> Tuple[List[Image.Image], Tuple[int, int]]:    
    process_size = (size[0]-size[0]%8, size[1]-size[1]%8)
    frames = [f.resize(process_size) for f in frames]

    return frames, process_size


def convert_image_to_frames(images: torch.Tensor) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Convert a batch of PyTorch tensors into a list of PIL Image frames and returns the size of the frames.

    Args:
    images (torch.Tensor): A batch of images represented as tensors.

    Returns:
    Tuple[List[Image], Tuple[int, int]]: A tuple containing a list of images converted to PIL Image format and the size (width, height) of the frames.
    """
    frames = []
    for image in images:
        torch_frame = image.detach().cpu()
        np_frame = torch_frame.numpy()
        np_frame = (np_frame * 255).clip(0, 255).astype(np.uint8)
        frame = Image.fromarray(np_frame)
        frames.append(frame)
        
    # TODO: Handle no frames case?
    size = frames[0].size
    
    return frames, size

    
def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    
    return mask


def read_masks(masks, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    for mask in masks:
        torch_mask = mask.detach().cpu()
        print(f'Mask shape: {torch_mask.size()}')
        # torch_mask = torch_mask.permute(1, 0, 2)
        # print(f'Mask shape: {torch_mask.size()}')
        
        np_mask = torch_mask.numpy()
        np_mask = (np_mask * 255).clip(0, 255).astype(np.uint8)
        mask = Image.fromarray(np_mask)
        masks_img.append(mask)

    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated
        
    
def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    
    mask[H_start+dilate_h: H_start+imgH-dilate_h, 
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
  
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


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
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".

        """
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
                "mask_dilation": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 10}), # --mask_dilation
                "ref_stride": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100}), # --ref_stride
                "fp16": (["enable", "disable"],), #--fp16
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"
    # FUNCTION = "propainter_inpainting"

    #OUTPUT_NODE = False

    CATEGORY = "ProPainter"

    def test(self, image: torch.Tensor, mask: torch.Tensor, width, height, mask_dilation, ref_stride, fp16) -> Tuple[torch.Tensor]:
        print(f"""
            image type: {type(image)}
            image size: {image.size()}
            mask type: {type(mask)}
            mask size: {mask.size()}
        """)
        #do some processing on the image, in this example I just invert it
        device = get_device()
        print(device)
        
        resize_ratio = 1.0
        save_fps = 24
        output = 'results'
        mode = 'video_inpainting'
        scale_h = 1.0
        scale_w = 1.2
        raft_iter = 20
        subvideo_length = 80
        neighbor_length = 10
        save_frames = False
        
        # OLD READ FRAME FUNCTION VARIABLES
        video_name = "test"
        
        # Use fp16 precision during inference to reduce running memory cost
        use_half = True if fp16 else False 
        if device == torch.device('cpu'):
            use_half = False
            
        image = 1.0 - image
        
        frames, size = convert_image_to_frames(image)
        print(f"Type of frames: {type(frames)}")
        print(f"Size of frames: {len(frames)}")
        print(f"Type of frames item: {type(frames[0])}")
        print(f"Size of frames item: {frames[0].size}")
        print(f"Channels of frames item: {frames[0].mode}")
        print(f"Size 1: {size}")
        
        frames, size, out_size = resize_frames(frames, size)   
        
        transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)]) 
        
        result_images = [transform(frame) for frame in frames]
        print(f"Type of result_images: {type(result_images)}")
        print(f"Size of result_images item: {result_images[0].size()}")
        
        permuted_images = [image.permute(1, 2, 0) for image in result_images]
        print(f"Size of permuted_images: {len(permuted_images)}")
        print(f"Type of permuted_images: {type(permuted_images)}")
        print(f"Size of permuted_images item: {permuted_images[0].size()}")
        
        
        stack_images = torch.stack(permuted_images, dim=0)
        print(f"Size of stack_images: {stack_images.size()}")
        print(f"Size of stack_images item: {stack_images[0].size()}")
        

        
        return (stack_images,)

    def propainter_inpainting(self, image, mask, width, height, mask_dilation, ref_stride, fp16) -> Tuple[torch.Tensor]:
         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"""
            image type: {type(image)}
            image size: {image.size()}
            mask type: {type(mask)}
            mask size: {mask.size()}
        """)
        
        device = get_device()
        print(device)
        
        # parser = argparse.ArgumentParser()
        # parser.add_argument(
        #     '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
        # parser.add_argument(
        #     '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
        # parser.add_argument( # NOT IMPLEMMENT
        #     '-o', '--output', type=str, default='results', help='Output folder. Default: results')
        # parser.add_argument( # NOT IMPLEMMENT
        #     "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
        # parser.add_argument(
        #     '--height', type=int, default=-1, help='Height of the processing video.')
        # parser.add_argument(
        #     '--width', type=int, default=-1, help='Width of the processing video.')
        # parser.add_argument(
        #     '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
        # parser.add_argument(
        #     "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
        # parser.add_argument( # NOT IMPLEMMENT
        #     "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
        # parser.add_argument( # NOT IMPLEMMENT
        #     "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
        # parser.add_argument( # NOT IMPLEMMENT
        #     "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
        # parser.add_argument( # NOT IMPLEMMENT
        #     '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
        # parser.add_argument( # NOT IMPLEMMENT
        #     '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
        # parser.add_argument( # NOT IMPLEMMENT
        #     '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
        # parser.add_argument( # NOT IMPLEMMENT
        #     '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
        # parser.add_argument( # NOT IMPLEMMENT
        #     '--save_frames', action='store_true', help='Save output frames. Default: False')
        # parser.add_argument(
        #     '--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

        # args = parser.parse_args()
        
        # NOT IMPLEMENTED ARGPARSE VARIABLES
        resize_ratio = 1.0
        save_fps = 24
        output = 'results'
        mode = 'video_inpainting'
        scale_h = 1.0
        scale_w = 1.2
        raft_iter = 20
        subvideo_length = 80
        neighbor_length = 10
        save_frames = False
        
        # OLD READ FRAME FUNCTION VARIABLES
        video_name = "test"
        
        # Use fp16 precision during inference to reduce running memory cost
        use_half = True if fp16 else False 
        if device == torch.device('cpu'):
            use_half = False
        
        frames, size = convert_image_to_frames(image)
        print(f"Type of frames: {type(frames)}")
        print(f"Size of frames: {len(frames)}")
        print(f"Size 1: {size}")
      
        # if not width == -1 and not height == -1:
        #     size = (width, height)
        # if not resize_ratio == 1.0:
        #     size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))

        # print(f"Size 2: {size}")

        frames, size, out_size = resize_frames(frames, size)    

        
        # fps = save_fps if fps is None else 
        
        # save_root = os.path.join(output, video_name)
        # if not os.path.exists(save_root):
        #     os.makedirs(save_root, exist_ok=True)

        if mode == 'video_inpainting':
            frames_len = len(frames)
            flow_masks, masks_dilated = read_masks(mask, frames_len, size, 
                                                flow_mask_dilates=mask_dilation,
                                                mask_dilates=mask_dilation)
            w, h = size
            
        elif mode == 'video_outpainting':
            assert scale_h is not None and scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
            frames, flow_masks, masks_dilated, size = extrapolation(frames, (scale_h, scale_w))
            w, h = size
        else:
            raise NotImplementedError
        
        # for saving the masked frames or video
        masked_frame_for_save = []
        for i in range(len(frames)):
            mask_ = np.expand_dims(np.array(masks_dilated[i]),2).repeat(3, axis=2)/255.
            img = np.array(frames[i])
            green = np.zeros([h, w, 3]) 
            green[:,:,1] = 255
            alpha = 0.6
            # alpha = 1.0
            fuse_img = (1-alpha)*img + alpha*green
            fuse_img = mask_ * fuse_img + (1-mask_)*img
            masked_frame_for_save.append(fuse_img.astype(np.uint8))

        frames_inp = [np.array(f).astype(np.uint8) for f in frames]
        frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
        flow_masks = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

        
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
        video_length = frames.size(1)
        print(f'\nProcessing: {video_name} [{video_length} frames]...')
        with torch.no_grad():
            # ---- compute flow ----
            if frames.size(-1) <= 640: 
                short_clip_len = 12
            elif frames.size(-1) <= 720: 
                short_clip_len = 8
            elif frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2
            
            # use fp32 for RAFT
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                    
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

            
            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
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
            else:
                pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                torch.cuda.empty_cache()
                

            # ---- image propagation ----
            masked_frames = frames * (1 - masks_dilated)
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
                                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                    
                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()
                    
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()
                
        
        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = neighbor_length // 2
        if video_length > subvideo_length:
            ref_num = subvideo_length // ref_stride
        else:
            ref_num = -1
        
        # ---- feature propagation + transformer ----
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
                
                pred_img = pred_img.view(-1, 3, h, w)

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
                    
        # save each frame
        # if save_frames:
        #     for idx in range(video_length):
        #         f = comp_frames[idx]
        #         f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
        #         f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        #         img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4)+'.png')
        #         imwrite(f, img_save_root)
                        

        # if mode == 'video_outpainting':
        #     comp_frames = [i[10:-10,10:-10] for i in comp_frames]
        #     masked_frame_for_save = [i[10:-10,10:-10] for i in masked_frame_for_save]
        
        # save videos frame
        # masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
        # comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
        # imageio.mimwrite(os.path.join(save_root, 'masked_in.mp4'), masked_frame_for_save, fps=fps, quality=7)
        # imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=fps, quality=7)
        
        # print(f'\nAll results are saved in {save_root}')
        print(f"Type of comp_frames: {type(comp_frames)}")
        print(f"Type of comp_frames item: {type(comp_frames[0])}")
        print(f"Size of comp_frames item: {comp_frames[0].shape}")
        # result_images = [torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0) for frame in comp_frames]
        torch_frames = [torch.from_numpy(frame.astype(np.float32) / 255).unsqueeze(0) for frame in comp_frames]
        print(f"Type of torch_frames: {type(torch_frames)}")
        print(f"Type of torch_frames item: {type(torch_frames[0])}")
        print(f"Size of torch_frames item: {torch_frames[0].size()}")
        
        
        result_images = torch.cat(torch_frames, dim=0)
        result_images = torch.permute(result_images, (0, 2, 1, 3))
        print(f"Type of result_images: {type(result_images)}")
        print(f"Size of result_images: {result_images.size()}")
        
        
        
        # print(comp_frames.size())
        torch.cuda.empty_cache()
        
        return (result_images,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ProPainter": ProPainter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPainter": "ProPainter Inpainting"
}
