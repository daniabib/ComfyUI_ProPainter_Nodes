import numpy as np
import torch

from comfy import model_management

from .propainter_inference import compute_flow, complete_flow, image_propagation, feature_propagation
from .utils.image_utils import resize_images, convert_image_to_frames, read_masks, to_tensors
from .utils.model_utils import load_raft_model, load_recurrent_flow_model, load_inpaint_model

from icecream import ic


class ProPainter:
    """
    ProPainter Inpainter

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
                    "min": 1,
                    "max": 300}), # --subvideo_length
                "raft_iter": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100}), # --raft_iter
                "fp16": (["enable", "disable"],), #--fp16
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK",)
    RETURN_NAMES = ("IMAGE", "FLOW_MASK", "MASK_DILATE",)
    FUNCTION = "propainter_inpainting"
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
        
        device = model_management.get_torch_device()
        
        # Use fp16 precision during inference to reduce running memory cost
        use_half = True if fp16 == "enable" else False 
        if device == torch.device('cpu'):
            use_half = False

        frames = convert_image_to_frames(image)
        input_size = frames[0].size
        
        
        output_size = (width, height)
        
        frames, process_size = resize_images(frames, input_size, output_size)   
                
        flow_masks, masks_dilated = read_masks(mask,  input_size, output_size, mask.size(dim=0), flow_mask_dilates, mask_dilates)

        
        original_frames = [np.array(f).astype(np.uint8) for f in frames]
        frames: torch.Tensor = to_tensors()(frames).unsqueeze(0) * 2 - 1    
        flow_masks: torch.Tensor = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated: torch.Tensor = to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)
        

        fix_raft = load_raft_model(device)         
        fix_flow_complete = load_recurrent_flow_model(device)
        model = load_inpaint_model(device)
        
        video_length = frames.size(dim=1)
        print(f'\nProcessing  {video_length} frames...')
        
        with torch.no_grad():
            gt_flows_bi = compute_flow(fix_raft, frames, raft_iter, video_length)
            
            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                fix_flow_complete = fix_flow_complete.half()
                model = model.half()
            
            pred_flows_bi = complete_flow(fix_flow_complete, gt_flows_bi, flow_masks, subvideo_length)
                
                
            updated_frames, updated_masks = image_propagation(model, frames, masks_dilated, pred_flows_bi, video_length, subvideo_length, process_size)
           
        comp_frames = feature_propagation(model, updated_frames, updated_masks, masks_dilated, pred_flows_bi, original_frames, video_length, subvideo_length, neighbor_length, ref_stride, process_size)
            
        output_frames = [torch.from_numpy(frame.astype(np.float32) / 255.0) for frame in comp_frames]
        
        output_frames = torch.stack(output_frames)
        
        output_flow_masks = flow_masks.squeeze()
        output_masks_dilated = masks_dilated.squeeze()

        
        return (output_frames, output_flow_masks, output_masks_dilated)
    

NODE_CLASS_MAPPINGS = {
    "ProPainter": ProPainter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPainter": "ProPainter Inpainting"
}
