import os
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
from icecream import ic

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

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


def resize_images(images: list[Image.Image], 
                  input_size: tuple[int, int], 
                  output_size: tuple[int, int]) -> list[Image.Image]:
    """
    Resizes each image in the list to a new size divisible by 8.

    Returns:
        List[Image.Image]: A list of resized images with dimensions divisible by 8.
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
    
    # if images.dtype == torch.float32:
    #     # Convert from 0-1 to 0-255 range if necessary
    #     video_tensor = (images * 255).byte()
    #     ic(video_tensor.size())
    # # Convert tensor to list of PIL Images for compatibility with the rest of your pipeline
    # frames = [Image.fromarray(frame.numpy().transpose(1, 2, 0)) for frame in video_tensor]
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
        # if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for channel-first format
        ic("image before permute:", image.size())
        # image = image.permute(1, 2, 0)  # Convert to H x W x C for PIL
        # ic("image after permute:", image.size())
        
        image = image.detach().cpu()

        # Adjust the scaling based on the data type
        if image.dtype == torch.float32:
            image = (image * 255).clamp(0, 255).byte()  # Scale float images from 0-1 to 0-255

        image = 255 - image
        frame = to_pil_image(image)
        frames.append(frame)
        
    save_root = "custom_nodes/ComfyUI-ProPainter-Nodes/results"
    for i, mask in enumerate(frames):
        mask.save(os.path.join(save_root, 'test_pil_masks', f"pil_mask_{i}.png"))
    
    return frames
    
def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    
    return mask

def read_mask_from_tensor(mask_tensor, size, flow_mask_dilates=8, mask_dilates=5):
    # Resize masks if needed, ensure correct data type and apply dilations
    resized_masks = []
    flow_masks = []
    masks_dilated = []
    
    # mask_tensor = mask_tensor.permute(0, 3, 1, 2)
    # Handle tensor resizing and dilation
    for mask in mask_tensor:
        if size:
            # Resize mask to target size
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0).float().unsqueeze(0),  # Add batch and channel dimension
                                                   size=size, mode='nearest').squeeze()  # Remove batch and channel dimension after resize
        mask_np = mask.cpu().numpy().astype(np.uint8)  # Ensure it's numpy array of type uint8

        # Dilate masks for flow calculations
        if flow_mask_dilates > 0:
            flow_mask_np = scipy.ndimage.binary_dilation(mask_np, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_np = binary_mask(mask_np)
        flow_masks.append(torch.from_numpy(flow_mask_np))

        # Dilate masks for general mask usage
        if mask_dilates > 0:
            mask_dilated_np = scipy.ndimage.binary_dilation(mask_np, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_dilated_np = binary_mask(mask_np)
        masks_dilated.append(torch.from_numpy(mask_dilated_np))
        
    save_root = "custom_nodes/ComfyUI-ProPainter-Nodes/results"
    for i, mask in enumerate(flow_masks):
        mask.save(os.path.join(save_root, 'mask_frames', f"flow_mask_{i}.png"))

    return flow_masks, masks_dilated

def read_masks(masks: torch.Tensor, 
               input_size: tuple[int, int], 
               output_size: tuple[int, int], 
               length, 
               flow_mask_dilates=8, 
               mask_dilates=5) -> tuple[list[Image.Image], list[Image.Image]]:
    """
    TODO: Mask image is inverted and diffent from input.
    """
    mask_imgs = convert_mask_to_frames(masks)
    mask_imgs, _ = resize_images(mask_imgs, input_size, output_size)
    masks_dilated = []
    flow_masks = []

    for mask_img in mask_imgs:
        # if size is not None:
        #     mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))
        ic("Initial mask values:")
        ic( np.unique(mask_img))  
        ic(mask_img.shape)

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        ic("Flow mask values after dilation:")
        ic(np.unique(flow_mask_img))
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(mask_imgs) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

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
                "flow_mask_dilates": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 10}), # --flow_mask_dilates
                "ref_stride": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100}), # --ref_stride
                "fp16": (["enable", "disable"],), #--fp16
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "FLOW_MASK", "MASK_DILATE")

    FUNCTION = "test"
    # FUNCTION = "propainter_inpainting"

    #OUTPUT_NODE = False

    CATEGORY = "ProPainter"

    def test(self, image: torch.Tensor, mask: torch.Tensor, width, height, flow_mask_dilates, ref_stride, fp16) -> tuple[torch.Tensor]:
        print(f"""
            image type: {type(image)}
            image size: {image.size()}
            mask type: {type(mask)}
            mask size: {mask.size()}
        """)
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
            
        # image = 1.0 - image
        
        # frames = convert_image_to_frames(image.permute(0, 3, 1, 2))
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
        print(f"Size of resized frame: {frames[0].size}")
        
        process_width, process_height = process_size
        
        flow_masks, masks_dilated = read_masks(mask,  input_size, output_size, mask.size(dim=0), flow_mask_dilates)
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
        
        ### OUTPUT HANDLING
        # transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)]) 
        
        # tensor_images = [transform(frame) for frame in frames]
        # print(f"Type of tensor_images: {type(tensor_images)}")
        # print(f"Size of tensor_images item: {tensor_images[0].size()}")
        
        # permuted_images = [image.permute(1, 2, 0) for image in tensor_images]
        # print(f"Size of permuted_images: {len(permuted_images)}")
        # print(f"Type of permuted_images: {type(permuted_images)}")
        # print(f"Size of permuted_images item: {permuted_images[0].size()}")
        
        # stack_images = torch.stack(permuted_images, dim=0)
        # print(f"Size of stack_images: {stack_images.size()}")
        # print(f"Size of stack_images item: {stack_images[0].size()}")
        
        # tensor_flow_mask = [transform(frame_mask) for frame_mask in flow_masks]
        # print(f"Type of tensor_flow_mask: {type(tensor_flow_mask)}")
        # print(f"Size of tensor_flow_mask item: {tensor_flow_mask[0].size()}")
        
        # stack_flow_masks = torch.stack(tensor_flow_mask, dim=0)
        # print(f"Size of stack_flow_masks: {stack_flow_masks.size()}")
        # print(f"Size of stack_flow_masks item: {stack_flow_masks[0].size()}")
        
        # tensor_mask_dilated = [transform(frame_mask) for frame_mask in masks_dilated]
        # print(f"Type of tensor_mask_dilated: {type(tensor_mask_dilated)}")
        # print(f"Size of tensor_mask_dilated item: {tensor_mask_dilated[0].size()}")
        
        # stack_dilated_masks = torch.stack(tensor_mask_dilated, dim=0)
        # print(f"Size of stack_dilated_masks: {stack_dilated_masks.size()}")
        # print(f"Size of stack_dilated_masks item: {stack_dilated_masks[0].size()}")
        
        # output_frames = frames.squeeze().permute(0, 2, 3, 1)
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
