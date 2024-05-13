import torch
import scipy
import numpy as np
from torchvision.transforms.functional import to_pil_image
from PIL import Image

# For Debugging
import os
from icecream import ic

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

def binary_mask(mask: np.ndarray, 
                th: float = 0.1) -> np.ndarray:
    mask[mask>th] = 1
    mask[mask<=th] = 0
    
    return mask

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