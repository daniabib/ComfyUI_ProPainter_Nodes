import torch
import scipy
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from numpy.typing import NDArray

from ..propainter_inference import ProPainterConfig


class Stack:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group) -> NDArray:
        mode = img_group[0].mode
        if mode == "1":
            img_group = [img.convert("L") for img in img_group]
            mode = "L"
        if mode == "L":
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == "RGB":
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor:
    """Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]."""

    # TODO: Check how this function is working with the comfy workflow.
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic) -> torch.Tensor:
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


def resize_images(
    images: list[Image.Image], config: ProPainterConfig
) -> list[Image.Image]:
    """Resizes each image in the list to a new size divisible by 8."""
    if config.process_size != config.input_size:
        images = [f.resize(config.process_size) for f in images]

    return images


def convert_image_to_frames(images: torch.Tensor) -> list[Image.Image]:
    """Convert a batch of PyTorch tensors into a list of PIL Image frames."""
    frames = []
    for image in images:
        torch_frame = image.detach().cpu()
        np_frame = torch_frame.numpy()
        np_frame = (np_frame * 255).clip(0, 255).astype(np.uint8)
        frame = Image.fromarray(np_frame)
        frames.append(frame)

    return frames


def binary_mask(mask: np.ndarray, th: float = 0.1) -> np.ndarray:
    mask[mask > th] = 1
    mask[mask <= th] = 0

    return mask


def convert_mask_to_frames(images: torch.Tensor) -> list[Image.Image]:
    """Convert a batch of PyTorch tensors into a list of PIL Image frames."""
    frames = []
    for image in images:
        image = image.detach().cpu()

        # Adjust the scaling based on the data type
        if image.dtype == torch.float32:
            image = (image * 255).clamp(0, 255).byte()

        frame: Image.Image = to_pil_image(image)
        frames.append(frame)

    return frames


def read_masks(
    masks: torch.Tensor, config: ProPainterConfig
) -> tuple[list[Image.Image], list[Image.Image]]:
    """TODO: Docstring."""
    mask_imgs = convert_mask_to_frames(masks)
    mask_imgs = resize_images(mask_imgs, config)
    masks_dilated = []
    flow_masks = []

    for mask_img in mask_imgs:
        mask_img = np.array(mask_img.convert("L"))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if config.flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(
                mask_img, iterations=config.flow_mask_dilates
            ).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))

        if config.mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(
                mask_img, iterations=config.mask_dilates
            ).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))

    if len(mask_imgs) == 1:
        flow_masks = flow_masks * config.length
        masks_dilated = masks_dilated * config.length

    return flow_masks, masks_dilated


def to_tensors():
    return transforms.Compose([Stack(), ToTorchFormatTensor()])


def prepare_frames_and_masks(frames, mask, node_config, device):
    frames = resize_images(frames, node_config)

    flow_masks, masks_dilated = read_masks(mask, node_config)

    original_frames = [np.array(f).astype(np.uint8) for f in frames]
    frames: torch.Tensor = to_tensors()(frames).unsqueeze(0) * 2 - 1
    flow_masks: torch.Tensor = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated: torch.Tensor = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = (
        frames.to(device),
        flow_masks.to(device),
        masks_dilated.to(device),
    )
    return frames, flow_masks, masks_dilated, original_frames


def handle_output(composed_frames, flow_masks, masks_dilated):
    output_frames = [
        torch.from_numpy(frame.astype(np.float32) / 255.0) for frame in composed_frames
    ]

    output_frames = torch.stack(output_frames)

    output_flow_masks = flow_masks.squeeze()
    output_masks_dilated = masks_dilated.squeeze()
    
    return output_frames, output_flow_masks, output_masks_dilated