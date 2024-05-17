import torch

from comfy import model_management

from .propainter_inference import (
    ProPainterConfig,
    process_inpainting,
    feature_propagation,
)
from .utils.image_utils import (
    convert_image_to_frames,
    prepare_frames_and_masks,
    handle_output,
)
from .utils.model_utils import initialize_models


class ProPainterInpaint:
    """ComfyUI Node for performing inpainting on video frames using ProPainter."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # --video
                "mask": ("MASK",),  # --mask
                "width": ("INT", {"default": 640, "min": 0, "max": 2560}),  # --width
                "height": ("INT", {"default": 360, "min": 0, "max": 2560}),  # --height
                "mask_dilates": (
                    "INT",
                    {"default": 5, "min": 0, "max": 100},
                ),  # --mask_dilates
                "flow_mask_dilates": (
                    "INT",
                    {"default": 8, "min": 0, "max": 100},
                ),  # arg dont exist on original code
                "ref_stride": (
                    "INT",
                    {"default": 10, "min": 1, "max": 100},
                ),  # --ref_stride
                "neighbor_length": (
                    "INT",
                    {"default": 10, "min": 2, "max": 300},
                ),  # --neighbor_length
                "subvideo_length": (
                    "INT",
                    {"default": 80, "min": 1, "max": 300},
                ),  # --subvideo_length
                "raft_iter": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100},
                ),  # --raft_iter
                "fp16": (["enable", "disable"],),  # --fp16
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "MASK",
    )
    RETURN_NAMES = (
        "IMAGE",
        "FLOW_MASK",
        "MASK_DILATE",
    )
    FUNCTION = "propainter_inpainting"
    CATEGORY = "ProPainter"

    def propainter_inpainting(
        self,
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
        fp16: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform inpainting on images input using the ProPainter model inference."""
        device = model_management.get_torch_device()

        frames = convert_image_to_frames(image)
        video_length = image.size(dim=0)
        input_size = frames[0].size

        node_config = ProPainterConfig(
            width,
            height,
            mask_dilates,
            flow_mask_dilates,
            ref_stride,
            neighbor_length,
            subvideo_length,
            raft_iter,
            fp16,
            video_length,
            input_size,
            device
        )

        frames, flow_masks, masks_dilated, original_frames = prepare_frames_and_masks(
            frames, mask, node_config, device
        )

        models = initialize_models(device, node_config.fp16)
        print(f"\nProcessing  {node_config.video_length} frames...")

        updated_frames, updated_masks, pred_flows_bi = process_inpainting(
            models,
            frames,
            flow_masks,
            masks_dilated,
            node_config,
        )

        composed_frames = feature_propagation(
            models.inpaint_model,
            updated_frames,
            updated_masks,
            masks_dilated,
            pred_flows_bi,
            original_frames,
            node_config,
        )

        return handle_output(composed_frames, flow_masks, masks_dilated)


NODE_CLASS_MAPPINGS = {"ProPainterInpaint": ProPainterInpaint}

NODE_DISPLAY_NAME_MAPPINGS = {"ProPainterInpaint": "ProPainter Inpainting"}
