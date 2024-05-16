import numpy as np
import torch

from comfy import model_management

from .propainter_inference import (
    ProPainterConfig,
    compute_flow,
    complete_flow,
    image_propagation,
    feature_propagation,
)
from .utils.image_utils import (
    resize_images,
    convert_image_to_frames,
    read_masks,
    to_tensors,
)
from .utils.model_utils import (
    load_raft_model,
    load_recurrent_flow_model,
    load_inpaint_model,
)


class ProPainterInpaint:
    """ComfyUI Node for performing inpainting on video frames using the ProPainter model."""

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
    ) -> tuple[torch.Tensor]:
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
        )

        # Use fp16 precision during inference to reduce running memory cost
        use_half = node_config.fp16 == "enable"
        if device == torch.device("cpu"):
            use_half = False

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

        raft_model = load_raft_model(device)
        flow_model = load_recurrent_flow_model(device)
        inpaint_model = load_inpaint_model(device)

        print(f"\nProcessing  {node_config.video_length} frames...")

        with torch.no_grad():
            gt_flows_bi = compute_flow(raft_model, frames, node_config)

            if use_half:
                frames, flow_masks, masks_dilated = (
                    frames.half(),
                    flow_masks.half(),
                    masks_dilated.half(),
                )
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                flow_model = flow_model.half()
                inpaint_model = inpaint_model.half()

            pred_flows_bi = complete_flow(
                flow_model, gt_flows_bi, flow_masks, node_config.subvideo_length
            )

            updated_frames, updated_masks = image_propagation(
                inpaint_model, frames, masks_dilated, pred_flows_bi, node_config
            )

        comp_frames = feature_propagation(
            inpaint_model,
            updated_frames,
            updated_masks,
            masks_dilated,
            pred_flows_bi,
            original_frames,
            node_config,
        )

        output_frames = [
            torch.from_numpy(frame.astype(np.float32) / 255.0) for frame in comp_frames
        ]

        output_frames = torch.stack(output_frames)

        output_flow_masks = flow_masks.squeeze()
        output_masks_dilated = masks_dilated.squeeze()

        return (output_frames, output_flow_masks, output_masks_dilated)


NODE_CLASS_MAPPINGS = {"ProPainterInpaint": ProPainterInpaint}

NODE_DISPLAY_NAME_MAPPINGS = {"ProPainterInpaint": "ProPainter Inpainting"}
