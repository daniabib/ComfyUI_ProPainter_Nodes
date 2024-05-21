# 🎨 ProPainter Nodes for ComfyUI

This repository contains custom nodes for ComfyUI implementing ProPainter models for video inpainting and outpainting. The ProPainter models enable advanced video frame editing by leveraging deep learning techniques for seamless inpainting and outpainting tasks.

## ✨ Features

- **Inpainting**: Fill in missing regions in video frames using the ProPainter model.
- **Outpainting**: Expand the boundaries of video frames by synthesizing realistic content beyond the original frame edges.

## 🛠️ Installation

To use the ProPainter nodes in your ComfyUI setup, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/ComfyUI-ProPainter-Nodes.git
    cd ComfyUI-ProPainter-Nodes
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Copy the `propainter_nodes.py` script to your ComfyUI nodes directory:
    ```bash
    cp propainter_nodes.py path/to/comfyui/nodes/
    ```

## 📖 Usage

### 🖌️ ProPainter Inpainting

The ProPainter Inpainting node performs inpainting on video frames, filling in missing regions based on the surrounding content.

#### Input Parameters:
- `image` (🖼️ IMAGE): The video frames to be inpainted.
- `mask` (🎭 MASK): The mask indicating the regions to be inpainted.
- `width` (🔢 INT): Width of the video frames (default: 640).
- `height` (🔢 INT): Height of the video frames (default: 360).
- `mask_dilates` (🔢 INT): Dilation size for the mask (default: 5).
- `flow_mask_dilates` (🔢 INT): Dilation size for the flow mask (default: 8).
- `ref_stride` (🔢 INT): Stride for reference frames (default: 10).
- `neighbor_length` (🔢 INT): Length of the neighborhood for inpainting (default: 10).
- `subvideo_length` (🔢 INT): Length of subvideos for processing (default: 80).
- `raft_iter` (🔢 INT): Number of iterations for RAFT model (default: 20).
- `fp16` (🔀 STRING): Enable or disable FP16 precision (default: "disable").

#### Output:
- `IMAGE` (🖼️): The inpainted video frames.
- `FLOW_MASK` (🎭): The flow mask used during inpainting.
- `MASK_DILATE` (🎭): The dilated mask used during inpainting.

### 🖼️ ProPainter Outpainting

The ProPainter Outpainting node extends the boundaries of video frames, generating new content beyond the original edges.

#### Input Parameters:
- `image` (🖼️ IMAGE): The video frames to be outpainted.
- `width` (🔢 INT): Width of the video frames (default: 640).
- `height` (🔢 INT): Height of the video frames (default: 360).
- `width_scale` (🔢 FLOAT): Scale factor for width expansion (default: 1.2).
- `height_scale` (🔢 FLOAT): Scale factor for height expansion (default: 1.0).
- `mask_dilates` (🔢 INT): Dilation size for the mask (default: 5).
- `flow_mask_dilates` (🔢 INT): Dilation size for the flow mask (default: 8).
- `ref_stride` (🔢 INT): Stride for reference frames (default: 10).
- `neighbor_length` (🔢 INT): Length of the neighborhood for outpainting (default: 10).
- `subvideo_length` (🔢 INT): Length of subvideos for processing (default: 80).
- `raft_iter` (🔢 INT): Number of iterations for RAFT model (default: 20).
- `fp16` (🔀 STRING): Enable or disable FP16 precision (default: "disable").

#### Output:
- `IMAGE` (🖼️): The outpainted video frames.
- `OUTPAINT_MASK` (🎭): The mask used during outpainting.
- `output_width` (🔢 INT): The width of the outpainted frames.
- `output_height` (🔢 INT): The height of the outpainted frames.

## 🧪 Example

Here is an example of how to use the ProPainter Inpainting node:


