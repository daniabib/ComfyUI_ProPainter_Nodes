# ProPainter Nodes for ComfyUI

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementation of [ProPainter](https://github.com/sczhou/ProPainter) for video inpainting. ProPainter is a framework that utilizes flow-based propagation and spatiotemporal transformer to enable advanced video frame editing for seamless inpainting tasks.

## Features

#### 👨🏻‍🎨 Object Removal
<table>
<tr>
   <td> 
      <img src="https://github.com/daniabib/ComfyUI_ProPainter_Nodes/assets/33937060/10feb7fd-4368-4573-9279-3b08bfff6da6">
   </td>
   <td> 
      <img src="https://github.com/daniabib/ComfyUI_ProPainter_Nodes/assets/33937060/48a42ba6-61de-4ca3-bdda-ad56fa9433d5">
   </td>
</tr>
</table>

#### 🎨 Video Completion
<table>
<tr>
   <td> 
      <img src="https://github.com/daniabib/ComfyUI_ProPainter_Nodes/assets/33937060/25704f66-e86c-4c1b-8c95-02e42b4ee34e">
   </td>
   <td> 
      <img src="https://github.com/daniabib/ComfyUI_ProPainter_Nodes/assets/33937060/5cfa7563-d935-42b6-9feb-c6afe443348e">
   </td>
</tr>
</table>

## Installation
### ComfyUI Manager:
You can use [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) to install the nodes:
1. Search for `ComfyUI ProPainter Nodes` and author `daniabib`. 

### Manual Installation:
1. Clone this repository to `ComfyUI/custom_nodes`:
    ```bash
    git clone https://github.com/daniabib/ComfyUI_ProPainter_Nodes
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Models will be automatically downloaded to the `weights` folder.

## Examples
**Basic Inpainting Workflow**

https://github.com/daniabib/ComfyUI_ProPainter_Nodes/assets/33937060/56244d09-fe89-4af2-916b-e8d903752f0d

https://github.com/daniabib/ComfyUI_ProPainter_Nodes/blob/main/examples/propainter-inpainting-workflow.json

## Others suggested nodes
* [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for loading and saving the video frames.
* [ComfyUI-YoloWorld-EfficientSAM](https://github.com/ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM) for masking images using prompt.

## Nodes Reference 
**🚧 Section under construction**
### ProPainter Inpainting

#### Input Parameters:
- `image`: The video frames to be inpainted.
- `mask`: The mask indicating the regions to be inpainted. Mask must have same size of video frames.
- `width`: Width of the output images. (default: 640).
- `height`: Height of the output images. (default: 360).
- `mask_dilates`: Dilation size for the mask (default: 5).
- `flow_mask_dilates`: Dilation size for the flow mask (default: 8).
- `ref_stride`: Stride for reference frames (default: 10).
- `neighbor_length`: Length of the neighborhood for inpainting (default: 10).
- `subvideo_length`: Length of subvideos for processing (default: 80).
- `raft_iter`): Number of iterations for RAFT model (default: 20).
- `fp16`: Enable or disable FP16 precision (default: "enable").

#### Output:
- `IMAGE`: The inpainted video frames.
- `FLOW_MASK`: The flow mask used during inpainting.
- `MASK_DILATE`: The dilated mask used during inpainting.

### ProPainter Outpainting
**Note**: The authors of the paper didn't mention the outpainting task for their framework, but there is an option for it in the original code. The results aren't very good but I decided to implement a node for it anyway.

#### Input Parameters:
- `image`: The video frames to be outpainted.
- `width`: Width of the video frames (default: 640).
- `height`: Height of the video frames (default: 360).
- `width_scale`: Scale factor for width expansion (default: 1.2).
- `height_scale`: Scale factor for height expansion (default: 1.0).
- `mask_dilates`: Dilation size for the mask (default: 5).
- `flow_mask_dilates`: Dilation size for the flow mask (default: 8).
- `ref_stride`: Stride for reference frames (default: 10).
- `neighbor_length`: Length of the neighborhood for outpainting (default: 10).
- `subvideo_length`: Length of subvideos for processing (default: 80).
- `raft_iter`: Number of iterations for RAFT model (default: 20).
- `fp16`: Enable or disable FP16 precision (default: "disable").

#### Output:
- `IMAGE`: The outpainted video frames.
- `OUTPAINT_MASK`: The mask used during outpainting.
- `output_width`: The width of the outpainted frames.
- `output_height`: The height of the outpainted frames.

## License
The ProPainter models and code are licensed under [NTU S-Lab License 1.0](https://github.com/sczhou/ProPainter/blob/main/LICENSE).
