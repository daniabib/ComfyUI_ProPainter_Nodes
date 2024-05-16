import os

from torch import device

from .download_utils import load_file_from_url
from ..model.modules.flow_comp_raft import RAFT_bi
from ..model.recurrent_flow_completion import RecurrentFlowCompleteNet
from ..model.propainter import InpaintGenerator


pretrain_model_url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


def download_model(model_url: str, model_name: str) -> str:
    """Downloads a model from a URL and returns the local path to the downloaded model."""
    return load_file_from_url(
        url=os.path.join(model_url, model_name),
        model_dir="custom_nodes/ComfyUI_ProPainter_Nodes/weights",
        progress=True,
        file_name=None,
    )


def load_raft_model(device: device) -> RAFT_bi:
    """Loads the RAFT bi-directional model."""
    model_path = download_model(pretrain_model_url, "raft-things.pth")
    raft_model = RAFT_bi(model_path, device)
    return raft_model


def load_recurrent_flow_model(device: device) -> RecurrentFlowCompleteNet:
    """Loads the Recurrent Flow Completion Network model."""
    model_path = download_model(pretrain_model_url, "recurrent_flow_completion.pth")
    flow_model = RecurrentFlowCompleteNet(model_path)
    for p in flow_model.parameters():
        p.requires_grad = False
    flow_model.to(device)
    flow_model.eval()
    return flow_model


def load_inpaint_model(device: device) -> InpaintGenerator:
    """Loads the Inpaint Generator model."""
    model_path = download_model(pretrain_model_url, "ProPainter.pth")
    inpaint_model = InpaintGenerator(model_path=model_path).to(device)
    inpaint_model.eval()
    return inpaint_model


def initialize_models(
    device: device,
) -> tuple[RAFT_bi, RecurrentFlowCompleteNet, InpaintGenerator]:
    "Return initialized inference models."
    raft_model = load_raft_model(device)
    flow_model = load_recurrent_flow_model(device)
    inpaint_model = load_inpaint_model(device)
    return raft_model, flow_model, inpaint_model
