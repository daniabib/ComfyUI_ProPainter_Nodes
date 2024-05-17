from dataclasses import dataclass

import torch

from ..model.modules.flow_comp_raft import RAFT_bi
from ..model.propainter import InpaintGenerator
from ..model.recurrent_flow_completion import RecurrentFlowCompleteNet
from .download_utils import download_model


@dataclass
class Models:
    raft_model: RAFT_bi
    flow_model: RecurrentFlowCompleteNet
    inpaint_model: InpaintGenerator


PRETRAIN_MODEL_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


def load_raft_model(device: torch.device) -> RAFT_bi:
    """Loads the RAFT bi-directional model."""
    model_path = download_model(PRETRAIN_MODEL_URL, "raft-things.pth")
    raft_model = RAFT_bi(model_path, device)
    return raft_model


def load_recurrent_flow_model(device: torch.device) -> RecurrentFlowCompleteNet:
    """Loads the Recurrent Flow Completion Network model."""
    model_path = download_model(PRETRAIN_MODEL_URL, "recurrent_flow_completion.pth")
    flow_model = RecurrentFlowCompleteNet(model_path)
    for p in flow_model.parameters():
        p.requires_grad = False
    flow_model.to(device)
    flow_model.eval()
    return flow_model


def load_inpaint_model(device: torch.device) -> InpaintGenerator:
    """Loads the Inpaint Generator model."""
    model_path = download_model(PRETRAIN_MODEL_URL, "ProPainter.pth")
    inpaint_model = InpaintGenerator(model_path=model_path).to(device)
    inpaint_model.eval()
    return inpaint_model


def initialize_models(device: torch.device, use_half: str) -> Models:
    """Return initialized inference models."""
    raft_model = load_raft_model(device)
    flow_model = load_recurrent_flow_model(device)
    inpaint_model = load_inpaint_model(device)

    if use_half == "enable":
        # raft_model = raft_model.half()
        flow_model = flow_model.half()
        inpaint_model = inpaint_model.half()
    return Models(raft_model, flow_model, inpaint_model)
