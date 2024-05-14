import os

from torch import device

from .download_util import load_file_from_url
from ..model.modules.flow_comp_raft import RAFT_bi
from ..model.recurrent_flow_completion import RecurrentFlowCompleteNet
from ..model.propainter import InpaintGenerator

from icecream import ic

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def download_model(model_url: str, model_name: str) -> str:
    """
    Downloads a model from a URL and returns the local path to the downloaded model.
    """
    return load_file_from_url(url=os.path.join(model_url, model_name), 
                                        model_dir='custom_nodes/ComfyUI-ProPainter-Nodes/weights', progress=True, file_name=None)

def load_raft_model(device: device) -> RAFT_bi:
    """
    Loads the RAFT bi-directional model.
    """
    model_path = download_model(pretrain_model_url, 'raft-things.pth')
    raft_model = RAFT_bi(model_path, device)
    return raft_model

def load_recurrent_flow_model(device: device) -> RecurrentFlowCompleteNet:
    """
    Loads the Recurrent Flow Completion Network model.
    """
    model_path = download_model(pretrain_model_url, 'recurrent_flow_completion.pth')
    flow_model = RecurrentFlowCompleteNet(model_path)
    for p in flow_model.parameters():
        p.requires_grad = False
    flow_model.to(device)
    flow_model.eval()
    return flow_model

def load_inpaint_model(device: device) -> InpaintGenerator:
    """
    Loads the Inpaint Generator model.
    """
    model_path = download_model(pretrain_model_url, 'ProPainter.pth')
    ic(model_path)
    inpaint_model = InpaintGenerator(model_path=model_path).to(device)
    inpaint_model.eval()
    return inpaint_model
