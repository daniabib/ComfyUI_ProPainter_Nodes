import os
from urllib.parse import urlparse

from torch.hub import download_url_to_file, get_dir


def load_file_from_url(
    url: str,
    model_dir: str | None = None,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Load file form http url, will download models if necessary."""
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def download_model(model_url: str, model_name: str) -> str:
    """Downloads a model from a URL and returns the local path to the downloaded model."""
    return load_file_from_url(
        url=os.path.join(model_url, model_name),
        model_dir="custom_nodes/ComfyUI_ProPainter_Nodes/weights",
        progress=True,
        file_name=None,
    )
