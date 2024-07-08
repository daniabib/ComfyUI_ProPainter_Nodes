from pathlib import Path
from urllib.parse import urljoin

from torch.hub import download_url_to_file


def load_file_from_url(
    url: str,
    model_dir: Path | None = None,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Load file form http url, will download models if necessary."""
    file_name = Path(file_name)
    model_dir.mkdir(exist_ok=True)
    cached_file = model_dir / file_name
    if not cached_file.exists():
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def download_model(model_url: str, model_name: str) -> str:
    """Downloads a model from a URL and returns the local path to the downloaded model."""
    base_dir = Path(__file__).parents[1].resolve()
    target_dir = base_dir / "weights"
    return load_file_from_url(
        url=urljoin(model_url, model_name),
        model_dir=target_dir,
        progress=True,
        file_name=model_name,
    )
