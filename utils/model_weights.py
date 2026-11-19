import os
from pathlib import Path


MODEL_WEIGHTS_ROOT = Path(
    os.getenv("MODEL_WEIGHTS_DIR", Path(__file__).resolve().parents[1] / "weights")
)


def model_weights_dir(*parts: str) -> Path:
    directory = MODEL_WEIGHTS_ROOT.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
