from pathlib import Path, PosixPath
from typing import Any

from omegaconf import OmegaConf


class PathConfig:
    PROJECT_DIR: PosixPath = Path(__file__).parent.parent.absolute()
    DATA_DIR: PosixPath = PROJECT_DIR / "data"
    PROCESSED_DATA_PATH: PosixPath = DATA_DIR / "youtube_transcripts.parquet"
    YOUTUBE_DATA_API: str = "https://www.googleapis.com/youtube/v3/search"


def load_config(path: PosixPath = PathConfig.PROJECT_DIR / "config.yaml") -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path))
