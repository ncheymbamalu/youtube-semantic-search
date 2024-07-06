import logging

from datetime import datetime
from pathlib import PosixPath

from src.config import Config

DIRECTORY: PosixPath = Config.Path.LOGS_DIR
DIRECTORY.mkdir(parents=True, exist_ok=True)
FILENAME: str = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logging.basicConfig(
    filename=DIRECTORY / FILENAME,
    format="[%(asctime)s] - %(pathname)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
