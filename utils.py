from pathlib import Path
import logging

import numpy as np

import imageio.v3 as iio
from torchvision import tv_tensors
MAX_8BIT = 255

def validate_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    else:
        raise TypeError(f"Provided path {path_str} does not exist")

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # Create a console handler
        console_handler = logging.StreamHandler()

        # Define and set formatter with clean timestamp
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def read_8bit_image(path:Path):
    image = iio.imread(path)
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)

    image = image.astype(np.float32)
    image = image/255
    image = image.transpose(2, 0, 1)
    return tv_tensors.Image(image)
