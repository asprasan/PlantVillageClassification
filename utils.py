from pathlib import Path
import logging

def validate_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    else:
        raise TypeError(f"Provided path {path_str} does not exist")

def get_logger(name):
    logger = logging.Logger(name)
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