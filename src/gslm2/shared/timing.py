import logging
import shutil
import time
from functools import wraps
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def copy_folder_contents(src_path: Path, dst_path: Path) -> None:
    if not src_path.is_dir():
        raise ValueError(f"Source directory {src_path} does not exist.")
    dst_path.mkdir(parents=True, exist_ok=True)
    for item in src_path.iterdir():
        dest_item = dst_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)


def timing(func):  # noqa: ANN001, ANN201
    """
    Decorator to measure the execution time of a function.
    Args:
        func: The function to be measured.
    Returns:
        A wrapper function that prints the execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug("Function '%s' executed in %.4f seconds", func.__name__, execution_time)
        return result

    return wrapper


def get_free_gpu_memory(prefix: str) -> None:
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties().total_memory - torch.cuda.memory_reserved()
        logger.debug("%s Free GPU Memory: %.2f GB", prefix, free_memory / (1024**3))
    else:
        logger.warning("CUDA is not available.")
