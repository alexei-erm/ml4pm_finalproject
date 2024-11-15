import torch
import random
import numpy as np


def set_seed(seed: int) -> None:
    """
    Seeds all RNGs for all used libraries
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Set deterministic backend for CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mps_is_available() -> bool:
    """
    Analogous to `torch.cuda.is_available()` but for MPS
    """

    try:
        torch.ones(1).to("mps")
        return True
    except Exception:
        return False


def select_device() -> torch.device:
    """
    Returns the best available device
    """

    if mps_is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return torch.device(device)
