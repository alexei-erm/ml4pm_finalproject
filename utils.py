import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set deterministic backend for CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create generator for DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return generator

def mps_is_available():
    """
    a function analogous to `torch.cuda.is_available()` but for MPS
    """
    try:
        torch.ones(1).to('mps')
        return True
    except Exception:
        return False


def device_selection():
    """
    a function to select the device: mps -> cuda -> cpu
    """
    if mps_is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'