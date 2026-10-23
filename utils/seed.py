import os
import random
import logging
import numpy as np
import torch

logger = logging.getLogger("BreastCancerPipeline")

def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Sets deterministic random seeds across Python, NumPy, and PyTorch for scientific reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, configures cuDNN deterministic flags.

    Returns:
        The configured seed integer.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    logger.info(f"Reproducibility seed set to: {seed} (Deterministic: {deterministic})")
    return seed


def get_device(preference: str = "auto") -> torch.device:
    """
    Determines and returns the torch device based on preference and availability.

    Args:
        preference: "auto", "cuda", or "cpu".

    Returns:
        torch.device instance.
    """
    if preference == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif preference == "cpu":
        device = torch.device("cpu")
    else: # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    logger.info(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    return device
