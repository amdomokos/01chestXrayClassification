import random

import numpy as np
import torch


def get_device(prefer=None):
    """Pick the best available device: Intel XPU > CUDA > CPU (or honor `prefer`)."""
    if prefer and prefer != "auto":
        dev = torch.device(prefer)
        if dev.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU requested but torch.xpu is not available (need the +xpu PyTorch build).")
        return dev
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device):
    if device.type == "xpu":
        return torch.xpu.get_device_name(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
