"""Pick the accelerator used by tests that apply token bitmasks on a device."""

import torch


def get_test_device() -> str:
    """Return "cuda" or "xpu" if available, otherwise "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def synchronize(device: str) -> None:
    """Wait for all kernels on the device to finish. No-op on CPU."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()
