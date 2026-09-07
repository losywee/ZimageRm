from __future__ import annotations


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def vram_gb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except ImportError:
        return None
    return None


def require_cuda() -> str:
    if not cuda_available():
        raise RuntimeError(
            "diffusion regeneration requires a working CUDA device; install "
            "torch with CUDA support and run on an NVIDIA GPU"
        )
    return "cuda"
