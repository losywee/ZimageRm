from __future__ import annotations


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def bf16_supported() -> bool:
    """Turing and older (T4, sm75) have no native bf16; bf16 GEMMs there
    fail with cuBLASLt 'GET was unable to find an engine'. Those cards
    must run fp16."""
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        return bool(torch.cuda.is_bf16_supported())
    except ImportError:
        return True


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
