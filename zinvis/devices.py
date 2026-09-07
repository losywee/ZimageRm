from __future__ import annotations


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def require_cuda() -> str:
    if not cuda_available():
        raise RuntimeError(
            "diffusion regeneration requires a working CUDA device; install "
            "torch with CUDA support and run on an NVIDIA GPU"
        )
    return "cuda"


def torch_dtype():
    import torch

    return torch.bfloat16
