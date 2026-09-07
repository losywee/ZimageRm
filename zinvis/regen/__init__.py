from __future__ import annotations

from .chroma import ChromaBackend
from .lcm import LcmBackend
from .sana import SanaBackend
from .sdxl import SDXLBackend
from .sdxl_canny import SDXLCannyBackend
from .vae import VAEBackend
from .zimage import ZImageBackend

BACKEND_NAMES = ("duo", "chroma", "zimage", "sdxl", "sdxl-canny", "vae",
                 "sana", "lcm")


class DuoBackend:
    """Chroma1 global regeneration followed by a Z-Image Turbo refinement pass."""

    name = "duo"

    def __init__(self, device: str = "cuda", hf_token: str | None = None,
                 refine_strength: float = 0.18, psnr_floor: float = 24.0,
                 low_vram: bool = False, stream: bool | None = None):
        self.chroma = ChromaBackend(device=device, hf_token=hf_token)
        self.zimage = ZImageBackend(
            device=device, hf_token=hf_token,
            prefer="diffsynth" if low_vram else "diffusers",
            stream=stream,
        )
        self.refine_strength = refine_strength
        self.psnr_floor = psnr_floor
        self.low_vram = low_vram

    def run(self, image, strength: float, seed: int):
        from ..io_utils import psnr

        stages: list[str] = []
        out = self.chroma.run(image, strength, seed)
        stages.append(f"chroma(s={strength:.3f},psnr={psnr(image, out):.1f})")

        if self.low_vram:
            self.chroma.unload()
            stages.append("chroma_unloaded(low_vram)")

        refined = self.zimage.run(out, self.refine_strength, seed)
        drop = psnr(image, refined)
        if drop >= self.psnr_floor:
            stages.append(f"zimage_refine(s={self.refine_strength:.3f},psnr={drop:.1f})")
            out = refined
        else:
            stages.append(f"zimage_refine_skipped(psnr={drop:.1f}<{self.psnr_floor:.1f})")
        if self.low_vram:
            self.zimage.unload()
            stages.append("zimage_unloaded(low_vram)")
        out.info.setdefault("zinvis_stages", stages)
        return out


def build_backend(name: str, device: str = "cuda", hf_token: str | None = None,
                  low_vram: bool = False, stream: bool | None = None,
                  **kwargs):
    if name == "chroma":
        return ChromaBackend(device=device, hf_token=hf_token)
    if name == "zimage":
        return ZImageBackend(
            device=device, hf_token=hf_token,
            prefer="diffsynth" if low_vram else "diffusers",
            stream=stream,
        )
    if name == "sdxl":
        return SDXLBackend(device=device, hf_token=hf_token, low_vram=low_vram)
    if name == "sdxl-canny":
        return SDXLCannyBackend(device=device, hf_token=hf_token,
                                low_vram=low_vram)
    if name == "sana":
        return SanaBackend(device=device, hf_token=hf_token, low_vram=low_vram)
    if name == "lcm":
        return LcmBackend(device=device, hf_token=hf_token, low_vram=low_vram)
    if name == "vae":
        return VAEBackend(device=device, hf_token=hf_token)
    if name == "duo":
        return DuoBackend(device=device, hf_token=hf_token,
                          low_vram=low_vram, stream=stream, **kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_NAMES}")
