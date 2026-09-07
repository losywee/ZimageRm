from __future__ import annotations

from PIL import Image

from ..profiles import requested_steps

SANA_MODEL_ID = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
SANA_EFFECTIVE_STEPS = 2
SANA_CFG = 1.0
SANA_PROMPT = "high quality, sharp, detailed, faithful to the original"
SANA_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"
LATENT_GRID = 32


def sana_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


class SanaBackend:
    """SANA-Sprint 0.6B img2img — the 2-step lightweight tier
    (~7.7 GB fetch: Gemma-2 encoder 5.2 + DiT 1.2 + VAE 1.25; Apache-2.0).

    Floors are uncalibrated defaults (like zimage): pass --strength for
    known-hard watermarks until an oracle calibration exists.
    """

    name = "sana"

    def __init__(self, device: str = "cuda", hf_token: str | None = None,
                 low_vram: bool = False):
        self.device = device
        self.hf_token = hf_token
        self.low_vram = low_vram
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import SanaSprintImg2ImgPipeline

        kwargs = {"torch_dtype": torch.bfloat16}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        try:
            pipe = SanaSprintImg2ImgPipeline.from_pretrained(
                SANA_MODEL_ID, variant="bf16", **kwargs)
        except Exception:
            pipe = SanaSprintImg2ImgPipeline.from_pretrained(
                SANA_MODEL_ID, **kwargs)
        if self.low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        self._pipe = pipe
        return self._pipe

    def unload(self):
        if self._pipe is None:
            return
        del self._pipe
        self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def run(self, image, strength: float, seed: int):
        import torch

        pipe = self._load()
        orig_size = image.size
        target = sana_target_size(*orig_size)
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        steps = requested_steps(SANA_EFFECTIVE_STEPS, strength)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            prompt=SANA_PROMPT,
            image=prepared,
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=SANA_CFG,
            generator=generator,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
