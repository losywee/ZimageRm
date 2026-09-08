from __future__ import annotations

from PIL import Image

SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
# Full-schedule SD1.5 img2img: fixed 30 steps, strength slices the
# executed window (0.3 -> ~9 steps). Guidance 7.5 is the SD1.5 default.
SD15_STEPS = 30
SD15_CFG = 7.5
SD15_PROMPT = "high quality, sharp, detailed, faithful to the original"
SD15_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"
LATENT_GRID = 8
SD15_NATIVE_MAX_SIDE = 768


def sd15_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


class SD15Backend:
    """SD1.5 full-schedule img2img — the thorough lightweight tier
    (~2.2 GB fetch fp16, ~3 GB VRAM with cpu offload).

    Unlike the distilled tiers, this runs a real multi-step schedule, so
    strength maps smoothly to disruption depth. Slow (~10x lcm per image)
    but the deepest clean available under 3 GB. Floors are uncalibrated
    defaults: pass --strength for known-hard watermarks.
    """

    name = "sd15"

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
        from diffusers import StableDiffusionImg2ImgPipeline

        from ..devices import bf16_supported

        dtype = torch.bfloat16 if bf16_supported() else torch.float16
        kwargs = {"torch_dtype": dtype}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                SD15_MODEL_ID, variant="fp16", **kwargs)
        except Exception:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                SD15_MODEL_ID, **kwargs)
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
        except Exception:
            pass

    def run(self, image, strength: float, seed: int):
        import torch

        pipe = self._load()
        orig_size = image.size
        working = image
        if max(orig_size) > SD15_NATIVE_MAX_SIDE:
            scale = SD15_NATIVE_MAX_SIDE / max(orig_size)
            working = image.resize(
                (round(orig_size[0] * scale), round(orig_size[1] * scale)),
                Image.Resampling.LANCZOS,
            )
        target = sd15_target_size(*working.size)
        prepared = working if working.size == target else working.resize(
            target, Image.Resampling.LANCZOS
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            prompt=SD15_PROMPT,
            negative_prompt=SD15_NEGATIVE,
            image=prepared,
            strength=float(strength),
            num_inference_steps=SD15_STEPS,
            guidance_scale=SD15_CFG,
            generator=generator,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
