from __future__ import annotations

from PIL import Image

from ..profiles import strength_safe_steps

SDTURBO_MODEL_ID = "stabilityai/sd-turbo"
# SD-Turbo is distilled for 1-4 unguided steps; img2img runs a fixed
# 2-step schedule and strength slices how many of them execute
# (<0.5 -> 1 of 2 steps, >=0.5 -> both).
SDTURBO_STEPS = 2
SDTURBO_CFG = 0.0
SDTURBO_PROMPT = "high quality, sharp, detailed, faithful to the original"
LATENT_GRID = 8
SDTURBO_NATIVE_MAX_SIDE = 768


def sdturbo_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


class SdTurboBackend:
    """SD2.1-Turbo 2-step img2img — the few-step lightweight tier
    (~2.6 GB fetch fp16, ~4 GB VRAM with cpu offload).

    Guidance is 0.0 (SD-Turbo convention: no CFG). Floors are
    uncalibrated defaults (like zimage): pass --strength for known-hard
    watermarks until an oracle calibration exists. Note the 2-step lock:
    strengths below 0.5 run only one of the two steps.
    """

    name = "sd-turbo"

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
        from diffusers import AutoPipelineForImage2Image

        from ..devices import bf16_supported

        dtype = torch.bfloat16 if bf16_supported() else torch.float16
        kwargs = {"torch_dtype": dtype}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                SDTURBO_MODEL_ID, variant="fp16", **kwargs)
        except Exception:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                SDTURBO_MODEL_ID, **kwargs)
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
        if max(orig_size) > SDTURBO_NATIVE_MAX_SIDE:
            scale = SDTURBO_NATIVE_MAX_SIDE / max(orig_size)
            working = image.resize(
                (round(orig_size[0] * scale), round(orig_size[1] * scale)),
                Image.Resampling.LANCZOS,
            )
        target = sdturbo_target_size(*working.size)
        prepared = working if working.size == target else working.resize(
            target, Image.Resampling.LANCZOS
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        steps = strength_safe_steps(SDTURBO_STEPS, strength)
        result = pipe(
            prompt=SDTURBO_PROMPT,
            image=prepared,
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=SDTURBO_CFG,
            generator=generator,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
