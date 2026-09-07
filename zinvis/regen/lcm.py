from __future__ import annotations

from PIL import Image

LCM_MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
# LCM is distilled for few steps; the img2img schedule window is
# original_steps * strength (50-step base), so num_inference_steps must
# fit inside it. 4 steps with a window clamp covers all legal strengths.
LCM_STEPS = 4
LCM_CFG = 1.0
LCM_PROMPT = "high quality, sharp, detailed, faithful to the original"
LATENT_GRID = 8
LCM_NATIVE_MAX_SIDE = 768


def lcm_steps(strength: float, original_steps: int = 50) -> int:
    """Fixed LCM step count clamped into the strength-sliced window."""
    window = max(1, int(original_steps * float(strength)))
    return max(1, min(LCM_STEPS, window))


def lcm_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


class LcmBackend:
    """SD1.5 Dreamshaper + LCM img2img — the ultra-light diffusion tier
    (~4.3 GB fetch fp32, MIT, ~4 GB VRAM with cpu offload).

    SD1.5 is 512-768 native: larger inputs soften after the round-trip.
    Pass --max-side 768 for best per-pixel fidelity. Disruptor tier —
    weakest diffusion cleaner, intended for weak carriers and pre-passes.
    """

    name = "lcm"

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
        from diffusers import LatentConsistencyModelImg2ImgPipeline

        kwargs = {"torch_dtype": torch.float16}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        # No fp16 variant files in the repo; single load, cast in memory.
        pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
            LCM_MODEL_ID, **kwargs)
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
        working = image
        if max(orig_size) > LCM_NATIVE_MAX_SIDE:
            scale = LCM_NATIVE_MAX_SIDE / max(orig_size)
            working = image.resize(
                (round(orig_size[0] * scale), round(orig_size[1] * scale)),
                Image.Resampling.LANCZOS,
            )
        target = lcm_target_size(*working.size)
        prepared = working if working.size == target else working.resize(
            target, Image.Resampling.LANCZOS
        )
        steps = lcm_steps(
            strength,
            int(getattr(pipe.scheduler.config, "original_inference_steps", 50)
                or 50),
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            prompt=LCM_PROMPT,
            image=prepared,
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=LCM_CFG,
            generator=generator,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
