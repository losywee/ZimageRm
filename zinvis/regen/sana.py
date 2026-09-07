from __future__ import annotations

from PIL import Image

SANA_MODEL_ID = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
# SCM is hard-locked to exactly 2 steps (check_inputs raises otherwise);
# strength scales the denoising fraction over that fixed schedule
# (<0.5 -> 1 of 2 steps, >=0.5 -> both).
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

    Guidance is 1.0 (below the pipeline default 4.5) to stay faithful to
    the input frame. Floors are uncalibrated defaults (like zimage):
    pass --strength for known-hard watermarks until an oracle
    calibration exists. Note the 2-step SCM lock: strengths below 0.5
    run only one of the two steps.
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

        from ..devices import bf16_supported

        # Weights are fp16-native; bf16 only where the GPU supports it
        # (T4/sm75 bf16 GEMMs fail with "GET was unable to find an engine").
        dtype = torch.bfloat16 if bf16_supported() else torch.float16
        kwargs = {"torch_dtype": dtype}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        # No bf16 variant files in the repo (fp16 weights); single load.
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
        except Exception:
            pass

    def run(self, image, strength: float, seed: int):
        import torch

        pipe = self._load()
        orig_size = image.size
        target = sana_target_size(*orig_size)
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        steps = SANA_EFFECTIVE_STEPS
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            prompt=SANA_PROMPT,
            image=prepared,
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=SANA_CFG,
            generator=generator,
            # Work at the input's native size: binning would snap to a
            # 1024-bin and squarify non-square frames.
            height=target[1],
            width=target[0],
            use_resolution_binning=False,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
