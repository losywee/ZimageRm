from __future__ import annotations

from PIL import Image

from ..profiles import requested_steps

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_LORA = "sdxl_lightning_4step_lora.safetensors"
SDXL_EFFECTIVE_STEPS = 4
SDXL_CFG = 1.0
SDXL_MIN_STRENGTH = 0.15
SDXL_PROMPT = "high quality, sharp, detailed, faithful to the original"
SDXL_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"
LATENT_GRID = 8


def sdxl_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


def peft_torchao_compat() -> None:
    """Silence peft's hard torchao gate.

    peft >= 0.20 raises ImportError when torchao < 0.16.0 is installed
    (Colab ships 0.10). zinvis never uses torchao quantization, so if an
    old torchao is present we stub peft's availability check to False —
    the same outcome as not having torchao installed at all.
    """
    try:
        import importlib.metadata as im

        import peft.import_utils as iu
    except Exception:
        return
    try:
        ver = im.version("torchao")
        parts = tuple(int(p) for p in ver.split(".")[:2])
    except Exception:
        return
    if parts < (0, 16):
        iu.is_torchao_available = lambda: False


class SDXLBackend:
    """SDXL-base + SDXL-Lightning 4-step LoRA img2img — the calibrated
    lightweight global pass (floors: google 0.25 / openai 0.15 / unknown 0.25).

    Auto-calibrated boundary data comes from the published sdxl-zimage oracle
    tests (fixed seed 0, 4 effective steps, CFG 1.0, neutral prompt). The
    floors are bound to exactly this configuration — changing any of them
    invalidates the calibration.
    """

    name = "sdxl"

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
        peft_torchao_compat()
        from diffusers import (
            AutoPipelineForImage2Image,
            EulerDiscreteScheduler,
        )

        from ..devices import bf16_supported

        dtype = torch.bfloat16 if bf16_supported() else torch.float16
        kwargs = {"torch_dtype": dtype}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                SDXL_MODEL_ID, variant="fp16", **kwargs,
            )
        except Exception:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                SDXL_MODEL_ID, **kwargs,
            )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        peft_torchao_compat()
        pipe.load_lora_weights(
            SDXL_LIGHTNING_MODEL_ID,
            weight_name=SDXL_LIGHTNING_LORA,
        )
        if self.low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        self._pipe = pipe
        return self._pipe

    def run(self, image, strength: float, seed: int):
        import torch

        pipe = self._load()
        # Clamp to the distillation floor: below it the executed timesteps
        # fall outside Lightning's trained sigma range and output noise.
        strength = max(float(strength), SDXL_MIN_STRENGTH)
        orig_size = image.size
        target = sdxl_target_size(*orig_size)
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        steps = requested_steps(SDXL_EFFECTIVE_STEPS, strength)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            prompt=SDXL_PROMPT,
            negative_prompt=SDXL_NEGATIVE,
            image=prepared,
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=SDXL_CFG,
            generator=generator,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
