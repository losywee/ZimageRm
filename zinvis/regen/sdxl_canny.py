from __future__ import annotations

import numpy as np
from PIL import Image

from ..profiles import requested_steps
from .sdxl import (
    SDXL_LIGHTNING_LORA,
    SDXL_LIGHTNING_MODEL_ID,
    SDXL_MODEL_ID,
    SDXL_CFG,
    SDXL_MIN_STRENGTH,
    peft_torchao_compat,
)

CONTROLNET_CANNY_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
SDXL_CANNY_CONTROL_SCALE = 0.5
SDXL_CANNY_PROMPT = "high quality, sharp, detailed, faithful to the original"
SDXL_CANNY_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"


def canny_image(image: Image.Image) -> Image.Image:
    """White-on-black edge map of the input (glyph geometry signal).

    Uses OpenCV when installed (Colab ships it); falls back to a
    dependency-free numpy Sobel edge map otherwise.
    """
    gray = np.asarray(image.convert("L"))
    try:
        import cv2

        edges = cv2.Canny(gray, 100, 200)
    except Exception:
        gy, gx = np.gradient(gray.astype(np.float32))
        mag = np.hypot(gx, gy)
        thr = max(float(np.percentile(mag, 88)), 8.0)
        edges = (mag >= thr).astype(np.uint8) * 255
    return Image.fromarray(edges, "L").convert("RGB")


class SDXLCannyBackend:
    """SDXL-base + SDXL-Lightning 4-step LoRA + Canny ControlNet img2img.

    The edge map pins glyph geometry through regeneration, so small text
    stays legible while pixels are still re-synthesized (the edge map
    carries no watermark payload). Uncalibrated floors — structure
    conditioning changes the disruption calibration.
    """

    name = "sdxl-canny"

    def __init__(self, device: str = "cuda", hf_token: str | None = None,
                 low_vram: bool = False, control_scale: float = 0.5):
        self.device = device
        self.hf_token = hf_token
        self.low_vram = low_vram
        self.control_scale = control_scale
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import (
            ControlNetModel,
            EulerDiscreteScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )

        from ..devices import bf16_supported

        dtype = torch.bfloat16 if bf16_supported() else torch.float16
        kwargs = {"torch_dtype": dtype}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        try:
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_CANNY_MODEL_ID, variant="fp16", **kwargs)
        except Exception:
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_CANNY_MODEL_ID, **kwargs)
        try:
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                SDXL_MODEL_ID, controlnet=controlnet, variant="fp16",
                **kwargs)
        except Exception:
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                SDXL_MODEL_ID, controlnet=controlnet, **kwargs)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config)
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
        # Same distillation floor as sdxl: below it the executed timesteps
        # fall outside Lightning's trained sigma range and output noise.
        strength = max(float(strength), SDXL_MIN_STRENGTH)
        orig_size = image.size
        grid = 8

        def _grid(v):
            return max(grid, (v // grid) * grid)

        target = (_grid(orig_size[0]), _grid(orig_size[1]))
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        control = canny_image(prepared)
        steps = requested_steps(4, strength)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            prompt=SDXL_CANNY_PROMPT,
            negative_prompt=SDXL_CANNY_NEGATIVE,
            image=prepared,
            control_image=control,
            controlnet_conditioning_scale=self.control_scale,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=SDXL_CFG,
            generator=generator,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
