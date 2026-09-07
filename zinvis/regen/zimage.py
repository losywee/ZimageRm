from __future__ import annotations

import os

from PIL import Image

from ..profiles import requested_steps

ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_STEPS = 8
ZIMAGE_CFG = 1.0
ZIMAGE_PROMPT = "high quality, sharp, detailed, faithful to the original"
ZIMAGE_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"
LATENT_GRID = 8


def zimage_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


class ZImageBackend:
    name = "zimage"

    def __init__(self, device: str = "cuda", hf_token: str | None = None,
                 prefer: str = "diffusers"):
        self.device = device
        self.hf_token = hf_token
        self.prefer = prefer
        self._pipe = None
        self.mode: str | None = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        if self.prefer != "diffsynth":
            try:
                return self._load_diffusers()
            except ImportError:
                pass
        return self._load_diffsynth()

    def _load_diffusers(self):
        import torch

        try:
            from diffusers import ZImageImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError(
                "this diffusers build has no ZImageImg2ImgPipeline; install "
                "DiffSynth for the streaming path: pip install diffsynth"
            ) from exc

        pipe = ZImageImg2ImgPipeline.from_pretrained(
            ZIMAGE_MODEL_ID, torch_dtype=torch.bfloat16,
        )
        self._pipe = pipe.to(self.device)
        self.mode = "diffusers"
        return self._pipe

    def _load_diffsynth(self):
        import torch

        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
        try:
            from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline
        except ImportError as exc:
            raise RuntimeError(
                "Z-Image streaming mode needs DiffSynth-Studio. Install it "
                "with: pip install diffsynth"
            ) from exc

        config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        model_configs = [
            ModelConfig(
                model_id=ZIMAGE_MODEL_ID,
                origin_file_pattern="transformer/*.safetensors",
                **config,
            ),
            ModelConfig(
                model_id=ZIMAGE_MODEL_ID,
                origin_file_pattern="text_encoder/*.safetensors",
                **config,
            ),
            ModelConfig(
                model_id=ZIMAGE_MODEL_ID,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **config,
            ),
        ]
        pipe = ZImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(
                model_id=ZIMAGE_MODEL_ID, origin_file_pattern="tokenizer/",
            ),
        )
        self._pipe = pipe
        self.mode = "diffsynth"
        return self._pipe

    def run(self, image, strength: float, seed: int):
        self._load()
        if self.mode == "diffsynth":
            return self._run_diffsynth(image, strength, seed)
        return self._run_diffusers(image, strength, seed)

    def _run_diffusers(self, image, strength: float, seed: int):
        import torch

        orig_size = image.size
        target = zimage_target_size(*orig_size)
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self._pipe(
            image=prepared,
            strength=float(strength),
            prompt=ZIMAGE_PROMPT,
            negative_prompt=ZIMAGE_NEGATIVE,
            num_inference_steps=requested_steps(ZIMAGE_STEPS, strength),
            guidance_scale=ZIMAGE_CFG,
            generator=generator,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")

    def _run_diffsynth(self, image, strength: float, seed: int):
        orig_size = image.size
        target = zimage_target_size(*orig_size)
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        result = self._pipe(
            prompt=ZIMAGE_PROMPT,
            negative_prompt=ZIMAGE_NEGATIVE,
            cfg_scale=ZIMAGE_CFG,
            input_image=prepared,
            denoising_strength=float(strength),
            height=target[1],
            width=target[0],
            seed=seed,
            rand_device="cpu",
            num_inference_steps=ZIMAGE_STEPS,
        )
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
