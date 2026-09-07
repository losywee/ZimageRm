from __future__ import annotations

import os

from PIL import Image


ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_STEPS = 8
ZIMAGE_CFG = 1.0
# DiffSynth skips CFG at cfg_scale==1.0, but diffusers' ZImage pipelines
# enable CFG for any guidance_scale > 0 — so the diffusers path must pass
# 0.0 to stay on the same unguided turbo recipe as the reference.
ZIMAGE_DIFFUSERS_GUIDANCE = 0.0
ZIMAGE_PROMPT = "high quality, sharp, detailed, faithful to the original"
ZIMAGE_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"
# Flux VAE scale 8 -> the diffusers pipeline requires sizes divisible by
# 2x that (16); DiffSynth enforces 16 internally too.
LATENT_GRID = 16


def zimage_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


def vram_limit_gb():
    """VRAM budget for DiffSynth's paging manager: free GiB minus headroom."""
    import torch

    try:
        free, _total = torch.cuda.mem_get_info("cuda")
        return max(1.0, free / 1024**3 - 0.5)
    except Exception:
        return None


def _safe_bf16_supported() -> bool:
    try:
        from ..devices import bf16_supported

        return bf16_supported()
    except Exception:
        return True


_CPU_BF16_CONFIG = None


def _cpu_bf16_config():
    import torch

    from ..devices import bf16_supported

    comp = torch.bfloat16 if bf16_supported() else torch.float16
    return {
        "offload_dtype": comp,
        "offload_device": "cpu",
        "onload_dtype": comp,
        "onload_device": "cpu",
        "preparing_dtype": comp,
        "preparing_device": "cuda",
        "computation_dtype": comp,
        "computation_device": "cuda",
    }


def _disk_stream_config():
    import torch

    from ..devices import bf16_supported

    comp = torch.bfloat16 if bf16_supported() else torch.float16
    return {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": comp,
        "preparing_device": "cuda",
        "computation_dtype": comp,
        "computation_device": "cuda",
    }


class ZImageBackend:
    name = "zimage"

    def __init__(self, device: str = "cuda", hf_token: str | None = None,
                 prefer: str = "diffusers", stream: bool | None = None):
        self.device = device
        self.hf_token = hf_token
        self.prefer = prefer
        # None = auto: disk streaming when the DiffSynth path is selected
        # for a small card (low RAM-safe), CPU bf16 offload otherwise.
        # Disk streaming needs ~10 GB RAM; CPU bf16 needs ~20 GB.
        self.stream = stream
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
        if self.stream is None:
            # Auto: the DiffSynth path is only selected for small cards,
            # where disk streaming is the RAM-safe choice.
            self.stream = True
        return self._load_diffsynth()

    def _load_diffusers(self):
        import torch

        from ..devices import bf16_supported

        try:
            from diffusers import ZImageImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError(
                "this diffusers build has no ZImageImg2ImgPipeline; install "
                "DiffSynth for the streaming path: pip install diffsynth"
            ) from exc

        dtype = torch.bfloat16 if bf16_supported() else torch.float16
        kwargs = {"torch_dtype": dtype}
        if self.hf_token:
            kwargs["token"] = self.hf_token

        pipe = ZImageImg2ImgPipeline.from_pretrained(
            ZIMAGE_MODEL_ID, **kwargs
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

        config = _disk_stream_config() if self.stream else _cpu_bf16_config()
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
            torch_dtype=(
                torch.bfloat16
                if _safe_bf16_supported() else torch.float16
            ),
            device=self.device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(
                model_id=ZIMAGE_MODEL_ID, origin_file_pattern="tokenizer/",
            ),
            vram_limit=vram_limit_gb(),
        )
        # Force materialization of exactly the img2img stack (the pipeline
        # pages per-stage, but an explicit preload avoids unmaterialized
        # meta params slipping through in disk-streaming mode).
        try:
            pipe.load_models_to_device(
                ["text_encoder", "dit", "vae_encoder", "vae_decoder"])
        except Exception:
            pass
        self._pipe = pipe
        self.mode = "diffsynth"
        return self._pipe

    def run(self, image, strength: float, seed: int):
        self._load()
        if self.mode == "diffsynth":
            try:
                return self._run_diffsynth(image, strength, seed)
            except NotImplementedError as exc:
                if "meta" not in str(exc) or self.stream is False:
                    raise
                # Disk streaming hit unmaterialized meta params: rebuild once
                # with CPU bf16 offload (no meta tensors) and retry.
                self.stream = False
                self._pipe = None
                self._load()
                return self._run_diffsynth(image, strength, seed)
        return self._run_diffusers(image, strength, seed)

    def unload(self):
        if self._pipe is None:
            return
        del self._pipe
        self._pipe = None
        self.mode = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
            # Fixed 8 steps like the DiffSynth path and the reference
            # (FACE_STEPS=8): step compensation would break the strength
            # calibration the floors are bound to.
            num_inference_steps=ZIMAGE_STEPS,
            guidance_scale=ZIMAGE_DIFFUSERS_GUIDANCE,
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
