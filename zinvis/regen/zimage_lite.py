from __future__ import annotations

from PIL import Image

ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_GGUF_REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF_FILES = {
    "q8": "z-image-turbo-Q8_0.gguf",
    "q4": "z-image-turbo-Q4_K_M.gguf",
}
ZIMAGE_LITE_STEPS = 8
ZIMAGE_LITE_CFG = 1.0
ZIMAGE_LITE_PROMPT = "high quality, sharp, detailed, faithful to the original"
# Same 16-divisibility requirement as zimage (Flux VAE scale 8 x 2).
LATENT_GRID = 16


def zimage_lite_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


class ZImageLiteBackend:
    """Z-Image Turbo with a GGUF-quantized transformer (unsloth) — the
    lightweight zimage tier: ~15.4 GB fetch (Q8) vs ~33 GB for the
    official fp32 repo, ~8 GB VRAM.

    The watermark prompt is a fixed constant, so the 8 GB Qwen3 text
    encoder is loaded, encodes once, and is freed BEFORE the GGUF
    transformer loads — peak system RAM stays ~8 GB (fits free Colab).
    Same 8-step schedule and uncalibrated floors as zimage.
    """

    name = "zimage-lite"

    def __init__(self, device: str = "cuda", hf_token: str | None = None,
                 gguf: str = "q8"):
        self.device = device
        self.hf_token = hf_token
        gguf = (gguf or "q8").lower()
        if gguf not in GGUF_FILES:
            raise ValueError(f"unknown gguf preset {gguf!r}; "
                             f"choose from {sorted(GGUF_FILES)}")
        self.gguf = gguf
        self._pipe = None
        self._embeds = None

    def _encode_once(self, tokenizer, text_encoder, torch, device):
        """Replicates ZImageImg2ImgPipeline._encode_prompt for the fixed
        watermark prompt (single sample, chat template, hidden_states[-2])."""
        messages = [{"role": "user", "content": ZIMAGE_LITE_PROMPT}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer([text], padding="max_length", max_length=512,
                           truncation=True, return_tensors="pt")
        ids = inputs.input_ids.to(device)
        masks = inputs.attention_mask.to(device).bool()
        hidden = text_encoder(input_ids=ids, attention_mask=masks,
                              output_hidden_states=True).hidden_states[-2]
        return [hidden[i][masks[i]] for i in range(len(hidden))]

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import gc

        import torch
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            GGUFQuantizationConfig,
            ZImageImg2ImgPipeline,
            ZImageTransformer2DModel,
        )
        from transformers import AutoTokenizer, Qwen3Model

        from ..devices import bf16_supported

        dtype = torch.bfloat16 if bf16_supported() else torch.float16

        tok_kwargs = {"token": self.hf_token} if self.hf_token else {}

        # 1. Text encoder pass first (8 GB), then free it entirely so the
        #    GGUF transformer never stacks on top of it in RAM.
        tokenizer = AutoTokenizer.from_pretrained(
            ZIMAGE_MODEL_ID, subfolder="tokenizer", **tok_kwargs)
        te = Qwen3Model.from_pretrained(
            ZIMAGE_MODEL_ID, subfolder="text_encoder",
            torch_dtype=dtype, **tok_kwargs).to(self.device)
        embeds = self._encode_once(tokenizer, te, torch, self.device)
        del te
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 2. Small components + GGUF transformer.
        vae = AutoencoderKL.from_pretrained(
            ZIMAGE_MODEL_ID, subfolder="vae", torch_dtype=dtype, **tok_kwargs)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            ZIMAGE_MODEL_ID, subfolder="scheduler", **tok_kwargs)
        gguf_file = GGUF_FILES[self.gguf]
        kwargs = {
            "quantization_config": GGUFQuantizationConfig(
                compute_dtype=dtype),
            "torch_dtype": dtype,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token
        transformer = ZImageTransformer2DModel.from_single_file(
            f"{ZIMAGE_GGUF_REPO}/{gguf_file}",
            config=ZIMAGE_MODEL_ID, subfolder="transformer", **kwargs)

        pipe = ZImageImg2ImgPipeline(
            scheduler=scheduler, vae=vae, text_encoder=None,
            tokenizer=None, transformer=transformer)
        pipe.to(self.device)
        self._pipe = pipe
        self._embeds = embeds
        return self._pipe

    def unload(self):
        if self._pipe is None:
            return
        del self._pipe
        self._pipe = None
        self._embeds = None
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
        target = zimage_lite_target_size(*orig_size)
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            prompt=None,
            prompt_embeds=self._embeds,
            image=prepared,
            strength=float(strength),
            num_inference_steps=ZIMAGE_LITE_STEPS,
            guidance_scale=ZIMAGE_LITE_CFG,
            generator=generator,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
