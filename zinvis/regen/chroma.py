from __future__ import annotations

from PIL import Image

from ..profiles import requested_steps

CHROMA_MODEL_ID = "lodestones/Chroma1-HD"
CHROMA_EFFECTIVE_STEPS = 4
CHROMA_GUIDANCE = 5.0
CHROMA_PROMPT = "high quality, sharp, detailed, faithful to the original"
CHROMA_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"
LATENT_GRID = 16


def chroma_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


class ChromaBackend:
    name = "chroma"

    def __init__(self, device: str = "cuda", hf_token: str | None = None):
        self.device = device
        self.hf_token = hf_token
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import ChromaImg2ImgPipeline

        kwargs = {"torch_dtype": torch.bfloat16}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        pipe = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, **kwargs)
        self._pipe = pipe.to(self.device)
        return self._pipe

    def unload(self):
        if self._pipe is None:
            return
        import torch

        del self._pipe
        self._pipe = None
        torch.cuda.empty_cache()

    def run(self, image, strength: float, seed: int):
        import torch

        pipe = self._load()
        orig_size = image.size
        target = chroma_target_size(*orig_size)
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        steps = requested_steps(CHROMA_EFFECTIVE_STEPS, strength)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            width=target[0],
            height=target[1],
            image=prepared,
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=CHROMA_GUIDANCE,
            generator=generator,
            prompt=CHROMA_PROMPT,
            negative_prompt=CHROMA_NEGATIVE,
        ).images[0]
        if result.size != orig_size:
            result = result.resize(orig_size, Image.Resampling.LANCZOS)
        return result.convert("RGB")
