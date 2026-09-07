from __future__ import annotations

from PIL import Image

VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
LATENT_GRID = 8


def vae_target_size(width: int, height: int) -> tuple[int, int]:
    return (
        max(LATENT_GRID, (width // LATENT_GRID) * LATENT_GRID),
        max(LATENT_GRID, (height // LATENT_GRID) * LATENT_GRID),
    )


class VAEBackend:
    """SD VAE round-trip with optional latent noise — the lightest disruptor.

    Projects the image through the VAE latent space the watermark decoder was
    never trained against (Gowal et al. 2026, §6.1); optional gaussian noise
    on the latents (strength, interpreted as std in latent units) widens the
    disruption. Weakest tier: weak for hard carriers, near-free to run.
    """

    name = "vae"

    def __init__(self, device: str = "cuda", hf_token: str | None = None):
        self.device = device
        self.hf_token = hf_token
        self._vae = None

    def _load(self):
        if self._vae is not None:
            return self._vae
        import torch
        from diffusers import AutoencoderKL

        kwargs = {"torch_dtype": torch.bfloat16}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, **kwargs)
        self._vae = vae.to(self.device).eval()
        return self._vae

    def run(self, image, strength: float, seed: int):
        import torch

        vae = self._load()
        orig_size = image.size
        target = vae_target_size(*orig_size)
        prepared = image if image.size == target else image.resize(
            target, Image.Resampling.LANCZOS
        )
        noise_std = max(0.0, float(strength))

        from diffusers.image_processor import VaeImageProcessor

        proc = VaeImageProcessor(do_resize=False, do_normalize=True)
        latents_in = proc.preprocess(prepared).to(
            device=self.device, dtype=next(vae.parameters()).dtype
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            encoded = vae.encode(latents_in)
            latents = (encoded.latent_dist.mode()
                       if hasattr(encoded, "latent_dist")
                       else encoded.latents)
            if noise_std > 0:
                latents = latents + noise_std * torch.randn(
                    latents.shape, generator=generator, device=latents.device,
                    dtype=latents.dtype,
                )
            decoded = vae.decode(latents, return_dict=False)[0]
        out = proc.postprocess(decoded, output_type="pil")[0]
        if out.size != orig_size:
            out = out.resize(orig_size, Image.Resampling.LANCZOS)
        return out.convert("RGB")
