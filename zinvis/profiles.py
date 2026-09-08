from __future__ import annotations

PROFILES = ("duo", "chroma", "zimage", "zimage-lite", "sdxl", "sdxl-canny",
            "vae", "sana", "lcm", "sd-turbo", "sd15")
PROFILE_SEED = 0

CHROMA_FLOORS = {
    "google": 0.40,
    "openai": 0.09,
    "microsoft": 0.125,
    "meta": 0.17,
}
CHROMA_UNKNOWN_FLOOR = 0.40

ZIMAGE_FLOORS = {
    "google": 0.30,
    "openai": 0.12,
    "microsoft": 0.15,
    "meta": 0.15,
}
ZIMAGE_UNKNOWN_FLOOR = 0.30

SDXL_FLOORS = {
    "google": 0.25,
    "openai": 0.15,
}
SDXL_UNKNOWN_FLOOR = 0.25
# Lightning's 4-step distillation bounds the usable sigma range: strengths
# below the lowest calibrated floor push the executed timesteps outside the
# training distribution and the UNet emits noise instead of content.
SDXL_MIN_STRENGTH = 0.15

# Canny conditioning pins structure (and glyph geometry), which also
# shields watermark signal — be conservative until calibrated.
SDXL_CANNY_FLOOR = 0.30

SANA_FLOORS = {
    "google": 0.30,
    "openai": 0.15,
}
SANA_UNKNOWN_FLOOR = 0.30

LCM_FLOORS = {
    "google": 0.35,
    "openai": 0.20,
}
LCM_UNKNOWN_FLOOR = 0.35

SDTURBO_FLOORS = {
    "google": 0.35,
    "openai": 0.20,
}
SDTURBO_UNKNOWN_FLOOR = 0.35

SD15_FLOORS = {
    "google": 0.35,
    "openai": 0.20,
}
SD15_UNKNOWN_FLOOR = 0.35

VAE_DEFAULT_NOISE = 0.15

DUO_REFINE_STRENGTH = 0.18
DEFAULT_PSNR_FLOOR = 24.0

_engine_by_vendor = {
    "openai": "chroma",
    "microsoft": "chroma",
}


def resolve_pipeline(pipeline: str, vendor: str | None,
                     route: bool = False) -> str:
    value = pipeline.strip().casefold().replace("_", "-")
    if value not in PROFILES:
        raise ValueError(f"unknown pipeline {pipeline!r}; choose from {PROFILES}")
    if not route or value != "duo":
        return value
    return _engine_by_vendor.get((vendor or "").casefold(), "duo")


def resolve_strength(
    pipeline: str,
    vendor: str | None,
    strength: float | None = None,
) -> float:
    if strength is not None:
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"strength must be in (0, 1]; got {strength}")
        return float(strength)
    v = (vendor or "").casefold()
    if pipeline in ("chroma", "duo"):
        return CHROMA_FLOORS.get(v, CHROMA_UNKNOWN_FLOOR)
    if pipeline == "sdxl":
        return SDXL_FLOORS.get(v, SDXL_UNKNOWN_FLOOR)
    if pipeline == "sdxl-canny":
        return SDXL_CANNY_FLOOR
    if pipeline == "sana":
        return SANA_FLOORS.get(v, SANA_UNKNOWN_FLOOR)
    if pipeline == "lcm":
        return LCM_FLOORS.get(v, LCM_UNKNOWN_FLOOR)
    if pipeline == "sd-turbo":
        return SDTURBO_FLOORS.get(v, SDTURBO_UNKNOWN_FLOOR)
    if pipeline == "sd15":
        return SD15_FLOORS.get(v, SD15_UNKNOWN_FLOOR)
    if pipeline == "vae":
        return VAE_DEFAULT_NOISE
    if pipeline in ("zimage", "zimage-lite"):
        return ZIMAGE_FLOORS.get(v, ZIMAGE_UNKNOWN_FLOOR)
    return ZIMAGE_FLOORS.get(v, ZIMAGE_UNKNOWN_FLOOR)


def resolve_seed(seed: int | None) -> int:
    return PROFILE_SEED if seed is None else seed


MAX_REQUESTED_STEPS = 100


def requested_steps(effective_steps: int, strength: float) -> int:
    import math

    steps = math.ceil(effective_steps / max(float(strength), 1e-6))
    return max(1, min(steps, MAX_REQUESTED_STEPS))


def strength_safe_steps(steps: int, strength: float) -> int:
    """Scale an img2img schedule up so at least one step executes.

    diffusers' SD-family img2img pipelines truncate the executed steps to
    `int(num_inference_steps * strength)`; when that yields 0 (few-step
    schedule x low strength) the pipeline produces empty latents and the
    VAE attention crashes on a 0-element batch ("cannot reshape tensor of
    0 elements into shape [0, -1, 1, 512]").
    """
    import math

    return max(int(steps), math.ceil(1.0 / max(float(strength), 1e-6)))
