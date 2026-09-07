import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zinvis import profiles


def test_resolve_pipeline():
    assert profiles.resolve_pipeline("duo", None) == "duo"
    assert profiles.resolve_pipeline("duo", "openai") == "duo"
    assert profiles.resolve_pipeline("duo", "openai", route=True) == "chroma"
    assert profiles.resolve_pipeline("chroma", "google") == "chroma"
    assert profiles.resolve_pipeline("zimage", "openai") == "zimage"


def test_resolve_strength():
    assert profiles.resolve_strength("chroma", "google") == 0.40
    assert profiles.resolve_strength("chroma", "openai") == 0.09
    assert profiles.resolve_strength("chroma", "microsoft") == 0.125
    assert profiles.resolve_strength("chroma", "meta") == 0.17
    assert profiles.resolve_strength("chroma", None) == 0.40
    assert profiles.resolve_strength("zimage", "google") == 0.30
    assert profiles.resolve_strength("zimage", None) == 0.30
    assert profiles.resolve_strength("chroma", "google", 0.55) == 0.55
    assert profiles.resolve_strength("sdxl", "google") == 0.25
    assert profiles.resolve_strength("sdxl", "openai") == 0.15
    assert profiles.resolve_strength("sdxl", "meta") == 0.25
    assert profiles.resolve_strength("sana", "google") == 0.30
    assert profiles.resolve_strength("sana", "openai") == 0.15
    assert profiles.resolve_strength("sana", "meta") == 0.30
    assert profiles.resolve_strength("sdxl-canny", "google") == 0.30
    assert profiles.resolve_strength("sdxl-canny", None) == 0.30
    assert profiles.resolve_strength("zimage-lite", "google") == 0.30
    assert profiles.resolve_strength("zimage-lite", "openai") == 0.12
    assert profiles.resolve_strength("lcm", "google") == 0.35
    assert profiles.resolve_strength("lcm", "openai") == 0.20
    assert profiles.resolve_strength("lcm", None) == 0.35
    assert profiles.resolve_strength("vae", None) == 0.15
    assert profiles.resolve_strength("vae", None, 0.05) == 0.05
    try:
        profiles.resolve_strength("chroma", "google", 0.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_requested_steps():
    assert profiles.requested_steps(4, 0.40) == 10
    assert profiles.requested_steps(4, 0.09) == 45
    assert profiles.requested_steps(4, 1.0) == 4
    assert profiles.requested_steps(8, 0.18) == 45
    assert profiles.requested_steps(8, 0.30) == 27
    assert profiles.requested_steps(4, 2.0) == 2
    assert profiles.requested_steps(4, 0.01) == profiles.MAX_REQUESTED_STEPS


def test_duo_uses_chroma_floors():
    assert profiles.resolve_strength("duo", "google") == 0.40
    assert profiles.resolve_strength("duo", "openai") == 0.09
    assert profiles.resolve_strength("duo", None) == 0.40


def test_routing_only_when_requested():
    assert profiles.resolve_pipeline("duo", "openai") == "duo"
    assert profiles.resolve_pipeline("duo", "openai", route=True) == "chroma"
    assert profiles.resolve_pipeline("duo", "google", route=True) == "duo"


def test_resolve_seed():
    assert profiles.resolve_seed(None) == 0
    assert profiles.resolve_seed(7) == 7


def test_lcm_steps():
    from zinvis.regen.lcm import lcm_steps

    assert lcm_steps(0.20) == 4      # window 10 fits 4
    assert lcm_steps(0.35) == 4      # window 17
    assert lcm_steps(0.05) == 2      # window 2 clamps
    assert lcm_steps(0.01) == 1      # minimum 1 step
    assert lcm_steps(0.0) == 1       # window floor, never 0
    assert lcm_steps(0.9, 50) == 4   # never above the LCM step count


def test_target_size_grids():
    from zinvis.regen import chroma, lcm, sana, sdxl, sdxl_canny, vae, zimage
    from zinvis.regen import zimage_lite

    mods = {
        # module: required divisor
        chroma: 16, sdxl: 8, sdxl_canny: 8, vae: 8, lcm: 8,
        zimage: 16, zimage_lite: 16, sana: 32,
    }
    fns = {
        chroma: chroma.chroma_target_size,
        sdxl: sdxl.sdxl_target_size,
        sdxl_canny: sdxl_canny.sdxl_canny_target_size,
        lcm: lcm.lcm_target_size,
        vae: vae.vae_target_size,
        zimage: zimage.zimage_target_size,
        zimage_lite: zimage_lite.zimage_lite_target_size,
        sana: sana.sana_target_size,
    }
    import random

    rng = random.Random(7)
    for mod, div in mods.items():
        fn = fns[mod]
        if fn is None:
            continue
        for _ in range(50):
            w, h = rng.randint(1, 4096), rng.randint(1, 4096)
            tw, th = fn(w, h)
            assert tw % div == 0 and th % div == 0, (mod, w, h, tw, th)
            assert tw >= div and th >= div


def test_gguf_filename_resolution():
    from zinvis.regen.zimage_lite import GGUF_FILES, pick_gguf_filename

    assert pick_gguf_filename(["z-image-turbo-Q8_0.gguf", "readme.md"],
                              "q8") == "z-image-turbo-Q8_0.gguf"
    assert pick_gguf_filename(["Z-Image-Turbo-Q8_0.GGUF"], "q8") == \
        "Z-Image-Turbo-Q8_0.GGUF"
    assert pick_gguf_filename(["z-image-turbo-Q4_K_M.gguf"], "q4") == \
        "z-image-turbo-Q4_K_M.gguf"
    try:
        pick_gguf_filename(["totally-other-model-Q8_0.gguf"], "q8")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert GGUF_FILES["q8"].endswith(".gguf")
    assert GGUF_FILES["q4"].endswith(".gguf")


def test_zimage_lite_embeds_cached_across_unload():
    from zinvis.regen.zimage_lite import ZImageLiteBackend

    be = ZImageLiteBackend(device="cpu")
    be._pipe = object()
    be._embeds = ["e"]
    be.unload()
    assert be._pipe is None
    assert be._embeds == ["e"]


def test_zimage_prefer_and_fallback():
    from zinvis.regen import build_backend
    from zinvis.regen import zimage as zm

    assert build_backend("zimage", device="cpu").prefer == "diffusers"
    assert build_backend("zimage", device="cpu",
                         stream=True).prefer == "diffsynth"
    assert build_backend("zimage", device="cpu",
                         stream=False).prefer == "diffsynth"
    assert build_backend("zimage", device="cpu",
                         low_vram=True).prefer == "diffsynth"
    duo = build_backend("duo", device="cpu", stream=True)
    assert duo.zimage.prefer == "diffsynth", duo.zimage.prefer

    be1 = zm.ZImageBackend(device="cpu", prefer="diffusers")
    be1._load_diffsynth = lambda: "synth"
    orig = zm._diffusers_has_zimage
    try:
        zm._diffusers_has_zimage = lambda: False
        assert be1._load() == "synth"
        be2 = zm.ZImageBackend(device="cpu", prefer="diffusers")
        be2._load_diffusers = lambda: "diff"
        zm._diffusers_has_zimage = lambda: True
        assert be2._load() == "diff"
    finally:
        zm._diffusers_has_zimage = orig


def test_zimage_cfg_semantics():
    from zinvis.regen import zimage, zimage_lite

    assert zimage.ZIMAGE_DIFFUSERS_GUIDANCE == 0.0
    assert zimage_lite.ZIMAGE_LITE_CFG == 0.0
    assert zimage.ZIMAGE_CFG == 1.0


if __name__ == "__main__":
    test_resolve_pipeline()
    test_resolve_strength()
    test_requested_steps()
    test_resolve_seed()
    test_lcm_steps()
    test_target_size_grids()
    test_gguf_filename_resolution()
    test_zimage_lite_embeds_cached_across_unload()
    test_zimage_prefer_and_fallback()
    test_zimage_cfg_semantics()
    print("PROFILES OK")
