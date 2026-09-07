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


if __name__ == "__main__":
    test_resolve_pipeline()
    test_resolve_strength()
    test_requested_steps()
    test_resolve_seed()
    test_lcm_steps()
    print("PROFILES OK")
