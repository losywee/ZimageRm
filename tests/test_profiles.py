import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zinvis import profiles


def test_resolve_pipeline():
    assert profiles.resolve_pipeline("duo", None) == "duo"
    assert profiles.resolve_pipeline("duo", "openai") == "chroma"
    assert profiles.resolve_pipeline("duo", "microsoft") == "chroma"
    assert profiles.resolve_pipeline("duo", "google") == "duo"
    assert profiles.resolve_pipeline("duo", "meta") == "duo"
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


def test_resolve_seed():
    assert profiles.resolve_seed(None) == 0
    assert profiles.resolve_seed(7) == 7


if __name__ == "__main__":
    test_resolve_pipeline()
    test_resolve_strength()
    test_requested_steps()
    test_resolve_seed()
    print("PROFILES OK")
