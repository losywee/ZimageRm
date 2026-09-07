import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import zinvis.engine as engine_mod
from zinvis.engine import ZinvisEngine


class FakeBackend:
    name = "fake"

    def __init__(self, device="cuda", hf_token=None, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def run(self, image, strength, seed):
        self.calls.append((image.size, strength, seed))
        out = image.copy()
        out.putdata([(min(255, r + 3), g, b) for r, g, b in image.getdata()])
        out.info["zinvis_stages"] = ["fake(s=1)"]
        return out


def setup_engine(tmp, monkey_backend=True):
    eng = ZinvisEngine(device='cpu', hf_token=None)
    if monkey_backend:
        eng._backend_for = lambda name: FakeBackend()
    return eng


def make_image(path, size=(64, 48)):
    import numpy as np

    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return Path(path)


def test_run_file(tmp="/tmp/zinvis_engine_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = make_image(p / "in.png")
    eng = setup_engine(p)

    r = eng.run_file(str(src), str(p / "out.png"), "chroma",
                     vendor="google", seed=5)
    assert r.status == "cleaned", r.error
    assert (p / "out.png").is_file()
    assert r.vendor == "google"
    assert r.strength == 0.40
    assert r.seed == 5
    assert r.resolved_pipeline == "chroma"
    assert r.psnr > 30.0

    r = eng.run_file(str(src), str(p / "out2.png"), "duo")
    assert r.resolved_pipeline == "duo"
    assert r.strength == 0.30

    r = eng.run_file(str(src), str(p / "out4.png"), "zimage")
    assert r.resolved_pipeline == "zimage"
    assert any("zimage floors" in w for w in r.warnings), r.warnings

    r = eng.run_file(str(src), str(p / "out3.png"), "chroma",
                     strength=0.2, seed=3)
    assert r.strength == 0.2


def test_auto_pipeline(tmp="/tmp/zinvis_auto_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = make_image(p / "in.png")
    eng = setup_engine(p)

    engine_mod.vram_gb = lambda: 15.0
    plan = eng.plan(None, None, None, None)
    assert plan["pipeline"] == "zimage", plan
    assert any("auto:" in w for w in plan["warnings"])

    r = eng.run_file(str(src), str(p / "auto_out.png"), None)
    assert r.resolved_pipeline == "zimage"
    assert r.pipeline == "auto"
    assert r.strength == 0.30

    engine_mod.vram_gb = lambda: 80.0
    plan = eng.plan(None, "openai", None, None)
    assert plan["pipeline"] == "chroma", plan

    engine_mod.vram_gb = lambda: None
    plan = eng.plan(None, None, None, None)
    assert plan["pipeline"] == "duo", plan

    eng_small = ZinvisEngine(device="cpu", hf_token=None, low_vram=True)
    eng_small._backend_for = lambda name: FakeBackend()
    r = eng_small.run_file(str(src), str(p / "lv.png"), "zimage")
    assert r.status == "cleaned"


def test_batch(tmp="/tmp/zinvis_batch_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    make_image(p / "a.png")
    make_image(p / "b.png")
    eng = setup_engine(p)
    br = eng.run_dir(str(p), str(p / "out"), "chroma", "openai")
    assert br.summary["succeeded"] == 2, br.summary
    assert '"status"' in br.to_json()


if __name__ == "__main__":
    test_run_file()
    test_batch()
    print("ENGINE OK")
