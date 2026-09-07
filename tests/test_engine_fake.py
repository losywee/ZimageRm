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
        import numpy as np

        self.calls.append((image.size, strength, seed))
        arr = np.asarray(image).astype(np.int16)
        arr[..., 0] = np.clip(arr[..., 0] + 3, 0, 255)
        out = Image.fromarray(arr.astype(np.uint8))
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
    Image.fromarray(arr).save(path)
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
    assert r.strength == 0.40

    r = eng.run_file(str(src), str(p / "out4.png"), "zimage")
    assert r.resolved_pipeline == "zimage"
    assert any("zimage floors" in w for w in r.warnings), r.warnings

    r = eng.run_file(str(src), str(p / "out3.png"), "chroma",
                     strength=0.2, seed=3)
    assert r.strength == 0.2

    r = eng.run_file(str(src), str(p / "out9.png"), "sdxl",
                     strength=0.05)
    assert r.strength == 0.15, r.strength
    assert any("clamping" in w for w in r.warnings), r.warnings

    r = eng.run_file(str(src), str(p / "out5.png"), "sdxl")
    assert r.resolved_pipeline == "sdxl" and r.strength == 0.25

    r = eng.run_file(str(src), str(p / "out6.png"), "vae")
    assert r.resolved_pipeline == "vae" and r.strength == 0.15
    assert any("weakest tier" in w for w in r.warnings), r.warnings

    r = eng.run_file(str(src), str(p / "out7.png"), "sana")
    assert r.resolved_pipeline == "sana" and r.strength == 0.30
    assert any("sana floors" in w for w in r.warnings), r.warnings
    assert any("1 of 2 SCM steps" in w for w in r.warnings), r.warnings

    eng_hi = setup_engine(p)
    r = eng_hi.run_file(str(src), str(p / "out7b.png"), "sana",
                        strength=0.6)
    assert r.status == "cleaned", r.error
    assert not any("1 of 2 SCM" in w for w in r.warnings), r.warnings

    r = eng.run_file(str(src), str(p / "out8.png"), "lcm")
    assert r.resolved_pipeline == "lcm" and r.strength == 0.35
    assert any("lcm floors" in w for w in r.warnings), r.warnings

    r = eng.run_file(str(src), str(p / "out10.png"), "sdxl-canny")
    assert r.resolved_pipeline == "sdxl-canny" and r.strength == 0.30
    assert any("sdxl-canny floors" in w for w in r.warnings), r.warnings

    # clamped strength must not also emit the stale "very low" warning
    r = eng.run_file(str(src), str(p / "out11.png"), "sdxl",
                     strength=0.04)
    assert r.strength == 0.15, r.strength
    assert any("clamping" in w for w in r.warnings), r.warnings
    assert not any("very low" in w for w in r.warnings), r.warnings

    r = eng.run_file(str(src), str(p / "out12.png"), "zimage-lite")
    assert r.resolved_pipeline == "zimage-lite" and r.strength == 0.30
    assert any("zimage-lite" in w for w in r.warnings), r.warnings

    from zinvis.regen import build_backend

    b = build_backend("zimage-lite", device="cpu", gguf="q4")
    assert b.gguf == "q4", b.gguf
    try:
        build_backend("zimage-lite", device="cpu", gguf="q3")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # control-scale plumbing reaches the backend
    from zinvis.regen import build_backend

    b = build_backend("sdxl-canny", device="cpu", control_scale=0.8)
    assert b.control_scale == 0.8, b.control_scale


def test_auto_pipeline(tmp="/tmp/zinvis_auto_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = make_image(p / "in.png")
    eng = setup_engine(p)

    original = engine_mod.vram_gb
    try:
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
    finally:
        engine_mod.vram_gb = original

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
    assert br.summary["cleaned"] == 2, br.summary
    assert '"status"' in br.to_json()

    (p / "out" / "b.png").unlink()
    (p / "out" / "a.png").write_bytes(b"placeholder")
    seen = []
    br = eng.run_dir(str(p), str(p / "out"), "chroma", "openai",
                     skip_existing=True, progress=seen.append)
    assert [r.status for r in br.images] == ["skipped_existing", "cleaned"]
    assert len(seen) == 2
    assert br.summary["skipped"] == 1
    assert br.summary["cleaned"] == 1


def test_error_and_max_side(tmp="/tmp/zinvis_error_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = make_image(p / "in.png", size=(256, 192))
    eng = setup_engine(p)

    r = eng.run_file(str(p / "missing.png"), str(p / "x.png"), "chroma")
    assert r.status == "error" and r.error
    assert not (p / "x.png").exists()

    r = eng.run_file(str(src), str(p / "y.png"), "chroma", max_side=64)
    assert r.status == "cleaned", r.error
    assert r.psnr > 30.0
    out = Image.open(p / "y.png")
    assert out.size == (256, 192)


def test_vram_restore():
    import zinvis.engine as em

    original = em.vram_gb
    try:
        em.vram_gb = lambda: 15.0
        eng = em.ZinvisEngine(device="cpu")
        assert eng.plan(None, None, None, None)["pipeline"] == "zimage"

        eng_hard = em.ZinvisEngine(device="cpu")
        eng_hard._backend_for = lambda name: FakeBackend()
        r = eng_hard.run_file(str(p2_src()), str(p2_out()), "chroma")
        assert r.status == "error"
        assert "refusing to download" in r.error, r.error
    finally:
        em.vram_gb = original
    assert em.vram_gb is original


_SRC = "/tmp/zinvis_vram_gate_in.png"
_OUT = "/tmp/zinvis_vram_gate_out.png"


def p2_src():
    if not Path(_SRC).exists():
        make_image(Path(_SRC))
    return _SRC


def p2_out():
    return _OUT


def test_zimage_stream_fallback(tmp="/tmp/zinvis_fallback_test"):
    from zinvis.regen.zimage import ZImageBackend

    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = make_image(p / "in.png")

    class FlakyPipe:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise NotImplementedError(
                    "Cannot copy out of meta tensor; no data!")
            return Image.open(src).convert("RGB")

    be = ZImageBackend(device="cpu", stream=True)
    flaky = FlakyPipe()
    be._pipe = flaky
    be.mode = "diffsynth"
    loads = []

    def _fake_load():
        loads.append(be.stream)
        be._pipe = flaky
        return flaky

    be._load = _fake_load
    out = be.run(Image.open(src).convert("RGB"), 0.3, 0)
    assert out.size == (64, 48)
    assert be.stream is False, "backend must flip off disk streaming"
    assert loads == [True, False], loads


def test_oom_retry_and_backoff(tmp="/tmp/zinvis_recover_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = make_image(p / "in.png")

    class OomOnceBackend(FakeBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.tries = 0

        def run(self, image, strength, seed):
            self.tries += 1
            if self.tries == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate")
            return super().run(image, strength, seed)

    eng = setup_engine(p)
    eng._backend_for = lambda name: OomOnceBackend()
    r = eng.run_file(str(src), str(p / "oom.png"), "sdxl")
    assert r.status == "cleaned", r.error
    assert any("oom_retry" in w for w in r.warnings), r.warnings

    class DarkBackend(FakeBackend):
        def run(self, image, strength, seed):
            # damage proportional to strength: lower strength -> higher PSNR
            import numpy as np

            arr = np.asarray(image).astype(np.float64)
            shift = (strength / 0.4) * 200.0
            arr = np.clip(arr - shift, 0, 255).astype(np.uint8)
            out = Image.fromarray(arr)
            out.info["zinvis_stages"] = [f"dark(s={strength})"]
            return out

    eng2 = setup_engine(p)
    eng2._backend_for = lambda name: DarkBackend()
    r = eng2.run_file(str(src), str(p / "dark.png"), "sdxl",
                      strength=0.4)
    assert r.status == "cleaned", r.error
    assert r.strength < 0.4, r.strength
    assert any("strength_backoff" in s for s in r.stages), r.stages

    # SDXL at distillation floor (0.15) must not back off even if PSNR is low
    r_floor = eng2.run_file(str(src), str(p / "dark_floor.png"), "sdxl",
                            strength=0.15)
    assert r_floor.status == "cleaned", r_floor.error
    assert r_floor.strength == 0.15, r_floor.strength
    assert not any("strength_backoff" in s for s in r_floor.stages), r_floor.stages

    class OomBackoffBackend(FakeBackend):
        def run(self, image, strength, seed):
            import numpy as np

            if strength < 0.4:
                raise RuntimeError("CUDA out of memory. Tried to allocate")
            arr = np.asarray(image).astype(np.float64) - 200.0
            out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            out.info["zinvis_stages"] = ["oomback(s=1)"]
            return out

    eng3 = setup_engine(p)
    eng3._backend_for = lambda name: OomBackoffBackend()
    r = eng3.run_file(str(src), str(p / "oomback.png"), "sdxl",
                      strength=0.4)
    assert r.status == "cleaned", r.error
    assert r.strength == 0.4, r.strength
    assert any("backoff skipped" in w for w in r.warnings), r.warnings

    # sticky scale-down: second file starts at the reduced size
    seen_sizes = []

    class SizeSpyBackend(FakeBackend):
        def run(self, image, strength, seed):
            seen_sizes.append(image.size)
            if len(seen_sizes) == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate")
            return super().run(image, strength, seed)

    eng4 = setup_engine(p)
    eng4._backend_for = lambda name: SizeSpyBackend()
    make_image(p / "c.png", size=(256, 192))
    make_image(p / "d.png", size=(256, 192))
    br = eng4.run_dir(str(p), str(p / "sticky"), "sdxl")
    assert br.summary["cleaned"] >= 3, br.summary
    assert eng4._oom_scale < 1.0, eng4._oom_scale
    assert seen_sizes[1][0] < 256, seen_sizes


def test_ocr_capture(tmp="/tmp/zinvis_ocr_test"):
    import zinvis.ocr as ocr_mod

    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = make_image(p / "in.png")
    orig_et, orig_bn = ocr_mod.extract_text, ocr_mod.backend_name
    try:
        ocr_mod.extract_text = lambda img: ["caption line", "123"]
        ocr_mod.backend_name = lambda: "fake"

        eng = setup_engine(p)
        r = eng.run_file(str(src), str(p / "out.png"), "sdxl")
        assert r.status == "cleaned", r.error
        assert r.text == ["caption line", "123"], r.text
        import json

        assert json.loads(r.to_json())["text"] == ["caption line", "123"]

        eng_off = ZinvisEngine(device="cpu", ocr=False)
        eng_off._backend_for = lambda name: FakeBackend()
        r = eng_off.run_file(str(src), str(p / "out2.png"), "sdxl")
        assert r.text == [], r.text
    finally:
        ocr_mod.extract_text, ocr_mod.backend_name = orig_et, orig_bn


def test_ocr_skipped_files(tmp="/tmp/zinvis_ocr_skip_test"):
    import zinvis.ocr as ocr_mod

    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    make_image(p / "a.png")
    make_image(p / "b.png")
    orig_et, orig_bn = ocr_mod.extract_text, ocr_mod.backend_name
    try:
        ocr_mod.extract_text = lambda img: ["pre-existing caption"]
        ocr_mod.backend_name = lambda: "fake"

        eng = setup_engine(p)
        eng.run_dir(str(p), str(p / "out"), "sdxl")
        # second run: everything skipped, but text must still be captured
        br = eng.run_dir(str(p), str(p / "out"), "sdxl",
                         skip_existing=True)
        assert br.summary["skipped"] == 2, br.summary
        assert all(r.text == ["pre-existing caption"] for r in br.images), \
            [r.text for r in br.images]
    finally:
        ocr_mod.extract_text, ocr_mod.backend_name = orig_et, orig_bn


def test_new_fixes(tmp="/tmp/zinvis_new_fixes_test"):
    import json
    from zinvis.regen import build_backend
    import zinvis.cli as cli_mod

    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = make_image(p / "in.png")

    # 1. Test unload on backends
    duo = build_backend("duo", device="cpu")
    duo.unload()

    sdxl_b = build_backend("sdxl", device="cpu")
    sdxl_b.unload()

    vae_b = build_backend("vae", device="cpu")
    vae_b.unload()

    # 2. Test auto-plan with low_vram=True
    eng_low = ZinvisEngine(device="cpu", low_vram=True)
    orig_vram = engine_mod.vram_gb
    try:
        engine_mod.vram_gb = lambda: 80.0
        plan = eng_low.plan(None, None, None, None)
        assert plan["pipeline"] == "zimage", plan
        assert any("low_vram" in w or "auto:" in w for w in plan["warnings"])
    finally:
        engine_mod.vram_gb = orig_vram

    # 3. Test single-file report in cli
    report_file = p / "single_report.json"
    if report_file.exists():
        report_file.unlink()

    orig_engine_cls = cli_mod.ZinvisEngine
    class MockEngine(orig_engine_cls):
        def _backend_for(self, name):
            return FakeBackend()

        def _require_device(self):
            pass

    cli_mod.ZinvisEngine = MockEngine
    try:
        ret = cli_mod.main([str(src), str(p / "cli_out.png"),
                            "--no-ocr", "--report", str(report_file)])
        assert ret == 0
        assert report_file.is_file()
        data = json.loads(report_file.read_text())
        # 4. Test directory with 0 matching images but other images present
        import io
        from contextlib import redirect_stdout

        dir_in = p / "dir_with_jpg"
        dir_in.mkdir(parents=True, exist_ok=True)
        make_image(dir_in / "photo.jpg")
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = cli_mod.main([str(dir_in), str(p / "dir_out"), "--no-ocr"])
        out_str = buf.getvalue()
        assert "0 images matched" in out_str and "*.jpg" in out_str, out_str
        assert ret == 1, ret
    finally:
        cli_mod.ZinvisEngine = orig_engine_cls


if __name__ == "__main__":
    test_run_file()
    test_auto_pipeline()
    test_batch()
    test_error_and_max_side()
    test_vram_restore()
    test_zimage_stream_fallback()
    test_oom_retry_and_backoff()
    test_ocr_capture()
    test_ocr_skipped_files()
    test_new_fixes()
    print("ENGINE OK")
