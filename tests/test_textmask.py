import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from zinvis.textmask import composite_original, detect_text_mask


def make_text_image(path, size=(320, 200)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.text((20, 30), "Small caption text here 123", fill="black")
    d.text((20, 60), "Another line of tiny words", fill="black")
    img.save(path)
    return img


def make_smooth_image(path, size=(320, 200)):
    x = np.tile(np.linspace(0, 255, size[0], dtype=np.uint8), (size[1], 1))
    arr = np.stack([x, x, x], axis=-1)
    img = Image.fromarray(arr)
    img.save(path)
    return img


def test_detect_text(tmp="/tmp/zinvis_textmask_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    text_img = make_text_image(p / "text.png")
    mask = detect_text_mask(text_img)
    assert mask.size == text_img.size and mask.mode == "L"
    frac = np.asarray(mask).mean() / 255.0
    assert 0.001 < frac < 0.25, frac

    smooth_img = make_smooth_image(p / "smooth.png")
    mask2 = detect_text_mask(smooth_img)
    frac2 = np.asarray(mask2).mean() / 255.0
    assert frac2 < 0.001, frac2


def test_mask_from_boxes(tmp="/tmp/zinvis_box_test"):
    from zinvis.textmask import mask_from_boxes, union_masks

    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)

    m = mask_from_boxes([[[0, 0], [20, 0], [20, 10], [0, 10]]], (40, 30))
    assert m is not None and m.size == (40, 30) and m.mode == "L"
    arr = np.asarray(m)
    assert arr[5, 10] > 200, arr[5, 10]
    assert arr[25, 35] < 50, arr[25, 35]

    assert mask_from_boxes([], (40, 30)) is None

    # clipped box partially outside the frame must not raise
    m2 = mask_from_boxes([[[30, 25], [50, 25], [50, 40], [30, 40]]], (40, 30))
    assert m2 is not None and m2.size == (40, 30)

    edge = detect_text_mask(make_text_image(p / "text.png"))
    u = union_masks(edge, m)
    assert u.size == edge.size
    arr_u = np.asarray(u)
    assert arr_u[5, 10] > 200  # box region survived the union
    assert union_masks(None, m) is m
    assert union_masks(edge, None) is edge


def test_composite(tmp="/tmp/zinvis_textmask_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    orig = make_text_image(p / "text.png")
    cleaned = orig.filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(3))
    mask = detect_text_mask(orig)
    out = composite_original(cleaned, orig, mask)
    assert out.size == orig.size
    # text pixels restored: output closer to original than blurred input
    import math

    def mse(a, b):
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        return float(np.mean(d ** 2))

    assert mse(out, orig) < mse(cleaned, orig)


def test_keep_text_engine(tmp="/tmp/zinvis_keeptext_test"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_engine_fake import FakeBackend, make_image  # noqa

    from zinvis.engine import ZinvisEngine

    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = p / "in.png"
    make_text_image(src)
    eng = ZinvisEngine(device="cpu", keep_text=True)
    eng._backend_for = lambda name: FakeBackend()
    r = eng.run_file(str(src), str(p / "out.png"), "sdxl")
    assert r.status == "cleaned", r.error
    assert "keep_text" in r.stages, r.stages
    assert any("keep-text" in w for w in r.warnings), r.warnings


def test_keep_text_downscale(tmp="/tmp/zinvis_keeptext_downscale_test"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_engine_fake import FakeBackend  # noqa

    from zinvis.engine import ZinvisEngine

    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = p / "in.png"
    make_text_image(src, size=(640, 400))
    eng = ZinvisEngine(device="cpu", keep_text=True)
    eng._backend_for = lambda name: FakeBackend()
    r = eng.run_file(str(src), str(p / "out.png"), "sdxl", max_side=320)
    assert r.status == "cleaned", r.error
    assert "keep_text" in r.stages, r.stages
    out_img = Image.open(p / "out.png")
    assert out_img.size == (640, 400)


if __name__ == "__main__":
    test_detect_text()
    test_mask_from_boxes()
    test_composite()
    test_keep_text_engine()
    test_keep_text_downscale()
    print("TEXTMASK OK")
