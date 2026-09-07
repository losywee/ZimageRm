import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from zinvis.io_utils import load_rgb, psnr, save_stripped


def test_save_strips_metadata(tmp="/tmp/zinvis_io_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    src = p / "in.png"
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    img.save(src, pnginfo=_fake_pnginfo())
    loaded = load_rgb(src)
    assert loaded.info, "fixture must carry metadata"

    out = p / "out.png"
    save_stripped(loaded, out)
    assert not Image.open(out).info

    out_jpg = p / "out.jpg"
    save_stripped(loaded, out_jpg)
    raw = out_jpg.read_bytes().lower()
    assert b"exif" not in raw
    assert b"ns.adobe.com" not in raw
    assert not Image.open(out_jpg).info.get("exif")
    assert not Image.open(out_jpg).info.get("xmp")


def _fake_pnginfo():
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("Software", "FakeGen 1.0")
    info.add_text("prompt", "test")
    return info


def test_psnr():
    a = Image.new("RGB", (4, 4), (0, 0, 0))
    assert psnr(a, a) == float("inf")
    b = Image.new("RGB", (4, 4), (0, 0, 1))
    assert 52.0 < psnr(a, b) < 53.5


def test_exif_orientation(tmp="/tmp/zinvis_exif_test"):
    p = Path(tmp)
    p.mkdir(parents=True, exist_ok=True)
    # EXIF orientation 6 = "rotate 90 CW to display": stored pixels are
    # tall, load_rgb must return the wide display orientation.
    tall = Image.new("RGB", (40, 80), "red")
    exif = tall.getexif()
    exif[274] = 6
    src = p / "rot.jpg"
    tall.save(src, exif=exif)
    assert load_rgb(src).size == (80, 40)


def test_summary_json_safe(tmp="/tmp/zinvis_summary_test"):
    from zinvis.io_utils import BatchReport, ImageRecord

    br = BatchReport(in_dir="a", out_dir="b", images=[
        ImageRecord(input="x", output="y", pipeline="p",
                    resolved_pipeline="p", vendor=None, strength=0.1,
                    seed=0, status="cleaned", psnr=float("inf")),
        ImageRecord(input="x2", output="y2", pipeline="p",
                    resolved_pipeline="p", vendor=None, strength=0.1,
                    seed=0, status="cleaned", psnr=30.0),
    ])
    s = br.summary
    assert s["mean_psnr"] == 30.0, s
    assert "Infinity" not in br.to_json()


if __name__ == "__main__":
    test_save_strips_metadata()
    test_psnr()
    test_exif_orientation()
    test_summary_json_safe()
    print("IO OK")
