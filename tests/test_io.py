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


if __name__ == "__main__":
    test_save_strips_metadata()
    test_psnr()
    print("IO OK")
