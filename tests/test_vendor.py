import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zinvis.vendor import sniff_vendor


def write(data: bytes) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="zinvis_vendor_"))
    p = tmp / "img.png"
    p.write_bytes(data)
    return p


def test_none():
    assert sniff_vendor(write(b"\x89PNG\r\n\x1a\n plain image bytes") ) is None


def test_google():
    assert sniff_vendor(write(b"...\x00 c2pa: Google LLC / SynthID ...")) == "google"
    assert sniff_vendor(write(b"Gemini generated image")) == "google"


def test_openai():
    assert sniff_vendor(write(b"issuer: openai.com c2pa manifest")) == "openai"


def test_microsoft():
    assert sniff_vendor(write(b"InvisMark signature by Microsoft Corporation")) == "microsoft"


def test_meta():
    assert sniff_vendor(write(b"iptc: trainedAlgorithmicMedia")) == "meta"


def test_priority():
    assert sniff_vendor(write(b"trainedAlgorithmicMedia openai")) == "openai"


def test_tail_window():
    filler = b"\x00" * 1_100_000
    assert sniff_vendor(write(filler + b"provider: OPENAI")) is not None


if __name__ == "__main__":
    test_none()
    test_google()
    test_openai()
    test_microsoft()
    test_meta()
    test_priority()
    test_tail_window()
    print("VENDOR OK")
