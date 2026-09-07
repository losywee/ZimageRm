from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

JPEG_QUALITY = 95


def load_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def save_stripped(img: Image.Image, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = Image.new("RGB", img.size)
    clean.putdata(list(img.getdata()))
    if p.suffix.lower() in (".jpg", ".jpeg"):
        clean.save(p, quality=JPEG_QUALITY)
    else:
        clean.save(p)


def psnr(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((aa - bb) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def has_metadata(path: str | Path) -> bool:
    img = Image.open(path)
    return bool(img.info) or "exif" in img.info


@dataclass
class ImageRecord:
    input: str
    output: str
    pipeline: str
    resolved_pipeline: str
    vendor: str | None
    strength: float
    seed: int
    stages: list = field(default_factory=list)
    psnr: float = 0.0
    seconds: float = 0.0
    status: str = "cleaned"
    warnings: list = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> str:
        import json

        return json.dumps(asdict(self), indent=2, default=str)


@dataclass
class BatchReport:
    in_dir: str
    out_dir: str
    images: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def summary(self) -> dict:
        ok = [r for r in self.images if r.error is None]
        return {
            "total": len(self.images),
            "succeeded": len(ok),
            "failed": len(self.images) - len(ok),
            "mean_psnr": float(np.mean([r.psnr for r in ok])) if ok else None,
            "seconds": self.seconds,
        }

    def to_json(self) -> str:
        import json

        return json.dumps(
            {
                "in_dir": self.in_dir,
                "out_dir": self.out_dir,
                "summary": self.summary,
                "images": [asdict(r) for r in self.images],
            },
            indent=2,
            default=str,
        )


def now() -> float:
    return time.time()
