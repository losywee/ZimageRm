from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

JPEG_QUALITY = 95


def load_rgb(path: str | Path) -> Image.Image:
    from PIL.ImageOps import exif_transpose

    with Image.open(path) as img:
        return exif_transpose(img).convert("RGB")


def save_stripped(img: Image.Image, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = img.copy()
    clean.info = {}
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

        d = asdict(self)
        if not np.isfinite(d["psnr"]):
            d["psnr"] = None
        return json.dumps(d, indent=2, default=str)


@dataclass
class BatchReport:
    in_dir: str
    out_dir: str
    images: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def summary(self) -> dict:
        cleaned = [r for r in self.images if r.status == "cleaned"]
        failed = [r for r in self.images if r.error is not None]
        skipped = [r for r in self.images if r.status == "skipped_existing"]
        psnrs = [r.psnr for r in cleaned if np.isfinite(r.psnr)]
        return {
            "total": len(self.images),
            "cleaned": len(cleaned),
            "failed": len(failed),
            "skipped": len(skipped),
            "mean_psnr": (float(np.mean(psnrs)) if psnrs else None),
            "seconds": self.seconds,
        }

    def to_json(self) -> str:
        import json

        images = []
        for r in self.images:
            d = asdict(r)
            if not np.isfinite(d["psnr"]):
                d["psnr"] = None
            images.append(d)
        return json.dumps(
            {
                "in_dir": self.in_dir,
                "out_dir": self.out_dir,
                "summary": self.summary,
                "images": images,
            },
            indent=2,
            default=str,
        )


def now() -> float:
    return time.time()
