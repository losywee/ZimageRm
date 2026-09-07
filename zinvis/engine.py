from __future__ import annotations

from pathlib import Path

from PIL import Image

from . import profiles
from .io_utils import (
    BatchReport,
    ImageRecord,
    load_rgb,
    now,
    psnr,
    save_stripped,
)
from .regen import build_backend
from .vendor import sniff_vendor


class ZinvisEngine:
    """Whole-frame invisible-watermark removal via diffusion regeneration."""

    def __init__(self, device: str = "cuda", hf_token: str | None = None,
                 refine_strength: float = profiles.DUO_REFINE_STRENGTH,
                 psnr_floor: float = profiles.DEFAULT_PSNR_FLOOR):
        self.device = device
        self.hf_token = hf_token
        self.refine_strength = refine_strength
        self.psnr_floor = psnr_floor
        self._backend = None
        self._backend_name: str | None = None

    def _backend_for(self, name: str):
        if self._backend is None or self._backend_name != name:
            self._backend = build_backend(
                name, device=self.device, hf_token=self.hf_token,
                refine_strength=self.refine_strength,
                psnr_floor=self.psnr_floor,
            )
            self._backend_name = name
        return self._backend

    def plan(self, pipeline: str, vendor: str | None,
             strength: float | None, seed: int | None,
             image_path: Path | None = None) -> dict:
        warnings: list[str] = []
        v = vendor
        if v is None and image_path is not None:
            v = sniff_vendor(image_path)
            if v:
                warnings.append(f"vendor sniffed from provenance: {v}")
        resolved = profiles.resolve_pipeline(pipeline, v)
        s = profiles.resolve_strength(resolved, v, strength)
        seed_v = profiles.resolve_seed(seed)
        if strength is None and pipeline == "zimage":
            warnings.append(
                "zimage floors are uncalibrated defaults; pass --strength "
                "for known-hard watermarks"
            )
        return {
            "pipeline": resolved,
            "vendor": v,
            "strength": s,
            "seed": seed_v,
            "warnings": warnings,
        }

    def _require_device(self):
        if self.device == "cuda":
            from .devices import require_cuda

            require_cuda()

    def run_file(self, input_path: str, output_path: str,
                 pipeline: str, vendor: str | None = None,
                 strength: float | None = None, seed: int | None = None,
                 max_side: int = 0) -> ImageRecord:
        from .devices import require_cuda

        t0 = now()
        in_p, out_p = Path(input_path), Path(output_path)
        plan = self.plan(pipeline, vendor, strength, seed, in_p)

        img = load_rgb(in_p)
        orig_size = img.size
        if max_side and max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)),
                Image.Resampling.LANCZOS,
            )

        record = ImageRecord(
            input=str(in_p), output=str(out_p),
            pipeline=pipeline, resolved_pipeline=plan["pipeline"],
            vendor=plan["vendor"], strength=plan["strength"],
            seed=plan["seed"], warnings=list(plan["warnings"]),
        )

        try:
            self._require_device()
            backend = self._backend_for(plan["pipeline"])
            out = backend.run(img, plan["strength"], plan["seed"])
            if out.size != orig_size:
                out = out.resize(orig_size, Image.Resampling.LANCZOS)
            save_stripped(out, out_p)
            record.stages = list(getattr(out, "info", {}).get(
                "zinvis_stages", [plan["pipeline"]]))
            record.psnr = psnr(img, out)
            record.status = "cleaned"
        except Exception as exc:
            record.status = "error"
            record.error = f"{type(exc).__name__}: {exc}"
        record.seconds = now() - t0
        return record

    def run_dir(self, in_dir: str, out_dir: str, pipeline: str,
                vendor: str | None = None, strength: float | None = None,
                seed: int | None = None, glob_pattern: str = "*.png",
                max_side: int = 0) -> BatchReport:
        t0 = now()
        br = BatchReport(in_dir=in_dir, out_dir=out_dir)
        for p in sorted(Path(in_dir).glob(glob_pattern)):
            r = self.run_file(str(p), str(Path(out_dir) / p.name),
                              pipeline, vendor, strength, seed, max_side)
            br.images.append(r)
        br.seconds = now() - t0
        return br
