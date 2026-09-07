from __future__ import annotations

import traceback
from pathlib import Path

from PIL import Image

from . import profiles
from .devices import vram_gb
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

CHROMA_MIN_VRAM_GB = 30.0
CHROMA_HARD_FLOOR_GB = 24.0
STREAM_VRAM_GB = 20.0
OOM_RETRY_SCALES = (0.75, 0.5)
BACKOFF_PSNR_FLOOR = 20.0
BACKOFF_FACTOR = 0.7
BACKOFF_MIN_STRENGTH = 0.05


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "outofmemory" in name or "out of memory" in msg


class ZinvisEngine:
    """Whole-frame invisible-watermark removal via diffusion regeneration."""

    def __init__(self, device: str = "cuda", hf_token: str | None = None,
                 refine_strength: float = profiles.DUO_REFINE_STRENGTH,
                 psnr_floor: float = profiles.DEFAULT_PSNR_FLOOR,
                 low_vram: bool = False, stream: bool | None = None,
                 keep_text: bool = False):
        self.device = device
        self.hf_token = hf_token
        self.refine_strength = refine_strength
        self.psnr_floor = psnr_floor
        self.low_vram = low_vram
        self.stream = stream
        self.keep_text = keep_text
        self._backend = None
        self._backend_name: str | None = None

    def _backend_for(self, name: str):
        if self._backend is None or self._backend_name != name:
            self._backend = build_backend(
                name, device=self.device, hf_token=self.hf_token,
                low_vram=self.low_vram or (
                    (vram_gb() or 999.0) < STREAM_VRAM_GB
                ),
                stream=self.stream,
                refine_strength=self.refine_strength,
                psnr_floor=self.psnr_floor,
            )
            self._backend_name = name
        return self._backend

    def plan(self, pipeline: str | None, vendor: str | None,
             strength: float | None, seed: int | None,
             image_path: Path | None = None) -> dict:
        warnings: list[str] = []
        v = vendor
        if v is None and image_path is not None:
            v = sniff_vendor(image_path)
            if v:
                warnings.append(f"vendor sniffed from provenance: {v}")

        was_auto = pipeline is None
        resolved = pipeline
        if resolved is None:
            vram = vram_gb()
            if vram is not None and vram < CHROMA_MIN_VRAM_GB:
                resolved = "zimage"
                warnings.append(
                    f"auto: {vram:.0f} GiB VRAM (<{CHROMA_MIN_VRAM_GB:.0f}) -> "
                    "zimage (DiffSynth CPU offload); Chroma1 needs ~29 GiB"
                )
            else:
                resolved = "duo"

        vram = vram_gb()
        if resolved in ("duo", "chroma") and vram is not None \
                and vram < CHROMA_MIN_VRAM_GB:
            warnings.append(
                f"warning: {resolved} needs ~29 GiB VRAM; this card has "
                f"{vram:.0f} GiB and will OOM"
            )

        resolved = profiles.resolve_pipeline(resolved, v,
                                             route=was_auto and resolved == "duo")
        s = profiles.resolve_strength(resolved, v, strength)
        seed_v = profiles.resolve_seed(seed)
        if strength is None and resolved == "zimage":
            warnings.append(
                "zimage floors are uncalibrated defaults; pass --strength "
                "for known-hard watermarks"
            )
        if resolved == "vae":
            warnings.append(
                "vae is the weakest tier: fine for weak watermarks and as a "
                "pre-pass, insufficient alone for hard carriers like "
                "Gemini SynthID"
            )
        if self.keep_text:
            warnings.append(
                "keep-text: original pixels are restored over detected small "
                "text, so any watermark signal under those pixels survives"
            )
        if strength is not None and strength < 0.05:
            warnings.append(
                f"strength {strength} is very low; step count is capped at "
                f"{profiles.MAX_REQUESTED_STEPS}"
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
                 pipeline: str | None, vendor: str | None = None,
                 strength: float | None = None, seed: int | None = None,
                 max_side: int = 0, keep_text: bool | None = None) -> ImageRecord:
        t0 = now()
        in_p, out_p = Path(input_path), Path(output_path)
        plan = self.plan(pipeline, vendor, strength, seed, in_p)
        record = ImageRecord(
            input=str(in_p), output=str(out_p),
            pipeline="auto" if pipeline is None else pipeline,
            resolved_pipeline=plan["pipeline"],
            vendor=plan["vendor"], strength=plan["strength"],
            seed=plan["seed"], warnings=list(plan["warnings"]),
        )

        try:
            self._require_device()
            vram = vram_gb()
            if plan["pipeline"] in ("chroma", "duo") and vram is not None \
                    and vram < CHROMA_HARD_FLOOR_GB:
                raise RuntimeError(
                    f"{plan['pipeline']} needs ~29 GiB VRAM but this card has "
                    f"{vram:.0f} GiB; refusing to download ~33 GB of Chroma "
                    "weights that cannot run. Use --pipeline zimage."
                )
            img = load_rgb(in_p)
            orig_size = img.size
            working = img
            if max_side and max(img.size) > max_side:
                scale = max_side / max(img.size)
                working = img.resize(
                    (round(img.width * scale), round(img.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            backend = self._backend_for(plan["pipeline"])
            strength_v = plan["strength"]
            try:
                out = backend.run(working, strength_v, plan["seed"])
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                out = self._retry_oom(backend, working, strength_v,
                                      plan["seed"], record)
                if out is None:
                    raise
            if out.size != working.size:
                out = out.resize(working.size, Image.Resampling.LANCZOS)
            use_keep = self.keep_text if keep_text is None else keep_text
            if use_keep:
                from .textmask import composite_original, detect_text_mask

                mask = detect_text_mask(working)
                out = composite_original(out, working, mask)
            record.psnr = psnr(working, out)
            backoff_stage = None
            if (record.psnr < BACKOFF_PSNR_FLOOR
                    and strength_v > BACKOFF_MIN_STRENGTH):
                retry_s = max(BACKOFF_MIN_STRENGTH,
                              strength_v * BACKOFF_FACTOR)
                retry = backend.run(working, retry_s, plan["seed"])
                if retry.size != working.size:
                    retry = retry.resize(working.size,
                                         Image.Resampling.LANCZOS)
                if use_keep:
                    from .textmask import composite_original, detect_text_mask

                    retry = composite_original(
                        retry, working, detect_text_mask(working))
                retry_psnr = psnr(working, retry)
                if retry_psnr > record.psnr:
                    out = retry
                    strength_v = retry_s
                    record.psnr = retry_psnr
                    backoff_stage = (f"strength_backoff(s={retry_s:.3f},"
                                     f"psnr={retry_psnr:.1f})")
            record.strength = strength_v
            if orig_size != working.size:
                out = out.resize(orig_size, Image.Resampling.LANCZOS)
            save_stripped(out, out_p)
            record.stages = list(getattr(out, "info", {}).get(
                "zinvis_stages", [plan["pipeline"]]))
            if backoff_stage:
                record.stages.append(backoff_stage)
            if use_keep:
                record.stages.append("keep_text")
            record.status = "cleaned"
        except Exception as exc:
            record.status = "error"
            tb = traceback.format_exc().strip().splitlines()[-4:]
            record.error = f"{type(exc).__name__}: {exc} | {' / '.join(tb)}"
        record.seconds = now() - t0
        return record

    def _retry_oom(self, backend, working, strength, seed, record):
        """Halve the working resolution and retry after a CUDA OOM."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        for scale in OOM_RETRY_SCALES:
            small = working.resize(
                (max(64, round(working.width * scale)),
                 max(64, round(working.height * scale))),
                Image.Resampling.LANCZOS,
            )
            try:
                out = backend.run(small, strength, seed)
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                continue
            record.warnings.append(
                f"oom_retry: CUDA OOM at {working.size}, retried at "
                f"{small.size} (detail loss possible)")
            return out
        return None

    def run_dir(self, in_dir: str, out_dir: str, pipeline: str | None,
                vendor: str | None = None, strength: float | None = None,
                seed: int | None = None, glob_pattern: str = "*.png",
                max_side: int = 0, skip_existing: bool = False,
                progress=None, keep_text: bool | None = None) -> BatchReport:
        t0 = now()
        br = BatchReport(in_dir=in_dir, out_dir=out_dir)
        for p in sorted(Path(in_dir).glob(glob_pattern)):
            out_p = Path(out_dir) / p.name
            if skip_existing and out_p.is_file():
                r = ImageRecord(
                    input=str(p), output=str(out_p),
                    pipeline="auto" if pipeline is None else pipeline,
                    resolved_pipeline="", vendor=vendor,
                    strength=strength if strength is not None else 0.0,
                    seed=profiles.resolve_seed(seed),
                    status="skipped_existing",
                )
            else:
                r = self.run_file(str(p), str(out_p), pipeline, vendor,
                                  strength, seed, max_side, keep_text)
            br.images.append(r)
            if progress is not None:
                try:
                    progress(r)
                except Exception:
                    pass
        br.seconds = now() - t0
        return br
