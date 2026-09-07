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
                 keep_text: bool = False, ocr: bool = True,
                 control_scale: float = 0.5, gguf: str = "q8"):
        self.device = device
        self.hf_token = hf_token
        self.refine_strength = refine_strength
        self.psnr_floor = psnr_floor
        self.low_vram = low_vram
        self.stream = stream
        self.keep_text = keep_text
        self.ocr = ocr
        self.control_scale = control_scale
        self.gguf = gguf
        self._backend = None
        self._backend_name: str | None = None
        # Sticky downscale: once an OOM forces a smaller working size, later
        # batch files start there instead of re-OOMing one by one.
        self._oom_scale: float = 1.0

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
                control_scale=self.control_scale,
                gguf=self.gguf,
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
            if self.low_vram or (vram is not None and vram < CHROMA_MIN_VRAM_GB):
                resolved = "zimage"
                vram_str = (
                    f"{vram:.0f} GiB VRAM (<{CHROMA_MIN_VRAM_GB:.0f})"
                    if vram is not None else "low_vram=True"
                )
                warnings.append(
                    f"auto: {vram_str} -> "
                    "zimage (DiffSynth disk-streaming); Chroma1 needs ~29 GiB"
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
        if resolved in ("sdxl", "sdxl-canny") \
                and s < profiles.SDXL_MIN_STRENGTH:
            warnings.append(
                f"{resolved} strength {s} is below the Lightning distillation "
                f"floor ({profiles.SDXL_MIN_STRENGTH}); clamping — lower "
                "values produce out-of-schedule noise"
            )
            s = profiles.SDXL_MIN_STRENGTH
        seed_v = profiles.resolve_seed(seed)
        if strength is None and resolved in ("zimage", "zimage-lite"):
            warnings.append(
                f"{resolved} floors are uncalibrated defaults; pass "
                "--strength for known-hard watermarks"
            )
        if resolved == "zimage-lite":
            warnings.append(
                "zimage-lite: GGUF-quantized transformer (uncalibrated); "
                "encode-once design keeps peak RAM ~8 GB"
            )
        if resolved == "vae":
            warnings.append(
                "vae is the weakest tier: fine for weak watermarks and as a "
                "pre-pass, insufficient alone for hard carriers like "
                "Gemini SynthID"
            )
        if resolved == "sdxl-canny":
            warnings.append(
                "sdxl-canny floors are uncalibrated: edge conditioning "
                "preserves glyph geometry but also shields watermark "
                "signal more than plain sdxl"
            )
        if resolved == "sana":
            warnings.append(
                "sana floors are uncalibrated defaults; pass --strength "
                "for known-hard watermarks"
            )
            if s < 0.5:
                warnings.append(
                    f"sana runs 1 of 2 SCM steps at strength {s}; raise "
                    "--strength to 0.5+ to run both"
                )
        if resolved == "lcm":
            warnings.append(
                "lcm floors are uncalibrated defaults; SD1.5 is 512-768 "
                "native, pass --max-side 768 for best fidelity"
            )
        if self.keep_text:
            warnings.append(
                "keep-text: original pixels are restored over detected small "
                "text, so any watermark signal under those pixels survives"
            )
        if s < 0.05:
            warnings.append(
                f"strength {s} is very low; step count is capped at "
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
            if self.ocr:
                from . import ocr as ocr_mod

                record.text = ocr_mod.extract_text(img)
                if not record.text and ocr_mod.backend_name() is None:
                    record.warnings.append(
                        "ocr: no backend installed; install "
                        "rapidocr-onnxruntime to save image text")
            orig_size = img.size
            working = img
            eff_max = max_side
            if self._oom_scale < 1.0 and max(img.size) > 64:
                eff_max = min(eff_max or max(img.size),
                              round(max(img.size) * self._oom_scale))
            if eff_max and max(img.size) > eff_max:
                scale = eff_max / max(img.size)
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
            text_mask = None
            if use_keep:
                from .textmask import composite_original, detect_text_mask

                text_mask = detect_text_mask(working)
                out = composite_original(out, working, text_mask)
            record.psnr = psnr(working, out)
            backoff_stage = None
            backoff_floor = (
                profiles.SDXL_MIN_STRENGTH
                if plan["pipeline"] in ("sdxl", "sdxl-canny")
                else BACKOFF_MIN_STRENGTH
            )
            if (record.psnr < BACKOFF_PSNR_FLOOR
                    and strength_v > backoff_floor):
                retry_s = max(backoff_floor,
                              strength_v * BACKOFF_FACTOR)
                try:
                    retry = backend.run(working, retry_s, plan["seed"])
                except Exception as exc:
                    if not _is_oom(exc):
                        raise
                    record.warnings.append(
                        "strength_backoff skipped: retry OOMed; kept "
                        "first result")
                    retry = None
                if retry is not None:
                    if retry.size != working.size:
                        retry = retry.resize(working.size,
                                             Image.Resampling.LANCZOS)
                    if text_mask is not None:
                        from .textmask import composite_original

                        retry = composite_original(
                            retry, working, text_mask)
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
                if use_keep:
                    from .textmask import composite_original, detect_text_mask

                    orig_mask = detect_text_mask(img)
                    out = composite_original(out, img, orig_mask)
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
            self._oom_scale = min(self._oom_scale,
                                  small.width / working.width)
            record.warnings.append(
                f"oom_retry: CUDA OOM at {working.size}, retried at "
                f"{small.size} (detail loss possible; later files start "
                f"at {self._oom_scale:.2f}x)")
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
                if self.ocr:
                    from . import ocr as ocr_mod

                    try:
                        r.text = ocr_mod.extract_text(load_rgb(p))
                    except Exception:
                        r.text = []
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
