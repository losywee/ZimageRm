from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, profiles
from .engine import ZinvisEngine
from .io_utils import BatchReport, ImageRecord
from .vendor import sniff_vendor


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="zinvis",
        description=(
            "Remove invisible AI image watermarks by regenerating the frame "
            "with Z-Image Turbo and/or Chroma1 (CUDA required)."
        ),
    )
    ap.add_argument("input", help="image file or directory")
    ap.add_argument("output", help="output file or directory")
    ap.add_argument("--pipeline", default=None,
                    choices=list(profiles.PROFILES),
                    help="duo = Chroma1 + Z-Image refinement; default: auto "
                         "(zimage disk-streaming on GPUs <30 GiB, duo otherwise)")
    ap.add_argument("--low-vram", action="store_true",
                    help="force Z-Image DiffSynth disk-streaming (small cards)")
    ap.add_argument("--stream", action="store_true", default=None,
                    help="force Z-Image DiffSynth disk-streaming (fp8, ~10 GB "
                         "RAM; default on small cards)")
    ap.add_argument("--cpu-offload", action="store_true",
                    help="force Z-Image DiffSynth CPU bf16 offload (~20 GB "
                         "RAM, lower latency; needs a big-RAM host)")
    ap.add_argument("--vendor", default=None,
                    choices=["google", "openai", "microsoft", "meta"],
                    help="strength cohort; auto-sniffs provenance when omitted")
    ap.add_argument("--strength", default=None, type=float)
    ap.add_argument("--seed", default=None, type=int)
    ap.add_argument("--refine-strength", default=profiles.DUO_REFINE_STRENGTH,
                    type=float, help="duo: Z-Image Turbo refinement pass strength")
    ap.add_argument("--psnr-floor", default=profiles.DEFAULT_PSNR_FLOOR,
                    type=float, help="duo: drop the refinement pass below this PSNR")
    ap.add_argument("--max-side", default=0, type=int,
                    help="cap the long side before diffusion (0 = native)")
    ap.add_argument("--keep-text", action="store_true",
                    help="restore original pixels over detected small text "
                         "(prevents glyph garbling; watermark signal under "
                         "text survives)")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--skip-existing", action="store_true",
                    help="batch: skip files whose output already exists")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--report", default=None, help="write JSON report here")
    ap.add_argument("--version", action="version",
                    version=f"zinvis {__version__}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stream and args.cpu_offload:
        build_parser().error("--stream and --cpu-offload are mutually exclusive")
    stream = True if args.stream else (False if args.cpu_offload else None)
    eng = ZinvisEngine(
        hf_token=args.hf_token,
        refine_strength=args.refine_strength,
        psnr_floor=args.psnr_floor,
        low_vram=args.low_vram,
        stream=stream,
        keep_text=args.keep_text,
    )
    in_path = Path(args.input)

    if in_path.is_dir():
        report_path = args.report or str(Path(args.output) / "summary.json")
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        collected: list = []

        def _write_partial():
            Path(report_path).write_text(BatchReport(
                in_dir=str(in_path), out_dir=str(args.output),
                images=collected,
            ).to_json())

        def _progress(r):
            line = f"[{r.status}] {Path(r.input).name}"
            if r.error:
                line += f" ({r.error})"
            elif r.status == "cleaned":
                line += f" psnr={r.psnr:.1f}dB strength={r.strength}"
            print(line, flush=True)
            collected.append(r)
            _write_partial()

        br = eng.run_dir(args.input, args.output, args.pipeline,
                         args.vendor, args.strength, args.seed, args.glob,
                         args.max_side, skip_existing=args.skip_existing,
                         progress=_progress)
        Path(report_path).write_text(br.to_json())
        s = br.summary
        print(f"processed {s['total']} images in {s['seconds']:.1f}s | "
              f"ok={s['cleaned']} failed={s['failed']} skipped={s['skipped']}")
        print("report:", report_path)
        return 0 if s["failed"] == 0 else 1

    r = eng.run_file(args.input, args.output, args.pipeline, args.vendor,
                     args.strength, args.seed, args.max_side)
    _print_record(r)
    return 0 if r.error is None else 1


def _print_record(r: ImageRecord) -> None:
    print(f"input   : {r.input}")
    print(f"output  : {r.output}")
    print(f"pipeline: {r.pipeline} -> {r.resolved_pipeline} "
          f"(vendor={r.vendor}, strength={r.strength}, seed={r.seed})")
    for w in r.warnings:
        print(f"warning : {w}")
    if r.error:
        print(f"error   : {r.error}")
        return
    print(f"psnr    : {r.psnr:.1f} dB")
    print(f"stages  : {', '.join(r.stages)}")
    print(f"time    : {r.seconds:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
