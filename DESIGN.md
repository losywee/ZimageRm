# zinvis — design

Independent, image-only invisible-watermark removal. Decisions and rationale.

## Goals

1. **Independent build**: only `diffusers` / `DiffSynth-Studio` as ML
   dependencies; no vendored research repos, no third-party watermark
   removers. Pure-Python parts (profiles, vendor sniffing, I/O, reports)
   run without torch.
2. **Z Image Turbo + Chroma**, whole-frame regeneration, **no face repair**:
   the watermark kill comes from regeneration strength, not from regional
   inpainting. Identity preservation is delegated to low strengths and the
   PSNR gate rather than face pipelines.
3. Reproducible: fixed seed (0) by default, fixed step counts per engine,
   strength resolved from a table, not vibes.

## Architecture

```
cli.py ──► ZinvisEngine (engine.py)
             ├─ plan(): resolve_pipeline + resolve_strength + sniff_vendor
             ├─ backend_for(name) ──► regen/
             │    ├─ ChromaBackend  (diffusers.ChromaImg2ImgPipeline,
             │    │                  lodestones/Chroma1-HD, bf16, CUDA)
             │    ├─ ZImageBackend  (diffusers.ZImageImg2ImgPipeline, falls
             │    │                  back to DiffSynth ZImagePipeline;
             │    │                  Tongyi-MAI/Z-Image-Turbo, CUDA)
             │    └─ DuoBackend     (chroma global → zimage refine, PSNR-gated)
             └─ run_file / run_dir ──► io_utils (strip-on-save, PSNR, JSON)
```

- **Backends** share one contract: `run(PIL.Image, strength, seed) -> PIL.Image`,
  lazy single-load, latent-grid size flooring with LANCZOS restore to the
  exact input size.
- **duo** is a composition, not a new model: stage 2 runs at 0.18 on the
  stage-1 output and is rolled back when combined PSNR falls under the floor
  (default 24 dB) — a fidelity guard, not a watermark verdict.
- **Step compensation**: diffusers img2img truncates the step count by
  `strength`, so the requested count is `ceil(effective_steps / strength)` to
  always spend the calibrated effective steps (4 for Chroma, 8 for Z-Image).
- **Metadata**: outputs are rebuilt pixel-fresh (`save_stripped`), so EXIF,
  XMP, C2PA-ish text chunks never survive the write.

## Strength policy

Per-vendor flat floors live in `profiles.py` as data, with the chroma column
taken from the published 2026-08 four-cohort oracle calibration of the
chroma-zimage recipe (same prompt/guidance/step semantics are pinned here so
the floors stay bound to this engine). The zimage column is an uncalibrated
conservative default — the CLI warns and points at `--strength`.

Vendor routing: `duo` → chroma for openai/microsoft (lower measured floors),
otherwise duo; explicit `--pipeline chroma|zimage` bypasses routing.
`vendor.py` is a self-contained head+tail byte scan (1 MB windows) with
ordered markers; it is a heuristic signal for strength selection only —
never a detection verdict.

## Failure behavior

- No CUDA / missing weights → per-image error record, exit 1, no partial
  output written for that image (save happens only after a successful pass).
- Backend load failures surface the import that failed and the install hint.
- Batch: errors are captured in `summary.json`; the run continues.

## Test seams

All ML glue sits behind `run()` boundaries and guarded imports, so the
profile math, vendor scan, metadata stripping, and engine orchestration are
unit-tested with fake backends in the CI venv (no torch, no downloads).
