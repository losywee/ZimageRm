# zinvis

Independent CLI that removes **invisible AI watermarks from images** by
regenerating the whole frame with diffusion models — **Z-Image Turbo** and
**Chroma1** — using calibrated per-vendor denoising strengths.

Standalone build: no vendored repos, no third-party watermark library. The
diffusion stack comes from `diffusers` / `DiffSynth-Studio` only.

> For research and for use on images you own or generated. Watermark systems
> change; verify important outputs with the provider's own verifier.

## How it works

Invisible pixel/frequency-domain watermarks are fragile against
regeneration: an img2img diffusion pass at sufficient denoising strength
re-projects the image off the manifold the watermark decoder expects, while
`strength`, prompt, steps, and seed are kept at values that preserve visual
content.

```
input.png ──► [stage 1: Chroma1-HD img2img, strength = vendor floor]
          ──► [stage 2 (duo): Z-Image Turbo refinement, low strength, PSNR-gated]
          ──► metadata-stripped output + JSON report
```

| Pipeline | Stage 1 | Stage 2 |
|---|---|---|
| `chroma` | Chroma1-HD (`lodestones/Chroma1-HD`, Apache-2.0) img2img, 4 effective steps, guidance 5.0 | — |
| `zimage` | Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`) img2img, 8 steps, guidance 1.0 | — |
| `duo` (default) | Chroma1 global pass | Z-Image Turbo refinement (default 0.18, PSNR-floor gated) |
| `sdxl` | SDXL-base + **SDXL-Lightning** 4-step LoRA img2img, guidance 1.0 | — |
| `sdxl-canny` | sdxl + **Canny ControlNet** (`diffusers/controlnet-canny-sdxl-1.0`, ~2.5 GB): edge conditioning pins glyph geometry, so small text survives regeneration | — |
| `sana` | SANA-Sprint 0.6B 2-step img2img (`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`), guidance 1.0, SCM-locked to 2 steps (strength <0.5 runs only 1 of them) | — |
| `lcm` | SD1.5 Dreamshaper + LCM img2img (`SimianLuo/LCM_Dreamshaper_v7`), guidance 1.0, 4 steps clamped to the strength window (`original_steps × strength`) | — |
| `vae` | SD VAE (`sd-vae-ft-mse`) round-trip + latent noise (`strength` = noise std) | — |

`duo` auto-collapses to `chroma` for OpenAI/Microsoft provenance (measured
lower floors there), mirroring published cross-engine calibration.

## Lightweight tiers

| Pipeline | Fetch | VRAM | Use when |
|---|---|---|---|
| `vae` | **~350 MB** | minimal | weak watermarks, quick sweeps, pre-pass before a heavier engine |
| `lcm` | **~4.3 GB** | ~4 GB | lightest diffusion tier (SD1.5, 512-768 native; use `--max-side 768`) |
| `sana` | ~7.7 GB | ~7 GB | 2-step, ~10x faster per image than sdxl, 1024px native |
| `sdxl` | ~8 GB | ~10 GB (cpu-offload with `--low-vram`) | calibrated middle tier — best strength/fidelity-per-GB |
| `sdxl-canny` | ~10.5 GB | ~12 GB | text/glyph preservation variant (~2x slower; uncalibrated floors) |
| `zimage` | ~21 GB | 8–10 GB (fp8 streaming) | best fidelity on 15 GB-class cards |
| `chroma`/`duo` | ~33 GB | ~29 GiB | strongest disruption, big cards only |

## Strength policy (vendor floors)

| Vendor | chroma / duo | sdxl | sdxl-canny | sana | lcm | zimage | vae |
|---|---|---|---|---|---|---|---|
| google | 0.40 | 0.25 | 0.30 | 0.30 | 0.35 | 0.30 | 0.15 (noise std) |
| openai | 0.09 | 0.15 | 0.30 | 0.15 | 0.20 | 0.12 | 0.15 |
| microsoft | 0.125 | 0.25* | 0.30 | 0.30* | 0.35* | 0.15 | 0.15 |
| meta | 0.17 | 0.25* | 0.30 | 0.30* | 0.35* | 0.15 | 0.15 |
| unknown | 0.40 | 0.25 | 0.30 | 0.30 | 0.35 | 0.30 | 0.15 |

Chroma and SDXL floors follow published oracle calibrations (fixed seed 0,
fixed steps/guidance/prompt — the floors are bound to exactly that
configuration). Z-Image, SANA and LCM floors are **uncalibrated defaults**;
`*` = no measured cohort, falls to the unknown floor (conservative). Vendor
is auto-sniffed from C2PA/XMP provenance in the file (or pass `--vendor`).

SDXL strengths below `0.15` are clamped up: the Lightning 4-step
distillation bounds the usable sigma range, and lower strengths make the
executed timesteps fall outside it (output degenerates into noise).

## Pre-processing text capture

Before the model touches an image, zinvis runs OCR on the original and
stores every detected text string in that image's `summary.json` record
(`"text": [...]`, reading order). This preserves what the image said even
if regeneration later garbles small glyphs. Uses the first installed
backend: `rapidocr-onnxruntime` (recommended, CPU, no system deps),
`easyocr`, or `pytesseract` (+ system tesseract). Disable with
`--no-ocr`; without any backend the run proceeds and records a warning.

## Small text (`--keep-text`)

Diffusion regeneration redraws glyphs; small text often comes back garbled.
`--keep-text` detects probable small-text regions (edge-density detector, no
extra dependencies) and composites the **original** pixels back over the
cleaned output with a feathered seam:

```bash
zinvis in.png out.png --pipeline sdxl --keep-text
```

Tradeoff: any watermark signal under the restored text pixels survives. The
mask is conservative (small dense edge clusters only; large display type
still regenerates). Without the flag, lowering `--strength` or using
`--pipeline vae` also reduces glyph damage.

## Automatic recovery

- **CUDA OOM**: the engine empties the CUDA cache and retries the failed
  image at 0.75x then 0.5x resolution instead of erroring (a warning is
  recorded; some detail loss is possible).
- **Overcooking**: if output PSNR drops below 20 dB, the engine retries once
  at 0.7x strength and keeps the better result (`strength_backoff` stage).
- **EXIF orientation**: phone photos are auto-rotated to display orientation
  on load, so outputs never come out sideways.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt   # CUDA torch required
pip install -e .
```

An NVIDIA CUDA GPU is required for the diffusion stages (bf16). Everything
else (provenance sniffing, metadata stripping, reports) runs anywhere.

### Google Colab (15 GB card)

Colab already ships CUDA torch — do **not** install `requirements-ml.txt`
(it would reinstall torch and risk a CUDA mismatch):

```python
!git clone https://github.com/losywee/ZimageRm /content/ZimageRm
%cd /content/ZimageRm
!pip uninstall -y torchao
!pip install -r requirements.txt diffusers transformers accelerate peft safetensors 'diffsynth>=2.0.17,<3' beautifulsoup4 ftfy rapidocr-onnxruntime
!pip install -e . --no-deps
!zinvis /content/drive/MyDrive/2026090701 /content/drive/MyDrive/2026090701clean/ \
  --glob '*.png' --pipeline sdxl --low-vram --skip-existing
```

(`pip uninstall -y torchao` matters: Colab ships torchao 0.10, and peft >= 0.20
raises an ImportError while loading LoRA adapters when torchao < 0.16 is
installed. zinvis never uses torchao quantization, so removing it is safe;
the sdxl backend also patches this gate at runtime as a fallback.)

### Models and disk usage

| Pipeline | Fetches | Size |
|---|---|---|
| `chroma` / `duo` | `lodestones/Chroma1-HD` (bf16) | ~27.5 GB |
| `zimage` | `Tongyi-MAI/Z-Image-Turbo` | ~21 GB |

Weights are downloaded once into the Hugging Face cache
(`~/.cache/huggingface/hub`) and reused from disk on every later run.
Want a light first run? `--pipeline zimage` skips the Chroma download
entirely. Set `HF_HOME` to relocate the cache, and export
`HF_HUB_ENABLE_HF_TRANSFER=1` (with `pip install hf_transfer`) for faster
downloads.

## Usage

```bash
zinvis in.png out.png                                  # auto pipeline, auto vendor, seed 0
zinvis in.png out.png --pipeline chroma --vendor google
zinvis in.png out.png --pipeline zimage --strength 0.35 --seed 7
zinvis ./images/ ./clean/ --glob '*.png' --report clean/summary.json
```

Exit codes: `0` ok, `1` error (per-image errors are captured in batch
reports and the run continues). Output metadata (EXIF/XMP/C2PA-style text
chunks) is always stripped on write.

## Small GPUs (lightweight mode)

`--pipeline` defaults to **auto**, resolved from your VRAM:

| VRAM | auto resolves to | Weight mode |
|---|---|---|
| < 30 GiB | `zimage` | **fp8 disk-streaming** (DiffSynth): ~8–10 GiB VRAM, slower per image |
| >= 30 GiB | `duo` | bf16 resident |

On a 15 GB card: `auto` (or explicit `--pipeline zimage`) runs Z-Image Turbo
fp8-streaming end-to-end. `chroma`/`duo` need ~29 GiB and will OOM — the app
warns loudly before attempting. `--low-vram` forces streaming mode
everywhere it applies. Trade-off: streaming is noticeably slower per image
(weights shuttle disk -> CPU -> GPU), but nothing else changes.

## Tests

```bash
python tests/test_profiles.py        # floors, routing, step compensation math
python tests/test_vendor.py          # provenance byte-scan
python tests/test_io.py              # metadata stripping, PSNR, EXIF, JSON
python tests/test_engine_fake.py     # orchestration with a faked backend
python tests/test_textmask.py        # keep-text detector + composite
python tests/test_torchao_compat.py  # peft/torchao gate shim
```

ML-boundary code is guarded behind imports so all of the above runs without
torch; end-to-end regeneration needs a CUDA machine with the weights
(downloaded automatically on first run via Hugging Face).
