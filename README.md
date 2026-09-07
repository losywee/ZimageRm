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
| `vae` | SD VAE (`sd-vae-ft-mse`) round-trip + latent noise (`strength` = noise std) | — |

`duo` auto-collapses to `chroma` for OpenAI/Microsoft provenance (measured
lower floors there), mirroring published cross-engine calibration.

## Lightweight tiers

| Pipeline | Fetch | VRAM | Use when |
|---|---|---|---|
| `vae` | **~350 MB** | minimal | weak watermarks, quick sweeps, pre-pass before a heavier engine |
| `sdxl` | ~8 GB | ~10 GB (cpu-offload with `--low-vram`) | calibrated middle tier — best strength/fidelity-per-GB |
| `zimage` | ~21 GB | 8–10 GB (fp8 streaming) | best fidelity on 15 GB-class cards |
| `chroma`/`duo` | ~33 GB | ~29 GiB | strongest disruption, big cards only |

## Strength policy (vendor floors)

| Vendor | chroma / duo | sdxl | zimage | vae |
|---|---|---|---|---|
| google | 0.40 | 0.25 | 0.30 | 0.15 (noise std) |
| openai | 0.09 | 0.15 | 0.12 | 0.15 |
| microsoft | 0.125 | 0.25* | 0.15 | 0.15 |
| meta | 0.17 | 0.25* | 0.15 | 0.15 |
| unknown | 0.40 | 0.25 | 0.30 | 0.15 |

Chroma and SDXL floors follow published oracle calibrations (fixed seed 0,
fixed steps/guidance/prompt — the floors are bound to exactly that
configuration). Z-Image floors are **uncalibrated defaults**; `*` = no
measured SDXL cohort, falls to the unknown floor (conservative). Vendor is
auto-sniffed from C2PA/XMP provenance in the file (or pass `--vendor`).

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt   # CUDA torch required
pip install -e .
```

An NVIDIA CUDA GPU is required for the diffusion stages (bf16). Everything
else (provenance sniffing, metadata stripping, reports) runs anywhere.

### Models and disk usage

| Pipeline | Fetches | Size |
|---|---|---|
| `chroma` / `duo` | `lodestones/Chroma1-HD` (bf16) | ~27.5 GB |
| `zimage` | `Tongyi-MAI/Z-Image-Turbo` | ~12 GB |

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
python tests/test_profiles.py     # floors, routing, step compensation math
python tests/test_vendor.py       # provenance byte-scan
python tests/test_io.py           # metadata stripping, PSNR
python tests/test_engine_fake.py  # orchestration with a faked backend
```

ML-boundary code is guarded behind imports so all of the above runs without
torch; end-to-end regeneration needs a CUDA machine with the weights
(downloaded automatically on first run via Hugging Face).
