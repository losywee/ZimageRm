# Technical Analysis — Portable AI Watermark Remover Pro (Windows v1.0.0)

Reverse-engineering notes for the unpacked portable application at
`/private/tmp/Portable_AI_Watermark_Remover_Pro_win_v1.0.0/`. All findings
below are derived from on-disk evidence; nothing is inferred from the vendor's
marketing.

## 1. Executive summary

A portable Windows desktop application that removes **visible** watermarks
from images and videos by **mask-and-inpaint**, not by regeneration. It ships
two classical inpainting networks with renamed weights:

| Media | Engine | Origin |
|---|---|---|
| Images | **Big-LaMa** | `resources/big-lama` (torch.package) + `core.bin` |
| Video | **ProPainter** | `engine/vcore/` (S-Lab licensed framework) |

There is **no** diffusion regeneration, **no** invisible-watermark detection
(SynthID/C2PA), and **no** metadata handling anywhere in the bundle.

## 2. Packaging

| Component | Evidence |
|---|---|
| PyInstaller onefile exe | `AI Watermark Remover Pro/` — split PE sections (`.text` 55 MB, `.rdata` 16 MB, `.data`, `.pdata`, `.rsrc`, `.reloc`); the app's own Python bytecode lives in the PyInstaller archive inside these sections |
| Embedded Python 3.12 | `python312.dll`, `python3.dll`, stdlib in `_pylibs/` (164 top-level modules) |
| GUI framework | PySide6 / Qt6 (`qt6core/gui/widgets/pdf/svg/network.dll`, `shiboken6`, `pyside6.abi3.dll`) |
| CV stack | `torch` + `torchgen` + `torchvision` (4.1 GB, CUDA build), `cv2/`, `numpy/`, `PIL/`, `imageio_ffmpeg/`, torch deps (`sympy`, `networkx`, `jinja2`, `fsspec`, `filelock`) |

The directory layout is characteristic of a PyInstaller onefile build that
was unpacked (e.g. with a PE-section splitter), leaving the PYZ archive
embedded in the sections rather than extracted.

## 3. Image pipeline — Big-LaMa

- `resources/big-lama/`: a **torch.package** archive — `code/`,
  `constants.pkl`, `data.pkl`, 991 `data/` shards, `version`.
- `resources/core.bin` (196 MB): the **same archive** (ZIP with
  `big-lama/data/...` entry names), renamed to hide the model identity.
- Big-LaMa is a Fourier-convolutions inpainting network. In this app it
  fills masked regions of still images: watermark mask in, clean pixels out.
- The mask source is the app's own logic (compiled into the PYZ): either a
  user-painted/box mask or automatic detection of registered visible marks.
  That logic is **not** recoverable from the bundle without decompiling the
  PYZ (see §7).

## 4. Video pipeline — ProPainter

`engine/vcore/` is the complete **ProPainter** video-inpainting framework
(NUS S-Lab), plus its Hugging Face web-demo code:

```
engine/vcore/
├── RAFT/                 optical flow extraction (corr, extractor, raft.py…)
├── model/
│   ├── propainter.py     InpaintGenerator (the flow-guided video inpainter)
│   ├── recurrent_flow_completion.py   flow completion network
│   ├── modules/, misc.py, vgg_arch.py
├── core/                 training utilities (dataset, trainer, metrics…)
├── web-demos/hugging_face/inpainter/base_inpainter.py
└── weights/  m0.bin (150 MB)  m1.bin (20 MB)  m2.bin (19 MB)
```

- `base_inpainter.py` is ProPainter's official HF demo inpainter, modified:
  comments are in **Vietnamese**, and `scipy.ndimage.binary_dilation` was
  replaced with an equivalent cv2 implementation ("Thay scipy … khỏi phụ
  thuộc scipy") to drop a dependency.
- The three `m*.bin` weights match ProPainter's three checkpoints
  (inpaint generator / RAFT / flow-completion) with the original names
  stripped.
- Pipeline semantics (from the demo code): decode video (torchvision /
  imageio-ffmpeg) → resize to /8 grid → RAFT flow → flow completion →
  recurrent propagation + inpainting → re-encode with audio.

## 5. Runtime characteristics

- Ships a full CUDA torch build (4.1 GB) — GPU expected; no CPU-fallback
  logic is visible outside the compiled app code.
- ffmpeg arrives via `imageio_ffmpeg` (bundled binary), so video
  decode/encode needs no system ffmpeg.
- Everything is offline; no downloader or telemetry appears in the bundle.

## 6. Licensing and legal flags

1. **ProPainter / S-Lab License 1.0** (`engine/vcore/LICENSE`): explicitly
   **non-commercial**; commercial use requires written permission from the
   authors. Redistributing the weights inside a paid "Pro" product is, at
   minimum, license-questionable.
2. **Big-LaMa** (Apache-2.0, SAIC) is fine to redistribute, but the renaming
   to `core.bin` obscures attribution.
3. Removing provenance marks from media the user does not own remains the
   user's legal responsibility; the tool itself performs no ownership checks.

## 7. Open questions (recoverable)

The GUI wiring, mask-detection heuristics, and any registered-mark templates
live in the PyInstaller PYZ inside the `.text`/`.rdata` sections. They are
recoverable with `pyinstxtractor` + `decompyle3`/`pycdc` on the extracted
`.pyc` entries. Likely findings: a template-matching detector for a handful
of vendor logos, rect-mask suggestion, and Qt threading around the two
inpainting engines.

## 8. Comparison with zinvis (our approach)

| | This app | zinvis |
|---|---|---|
| Target | **Visible** marks (logo/text overlays) | **Invisible** pixel/frequency watermarks |
| Method | mask + LaMa/ProPainter inpainting | diffusion regeneration (Z-Image Turbo, Chroma1) |
| Invisible watermarks | unaffected | disrupted by regeneration strength |
| GPUs | CUDA torch, ~6–8 GB | 15 GB fp8-streaming (Z-Image) / ~29 GiB (Chroma) |
| Weight licensing | S-Lab non-commercial (ProPainter) | Apache-2.0 models only |

The two are complementary: visible overlay removal (inpainting) vs provenance
signal removal (regeneration). Neither replaces the other.
