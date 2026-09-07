"""Small-text preservation mask for diffusion regeneration.

Diffusion img2img redraws glyphs; below a legibility threshold the redraw
hallucinates (garbled small text). This module detects probable text regions
with a dependency-free edge-density detector (numpy + PIL only) and composites
the *original* pixels back over the regenerated output, so small text stays
byte-identical while the rest of the frame is cleaned.

Tradeoff: any watermark signal under the restored text pixels survives. The
mask is therefore conservative by design — it only fires on dense,
high-contrast edge clusters typical of rendered glyphs, and only on small
components (large display type regenerates fine and stays cleaned).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Components taller than this fraction of the frame are left to regenerate.
MAX_TEXT_HEIGHT_FRAC = 0.06
# Components smaller than this are noise, not glyphs.
MIN_COMPONENT_PX = 12
# Edge-density threshold inside a component's bbox to count as text.
MIN_EDGE_DENSITY = 0.08
# Feather radius (px) for the composite seam.
FEATHER_RADIUS = 2


def _gray(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]).astype(np.float32)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    mag = np.hypot(gx, gy)
    thr = float(np.percentile(mag, 92))
    return (mag >= max(thr, 8.0)).astype(np.uint8)


def _connected_components(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Two-pass union-find labeling; returns bounding boxes (x0, y0, x1, y1)."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]
    nxt = 0

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for y in range(h):
        row = binary[y]
        lab = labels[y]
        for x in range(w):
            if not row[x]:
                continue
            left = int(lab[x - 1]) if x > 0 else 0
            up = int(labels[y - 1, x]) if y > 0 else 0
            if not left and not up:
                nxt += 1
                parent.append(nxt)
                lab[x] = nxt
            elif left and up:
                lab[x] = left
                rl, ru = find(left), find(up)
                if rl != ru:
                    parent[max(rl, ru)] = min(rl, ru)
            else:
                lab[x] = left or up

    boxes: dict[int, list[int]] = {}
    for y in range(h):
        for x in range(w):
            v = int(labels[y, x])
            if not v:
                continue
            r = find(v)
            labels[y, x] = r
            if r in boxes:
                b = boxes[r]
                if x < b[0]:
                    b[0] = x
                if y < b[1]:
                    b[1] = y
                if x > b[2]:
                    b[2] = x
                if y > b[3]:
                    b[3] = y
            else:
                boxes[r] = [x, y, x, y]
    return [(b[0], b[1], b[2], b[3]) for b in boxes.values()]


def detect_text_mask(image: Image.Image) -> Image.Image:
    """Return an L-mode mask (255 = probable small text) at image size."""
    w, h = image.size
    arr = np.asarray(image.convert("RGB"))
    edges = _edge_map(_gray(arr))
    boxes = _connected_components(edges)
    max_h = max(4, int(h * MAX_TEXT_HEIGHT_FRAC))
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        area = bw * bh
        if area < MIN_COMPONENT_PX or bh > max_h:
            continue
        region = edges[y0:y1 + 1, x0:x1 + 1]
        if region.size and region.mean() >= MIN_EDGE_DENSITY:
            mask[y0:y1 + 1, x0:x1 + 1] = 255
    m = Image.fromarray(mask, "L")
    if FEATHER_RADIUS > 0:
        m = m.filter(ImageFilter.GaussianBlur(FEATHER_RADIUS))
    return m


def composite_original(cleaned: Image.Image, original: Image.Image,
                       mask: Image.Image) -> Image.Image:
    """Paste original pixels over cleaned where mask is bright (feathered)."""
    if cleaned.size != original.size:
        original = original.resize(cleaned.size, Image.Resampling.LANCZOS)
    if mask.size != cleaned.size:
        mask = mask.resize(cleaned.size, Image.Resampling.LANCZOS)
    base = cleaned.convert("RGB").copy()
    base.paste(original.convert("RGB"), (0, 0), mask.convert("L"))
    return base
