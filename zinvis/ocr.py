"""Optional text extraction (OCR) captured before model processing.

Backends are probed in order and the first installed one wins; none are
required dependencies. If nothing is installed, extraction returns an
empty list and the engine records a warning instead.

    rapidocr-onnxruntime  (pip install rapidocr-onnxruntime)
    easyocr               (pip install easyocr)
    pytesseract           (pip install pytesseract + system tesseract)
"""

from __future__ import annotations

import threading

_backend = None
_backend_name: str | None = None
_lock = threading.Lock()


def backend_name() -> str | None:
    """Name of the OCR backend that would be used, None if none installed."""
    _get_backend()
    return _backend_name


def _get_backend():
    global _backend, _backend_name
    with _lock:
        if _backend_name is not None or _backend == "none":
            return _backend if _backend != "none" else None
        try:
            from rapidocr_onnxruntime import RapidOCR

            _backend = RapidOCR()
            _backend_name = "rapidocr"
            return _backend
        except Exception:
            pass
        try:
            import easyocr

            _backend = ("easyocr", easyocr.Reader(["en"], gpu=False))
            _backend_name = "easyocr"
            return _backend
        except Exception:
            pass
        try:
            import pytesseract  # noqa: F401

            _backend = "pytesseract"
            _backend_name = "pytesseract"
            return _backend
        except Exception:
            pass
        _backend = "none"
        return None


def extract_text(image) -> list[str]:
    """Best-effort text extraction; never raises."""
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    be = _get_backend()
    if be is None:
        return []
    try:
        if _backend_name == "rapidocr":
            result, _ = be(arr)
            if not result:
                return []
            dets = sorted(result, key=lambda d: (d[0][0][1], d[0][0][0]))
            return [str(d[1]).strip() for d in dets if str(d[1]).strip()]
        if _backend_name == "easyocr":
            _, reader = be
            result = reader.readtext(arr)
            dets = sorted(result, key=lambda d: (d[0][0][1], d[0][0][0]))
            return [str(d[1]).strip() for d in dets if str(d[1]).strip()]
        if _backend_name == "pytesseract":
            import pytesseract

            data = pytesseract.image_to_data(
                image.convert("RGB"),
                output_type=pytesseract.Output.DICT,
            )
            words = []
            for text, conf in zip(
                    data["text"], data["conf"], strict=False):
                t = str(text).strip()
                if t and float(conf) > 0:
                    words.append(t)
            return words
    except Exception:
        return []
    return []
