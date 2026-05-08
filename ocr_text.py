"""
Step 3: OCR the left-side text panel.
Uses PaddleOCR for Traditional Chinese (supports vertical text).
Runs on AutoDL with GPU, or locally on M3 (see config).

Tall vertical strips (e.g. 连环画) often OCR poorly as raw vertical layout; optional
90° CW rotation + horizontal reading order is toggled via phase2_manifest.json (see main_autodl.py).
"""
from __future__ import annotations

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from configs.config import (
    OCR_DET_LIMIT_SIDE_LEN,
    OCR_LANG,
    OCR_RETRY_WITH_ENHANCEMENT,
    OCR_USE_GPU,
)


_ocr = None
# Filled by main_autodl from crops_dir/phase2_manifest.json (packaged in crops_left.zip).
_phase2_manifest: dict = {}


def set_phase2_manifest(manifest: dict | None) -> None:
    """AutoDL Phase 2: apply book-specific OCR hints from uploaded manifest."""
    global _phase2_manifest
    _phase2_manifest = dict(manifest or {})


def _unwrap_ocr_payload(result):
    if result is None:
        return None

    if isinstance(result, dict):
        payload = result
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        payload = result[0]
    else:
        payload = result
    if payload is None:
        return None

    for attr in ("json", "to_dict"):
        fn = getattr(payload, attr, None)
        if callable(fn):
            try:
                payload = fn()
                break
            except Exception:
                pass

    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]
    return payload


def _poly_bounds_xy(poly_like) -> tuple[float, float, float, float]:
    if poly_like is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        arr = np.asarray(poly_like, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        xs = arr[..., 0]
        ys = arr[..., 1]
        return (
            float(np.min(xs)),
            float(np.min(ys)),
            float(np.max(xs)),
            float(np.max(ys)),
        )
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0, 0.0, 0.0


def _poly_x_center(poly_like) -> float:
    xmin, _, xmax, _ = _poly_bounds_xy(poly_like)
    return (xmin + xmax) / 2.0


def _poly_y_min(poly_like) -> float:
    _, ymin, _, _ = _poly_bounds_xy(poly_like)
    return ymin


def _poly_x_min(poly_like) -> float:
    xmin, _, _, _ = _poly_bounds_xy(poly_like)
    return xmin


def _poly_text_pairs_from_result(result) -> list[tuple[object, str]]:
    """Parse PaddleOCR v2 / v3 style output into (polygon, text) pairs."""
    payload = _unwrap_ocr_payload(result)
    if payload is None:
        return []

    if isinstance(payload, dict) and "rec_texts" in payload:
        texts = payload["rec_texts"]
        polys = payload.get("rec_polys")
        if polys is None:
            polys = payload.get("dt_polys")
        if polys is not None and len(polys) == len(texts):
            pairs: list[tuple[object, str]] = []
            for poly, t in zip(polys, texts):
                if t:
                    pairs.append((poly, str(t)))
            return pairs
        return [(None, str(t)) for t in texts if t]

    if not isinstance(payload, (list, tuple)):
        return []

    pairs = []
    for item in payload:
        if not item:
            continue
        if not isinstance(item, (list, tuple)):
            continue
        if len(item) < 2:
            continue
        a, b = item[0], item[1]
        box, text = None, None
        if isinstance(a, str):
            text = a
            box = b if not isinstance(b, str) else None
        elif isinstance(b, str):
            text = b
            box = a if not isinstance(a, str) else None
        else:
            box = a
            if isinstance(b, (list, tuple)) and len(b) >= 1:
                text = b[0]
            elif isinstance(b, str):
                text = b
        if text and box is not None:
            pairs.append((box, str(text)))
    return pairs


def _lines_from_ocr_result(result, *, reading_order: str) -> list[str]:
    pairs = _poly_text_pairs_from_result(result)
    if not pairs:
        return []

    if reading_order == "horizontal_tb_lr":
        pairs.sort(key=lambda p: (_poly_y_min(p[0]), _poly_x_min(p[0])))
    else:
        # Vertical comic strip: columns right-to-left, top-to-bottom within a column.
        pairs.sort(key=lambda p: (-_poly_x_center(p[0]), _poly_y_min(p[0])))
    return [t for _, t in pairs if t]


def _scores_from_ocr_result(result) -> list[float]:
    """Extract per-line confidence scores when PaddleOCR provides them."""
    payload = _unwrap_ocr_payload(result)
    if payload is None:
        return []

    if isinstance(payload, dict):
        scores = payload.get("rec_scores")
        if isinstance(scores, (list, tuple)):
            out = []
            for s in scores:
                try:
                    out.append(float(s))
                except (TypeError, ValueError):
                    continue
            return out
        return []

    if not isinstance(payload, (list, tuple)):
        return []

    out = []
    for item in payload:
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[1], (list, tuple))
            and len(item[1]) >= 2
        ):
            try:
                out.append(float(item[1][1]))
            except (TypeError, ValueError):
                continue
    return out


def get_ocr():
    global _ocr
    if _ocr is None:
        # PaddleOCR constructor args changed across versions (pipelines API: device=, not use_gpu=).
        # Try modern args first, then legacy.
        dev = "gpu" if OCR_USE_GPU else "cpu"
        det_kw: dict = {}
        if OCR_DET_LIMIT_SIDE_LEN and int(OCR_DET_LIMIT_SIDE_LEN) > 0:
            det_kw["det_limit_side_len"] = int(OCR_DET_LIMIT_SIDE_LEN)

        candidates = [
            dict(lang=OCR_LANG, use_angle_cls=True, device=dev, show_log=False, **det_kw),
            dict(lang=OCR_LANG, use_angle_cls=True, show_log=False, **det_kw),
            dict(lang=OCR_LANG, show_log=False, **det_kw),
            dict(
                lang=OCR_LANG,
                use_angle_cls=True,
                use_gpu=OCR_USE_GPU,
                show_log=False,
                **det_kw,
            ),
        ]
        # If det_limit_side_len is unsupported, retry without it.
        candidates.append(
            dict(lang=OCR_LANG, use_angle_cls=True, device=dev, show_log=False)
        )
        candidates.append(
            dict(lang=OCR_LANG, use_angle_cls=True, use_gpu=OCR_USE_GPU, show_log=False)
        )

        last_err = None
        for kwargs in candidates:
            try:
                _ocr = PaddleOCR(**kwargs)
                break
            except Exception as e:
                last_err = e
        if _ocr is None and last_err is not None:
            raise last_err
    return _ocr


def _count_cjk(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _ocr_once(ocr: PaddleOCR, arr: np.ndarray, *, reading_order: str) -> tuple[str, float]:
    try:
        result = ocr.ocr(arr, cls=True)
    except TypeError:
        # Some PaddleOCR versions no longer support cls=...
        result = ocr.ocr(arr)
    lines = _lines_from_ocr_result(result, reading_order=reading_order)
    text = "\n".join(lines)
    scores = _scores_from_ocr_result(result)
    conf = float(np.mean(scores)) if scores else 0.0
    return text, conf


def _enhance_for_ocr(arr: np.ndarray) -> np.ndarray:
    """Simple deterministic enhancement for old scanned pages."""
    if arr.ndim == 3:
        gray = (
            0.299 * arr[..., 0].astype(np.float32)
            + 0.587 * arr[..., 1].astype(np.float32)
            + 0.114 * arr[..., 2].astype(np.float32)
        )
    else:
        gray = arr.astype(np.float32)

    # Stretch contrast and slightly darken text strokes.
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    span = max(1.0, p95 - p5)
    norm = np.clip((gray - p5) * (255.0 / span), 0, 255)
    enhanced = np.clip(norm * 0.92, 0, 255).astype(np.uint8)
    return enhanced


def ocr_image(pil_image: Image.Image) -> str:
    rotate_cw90 = bool(_phase2_manifest.get("ocr_rotate_left_cw90"))
    reading_order = "horizontal_tb_lr" if rotate_cw90 else "vertical_rl_tt"

    img = pil_image.convert("RGB")
    if rotate_cw90:
        # 90° clockwise: vertical columns become horizontal rows (much better Paddle accuracy).
        img = img.transpose(Image.Transpose.ROTATE_270)

    arr = np.array(img)
    ocr = get_ocr()

    base_text, base_conf = _ocr_once(ocr, arr, reading_order=reading_order)
    if not OCR_RETRY_WITH_ENHANCEMENT:
        return base_text

    # Retry not only for "too few chars", but also for low-confidence long outputs.
    base_cjk = _count_cjk(base_text)
    if base_cjk >= 40 and base_conf >= 0.88:
        return base_text

    enhanced_arr = _enhance_for_ocr(arr)
    enhanced_text, enhanced_conf = _ocr_once(ocr, enhanced_arr, reading_order=reading_order)
    enhanced_cjk = _count_cjk(enhanced_text)

    if enhanced_conf > base_conf + 0.03:
        return enhanced_text
    if enhanced_cjk > base_cjk + 8:
        return enhanced_text
    return base_text
