"""
Step 3: OCR the left-side text panel.
Uses PaddleOCR for Traditional Chinese (supports vertical text).
Runs on AutoDL with GPU, or locally on M3 (see config).
"""
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from configs.config import OCR_LANG, OCR_USE_GPU, OCR_RETRY_WITH_ENHANCEMENT


_ocr = None


def _ocr_sort_x(box_like) -> float:
    """Left-most x of a detection polygon (for right-to-left column order)."""
    try:
        arr = np.asarray(box_like, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        return float(np.min(arr[..., 0]))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _lines_from_ocr_result(result) -> list[str]:
    """Normalize PaddleOCR v2 list output and PaddleOCR 3 / PaddleX dict output."""
    if result is None:
        return []

    if isinstance(result, dict):
        payload = result
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        payload = result[0]
    else:
        payload = result
    if payload is None:
        return []

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

    if isinstance(payload, dict) and "rec_texts" in payload:
        texts = payload["rec_texts"]
        polys = payload.get("rec_polys")
        if polys is None:
            polys = payload.get("dt_polys")
        if polys is not None and len(polys) == len(texts):
            items = []
            for poly, t in zip(polys, texts):
                if not t:
                    continue
                items.append((_ocr_sort_x(poly), str(t)))
            items.sort(key=lambda x: -x[0])
            return [t for _, t in items]
        return [str(t) for t in texts if t]

    if not isinstance(payload, (list, tuple)):
        return []

    lines: list[tuple[float, str]] = []
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
        if not text:
            continue
        x = _ocr_sort_x(box) if box is not None else 0.0
        lines.append((x, str(text)))
    lines.sort(key=lambda x: -x[0])
    return [t for _, t in lines]


def _scores_from_ocr_result(result) -> list[float]:
    """Extract per-line confidence scores when PaddleOCR provides them."""
    if result is None:
        return []

    if isinstance(result, dict):
        payload = result
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        payload = result[0]
    else:
        payload = result
    if payload is None:
        return []

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
        candidates = [
            dict(lang=OCR_LANG, use_angle_cls=True, device=dev, show_log=False),
            dict(lang=OCR_LANG, use_angle_cls=True, show_log=False),
            dict(lang=OCR_LANG, show_log=False),
            dict(lang=OCR_LANG, use_angle_cls=True, use_gpu=OCR_USE_GPU, show_log=False),
        ]
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


def _ocr_once(ocr: PaddleOCR, arr: np.ndarray) -> tuple[str, float]:
    try:
        result = ocr.ocr(arr, cls=True)
    except TypeError:
        # Some PaddleOCR versions no longer support cls=...
        result = ocr.ocr(arr)
    lines = _lines_from_ocr_result(result)
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
    arr = np.array(pil_image)
    ocr = get_ocr()

    base_text, base_conf = _ocr_once(ocr, arr)
    if not OCR_RETRY_WITH_ENHANCEMENT:
        return base_text

    # Retry not only for "too short", but also for low confidence long outputs.
    base_cjk = _count_cjk(base_text)
    if base_cjk >= 40 and base_conf >= 0.88:
        return base_text

    enhanced_arr = _enhance_for_ocr(arr)
    enhanced_text, enhanced_conf = _ocr_once(ocr, enhanced_arr)
    enhanced_cjk = _count_cjk(enhanced_text)

    # Prefer enhanced result when it improves confidence enough, or recovers more text.
    if enhanced_conf > base_conf + 0.03:
        return enhanced_text
    if enhanced_cjk > base_cjk + 8:
        return enhanced_text
    return base_text
