"""
Step 3: OCR the left-side text panel.
Uses PaddleOCR for Traditional Chinese (supports vertical text).
Runs on AutoDL with GPU, or locally on M3 (see config).

Tall vertical strips (e.g. 连环画) often OCR poorly as raw vertical layout; optional
90° rotation to horizontal + horizontal reading order is toggled via phase2_manifest.json
`ocr_rotate_left_cw90` (see main_autodl.py). Implementation uses ROTATE_90 so the right column reads first.
"""
from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import configs.config as _proj_cfg
from configs.config import (
    OCR_DET_LIMIT_SIDE_LEN,
    OCR_LANG,
)


_ocr = None
# Forwarded into PaddleX `predict()` on PaddleOCR 3+ (e.g. `use_textline_orientation=False`).
_ocr_predict_kwargs: dict[str, object] = {}

# Filled by main_autodl from crops_dir/phase2_manifest.json (packaged in crops_left.zip).
_phase2_manifest: dict = {}


def set_ocr_predict_kwargs(kwargs: Mapping[str, object] | None) -> None:
    global _ocr_predict_kwargs
    _ocr_predict_kwargs = dict(kwargs or {})


def reset_ocr_engine() -> None:
    """Drop lazy Paddle singleton (e.g. after changing OCR_USE_GPU)."""
    global _ocr
    _ocr = None


def set_phase2_manifest(manifest: dict | None) -> None:
    """AutoDL Phase 2: apply book-specific OCR hints from uploaded manifest."""
    global _phase2_manifest
    _phase2_manifest = dict(manifest or {})


def _paddleocr_major_version() -> int | None:
    try:
        return int(version("paddleocr").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return None


def _normalize_ocr_payload(payload):
    """PaddleOCR 3.x returns OCRResult objects; v2 returns list of boxes. Normalize to dict or list."""
    if payload is None:
        return None
    if isinstance(payload, (list, tuple)):
        return payload
    if isinstance(payload, Mapping) and "rec_texts" in payload:
        return dict(payload)
    if hasattr(payload, "rec_texts") and hasattr(payload, "dt_polys"):
        scores = getattr(payload, "rec_scores", None)
        d: dict = {
            "rec_texts": list(payload.rec_texts),
            "rec_polys": list(payload.dt_polys),
        }
        if scores is not None:
            d["rec_scores"] = list(scores)
        return d
    return payload


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
    return _normalize_ocr_payload(payload)


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


def _poly_y_center(poly_like) -> float:
    _, ymin, _, ymax = _poly_bounds_xy(poly_like)
    return (ymin + ymax) / 2.0


def _poly_height(poly_like) -> float:
    _, ymin, _, ymax = _poly_bounds_xy(poly_like)
    return max(1.0, ymax - ymin)


def _cluster_horizontal_lines(
    pairs: list[tuple[object, str]],
    *,
    row_y_merge_ratio: float = 0.22,
) -> list[str]:
    """Merge boxes on the same horizontal row (L→R), rows top→bottom; fixes fragment ordering."""
    with_poly = [(po, t) for po, t in pairs if po is not None and t]
    no_poly = [t for po, t in pairs if po is None and t]
    if not with_poly:
        return no_poly

    heights = [_poly_height(po) for po, _ in with_poly]
    med_h = float(np.median(np.asarray(heights, dtype=np.float64)))
    y_thresh = max(6.0, med_h * row_y_merge_ratio)

    decorated: list[tuple[float, float, object, str]] = []
    for po, t in with_poly:
        decorated.append((_poly_y_center(po), _poly_x_min(po), po, t))
    decorated.sort(key=lambda z: (z[0], z[1]))

    rows: list[list[tuple[float, float, object, str]]] = []
    for item in decorated:
        yc, xm, po, t = item
        placed = False
        for row in rows:
            row_mean_y = sum(r[0] for r in row) / len(row)
            if abs(yc - row_mean_y) <= y_thresh:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    rows.sort(
        key=lambda r: float(np.median([_poly_y_min(p[2]) for p in r]))
        if r
        else 0.0
    )
    lines = ["".join(r[3] for r in sorted(row, key=lambda r: r[1])) for row in rows]
    return lines + no_poly


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
        lines = _cluster_horizontal_lines(pairs)
        if not lines:
            return []
        # Rotated 连环画 strip → one horizontal paragraph per crop (no spurious line breaks).
        return ["".join(lines)]
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
        dev = "gpu" if _proj_cfg.OCR_USE_GPU else "cpu"
        det_kw: dict = {}
        if OCR_DET_LIMIT_SIDE_LEN and int(OCR_DET_LIMIT_SIDE_LEN) > 0:
            det_kw["det_limit_side_len"] = int(OCR_DET_LIMIT_SIDE_LEN)

        major = _paddleocr_major_version()
        candidates: list[dict] = []

        # v2 __init__ does not accept these kwargs.
        doc_off_v3: dict = {}
        if (
            major is not None
            and major >= 3
            and not _proj_cfg.OCR_PADDLEX_USE_DOC_PREPROCESSOR
        ):
            doc_off_v3 = {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
            }

        lite_cpu_models = (
            not _proj_cfg.OCR_USE_GPU
            and getattr(_proj_cfg, "OCR_CPU_USE_LITE_MODELS", True)
        )

        if lite_cpu_models and (major is None or major >= 3):
            # YAML first: omit `lang` or Paddle merges in PP-OCRv5 server + 4000 max_side anyway.
            _cpu_pdx = (
                Path(__file__).resolve().parent / "configs" / "paddlex_ocr_cpu_lite.yaml"
            )
            if _cpu_pdx.is_file():
                candidates.append(
                    dict(
                        paddlex_config=str(_cpu_pdx),
                        device=dev,
                        use_angle_cls=True,
                        **doc_off_v3,
                        **det_kw,
                    )
                )
            # PP-OCRv5 server_* spikes RAM on CPU; PP-OCRv3 mobile supports chinese_cht.
            candidates.extend(
                [
                    dict(
                        lang=OCR_LANG,
                        ocr_version="PP-OCRv3",
                        device=dev,
                        use_angle_cls=True,
                        **doc_off_v3,
                        **det_kw,
                    ),
                    dict(
                        lang=OCR_LANG,
                        ocr_version="PP-OCRv3",
                        device=dev,
                        use_angle_cls=True,
                        **doc_off_v3,
                    ),
                    dict(
                        lang=OCR_LANG,
                        ocr_version="PP-OCRv3",
                        device=dev,
                        **doc_off_v3,
                    ),
                ]
            )
        else:
            _pdx_yaml = (
                Path(__file__).resolve().parent / "configs" / "paddlex_ocr_pipeline.yaml"
            )
            if (major is None or major >= 3) and _pdx_yaml.is_file():
                candidates.append(
                    dict(
                        paddlex_config=str(_pdx_yaml),
                        use_angle_cls=True,
                        device=dev,
                        **doc_off_v3,
                        **det_kw,
                    )
                )

            if major is None or major >= 3:
                # Paddle 3+: default PP‑OCRv5 server lang path (GPU / heavy).
                candidates.extend(
                    [
                        dict(
                            lang=OCR_LANG,
                            use_angle_cls=True,
                            device=dev,
                            **doc_off_v3,
                            **det_kw,
                        ),
                        dict(
                            lang=OCR_LANG,
                            use_angle_cls=True,
                            device=dev,
                            **doc_off_v3,
                        ),
                        dict(lang=OCR_LANG, device=dev, **doc_off_v3),
                        dict(
                            lang=OCR_LANG,
                            use_angle_cls=True,
                            **doc_off_v3,
                            **det_kw,
                        ),
                        dict(lang=OCR_LANG, use_angle_cls=True, **doc_off_v3),
                        dict(lang=OCR_LANG, **doc_off_v3),
                    ]
                )

        # PaddleOCR 2.x legacy (and fallbacks).
        candidates.extend(
            [
                dict(lang=OCR_LANG, use_angle_cls=True, device=dev, show_log=False, **det_kw),
                dict(lang=OCR_LANG, use_angle_cls=True, show_log=False, **det_kw),
                dict(lang=OCR_LANG, show_log=False, **det_kw),
                dict(
                    lang=OCR_LANG,
                    use_angle_cls=True,
                    use_gpu=_proj_cfg.OCR_USE_GPU,
                    show_log=False,
                    **det_kw,
                ),
                dict(lang=OCR_LANG, use_angle_cls=True, device=dev, show_log=False),
                dict(lang=OCR_LANG, use_angle_cls=True, use_gpu=_proj_cfg.OCR_USE_GPU, show_log=False),
            ]
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
    # PaddleOCR 3+ / PaddleX: predict() does not accept cls= (textline ori is in the pipeline).
    major = _paddleocr_major_version()
    if major is None or major >= 3:
        result = ocr.predict(arr, **_ocr_predict_kwargs)
    else:
        try:
            result = ocr.ocr(arr, cls=True)
        except TypeError:
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
    # PaddleOCR 3 / PaddleX expect H×W×3 uint8 images.
    if enhanced.ndim == 2:
        enhanced = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return enhanced


def _cap_rgb_long_edge(img: Image.Image) -> Image.Image:
    lim_raw = getattr(_proj_cfg, "OCR_MAX_INPUT_LONG_SIDE", 0)
    if not lim_raw or int(lim_raw) <= 0:
        return img
    lim = int(lim_raw)
    w, h = img.size
    m = max(w, h)
    if m <= lim:
        return img
    s = lim / m
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _upscale_for_ocr(img: Image.Image, min_long_side: int = 2700) -> Image.Image:
    """Upscale small crops so stroke height clears PP-OCRv3 mobile comfort zone (~16px)."""
    w, h = img.size
    long_side = max(w, h)
    if long_side >= min_long_side:
        return img
    scale = min_long_side / long_side
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _suppress_bottom_background(img: Image.Image) -> Image.Image:
    """
    Trim trailing bottom area with no strong text strokes.

    This removes faint seal/background art near the bottom without relying on fixed
    crop ratios. If text extends near the bottom, the cutoff naturally stays low.
    """
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = arr.shape
    if h < 120 or w < 60:
        return img

    # Inspect only lower half for the trailing non-text background block.
    start = h // 2
    dark = arr < 165
    row_dark = np.sum(dark, axis=1)
    min_dark_px = max(8, int(round(w * 0.02)))
    active = row_dark >= min_dark_px
    active[:start] = False

    # Find contiguous active runs and use the lowest meaningful one as text tail.
    runs: list[tuple[int, int]] = []
    run_start = -1
    for y, on in enumerate(active.tolist()):
        if on and run_start < 0:
            run_start = y
        elif not on and run_start >= 0:
            runs.append((run_start, y - 1))
            run_start = -1
    if run_start >= 0:
        runs.append((run_start, h - 1))

    min_run = max(6, int(round(h * 0.01)))
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_run]
    if not runs:
        return img

    last_text_row = max(r[1] for r in runs)
    pad = max(12, int(round(h * 0.02)))
    cutoff = min(h, last_text_row + pad)
    if cutoff >= h - 2:
        return img
    return img.crop((0, 0, w, cutoff)).copy()


def _choose_ocr_result(
    *,
    base_text: str,
    base_conf: float,
    base_cjk: int,
    enhanced_text: str,
    enhanced_conf: float,
    enhanced_cjk: int,
) -> str:
    """Pick base vs enhanced OCR output, penalising enhanced runs that inject non-CJK noise.

    `_enhance_for_ocr` boosts contrast which can make PaddleOCR confidently mis-recognise
    border/seal artefacts as digits/letters. The garbage budget below rejects an enhanced
    win when its non-CJK character count grew by more than 2 over the base output.
    """
    base_garbage = len(base_text) - base_cjk
    enhanced_garbage = len(enhanced_text) - enhanced_cjk

    if enhanced_conf > base_conf + 0.10 and enhanced_garbage <= base_garbage + 2:
        return enhanced_text
    if enhanced_cjk > base_cjk + 8 and enhanced_garbage <= base_garbage + 2:
        return enhanced_text
    return base_text


def ocr_image(pil_image: Image.Image) -> str:
    rotate_cw90 = bool(_phase2_manifest.get("ocr_rotate_left_cw90"))
    reading_order = "horizontal_tb_lr" if rotate_cw90 else "vertical_rl_tt"

    img = pil_image.convert("RGB")
    img = _suppress_bottom_background(img)
    if rotate_cw90:
        # Vertical strip → horizontal: use ROTATE_90 (CCW) so the traditional right column
        # lands above the left column after rotation (top→bottom read matches R→L columns).
        # ROTATE_270 (90°CW) puts the left column on top and reverses sentence order.
        img = img.transpose(Image.Transpose.ROTATE_90)

    img = _upscale_for_ocr(img)
    img = _cap_rgb_long_edge(img)

    arr = np.array(img)
    ocr = get_ocr()

    base_text, base_conf = _ocr_once(ocr, arr, reading_order=reading_order)
    if not _proj_cfg.OCR_RETRY_WITH_ENHANCEMENT:
        return base_text

    # Retry not only for "too few chars", but also for low-confidence long outputs.
    base_cjk = _count_cjk(base_text)
    if base_cjk >= 40 and base_conf >= 0.88:
        return base_text

    enhanced_arr = _enhance_for_ocr(arr)
    enhanced_text, enhanced_conf = _ocr_once(ocr, enhanced_arr, reading_order=reading_order)
    enhanced_cjk = _count_cjk(enhanced_text)

    return _choose_ocr_result(
        base_text=base_text,
        base_conf=base_conf,
        base_cjk=base_cjk,
        enhanced_text=enhanced_text,
        enhanced_conf=enhanced_conf,
        enhanced_cjk=enhanced_cjk,
    )
