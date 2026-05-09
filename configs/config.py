"""Wuxia project configuration: common defaults + per-book merged settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Common defaults shared by all books.
START_PAGE = 6
END_PAGE = -2
SPLIT_RATIO = 0.5
WORK_ROOT_BASE = "work"
OUTPUT_SUBDIR = "output"

# OCR (works on both local M3 and AutoDL GPU)
OCR_LANG = "chinese_cht"
OCR_USE_GPU = True
OCR_RETRY_WITH_ENHANCEMENT = True
OCR_OUTPUT_SIMPLIFIED = True
# Larger than default 960: 300 DPI left crops are tall; too-small limit hurts vertical 繁体.
OCR_DET_LIMIT_SIDE_LEN = 2048
# PaddleOCR 3 / PaddleX only. Doc orientation + UVDoc unwarp helps phone photos of curved paper;
# for flat PDF / 连环画 crops it often hurts text and is slow on CPU.
OCR_PADDLEX_USE_DOC_PREPROCESSOR = False
# If > 0: LANCZOS downscale longest edge before Paddle (OOM guard). Default 0 = full res for GPU servers.
# ocr_one.py --cpu defaults this to 4480 when unset.
OCR_MAX_INPUT_LONG_SIDE = 0
# When OCR_USE_GPU is False: PP-OCRv5 server det/rec + custom PaddleX YAML OOM‑kills on many laptops.
# Prefer PP-OCRv3 mobile (still chinese_cht) for local smoke tests; AutoDL GPU unchanged.
OCR_CPU_USE_LITE_MODELS = True

# Translation (Traditional → Simplified Chinese)
OPENCC_CONFIG = "t2s"

# Image output
IMAGE_FORMAT = "PNG"
IMAGE_DPI = 300

# PDF Step 1: embedded bitmap vs full-page render (see extract_pages.py)
# If embedded width / (page_pt_width/72) is below threshold, pixmap @ IMAGE_DPI is mostly
# interpolation blur for "screen JPG in letterbox PDF" pages; extract native JPEG instead.
EXTRACT_EMBEDDED_EFFECTIVE_DPI_THRESHOLD = 80.0
EXTRACT_EMBEDDED_TARGET_LONG_SIDE = 2700
# Default when book sidecar omits pdf_extract_mode: auto | render | embedded_native
PDF_EXTRACT_MODE_DEFAULT = "auto"

# Upscaling (optional — AutoDL only; requires realesrgan)
UPSCALE_ENABLED = False
UPSCALE_FACTOR = 4

# Fallback folder name when no chapter title detected
FOLDER_FALLBACK = "page_{page_num:03d}"

# Hybrid workflow paths (AutoDL side stays shared)
AUTODL_REMOTE_DIR = "/root/wuxia_crops"
AUTODL_OUTPUT_DIR = "/root/wuxia_output"

# Written into tmp_crops/ and packaged in crops_left.zip for Phase 2 on AutoDL.
PHASE2_MANIFEST_NAME = "phase2_manifest.json"


@dataclass(frozen=True)
class BookRuntimeConfig:
    pdf_path: Path
    slug: str
    start_page: int
    end_page: int | None
    split_ratio: float
    work_root: Path
    tmp_pages: Path
    tmp_crops: Path
    tmp_results: Path
    output_dir: Path
    crops_zip: Path
    ocr_left_crop: dict[str, float | int] | None
    ocr_rotate_left_cw90: bool
    pdf_extract_mode: str


def write_phase2_manifest(tmp_crops: Path, *, ocr_rotate_left_cw90: bool) -> None:
    """Drop AutoDL Phase 2 hints next to left crops so the zip carries book-specific OCR behavior."""
    path = tmp_crops / PHASE2_MANIFEST_NAME
    if ocr_rotate_left_cw90:
        path.write_text(
            json.dumps({"ocr_rotate_left_cw90": True}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif path.is_file():
        path.unlink()


def sanitize_slug(stem: str) -> str:
    stem = stem.replace("\0", "_").replace("/", "_").replace("\\", "_")
    stem = stem.strip()
    return stem or "book"


def resolve_book_argument(book: str, *, cwd: Path | None = None) -> Path:
    base = Path.cwd() if cwd is None else cwd
    p = Path(book)
    if len(p.parts) == 1 and not p.is_absolute():
        candidate = (base / "data" / book).resolve()
    else:
        candidate = p.resolve() if p.is_absolute() else (base / p).resolve()
    if not candidate.is_file():
        raise ValueError(f"PDF not found: {candidate}")
    if candidate.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file (.pdf): {candidate}")
    return candidate


def _book_config_path(pdf_path: Path, *, cwd: Path | None = None) -> Path:
    base = Path.cwd() if cwd is None else cwd
    return base / "configs" / "books" / f"{pdf_path.stem}.json"


def load_book_overrides(pdf_path: Path, *, cwd: Path | None = None) -> dict:
    sidecar = _book_config_path(pdf_path, cwd=cwd)
    legacy_sidecar = pdf_path.parent / f"{pdf_path.stem}.book.json"
    if not sidecar.is_file() and legacy_sidecar.is_file():
        sidecar = legacy_sidecar
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {sidecar}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{sidecar} must be a JSON object")
    return data


def _coerce_path(value: str | None, *, base: Path) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def build_book_runtime_config(book: str, *, cwd: Path | None = None) -> BookRuntimeConfig:
    base = Path.cwd() if cwd is None else cwd
    resolved_pdf = resolve_book_argument(book, cwd=base)
    overrides = load_book_overrides(resolved_pdf, cwd=base)

    override_pdf = overrides.get("pdf_path")
    if override_pdf:
        resolved_pdf = resolve_book_argument(str(override_pdf), cwd=base)

    slug = sanitize_slug(str(overrides.get("slug", resolved_pdf.stem)))
    default_work_root = (base / WORK_ROOT_BASE / slug).resolve()
    work_root = _coerce_path(overrides.get("work_root"), base=base) or default_work_root

    default_output_dir = work_root / OUTPUT_SUBDIR
    output_dir = _coerce_path(overrides.get("output_dir"), base=base) or default_output_dir

    start_page = int(overrides.get("start_page", START_PAGE))
    end_raw = overrides.get("end_page", END_PAGE)
    end_page = None if end_raw is None else int(end_raw)
    split_ratio = float(overrides.get("split_ratio", SPLIT_RATIO))
    ocr_left_crop = overrides.get("ocr_left_crop")
    if ocr_left_crop is not None and not isinstance(ocr_left_crop, dict):
        raise ValueError("ocr_left_crop must be a JSON object when provided")
    if isinstance(ocr_left_crop, dict):
        allowed = {"top", "left", "right"}
        extra = sorted(set(ocr_left_crop) - allowed)
        if extra:
            raise ValueError(
                f"ocr_left_crop only supports top/left/right; unsupported keys: {', '.join(extra)}"
            )
    ocr_rotate_left_cw90 = bool(overrides.get("ocr_rotate_left_cw90", False))

    pdf_extract_mode = str(overrides.get("pdf_extract_mode", PDF_EXTRACT_MODE_DEFAULT)).strip().lower()
    if pdf_extract_mode not in ("auto", "render", "embedded_native"):
        raise ValueError(
            f"pdf_extract_mode must be auto, render, or embedded_native (got {pdf_extract_mode!r})"
        )

    return BookRuntimeConfig(
        pdf_path=resolved_pdf,
        slug=slug,
        start_page=start_page,
        end_page=end_page,
        split_ratio=split_ratio,
        work_root=work_root,
        tmp_pages=work_root / "tmp_pages",
        tmp_crops=work_root / "tmp_crops",
        tmp_results=work_root / "tmp_results",
        output_dir=output_dir,
        crops_zip=work_root / "crops_left.zip",
        ocr_left_crop=ocr_left_crop,
        ocr_rotate_left_cw90=ocr_rotate_left_cw90,
        pdf_extract_mode=pdf_extract_mode,
    )
