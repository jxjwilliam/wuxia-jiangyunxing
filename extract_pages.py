"""
Step 1: Extract page images from PDF.

Default: PyMuPDF full-page pixmap at IMAGE_DPI.

For "embedded low-DPI raster in a large point-size page", pixmap does a giant upscale then
downstream OCR may cap → double blur. Mode ``auto`` uses the first embedded image's native
pixels + one PIL upscale instead (see tests/c1.md).
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

from configs.config import (
    EXTRACT_EMBEDDED_EFFECTIVE_DPI_THRESHOLD,
    EXTRACT_EMBEDDED_TARGET_LONG_SIDE,
    IMAGE_DPI,
)


def embedded_raster_effective_dpi(src_width_px: float, page_width_pt: float) -> float:
    """Effective horizontal DPI if the bitmap were stretched across the full page width."""
    if page_width_pt <= 0 or src_width_px <= 0:
        return 0.0
    return float(src_width_px) / (page_width_pt / 72.0)


def pdf_page_should_use_embedded_native(
    mode: str,
    *,
    has_embedded_image: bool,
    effective_dpi: float,
    threshold: float,
) -> bool:
    """Whether to decode the first embedded raster instead of get_pixmap for this page."""
    if mode == "render":
        return False
    if mode == "embedded_native":
        return has_embedded_image
    if mode == "auto":
        return has_embedded_image and effective_dpi > 0 and effective_dpi < threshold
    raise ValueError(f"unknown pdf_extract_mode: {mode!r}")


def _maybe_upscale_long_side(img: Image.Image, target_long_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side >= target_long_side:
        return img
    scale = target_long_side / long_side
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _pixmap_page_rgb(page: fitz.Page, dpi: int) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _first_embedded_rgb_and_dpi(doc: fitz.Document, page: fitz.Page) -> tuple[Image.Image | None, float]:
    """Decode first embedded image (PyMuPDF list order) + effective DPI by page width."""
    img_list = page.get_images(full=True)
    if not img_list:
        return None, 0.0
    try:
        base = doc.extract_image(img_list[0][0])
    except Exception:
        return None, 0.0
    raw = base.get("image")
    if raw is None:
        return None, 0.0
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None, 0.0
    src_w = float(base.get("width") or im.width)
    eff = embedded_raster_effective_dpi(src_w, float(page.rect.width))
    return im, eff


def extract_pages(
    pdf_path: Path,
    tmp_dir: Path,
    *,
    start_page: int,
    end_page: int | None,
    image_dpi: int | None = None,
    pdf_extract_mode: str | None = None,
    embedded_effective_dpi_threshold: float | None = None,
    embedded_target_long_side: int | None = None,
):
    """Write ``page_XXX.jpg`` under ``tmp_dir``; returns list of (1-based page num, path)."""
    dpi = IMAGE_DPI if image_dpi is None else image_dpi
    mode = (pdf_extract_mode or "auto").strip().lower()
    thr = (
        EXTRACT_EMBEDDED_EFFECTIVE_DPI_THRESHOLD
        if embedded_effective_dpi_threshold is None
        else embedded_effective_dpi_threshold
    )
    tgt = (
        EXTRACT_EMBEDDED_TARGET_LONG_SIDE
        if embedded_target_long_side is None
        else embedded_target_long_side
    )

    doc = fitz.open(str(pdf_path))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[int, Path]] = []

    n_pages = len(doc)
    end = n_pages
    if end_page is not None:
        end = n_pages + end_page if end_page < 0 else end_page
    end = max(0, min(end, n_pages))
    start_idx = start_page - 1
    if start_idx < 0 or start_idx >= n_pages or start_idx >= end:
        doc.close()
        raise ValueError(
            f"No pages to extract: page index range [{start_idx}, {end}) is empty or invalid "
            f"(document has {n_pages} pages, start_page={start_page}, end_page={end_page})"
        )

    try:
        for page_num in range(start_idx, end):
            page = doc[page_num]
            pil_img: Image.Image | None = None
            emb, eff = _first_embedded_rgb_and_dpi(doc, page)
            use_emb = pdf_page_should_use_embedded_native(
                mode,
                has_embedded_image=emb is not None,
                effective_dpi=eff,
                threshold=thr,
            )
            if use_emb and emb is not None:
                try:
                    pil_img = _maybe_upscale_long_side(emb, tgt)
                except Exception:
                    pil_img = None
            if pil_img is None:
                pil_img = _pixmap_page_rgb(page, dpi)
            out_path = tmp_dir / f"page_{page_num + 1:03d}.jpg"
            pil_img.save(str(out_path), "JPEG", quality=95)
            results.append((page_num + 1, out_path))
    finally:
        doc.close()

    return results
