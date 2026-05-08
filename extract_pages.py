"""
Step 1: Extract page images from PDF.
Rasterise each PDF page to a high-res JPEG using PyMuPDF.
"""
import fitz
from pathlib import Path
from config import IMAGE_DPI


def extract_pages(
    pdf_path: Path,
    tmp_dir: Path,
    *,
    start_page: int,
    end_page: int | None,
    image_dpi: int | None = None,
):
    if image_dpi is None:
        image_dpi = IMAGE_DPI
    doc = fitz.open(str(pdf_path))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results = []

    end = len(doc)
    if end_page is not None:
        end = len(doc) + end_page if end_page < 0 else end_page
    end = max(0, min(end, len(doc)))
    start_idx = start_page - 1
    if start_idx < 0 or start_idx >= len(doc) or start_idx >= end:
        doc.close()
        raise ValueError(
            f"No pages to extract: page index range [{start_idx}, {end}) is empty or invalid "
            f"(document has {len(doc)} pages, start_page={start_page}, end_page={end_page})"
        )
    for page_num in range(start_idx, end):  # fitz is 0-indexed
        page = doc[page_num]
        mat = fitz.Matrix(image_dpi / 72, image_dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        out_path = tmp_dir / f"page_{page_num + 1:03d}.jpg"
        pix.save(str(out_path))
        results.append((page_num + 1, out_path))

    doc.close()
    return results
