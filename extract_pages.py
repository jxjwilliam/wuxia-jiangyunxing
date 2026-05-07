"""
Step 1: Extract page images from PDF.
Rasterise each PDF page to a high-res JPEG using PyMuPDF.
"""
import fitz
from pathlib import Path
from config import PDF_PATH, START_PAGE, END_PAGE, IMAGE_DPI


def extract_pages(tmp_dir: Path):
    doc = fitz.open(PDF_PATH)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results = []

    end = END_PAGE or len(doc)
    for page_num in range(START_PAGE - 1, end):  # fitz is 0-indexed
        page = doc[page_num]
        # Scale to target DPI (native PDF is 72 PPI)
        mat = fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        out_path = tmp_dir / f"page_{page_num + 1:03d}.jpg"
        pix.save(str(out_path))
        results.append((page_num + 1, out_path))

    doc.close()
    return results
