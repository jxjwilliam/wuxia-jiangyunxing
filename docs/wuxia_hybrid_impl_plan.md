# Wuxia PDF Extraction — Hybrid Implementation Plan

Extends [`wuxia_extraction_plan.md`](./wuxia_extraction_plan.md) for the **Hybrid Local + AutoDL** approach.

All existing code in `wuxia_extraction_plan.md` remains valid. This plan adds:
- File restructuring for hybrid workflow
- AutoDL upload/download scripts
- Modified orchestrators (local vs AutoDL)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        HYBRID WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [LOCAL M3 MacBook]              [AUTODL GPU]    [LOCAL M3]     │
│  ┌───────────────┐               ┌──────────┐   ┌────────────┐  │
│  │ PDF Rasterize │──▶──upload──▶│ PaddleOCR│──▶│ Translate   │  │
│  │ Page Split    │               │ Upscale  │   │ Title Detect│  │
│  │ (1-2 min)     │               │ (3-5 min)│   │ Organize    │  │
│  └───────────────┘               └──────────┘   │ (1 min)     │  │
│                                                  └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
wuxia-extractor/
├── config.py               ← All settings (same as original plan)
├── extract_pages.py        ← Step 1: PDF → page JPEGs (LOCAL)
├── split_page.py           ← Step 2: Split JPEG left/right (LOCAL)
├── prepare_upload.py       ← NEW: Package crops for AutoDL (LOCAL)
├── ocr_text.py             ← Step 3: OCR (AUTODL — same code as plan)
├── translate.py            ← Step 4: 繁→簡 (LOCAL — same code as plan)
├── detect_title.py         ← Step 5: Chapter title (LOCAL — same code)
├── upscale.py              ← Step 6: Optional upscale (AUTODL)
├── organise.py             ← Step 7: Save files (LOCAL)
├── download_results.py     ← NEW: Process AutoDL output (LOCAL)
├── main_local.py           ← NEW: Local orchestrator
├── main_autodl.py          ← NEW: AutoDL orchestrator
├── requirements_local.txt  ← Local dependencies
├── requirements_gpu.txt    ← AutoDL dependencies
└── wuxia/                  ← Output folder (auto-created)
```

---

## File-by-File Implementation

### `config.py` — Updated for Hybrid

```python
# config.py — all settings

# PDF
PDF_PATH = "data/jiang-yun-xing.pdf"
OUTPUT_DIR = "wuxia"
START_PAGE = 6
END_PAGE = None              # None = process all

# Split
SPLIT_RATIO = 0.5            # Adjust if text/illustration boundary off-centre

# OCR
OCR_LANG = "chinese_cht"     # Traditional Chinese for PaddleOCR
OCR_USE_GPU = True           # True on AutoDL, False on local

# Translation
OPENCC_CONFIG = "t2s"        # Traditional → Simplified

# Image output
IMAGE_FORMAT = "PNG"
IMAGE_DPI = 300

# Upscaling (optional — AutoDL only)
UPSCALE_ENABLED = False
UPSCALE_FACTOR = 4

# AutoDL paths (used by prepare/download scripts)
AUTODL_CROPS_DIR = "tmp_crops"       # Where split crops are stored locally
AUTODL_REMOTE_DIR = "/root/wuxia_crops"  # Where crops land on AutoDL
AUTODL_OUTPUT_DIR = "/root/wuxia_output" # Where AutoDL writes results
AUTODL_RESULTS_DIR = "tmp_results"   # Where we store downloaded results

# Fallback folder name
FOLDER_FALLBACK = "page_{page_num:03d}"
```

### `prepare_upload.py` — NEW: Package for AutoDL

Purpose: Package the left (text) and right (illustration) crops into structured folders for upload to AutoDL.

```python
"""
prepare_upload.py — Package split crops for AutoDL transfer.
Run this AFTER split_page.py creates the tmp_crops/ folder.
Creates two zip files:
  - crops_left.zip  (all text-side images for OCR)
  - crops_right.zip (all illustration-side images for upscale, optional)
"""
from pathlib import Path
import zipfile
import shutil
from config import AUTODL_CROPS_DIR

def prepare_upload(crops_dir: Path):
    """
    Package tmp_crops/ into zip files for easy scp to AutoDL.
    """
    left_zip = Path("crops_left.zip")
    right_zip = Path("crops_right.zip")

    with zipfile.ZipFile(left_zip, 'w', zipfile.ZIP_DEFLATED) as zl, \
         zipfile.ZipFile(right_zip, 'w', zipfile.ZIP_DEFLATED) as zr:

        for img_path in sorted(crops_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            if '_left' in img_path.name:
                zl.write(img_path, img_path.name)
            elif '_right' in img_path.name:
                zr.write(img_path, img_path.name)

    print(f"  → {left_zip}: {left_zip.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  → {right_zip}: {right_zip.stat().st_size / 1024 / 1024:.1f} MB")
    print("Upload these to AutoDL via scp or web UI.")
```

### `main_local.py` — NEW: Local Orchestrator (Phase 1 + Phase 3)

```python
"""
main_local.py — Orchestrator for local execution.
Phase 1: Extract + Split (prepares crops for AutoDL)
Phase 3: Translate + Title Detect + Organise (after AutoDL results return)
"""
import sys
from pathlib import Path
from config import AUTODL_CROPS_DIR, AUTODL_RESULTS_DIR

TMP_DIR = Path("tmp_pages")
CROPS_DIR = Path(AUTODL_CROPS_DIR)
RESULTS_DIR = Path(AUTODL_RESULTS_DIR)

def phase1_preprocess():
    """Extract PDF pages and split into left/right crops."""
    from extract_pages import extract_pages
    from split_page import split_page

    print("=" * 50)
    print("PHASE 1: Local Preprocessing")
    print("=" * 50)

    print("\nStep 1: Extracting page images from PDF...")
    pages = extract_pages(TMP_DIR)
    print(f"  → {len(pages)} pages extracted")

    print("\nStep 2: Splitting pages into left/right crops...")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for page_num, img_path in pages:
        left_img, right_img = split_page(img_path)
        left_path = CROPS_DIR / f"page_{page_num:03d}_left.jpg"
        right_path = CROPS_DIR / f"page_{page_num:03d}_right.jpg"
        left_img.save(left_path, "JPEG", quality=95)
        right_img.save(right_path, "JPEG", quality=95)

    print(f"  → {len(pages)} pairs saved to {CROPS_DIR}/")

    print("\nStep 3: Packaging crops for AutoDL upload...")
    from prepare_upload import prepare_upload
    prepare_upload(CROPS_DIR)

    print("\n✅ Phase 1 complete. Upload crops_left.zip to AutoDL.")
    print("   Run 'python main_autodl.py' on AutoDL instance.")
    print("   Download results to tmp_results/ when done.")
    print("   Then run: python main_local.py --phase3\n")


def phase3_assemble():
    """After AutoDL results are downloaded, translate and organize."""
    from translate import translate
    from detect_title import detect_title
    from organise import save_pair
    from PIL import Image

    print("=" * 50)
    print("PHASE 3: Local Assembly")
    print("=" * 50)

    if not RESULTS_DIR.exists():
        print(f"❌ {RESULTS_DIR}/ not found. Download AutoDL results first.")
        sys.exit(1)

    # Expected structure from AutoDL:
    # tmp_results/
    #   page_006/
    #     ocr_text.txt        ← from PaddleOCR
    #     illustration.png    ← from right crop (optionally upscaled)
    #   page_007/
    #     ...

    page_dirs = sorted(RESULTS_DIR.iterdir())

    print(f"\nStep 4: Translating {len(page_dirs)} pages...")
    for page_dir in page_dirs:
        if not page_dir.is_dir():
            continue
        text_file = page_dir / "ocr_text.txt"
        if not text_file.exists():
            continue

        raw_text = text_file.read_text(encoding="utf-8")
        simplified = translate(raw_text)

        # Step 5: Detect title
        folder_name = detect_title(simplified, int(page_dir.name.split("_")[1]))

        # Step 7: Save paired files
        img_path = page_dir / "illustration.png"
        if img_path.exists():
            img = Image.open(img_path)
            save_pair(folder_name, simplified, img)
        else:
            save_pair(folder_name, simplified, None)

        print(f"  → Page {page_dir.name}: {folder_name}/")

    print(f"\n✅ Phase 3 complete. Output in wuxia/")
    print(f"   Total folders: {len(list(Path('wuxia').iterdir()))}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--phase3":
        phase3_assemble()
    else:
        phase1_preprocess()
```

### `main_autodl.py` — NEW: AutoDL GPU Orchestrator (Phase 2)

```python
"""
main_autodl.py — Orchestrator for AutoDL GPU execution.
Run this on the AutoDL instance after uploading crops.
Phase 2: OCR left crops + (optional) upscale right crops
"""
from pathlib import Path
from config import AUTODL_REMOTE_DIR, AUTODL_OUTPUT_DIR, UPSCALE_ENABLED

CROPS_DIR = Path(AUTODL_REMOTE_DIR)
OUT_DIR = Path(AUTODL_OUTPUT_DIR)

def phase2_gpu():
    from ocr_text import ocr_image
    from PIL import Image

    print("=" * 50)
    print("PHASE 2: AutoDL GPU Processing")
    print("=" * 50)

    if UPSCALE_ENABLED:
        from upscale import upscale_image

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crop_files = sorted(CROPS_DIR.iterdir())
    # Group by page number
    pages = {}
    for f in crop_files:
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        # filename: page_006_left.jpg or page_006_right.jpg
        parts = f.stem.split('_')
        page_num = parts[1]
        side = parts[2]
        pages.setdefault(page_num, {})[side] = f

    print(f"\nProcessing {len(pages)} pages...")

    for page_num in sorted(pages.keys()):
        page_dir = OUT_DIR / f"page_{page_num}"
        page_dir.mkdir(exist_ok=True)

        left_path = pages[page_num].get('left')
        right_path = pages[page_num].get('right')

        if left_path:
            print(f"  OCR page {page_num}...")
            left_img = Image.open(left_path)
            text = ocr_image(left_img)
            (page_dir / "ocr_text.txt").write_text(text, encoding="utf-8")

        if right_path:
            right_img = Image.open(right_path)
            if UPSCALE_ENABLED:
                print(f"  Upscaling page {page_num}...")
                right_img = upscale_image(right_img)
            right_img.save(page_dir / "illustration.png", "PNG")

        print(f"  → Page {page_num} done")

    print(f"\n✅ Phase 2 complete. Output in {OUT_DIR}/")
    print("Download the results to local tmp_results/")


if __name__ == "__main__":
    phase2_gpu()
```

### `download_results.py` — NEW: Helper to process AutoDL results

```python
"""
download_results.py — Helper to explain the download + post-processing flow.
After AutoDL completes phase2, download results to tmp_results/.
Then run: python main_local.py --phase3
"""
print("""
═══ How to download AutoDL results ═══

1. From AutoDL instance, zip the output:
   $ cd /root && zip -r wuxia_output.zip wuxia_output/

2. On your local Mac, download:
   $ scp -P <port> root@<autodl-ip>:/root/wuxia_output.zip ./
   $ unzip wuxia_output.zip -d tmp_results/

3. Run Phase 3 assembly:
   $ python main_local.py --phase3

Done!
""")
```

### Dependencies

**`requirements_local.txt`**
```
pymupdf
pillow
opencc-python-reimplemented
```

**`requirements_gpu.txt`** (used on AutoDL)
```
paddlepaddle-gpu
paddleocr
pillow
# Optional for upscaling:
basicsr
realesrgan
opencv-python-headless
```

---

## Implementation Order

### Step 1 — Create config and all Python files (LOCAL)

1. `config.py` — as above (updated for hybrid)
2. `extract_pages.py` — same as original plan (unchanged)
3. `split_page.py` — same as original plan (unchanged)
4. `ocr_text.py` — same as original plan (unchanged; runs on AutoDL with GPU)
5. `translate.py` — same as original plan (unchanged)
6. `detect_title.py` — same as original plan (unchanged)
7. `organise.py` — same as original plan (unchanged)
8. `upscale.py` — same as original plan (unchanged; runs on AutoDL)
9. `prepare_upload.py` — NEW
10. `main_local.py` — NEW
11. `main_autodl.py` — NEW
12. `download_results.py` — NEW
13. `requirements_local.txt` — NEW
14. `requirements_gpu.txt` — NEW

### Step 2 — Test Phase 1 Locally

```
python main_local.py
```

Verify: crops in `tmp_crops/`, zip files `crops_left.zip` + `crops_right.zip` created.

Do a visual check on a few crops to verify split is correct.

### Step 3 — Set Up AutoDL Instance

1. Create an RTX 4090 instance on AutoDL
2. Install dependencies:
   ```
   pip install -r requirements_gpu.txt
   ```
3. Upload `main_autodl.py`, `ocr_text.py`, `upscale.py`, `config.py`, and the crop zip files
4. Unzip crops:
   ```
   unzip crops_left.zip -d /root/wuxia_crops/
   unzip crops_right.zip -d /root/wuxia_crops/
   ```

### Step 4 — Run Phase 2 on AutoDL

Test with 5 pages first:
```
# Config: set END_PAGE = 10 temporarily
python main_autodl.py
```

Verify OCR output in `/root/wuxia_output/`. If good, remove END_PAGE limit and run all 207 pages.

### Step 5 — Download + Phase 3 Assembly

1. Zip and download AutoDL output
2. Unzip to `tmp_results/`
3. Run:
   ```
   python main_local.py --phase3
   ```

---

## Related Documents

- [Implementation Plan](./wuxia_extraction_plan.md) — Original full pipeline design
- [Feasibility Analysis](./wuxia_feasibility_analysis.md) — Local M3 vs AutoDL GPU analysis
- [Implementation Summary](./summary.md) — Brief project summary
- [README](../README.md) — Project overview

---

## Verification Checklist

| Check | How | Pass/Fail |
|---|---|---|
| Split boundary correct | Visual check on page 6, 10, 50 | ☐ |
| OCR produces Chinese text | Read `ocr_text.txt` from a few pages | ☐ |
| Translation to Simplified works | `grep"简体" text.txt` for simplified chars | ☐ |
| Title detection | `第九回_鐵槍破犁` folder should exist | ☐ |
| Output structure | `wuxia/` has 35+ folders (one per chapter) | ☐ |
| All 207 pages processed | Count files in `wuxia/` | ☐ |
