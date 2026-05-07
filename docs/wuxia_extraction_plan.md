# Wuxia PDF Extraction — Implementation Plan

## Key Discovery (Before You Start)

> ⚠️ **The PDF is fully image-based, not text-based.**
> Every page is a single embedded JPEG. `pdftotext` returns empty.
> This means **OCR is required** to extract the left-side text.
> Plan accounts for this.

---

## PDF Facts

| Property | Value |
|---|---|
| Total pages | 214 |
| Page size | 657 × 899 pts (single), 1315 × 899 pts (spreads) |
| Page 1 | Cover — 657 × 899 (single width, skip) |
| Pages 2–5 | Front matter — skip |
| Pages 6–214 | Process — each is a 1315 × 899 JPEG spread |
| Embedded format | JPEG (one image per page) |
| Text extractable | ❌ No — image-based, OCR needed |
| Resolution | 72 PPI (low — upscaling recommended for image-to-video use) |

---

## Output Structure

```
wuxia/
├── 第九回_鐵槍破犁/
│   ├── text.txt          ← Simplified Chinese (translated from Traditional)
│   └── illustration.png  ← Right-side comic image (high quality)
├── 第十回_寃家聚頭/
│   ├── text.txt
│   └── illustration.png
├── page_006/             ← Fallback name if no 第X回 detected
│   ├── text.txt
│   └── illustration.png
...
```

---

## Tech Stack

| Task | Tool | Why |
|---|---|---|
| Extract page images from PDF | **PyMuPDF** (`fitz`) | Fast, precise, outputs PIL-compatible images |
| Split page image left/right | **Pillow** | Simple crop at midpoint |
| OCR Traditional Chinese (vertical) | **PaddleOCR** | Best-in-class for vertical CJK text, free, offline |
| Translate 繁體 → 簡體 | **OpenCC** | Fast, offline, rule-based, well-maintained |
| Chapter title detection | **regex** | Match `第[零一二三四五六七八九十百]+回` pattern |
| Image upscaling (optional) | **Real-ESRGAN** or **Pillow** | AI upscale for image-to-video quality |
| Orchestration | **Python 3.10+** | Single pipeline script |

### Dependencies (pip install)

```bash
pip install pymupdf pillow paddlepaddle paddleocr opencc-python-reimplemented
# Optional upscaling:
pip install basicsr realesrgan
```

> PaddleOCR requires PaddlePaddle. On Apple Silicon use the CPU build.
> On GPU machines use paddlepaddle-gpu for 10× faster OCR.

---

## Project Structure

```
wuxia-extractor/
├── main.py               ← Orchestrator: runs the full pipeline
├── config.py             ← All tunable settings in one place
├── extract_pages.py      ← Step 1: PDF → page JPEGs
├── split_page.py         ← Step 2: Split each JPEG left/right
├── ocr_text.py           ← Step 3: OCR left-side crop
├── translate.py          ← Step 4: Traditional → Simplified Chinese
├── detect_title.py       ← Step 5: Extract chapter title from OCR text
├── upscale.py            ← Step 6 (optional): Upscale illustration
├── organise.py           ← Step 7: Write files into wuxia/ folder tree
├── requirements.txt
└── wuxia/                ← Output folder (auto-created)
```

---

## config.py

```python
# config.py — all settings in one place

PDF_PATH = "data/jiang-yun-xing.pdf"
OUTPUT_DIR = "wuxia"
START_PAGE = 6          # 1-indexed; pages 1–5 are skipped
END_PAGE = None         # None = process to end of PDF

# Page split: ratio of left/right (0.5 = exact midpoint)
# Adjust if text/illustration boundary is off-centre
SPLIT_RATIO = 0.5

# OCR
OCR_LANG = "chinese_cht"   # Traditional Chinese for PaddleOCR
OCR_USE_GPU = False         # Set True if CUDA available

# Translation
OPENCC_CONFIG = "t2s"       # Traditional → Simplified

# Image output
IMAGE_FORMAT = "PNG"        # PNG = lossless, best for AI tools
IMAGE_DPI = 300             # DPI for page rasterisation (72 is too low)

# Upscaling (optional)
UPSCALE_ENABLED = False
UPSCALE_FACTOR = 4          # 2× or 4×

# Fallback folder name when no chapter title detected
FOLDER_FALLBACK = "page_{page_num:03d}"
```

---

## Step-by-Step Implementation

### Step 1 — Extract Page Images from PDF (`extract_pages.py`)

Use PyMuPDF to rasterise each PDF page to a high-res image.

```python
import fitz  # PyMuPDF
from pathlib import Path
from config import PDF_PATH, START_PAGE, END_PAGE, IMAGE_DPI

def extract_pages(tmp_dir: Path):
    """
    Rasterise each page of the PDF to a JPEG in tmp_dir.
    Returns list of (page_num, image_path) tuples.
    """
    doc = fitz.open(PDF_PATH)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results = []

    end = END_PAGE or len(doc)
    for page_num in range(START_PAGE - 1, end):  # fitz is 0-indexed
        page = doc[page_num]
        mat = fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72)  # scale to target DPI
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        out_path = tmp_dir / f"page_{page_num + 1:03d}.jpg"
        pix.save(str(out_path))
        results.append((page_num + 1, out_path))

    doc.close()
    return results
```

**Key note:** The original PDF is 72 PPI. Setting `IMAGE_DPI = 300` scales up
the rasterisation matrix (300/72 ≈ 4.2×), giving you ~5500 × 3750px images
— much better for both OCR accuracy and illustration quality.

---

### Step 2 — Split Page Left/Right (`split_page.py`)

Each page spread is ~double-wide. Cut at midpoint.

```python
from PIL import Image
from config import SPLIT_RATIO

def split_page(image_path) -> tuple:
    """
    Returns (left_img, right_img) as PIL Image objects.
    Left = text panel, Right = illustration.
    """
    img = Image.open(image_path)
    w, h = img.size
    mid = int(w * SPLIT_RATIO)

    left  = img.crop((0,   0, mid, h))
    right = img.crop((mid, 0, w,   h))
    return left, right
```

**Tuning tip:** If the split cuts into text or illustration, adjust
`SPLIT_RATIO` in config.py. You can also add a visual check script
(see Testing section).

---

### Step 3 — OCR the Text Panel (`ocr_text.py`)

PaddleOCR handles vertical Traditional Chinese well out of the box.

```python
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from config import OCR_LANG, OCR_USE_GPU

# Initialise once — expensive to reload per page
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            lang=OCR_LANG,
            use_angle_cls=True,   # detect rotated/vertical text
            use_gpu=OCR_USE_GPU,
            show_log=False,
        )
    return _ocr

def ocr_image(pil_image: Image.Image) -> str:
    """
    Run OCR on a PIL image, return plain text string.
    """
    arr = np.array(pil_image)
    result = get_ocr().ocr(arr, cls=True)

    lines = []
    if result and result[0]:
        # Sort by vertical position (top-to-bottom reading order)
        # For vertical Chinese text, sort by x descending (right-to-left columns)
        boxes = sorted(result[0], key=lambda r: -r[0][0][0])
        for box in boxes:
            text = box[1][0]
            lines.append(text)

    return "\n".join(lines)
```

**Vertical text note:** Traditional Chinese in this book runs top-to-bottom,
right-to-left. The sorting key (`-r[0][0][0]` = rightmost x first) reconstructs
reading order. You may need to tune this depending on PaddleOCR's output
bounding box ordering.

---

### Step 4 — Translate Traditional → Simplified (`translate.py`)

OpenCC is fast, deterministic, and works offline.

```python
import opencc
from config import OPENCC_CONFIG

_converter = None

def get_converter():
    global _converter
    if _converter is None:
        _converter = opencc.OpenCC(OPENCC_CONFIG)
    return _converter

def translate(text: str) -> str:
    """Convert Traditional Chinese to Simplified Chinese."""
    return get_converter().convert(text)
```

**Alternative:** If you want more natural translation (not just character
substitution), swap `translate()` to call the Claude API or another LLM.
OpenCC is character-mapping only — idioms and regional vocabulary
stay Traditional in meaning; only glyphs change.

---

### Step 5 — Detect Chapter Title (`detect_title.py`)

Look for `第X回` pattern in the OCR text and extract the subtitle.

```python
import re

# Matches: 第九回, 第十二回, 第一百回, etc.
CHAPTER_PATTERN = re.compile(
    r"第[零一二三四五六七八九十百千]+回\s*([^\n]{2,12})"
)

def detect_title(ocr_text: str, page_num: int) -> str:
    """
    Returns folder name like '第九回_鐵槍破犁'.
    Falls back to 'page_009' if no match found.
    """
    match = CHAPTER_PATTERN.search(ocr_text)
    if match:
        full = match.group(0).strip()
        # Sanitise for filesystem: remove punctuation, spaces
        safe = re.sub(r'[\s/\\:*?"<>|]', "_", full)
        return safe
    return f"page_{page_num:03d}"
```

**Edge cases to handle:**
- OCR may introduce small errors in the 第X回 characters
- Some pages may be chapter openers; others are mid-chapter continuation
  (no title on those pages — fallback to `page_NNN`)
- Pages with the same chapter (continuation) get `page_NNN` names;
  only the opener page gets the named folder

---

### Step 6 — Upscale Illustration (Optional, `upscale.py`)

The raw illustration crop at 300 DPI rasterisation will be good enough
for most image-to-video tools. But if you want maximum quality, use
Real-ESRGAN (AI upscaler trained on anime/illustration style — ideal
for these comic-style images).

```python
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from config import UPSCALE_FACTOR

def upscale_image(pil_image):
    """
    Returns upscaled PIL Image using Real-ESRGAN.
    Use RealESRGAN_x4plus_anime_6B model for illustration style.
    """
    model = RRDBNet(num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=UPSCALE_FACTOR,
        model_path="weights/RealESRGAN_x4plus_anime_6B.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )
    import numpy as np, cv2
    arr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    out, _ = upsampler.enhance(arr, outscale=UPSCALE_FACTOR)
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
```

**If skipping Real-ESRGAN:** A simple 2× Lanczos upscale in Pillow
(`img.resize((w*2, h*2), Image.LANCZOS)`) is a reasonable fallback
with no extra dependencies.

---

### Step 7 — Organise Output (`organise.py`)

```python
from pathlib import Path
from config import OUTPUT_DIR, IMAGE_FORMAT

def save_pair(folder_name: str, text: str, illustration_img):
    """
    Creates wuxia/<folder_name>/text.txt and illustration.png
    """
    folder = Path(OUTPUT_DIR) / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    # Save text
    (folder / "text.txt").write_text(text, encoding="utf-8")

    # Save illustration
    ext = IMAGE_FORMAT.lower()
    illustration_img.save(folder / f"illustration.{ext}", IMAGE_FORMAT)
```

---

### Step 8 — Main Orchestrator (`main.py`)

```python
from pathlib import Path
from extract_pages import extract_pages
from split_page import split_page
from ocr_text import ocr_image
from translate import translate
from detect_title import detect_title
from organise import save_pair
from config import UPSCALE_ENABLED

TMP_DIR = Path("tmp_pages")

def run():
    print("Step 1: Extracting page images from PDF...")
    pages = extract_pages(TMP_DIR)
    print(f"  → {len(pages)} pages extracted")

    for page_num, img_path in pages:
        print(f"Processing page {page_num}...")

        # Step 2: Split
        left_img, right_img = split_page(img_path)

        # Step 3: OCR
        raw_text = ocr_image(left_img)

        # Step 4: Translate
        simplified_text = translate(raw_text)

        # Step 5: Detect title
        folder_name = detect_title(simplified_text, page_num)

        # Step 6: Optional upscale
        if UPSCALE_ENABLED:
            from upscale import upscale_image
            right_img = upscale_image(right_img)

        # Step 7: Save
        save_pair(folder_name, simplified_text, right_img)
        print(f"  → Saved as: {folder_name}/")

    print("\nDone! Output in wuxia/")

if __name__ == "__main__":
    run()
```

---

## Testing Strategy

### Before running the full 214-page pipeline:

**1. Visual split check** — verify the midpoint crop is correct:
```python
# quick_check.py
from extract_pages import extract_pages
from split_page import split_page
from pathlib import Path

pages = extract_pages(Path("tmp_pages"))
left, right = split_page(pages[0][1])   # first page
left.save("check_left.jpg")
right.save("check_right.jpg")
# Open and inspect — is left all text? Is right all illustration?
```

**2. OCR spot check** — run OCR on 3-4 pages manually and eyeball accuracy:
```python
# spot_check_ocr.py
from ocr_text import ocr_image
from PIL import Image
img = Image.open("check_left.jpg")
print(ocr_image(img))
```

**3. Title detection check** — confirm regex matches your actual OCR output:
```python
from detect_title import detect_title
sample = "第九回\n鐵槍破犁\n楊鐵心取下壁上掛著的..."
print(detect_title(sample, 6))  # expect: 第九回_鐵槍破犁
```

**4. Dry run (pages 6–10 only):**
Set `END_PAGE = 10` in config.py and run `main.py` to validate the full
pipeline before committing to all 214 pages.

---

## Performance Estimates

| Step | Time per page | 209 pages total |
|---|---|---|
| PDF rasterisation (PyMuPDF) | ~0.3s | ~1 min |
| Image split (Pillow) | ~0.1s | ~0.3 min |
| OCR (PaddleOCR, CPU) | ~3–8s | **10–28 min** |
| OCR (PaddleOCR, GPU) | ~0.5–1s | ~2–3 min |
| Translation (OpenCC) | ~0.01s | negligible |
| Save files | ~0.1s | ~0.3 min |
| **Total (CPU)** | | **~12–30 min** |
| **Total (GPU)** | | **~4–5 min** |

> OCR is by far the bottleneck. A GPU (even a modest one) gives 8–10×
> speedup. On Apple Silicon, PaddlePaddle runs on CPU but is well
> optimised.

---

## Known Edge Cases & How to Handle

| Situation | How to handle |
|---|---|
| Page has no text (illustration-only) | OCR returns empty string → title fallback → `page_NNN` |
| Chapter title spans 2 OCR lines | Regex captures up to 12 chars after `第X回`, adjust length |
| OCR misreads a character in `第X回` | Add fuzzy matching or manual override map in `detect_title.py` |
| Two pages belong to same chapter | Both get fallback `page_NNN` names (only opener gets titled folder) |
| Some pages rotated or skewed | PaddleOCR `use_angle_cls=True` handles most rotation cases |
| PDF page 1 is single-width (cover) | `START_PAGE = 6` already skips it |

---

## Image Quality Recommendations for Image-to-Video

The raw PDF is 72 PPI. After rasterising at 300 DPI and taking the right-side
crop, your illustration will be approximately **2750 × 3750 px** — more than
enough for 1080p/4K video generation.

**Recommended settings by use case:**

| AI Tool | Recommended input |
|---|---|
| Runway Gen-3 / Kling | PNG, 1024–2048px wide, no upscaling needed |
| Stable Video Diffusion | PNG, 1024 × 576 or 576 × 1024 |
| Pika Labs | PNG or JPG, any reasonable resolution |
| ComfyUI img2vid | Match model's native resolution |

**Do not JPEG-compress the illustrations** — save as PNG from Pillow to avoid
generational quality loss when the AI tool re-encodes.

---

## Related Documents

- [Feasibility Analysis](./wuxia_feasibility_analysis.md) — Local M3 vs AutoDL GPU analysis
- [Hybrid Implementation Plan](./wuxia_hybrid_impl_plan.md) — Hybrid local + cloud implementation
- [Implementation Summary](./summary.md) — Brief project summary
- [README](../README.md) — Project overview

---

## Recommended Build Order

1. `config.py` — set all paths and flags
2. `extract_pages.py` + quick visual check
3. `split_page.py` + visual check of crop
4. `ocr_text.py` + spot check a few pages
5. `translate.py` — trivial, test with a string
6. `detect_title.py` + unit test against sample OCR output
7. `organise.py` — trivial
8. `main.py` — wire it all together
9. Dry run on pages 6–10
10. Full run on all 209 pages

---

## Optional Enhancements (Post-MVP)

- **`manifest.json`** in `wuxia/` — index of all folders with page numbers,
  chapter titles, and file paths, for easy downstream processing
- **Progress bar** using `tqdm` around the main loop
- **Resume support** — skip folders that already exist (useful if pipeline crashes mid-run)
- **Claude API translation** — replace OpenCC with a Claude API call for more
  natural Simplified Chinese (better for idioms and literary language)
- **Image-to-video batch config** — auto-generate prompt files alongside
  each illustration based on the chapter text summary
