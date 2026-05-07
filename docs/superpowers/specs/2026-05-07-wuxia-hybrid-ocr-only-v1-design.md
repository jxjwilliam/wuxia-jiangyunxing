# Wuxia Hybrid OCR-Only v1 Design

**Goal:** Build a reliable first version that extracts left-side Traditional Chinese text from the PDF, converts it to Simplified Chinese, and pairs each text with the right-side image into structured `wuxia/` folders.

**Scope:** Hybrid pipeline only (local preprocess + AutoDL OCR + local assembly). This design explicitly excludes Real-ESRGAN upscaling for v1.

## 1. Context and Constraints

- Source file is image-based PDF (`data/jiang-yun-xing.pdf`), so OCR is required.
- Processing starts from page 6 (skip pages 1-5).
- Local machine is MacBook Pro M3 18GB; PP-OCR on Apple Silicon is not the preferred first-run path.
- AutoDL GPU is used only for OCR to minimize cloud cost and runtime.
- Final output must preserve one text-image pair per processed page under `wuxia/`.

## 2. Chosen Approach

The selected architecture is modular scripts plus a thin orchestrator:

1. Local phase A: rasterize pages and split left/right images.
2. Cloud phase B: run batch PaddleOCR on left crops in AutoDL.
3. Local phase C: convert OCR text to Simplified Chinese, detect folder names, and assemble final outputs.

This balances delivery speed and maintainability for `ocr_only_v1`.

## 3. Architecture

### 3.1 Components

- `config.py`
  - Central settings: paths, page range, split ratio, naming, OpenCC config.
- `extract_pages.py`
  - PDF to per-page raster images.
- `split_page.py`
  - Split each page image into `left` (text) and `right` (illustration).
- `cloud/ocr_autodl.py`
  - AutoDL-side OCR batch process over uploaded left crops.
- `translate.py`
  - Traditional to Simplified conversion via OpenCC (`t2s`).
- `detect_title.py`
  - Chapter title extraction using `第...回` pattern with fallback naming.
- `organise.py`
  - Build `wuxia/` folder tree and place paired artifacts.
- `main.py`
  - Phase orchestrator with explicit run modes.

### 3.2 Data Flow

1. `data/jiang-yun-xing.pdf`
2. `tmp/pages/page_XXX.jpg`
3. `tmp/left/page_XXX.jpg` and `tmp/right/page_XXX.jpg`
4. Upload `tmp/left` to AutoDL
5. AutoDL OCR output: `ocr_raw/page_XXX.txt`
6. Download `ocr_raw` back locally
7. Convert to `ocr_simplified/page_XXX.txt`
8. Detect folder name and assemble:
   - `wuxia/<chapter_or_page>/text.txt`
   - `wuxia/<chapter_or_page>/illustration.png`

### 3.3 Phase Boundaries

- **Local Prep Phase** must complete before any upload.
- **Cloud OCR Phase** must produce one text file per left crop.
- **Local Assembly Phase** runs only after OCR output parity check passes.

## 4. Interface Contracts

### 4.1 File Naming

- Page identity is canonicalized as `page_XXX` (3-digit, zero-padded).
- All intermediate artifacts use this identity for deterministic matching.

### 4.2 Folder Naming Rules

Priority:
1. Detected chapter title, normalized for filesystem safety.
2. Fallback: `page_XXX`.

Duplicate chapter names are disambiguated by suffix `_pXXX`.

### 4.3 Minimum Required Artifacts Per Page

- One right image crop.
- One OCR raw text.
- One simplified text.
- One final output folder containing:
  - `text.txt`
  - `illustration.png`

## 5. Error Handling and Recovery

## 5.1 Preflight Checks

- Validate input PDF existence and readability.
- Validate/create required local directories.
- Validate required dependency imports per phase.

## 5.2 Runtime Handling

- Local prep failure aborts pipeline before cloud operations.
- OCR count mismatch aborts assembly and reports missing `page_XXX`.
- OpenCC conversion failure on a page writes raw OCR text as fallback and logs warning.
- Any missing right image for a page aborts assembly for deterministic pairing.

## 5.3 Resumability (v1 level)

- Each phase can be run independently.
- Existing valid artifacts are reused when rerunning the same phase.
- No database/state service required; filesystem state is the source of truth.

## 6. Testing and Verification Strategy

### 6.1 Dry Run

- Run pages 6-10 first.
- Verify each page has complete artifacts across all three phases.

### 6.2 Full Run Validation

- Final folder count equals processed page count.
- No page id missing from final output.
- Spot-check 3 random pages for text-image correctness and chapter naming quality.

### 6.3 Acceptance Criteria (v1)

- Pipeline completes without manual data repair.
- Every processed page produces a valid pair in `wuxia/`.
- Text is Simplified Chinese output from OCR raw Traditional Chinese.
- Runtime path remains hybrid and excludes upscaling.

## 7. Non-Goals for v1

- Real-ESRGAN or any AI image upscaling.
- Full local PP-OCR on Apple Silicon.
- OCR confidence filtering, semantic cleanup, or LLM post-editing.
- Complex orchestration infra (queue, DB, distributed workers).

## 8. Future Extensions

- Optional AutoDL upscaling phase with Real-ESRGAN.
- Improved title extraction with multi-pattern ranking.
- Quality scoring report (OCR confidence, missing-character heuristics).
- Manifest file for richer resume and audit trails.
