# 武侠 · 姜云行金庸作品插画集 — PDF Extractor

Extract paired text + illustration content from Chinese comic PDFs in `data/`. Each page has Traditional Chinese text on the left and art on the right — split, OCR (GPU), translate to Simplified Chinese, and save chapter folders.

**Multi-book:** Every local run requires **`--book`** (basename under `data/` or path to `.pdf`). `configs/config.py` is the single loader that reads **`configs/books/<stem>.json`**, merges per-book values (e.g. `pdf_path`, `output_dir`, `start_page`, `end_page`, `split_ratio`) with common defaults, then runtime uses the merged config.

## Pipeline

```
PDF (pages 6-212)
    │
    ▼ [PyMuPDF]
Page JPEGs (300 DPI)
    │
    ▼ [Pillow]
Left crop (text)          Right crop (illustration)
    │                           │
    ▼ [PaddleOCR GPU]           ▼ [Real-ESRGAN optional]
Traditional Chinese        Upscaled illustration (PNG)
    │
    ▼ [OpenCC]
Simplified Chinese
    │
    ▼ [chapter title regex]
work/<slug>/output/第九回_鐵槍破犁/
    ├── text.txt
    └── illustration.png
```

## File Structure

```
├── book_run.py                 ← Resolve --book, sidecar JSON, work/<slug>/ paths
├── data/*.pdf                  ← Source PDFs (gitignored unless you force-add)
├── configs/books/<stem>.json   ← Optional per-PDF overrides (same stem as PDF)
├── configs/config.py           ← Global defaults (tunable settings)
├── extract_pages.py            ← PDF → page JPEGs
├── split_page.py               ← Split double-wide pages left/right
├── ocr_text.py                 ← PaddleOCR (runs on AutoDL GPU)
├── translate.py                ← OpenCC 繁→簡
├── detect_title.py             ← Chapter title regex
├── upscale.py                  ← Real-ESRGAN (optional, AutoDL)
├── organise.py                 ← Save paired files to a given output directory
├── prepare_upload.py           ← Package crops for AutoDL
├── main_local.py               ← Phase 1 + Phase 3 orchestrator
├── main_autodl.py              ← Phase 2 GPU orchestrator
├── download_results.py         ← Download instructions
├── requirements_local.txt      ← Local dependencies
├── requirements_gpu.txt        ← AutoDL dependencies
├── work/<slug>/                ← Per-PDF: tmp_pages, tmp_crops, tmp_results, output
│
├── docs/
│   └── superpowers/specs/      ← e.g. multi-PDF design spec
│
└── README.md
```

## Quick Start

### Phase 1 — Extract & Split (local)

```bash
python main_local.py --book jiang-yun-xing.pdf
```

Resume from split only (reuse existing `tmp_pages`):

```bash
python main_local.py --book jiang-yun-xing.pdf --start-step 2
```

Artifacts: `work/jiang-yun-xing/tmp_pages/`, `work/jiang-yun-xing/tmp_crops/`, `work/jiang-yun-xing/crops_left.zip`.

### Phase 2 — OCR (AutoDL GPU)

**Before each new book on the same instance:** empty `/root/wuxia_crops/` and `/root/wuxia_output/` so crops and OCR from two PDFs never mix.

Upload the zip to an [AutoDL](https://autodl.com) GPU instance:

```bash
$ GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/jxjwilliam/wuxia-jiangyunxing.git
$ scp -P 52290 work/jiang-yun-xing/crops_left.zip root@connect.westb.seetacloud.com:/root/
$ zip -o crops_left.zip -d /root/wuxia_crops/

$ pip install paddlepaddle-gpu paddleocr pillow
$ python main_autodl.py
```

Estimated cost: ¥1-2 (~$0.15-0.30) on an RTX 4090 instance.

### Phase 3 — Assemble (local)

```bash
# Download wuxia_output.zip from AutoDL, then:
unzip -o wuxia_output.zip -d work/jiang-yun-xing/tmp_results/

python main_local.py --book jiang-yun-xing.pdf --phase3
```

Output: `work/jiang-yun-xing/output/<chapter>/` with `text.txt` + `illustration.png`.

## Dependencies

**Local:**
```bash
$ pip install pymupdf pillow opencc-python-reimplemented
```

**AutoDL GPU:**
```
pip install paddlepaddle-gpu paddleocr pillow
# Optional: pip install basicsr realesrgan opencv-python-headless
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Hybrid local + cloud** | PaddleOCR (best Traditional Chinese OCR) lacks Apple Silicon support. Local for CPU work, AutoDL GPU for OCR. |
| **300 DPI rasterisation** | Native PDF is 72 PPI — upscaling to 300 DPI gives ~5500×3750px for better OCR and illustration quality. |
| **PNG for illustrations** | Lossless format avoids generational quality loss when image-to-video tools re-encode. |
| **OpenCC for translation** | Fast, offline, deterministic character mapping — no API calls needed. |
| **Chapter title folder names** | `第九回_鐵槍破犁` instead of numeric page numbers — human-readable. Falls back to `page_NNN` for mid-chapter pages. |

## Output Structure

```
work/<slug>/output/
├── 第九回_鐵槍破犁/
│   ├── text.txt           ← OCR'd Simplified Chinese
│   └── illustration.png   ← Right-side comic panel
├── 第十回_寃家聚頭/
│   ├── text.txt
│   └── illustration.png
├── page_028/               ← Fallback (no chapter title detected)
│   ├── text.txt
│   └── illustration.png
...
```

## Optional Enhancements

- **Auto-upscale:** Set `UPSCALE_ENABLED = True` in `configs/config.py` — Real-ESRGAN will upscale illustrations 4× on AutoDL.
- **Claude API translation:** Replace `translate.py` with an LLM call for more natural Simplified Chinese (literary idioms).
- **Image-to-video prompts:** Auto-generate prompt files alongside each illustration based on chapter text summary.
- **Resume support:** Skip already-processed folders if the pipeline crashes mid-run.

## Docs Index

| Doc | Description |
|---|---|
| [Implementation Plan](docs/wuxia_extraction_plan.md) | Original full pipeline design |
| [Feasibility Analysis](docs/wuxia_feasibility_analysis.md) | Local M3 vs AutoDL GPU analysis |
| [Hybrid Implementation Plan](docs/wuxia_hybrid_impl_plan.md) | This implementation |
| [Summary](docs/summary.md) | Implementation summary |
