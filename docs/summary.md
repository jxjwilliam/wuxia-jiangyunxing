# Wuxia PDF Extractor — Implementation Summary

## Project

Extract text and images from **姜云行金庸作品插画集** (`data/jiang-yun-xing.pdf`), a 214-page Chinese wuxia comic PDF. Each page has Traditional Chinese text (left) and a comic illustration (right).

## Approach: Hybrid Local + Cloud GPU

The pipeline is split into 3 phases to maximize efficiency — CPU-bound work runs locally, GPU-bound work runs on [AutoDL](https://autodl.com) cloud GPU instances.

| Phase | Where | What | Time | Cost |
|---|---|---|---|---|
| **Phase 1** ✅ | Local M3 | PDF → page JPEGs → split left/right → package zip | ~2 min | $0 |
| **Phase 2** | AutoDL GPU | PaddleOCR (left crops) + optional Real-ESRGAN (right crops) | ~3-5 min | ~¥1-2 |
| **Phase 3** | Local M3 | OpenCC translation → chapter title detection → organize folders | ~1 min | $0 |

## Files Created (14 files)

| File | Purpose |
|---|---|
| `config.py` | All settings: paths, flags, thresholds |
| `extract_pages.py` | PyMuPDF rasterisation (300 DPI) |
| `split_page.py` | Pillow midpoint crop |
| `ocr_text.py` | PaddleOCR module (GPU on AutoDL) |
| `translate.py` | OpenCC Traditional → Simplified |
| `detect_title.py` | Regex pattern `第[零一二三四五六七八九十百千]+回` |
| `upscale.py` | Real-ESRGAN anime upscaler (optional) |
| `organise.py` | Save paired text + PNG to folder tree |
| `prepare_upload.py` | Package crops into zip for AutoDL |
| `main_local.py` | Phase 1 (extract+split) + Phase 3 (assemble) orchestrator |
| `main_autodl.py` | Phase 2 GPU orchestrator |
| `download_results.py` | Download instructions helper |
| `requirements_local.txt` | Local pip deps |
| `requirements_gpu.txt` | AutoDL pip deps |

## Phase 1 Results

- **209 pages** extracted at 300 DPI (~0.5s per page)
- **418 crop files** (209 text + 209 illustration)
- `crops_left.zip` — **374 MB** (text panels for OCR)
- `crops_right.zip` — **599 MB** (illustration panels)

## Key Technical Decisions

### Why not fully local?

PaddleOCR (PP-OCR) **has no Apple Silicon support** — the `paddlepaddle` package is unavailable for arm64 macOS. Alternatives were evaluated:

| Engine | Apple Silicon | Traditional Chinese | Vertical Text |
|---|---|---|---|
| PaddleOCR-VL (MLX) | ✅ | ✅ | ✅ |
| EasyOCR (MPS) | ✅ | ✅ | ✅ |
| **PaddleOCR GPU** (AutoDL) | N/A | ✅✅ Best | ✅✅ Best |

PaddleOCR on GPU delivers the best accuracy for vertical Traditional Chinese and costs only ~¥1-2 for the full run.

### Image quality

- PDF native resolution: **72 PPI** (too low)
- Rasterisation at **300 DPI**: ~5500×3750px pages → ~2750×3750px illustration crops
- **PNG output**: Lossless, no generational quality loss for downstream image-to-video
- Optional **Real-ESRGAN 4× upscale** on AutoDL for maximum quality

### Translation

**OpenCC** is used for character-level Traditional → Simplified Chinese conversion (fast, offline, deterministic). For better literary translation of classical wuxia language, the `translate.py` function can be swapped to call the Claude API.

## What's Next

1. **Upload** `crops_left.zip` and `crops_right.zip` to AutoDL
2. **Run** `python main_autodl.py` on the GPU instance
3. **Download** AutoDL results to `tmp_results/`
4. **Run** `python main_local.py --phase3` to produce the final `wuxia/` output

## Related Documents

- [Implementation Plan](wuxia_extraction_plan.md) — Original full design
- [Feasibility Analysis](wuxia_feasibility_analysis.md) — Local vs cloud analysis
- [Hybrid Implementation Plan](wuxia_hybrid_impl_plan.md) — Detailed hybrid workflow
- [README](../README.md) — Project overview and quick start
