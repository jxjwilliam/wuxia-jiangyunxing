# Wuxia PDF Extraction — Local vs Cloud Feasibility Analysis

## Overview

This document evaluates whether the extraction pipeline can run on your **MacBook Pro M3 (18GB RAM)** or whether a **cloud GPU (AutoDL)** is needed for acceptable performance. It extends the existing [`wuxia_extraction_plan.md`](./wuxia_extraction_plan.md) with platform-specific findings and revised recommendations.

---

## 1. Hardware Profiles

### Local: MacBook Pro M3 (18GB)

| Component | Spec |
|---|---|
| CPU | 8-core (4P + 4E) |
| GPU | 10-core Apple GPU (unified memory) |
| RAM | 18 GB unified (shared CPU/GPU) |
| Memory bandwidth | ~100 GB/s |
| Neural Engine | 16-core |
| ML framework | PyTorch MPS (Metal), Core ML |
| Disk | Fast NVMe SSD |

### Cloud: AutoDL (Typical GPU Instance)

| GPU | VRAM | Price (¥/hr) | CUDA Cores | Notes |
|---|---|---|---|---|
| RTX 4090 | 24 GB | ~3-5 | 16,384 | Best perf for PaddleOCR |
| RTX 3090 | 24 GB | ~2-3 | 10,496 | Good value |
| A100 (40G) | 40 GB | ~8-12 | 6,912 | Overkill for this task |
| RTX 4060 Ti | 16 GB | ~1-2 | 4,352 | Budget option, still 10× faster than CPU OCR |

---

## 2. Component-by-Component Feasibility

### Pipeline Steps

| Step | Tool | Local (M3 18GB) | AutoDL (GPU) | Recommendation |
|---|---|---|---|---|
| ① PDF rasterisation | PyMuPDF (fitz) | ✅ Fast (~1 min) | ✅ Fast (~1 min) | **Local** — CPU-bound, negligible time |
| ② Image split | Pillow | ✅ Fast (~30s) | ✅ Fast (~30s) | **Local** — trivial crop operation |
| ③ OCR (Traditional Chinese) | PaddleOCR | ⚠️ **Problematic** | ✅✅ **Fast** (~2-3 min) | **AutoDL** (see §3 for details) |
| ④ Translation (繁→简) | OpenCC | ✅ Negligible | ✅ Negligible | **Local** — instant, no GPU needed |
| ⑤ Chapter title detection | regex | ✅ Negligible | ✅ Negligible | **Local** — pure string matching |
| ⑥ Upscaling (optional) | Real-ESRGAN | ⚠️ **Slow/unstable** | ✅✅ **Fast** (~5-10 min for 207 img) | **AutoDL** (or skip) |
| ⑦ File organization | Python | ✅ Negligible | ✅ Negligible | **Local** |

### 2.1 PyMuPDF + Pillow — Fully Local ✅

These are pure CPU/memory operations. PyMuPDF renders pages at 300 DPI in ~0.3s/page. All 207 pages → ~1 minute. No GPU needed.

The images are saved as JPEGs to `tmp_pages/` (~50-100 MB total for 207 pages at JPEG quality).

### 2.2 PaddleOCR — Problematic on Apple Silicon ⚠️

**Critical finding:** PaddleOCR (PP-OCR v2/v3) has **no official support for Apple Silicon (M-series)**. Multiple GitHub issues confirmed this:

- [GitHub Discussion #13036](https://github.com/PaddlePaddle/PaddleOCR/discussions/13036): Official response — "We currently do not support the M-series system"
- Install attempts on M3 Pro result in `paddlepaddle` not being available for `arm64` Python
- The `paddlepaddle` CPU package for macOS only supports `x86_64` (Intel)

**What works on Apple Silicon (as of 2025-2026):**

| Variant | Status | Notes |
|---|---|---|
| PP-OCR (paddlepaddle 2.x) | ❌ **No** | Not available for arm64 macOS |
| PP-OCR in Docker (Rosetta 2) | ⚠️ Partial | x86 emulation, CPU-only, slow |
| PaddleOCR-VL (paddlepaddle 3.x) | ✅ **Yes** | New VLM-based OCR, supports M-series via MLX acceleration |

**PaddleOCR-VL** is a newer vision-language-model based OCR that *does* support Apple Silicon through:
- `paddlepaddle==3.2.1` (arm64 macOS build available)
- Optional MLX acceleration (runs on Apple GPU via Metal)
- Official docs: [PaddleOCR-VL Apple Silicon Usage Tutorial](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html)

**However**, PaddleOCR-VL is much heavier than PP-OCR (VLM vs lightweight CNN model). With 18GB unified memory on M3, it will work but may be slower than a dedicated GPU solution.

### 2.3 OCR Alternatives That Work on M3 Locally

If you prefer to run everything locally without AutoDL:

| OCR Engine | M3 Support | Traditional Chinese | Vertical Text | Speed (per page) | Total (207 pages) |
|---|---|---|---|---|---|
| **PaddleOCR-VL** (MLX) | ✅ Official | ✅ Good | ✅ Good | ~2-5s | ~7-17 min |
| **EasyOCR** (CPU/PyTorch) | ✅ Works | ✅ Good | ✅ Good | ~5-15s | ~17-52 min |
| **Tesseract** (CPU) | ✅ Native | ⚠️ Limited | ❌ Poor | ~3-8s | ~10-28 min |
| **Surya OCR** (MPS) | ✅ MPS | ✅ Good | ✅ Good | ~3-10s | ~10-35 min |
| **Apple Vision** (Native) | ✅ Native | ⚠️ Limited | ❌ Poor | ~1-3s | ~3-10 min |

**Recommendation for local-only:** PaddleOCR-VL with MLX — best accuracy for vertical Traditional Chinese, and uses M3's GPU via Metal. Next best: EasyOCR.

### 2.4 Real-ESRGAN — GPU Recommended ⚠️

- **On M3 18GB:** Real-ESRGAN's MPS support is still experimental (unmerged PR [#902](https://github.com/xinntao/Real-ESRGAN/issues/902)). Running on CPU would be very slow — estimated 30-60 seconds per 2750×3750px image (hours for all 207).
- **On AutoDL GPU (RTX 4090):** ~5-10 seconds per image, total ~20-30 min for 207 images.
- **Alternative for local:** Pillow Lanczos upscale (2× is effectively free) if you just need larger dimensions without AI quality gain.

**Real-ESRGAN is optional.** The 300 DPI rasterisation already gives ~2750×3750px illustrations which is sufficient for most image-to-video tools. Only use Real-ESRGAN if you need maximum quality.

---

## 3. Recommended Pipeline Architectures

### Option A: Fully Local (Simplest — PaddleOCR-VL + MLX)

Recommended if: **You want zero cloud cost and can tolerate moderate speed.**

```
Local M3:
  PDF → [PyMuPDF] → page JPEGs → [Pillow] → split L/R
                                              ↓
  text.txt ← [OpenCC] ← raw text ← [PaddleOCR-VL] ← left crop
                ↓
          [detect_title] → folder name
                                              ↓
  illustration.png ← [Pillow Lanczos upscale] ← right crop
```

**Pros:** Everything offline, no setup on cloud, no data transfer.
**Cons:** PaddleOCR-VL is heavier than PP-OCR; M3 18GB may page/swAP under load.
**Estimated time:** ~15-30 min total.

### Option B: Hybrid (Local Prep → AutoDL OCR → Local Assembly) ✅ RECOMMENDED

Recommended if: **You want the fastest OCR with minimal cloud cost.**

```
[LOCAL M3]
  PDF → [PyMuPDF] → page JPEGs → [Pillow] → split L/R

[UPLOAD to AutoDL]
  left crops (207 images) → upload (~50 MB total)

[AUTODL GPU]
  left crops → [PaddleOCR GPU] → raw text files (tiny)
  right crops → [Real-ESRGAN GPU] → upscaled illustrations (optional)

[DOWNLOAD from AutoDL]
  raw text + illustrations → download

[LOCAL M3]
  raw text → [OpenCC] → simplified text
  simplified text → [detect_title] → folder names
  save pairs to wuxia/
```

**Pros:** Fastest OCR (PaddleOCR on GPU), minimal local compute, pay only for GPU runtime.
**Cons:** Requires AutoDL account, data upload/download steps.
**Estimated AutoDL cost:** ~¥5-15 total ($0.70-2.00).
**Estimated time:** ~5-10 min total (including transfer).

### Option C: Fully AutoDL (Simplest Setup)

Recommended if: **You don't want to manage local dependencies at all.**

```
Upload PDF to AutoDL → run full pipeline on GPU → download wuxia/ folder
```

**Pros:** Zero local setup (everything runs in a pre-configured Docker image).
**Cons:** GPU idle during CPU tasks (paying for wasted time); larger download.
**Estimated AutoDL cost:** ~¥10-30 ($1.50-4.00).
**Estimated time:** ~5-10 min total.

---

## 4. Cost Analysis

### AutoDL GPU Pricing (Approximate as of 2026)

| GPU Instance | ¥/hr | Est. runtime | Est. cost |
|---|---|---|---|
| RTX 4090 (24GB) | ~3-5 | 5-10 min | ¥0.5-1.0 (~$0.07-0.14) |
| RTX 3090 (24GB) | ~2-3 | 5-12 min | ¥0.3-0.6 (~$0.04-0.08) |
| RTX 4060 Ti (16GB) | ~1-2 | 10-20 min | ¥0.2-0.7 (~$0.03-0.10) |

**AutoDL billing:** Charged by the second, minimum 1 minute. Most instances cost ¥1-5/hr. A full OCR run (207 pages) on RTX 4090 costs **less than ¥1** ($0.15).

### Data Transfer

| Data | Size | Upload time (50 Mbps) | Download time |
|---|---|---|---|
| PDF | 39 MB | ~6s | N/A |
| Left crops (207 JPEGs) | ~50-100 MB | ~10-15s | N/A |
| Right crops (207 JPEGs) | ~50-100 MB | ~10-15s | N/A |
| Upscaled illustrations (207 PNGs) | ~2-5 GB | N/A | ~5-15 min |

> **Tip:** Only upload the PDF and do the page extraction ON AutoDL (Option C), or pre-extract + upload just the split halves (Option B — smaller, and you can verify the split quality locally first).

---

## 5. Environment Setup Guide

### 5.1 Local M3 Setup (Option A — PaddleOCR-VL)

```bash
# 1. Create virtual environment
python -m venv .venv_wuxia
source .venv_wuxia/bin/activate

# 2. Install PaddlePaddle 3.x (Apple Silicon build)
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 3. Install PaddleOCR-VL
python -m pip install -U "paddleocr[doc-parser]"

# 4. Install MLX acceleration (optional, for faster inference)
python -m pip install "mlx-vlm>=0.3.11"

# 5. Install other dependencies
pip install pymupdf pillow opencc-python-reimplemented

# 6. Start MLX inference server (in separate terminal)
mlx_vlm.server --port 8111

# 7. Update config.py
# OCR_LANG = "chinese_cht"
# Set PaddleOCR-VL backend to use MLX server
```

### 5.2 Local M3 Setup (Alternative — EasyOCR)

```bash
# 1. Create virtual environment
python -m venv .venv_wuxia
source .venv_wuxia/bin/activate

# 2. Install dependencies
pip install pymupdf pillow easyocr opencc-python-reimplemented

# 3. For faster inference, install PyTorch with MPS support
pip install torch torchvision torchaudio
# PyTorch on M3 will use MPS backend automatically
```

**Note on EasyOCR + M3:** EasyOCR uses PyTorch. On M3, PyTorch can use the MPS (Metal Performance Shaders) backend for GPU acceleration. This gives approximately 2-5× speedup over CPU. First run will download model files (~100 MB).

### 5.3 AutoDL Setup (Option B/C)

**Step 1: Create AutoDL instance**
1. Go to [autodl.com](https://www.autodl.com) and register
2. Create an instance with a GPU (recommended: RTX 4090 or RTX 3090)
3. Choose a PyTorch/PaddlePaddle community image (many come pre-configured)
4. Start the instance

**Step 2: Install dependencies**
```bash
# On AutoDL instance (already has CUDA + PyTorch typically)
pip install pymupdf pillow paddlepaddle-gpu paddleocr opencc-python-reimplemented
# Optional upscaling:
pip install basicsr realesrgan
```

**Step 3: File transfer**
```bash
# Upload files to AutoDL
scp -P <port> data/jiang-yun-xing.pdf root@<autodl-ip>:/root/
# Or use AutoDL's built-in file manager (drag & drop via web UI)

# Download results
scp -P <port> -r root@<autodl-ip>:/root/wuxia/ ./
```

**Step 4: Run the pipeline**
```bash
python main.py  # This will use GPU automatically via PaddleOCR
```

> **AutoDL Tip:** Save your environment as a custom image after setup. Next time, you can spin up a new instance from your image and skip all installation steps.

---

## 6. Decision Matrix

| Factor | Option A: Fully Local (PaddleOCR-VL) | Option A: Fully Local (EasyOCR) | Option B: Hybrid ✅ | Option C: Fully AutoDL |
|---|---|---|---|---|
| **Setup effort** | Medium (MLX, PaddlePaddle 3.x) | Low (pip install) | Medium (AutoDL + local) | Medium (AutoDL only) |
| **Total time** | ~15-30 min | ~20-60 min | ~5-10 min | ~5-10 min |
| **Cloud cost** | ¥0 | ¥0 | ~¥1-5 ($0.15-0.70) | ~¥3-10 ($0.50-1.50) |
| **OCR quality** | ✅ High (PaddleOCR-VL) | ✅ High | ✅✅ Highest (PaddleOCR GPU) | ✅✅ Highest |
| **Upscale quality** | 🟡 Lanczos only | 🟡 Lanczos only | ✅ Real-ESRGAN possible | ✅ Real-ESRGAN possible |
| **Offline capable** | ✅ Yes | ✅ Yes | ❌ No (needs cloud) | ❌ No (needs cloud) |
| **Best for** | Small batches, offline | Quick & dirty, no GPU | **Production quality, best value** | "Set and forget" |

---

## 7. Updated Implementation Plan

### Recommended Workflow (Hybrid — Option B)

```
Phase 1 — Local Setup (5 min)
├── Install Python dependencies (pymupdf, pillow, opencc)
├── Create config.py with paths
└── Test PDF can be opened

Phase 2 — Local Preprocessing (2 min)
├── Run extract_pages.py → page JPEGs
├── Run split_page.py → left/right crops
└── Visually verify split quality (check a few pages)

Phase 3 — AutoDL Setup (10 min)
├── Spin up GPU instance on AutoDL
├── Install PaddleOCR + dependencies
└── Upload split images (left/right crops)

Phase 4 — GPU Processing (3-5 min on RTX 4090)
├── Run PaddleOCR on all 207 left crops
├── (Optional) Run Real-ESRGAN on all 207 right crops
└── Download results (text JSONs + upscaled images)

Phase 5 — Local Assembly (1 min)
├── Run translate.py (OpenCC: 繁→簡)
├── Run detect_title.py (chapter title regex)
├── Run organise.py (create wuxia/ folder structure)
└── Verify output structure

Phase 6 — Dry Run + Full Run
├── Test with 5 pages (6-10) first
└── Full 207-page pipeline
```

### If Going Fully Local (Option A — PaddleOCR-VL with MLX)

The original plan at `wuxia_extraction_plan.md` needs these modifications:

| Original plan | Updated for PaddleOCR-VL on M3 |
|---|---|
| `paddlepaddle==2.x` | `paddlepaddle==3.2.1` (arm64 macOS build) |
| `paddleocr` (PP-OCR) | `paddleocr[doc-parser]` (PaddleOCR-VL) |
| `OCR_USE_GPU = False` | Use MLX backend instead |
| `ocr.py` with `PaddleOCR()` | Use `PaddleOCRVL()` with MLX-VLM server |

---

## 8. Recommendation

**Go with the Hybrid approach (Option B) for first run:**

1. **Do PDF extraction + page splitting locally** (minutes, zero GPU needed)
2. **Upload to AutoDL for OCR only** (¥1-2, < 5 minutes on RTX 4090)
3. **Download results and do translation/organization locally** (minutes)

This gives you the fastest PaddleOCR GPU quality at minimal cost. The local M3 handles everything except the GPU-bound OCR.

If you later want to run everything locally, switch to PaddleOCR-VL with MLX — but for the first full extraction, the hybrid path is the most reliable and fastest.

---

## Related Documents

- [Implementation Plan](./wuxia_extraction_plan.md) — Original full pipeline design
- [Hybrid Implementation Plan](./wuxia_hybrid_impl_plan.md) — Detailed hybrid workflow
- [Implementation Summary](./summary.md) — Brief project summary
- [README](../README.md) — Project overview

---

## Appendix: Useful Commands

### Check PaddlePaddle Apple Silicon Compatibility

```bash
# Check if paddlepaddle can be installed
pip install paddlepaddle==3.2.1  # arm64 build (PaddleOCR-VL compatible)

# Verify installation
python -c "import paddle; print(paddle.__version__); print(paddle.is_compiled_with_cuda())"
```

### Check PyTorch MPS (for EasyOCR or Surya alternatives)

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

### AutoDL Quick CMDS

```bash
# Install PaddleOCR GPU version
pip install paddlepaddle-gpu paddleocr

# Verify GPU
python -c "import paddle; print('GPU available:', paddle.is_compiled_with_cuda())"

# Upload files
scp -P 12345 data/*.jpg root@<ip>:/root/tmp_pages/

# Run OCR
python main.py

# Download results (from your local machine)
scp -P 12345 -r root@<ip>:/root/wuxia/ ./
```
