# OpenCode Review — 连环画 OCR Failure Analysis

**Date:** 2026-05-09  
**Reviewer:** Sisyphus  
**Scope:** OCR pipeline failures for 连环画-format PDFs (`01-桃园结义`, `14-煮酒论英雄`, `41-定军山`)

---

## 1. Problem Statement

The OCR pipeline successfully extracts Traditional Chinese text from `jiang-yun-xing.pdf` (standard layout, `split_ratio=0.5`, no rotation). However, three 连环画-format PDFs (`01-桃园结义`, `14-煮酒论英雄`, `41-定军山`) produce garbled/lost text with **both** AutoDL GPU (`main_autodl.py`) and local CPU (`ocr_one.py --cpu --rotate`). The 连环画 books share a narrow vertical text strip layout on the left side of each spread, with `split_ratio=0.2`, `ocr_rotate_left_cw90=true`, and custom crop margins.

### 1.1 Layout Comparison

| PDF | split_ratio | Rotation | Crop Margins | Text Layout | Result |
|---|---|---|---|---|---|
| `jiang-yun-xing` | 0.5 | No | None | Standard double-spread | ✅ Correct |
| `01-桃园结义` | 0.2 | cw90 | top=0.13, left=0.22, right=0.02 | Vertical strip, 连环画 | ❌ Garbled |
| `14-煮酒论英雄` | 0.2 | cw90 | top=0.13, left=0.22, right=0.02 | Vertical strip, 连环画 | ❌ Garbled |
| `41-定军山` | 0.2 | cw90 | top=0.17, left=0.27, right=0.06 | Vertical strip, 连环画 | ❌ Garbled |

---

## 2. Investigation Method

### 2.1 Tools Used

| Tool | Purpose |
|---|---|
| `golden_ocr_loop.py --cpu --rotate` | Cycles 20 OCR parameter variants against golden reference text |
| Direct `ocr_text.py` function calls | Isolates preprocessing (rotation, upscale, cap, enhance) from PaddleX |
| Image pixel analysis (Python/PIL/NumPy) | Measures dark pixel density, text column positions, stroke widths |
| Config diff & code path analysis | Traces per-book parameter differences through the pipeline |

### 2.2 Diagnostic Tests Performed

1. **With rotation, all 20 variants** — every variant produces the same 38-char output (deterministic, never matches golden reference)
2. **Without rotation, old margins** — output is completely scrambled (reading order wrong)
3. **Without rotation, new margins** — still fragmented (PaddleX detection fails on dense vertical text)
4. **CW rotation (ROTATE_270)** — reading order REVERSED (left-column-first instead of right-column-first)
5. **Increased upscale factor** (min_side 2700→3600→4500) — more noise, lower confidence
6. **Enhanced vs base comparison** — base text consistently cleaner despite lower confidence
7. **Crop margin sweep** (left=0.22→0.05→0.18) — border noise vs text preservation tradeoff

### 2.3 Key Instrumentation

The `_enhance_for_ocr()` confidence comparison was identified as a bug by comparing base vs enhanced outputs across all parameter variants. The confidence threshold of `+0.03` was found to select the noisier enhanced output.

---

## 3. Root Causes

### 3.1 Bug 1: Enhancement Retry Picks Noisier Output (HIGH severity)

**File:** `ocr_text.py:581-585`  
**Mechanism:** The `ocr_image()` function runs OCR twice — once on the original image, once on `_enhance_for_ocr()` (contrast stretch + darken). It selects the result with higher confidence or more CJK characters.

**Problem:** `_enhance_for_ocr()` stretches contrast, which enhances BOTH text strokes AND border artifacts (page edges, illustration bleed-through). PaddleX confidently recognizes these artifacts as characters (digits, letters), inflating both confidence AND CJK count. The threshold `+0.03` is too low — the noisy enhanced output wins.

**Evidence (page_010, CPU PP-OCRv3 mobile, all 20 variants identical):**

| Version | Chars | Conf | Text |
|---|---|---|---|
| Base | 41 | 0.593 | `四他們約定了回日子各地一齊向官兵進·政o因為他們以黃巾里頭XO為標幟所以叫作黃巾軍` |
| Enhanced (was selected) | 46 | 0.651 | `張四他們約定了個二了各地一齊向官兵進祀口口政o因為他們以黃巾里頭火O3為標幟所以叫作黃中軍0` |

Enhanced adds 5 extra chars of noise (`張`, `個`, `祀口口`, `火`, `3`, `0`) while only improving `日子→個二了` (arguably worse). The `0.651 > 0.593 + 0.03` condition always selects the noisy version.

### 3.2 Bug 2: Crop Margins Too Aggressive (MEDIUM severity)

**File:** `configs/books/01-桃园结义.json`  
**Parameters:** `ocr_left_crop: {top: 0.13, left: 0.22, right: 0.02}`

**Problem:** After `split_ratio=0.2`, the left crop is ~412px wide (old) or ~503px wide (after remargin). The `left: 0.22` margin removes 90px from the left edge.

Image analysis of `page_010_left.jpg` (412x1469 crop):
- **First text column at x=60-89** (24.3% dark pixels = text)
- **Left margin at 90px** → cuts ENTIRELY through the first text column
- **Top margin at 219px** (13% of 1688) → cuts through text starting at y=100 (17.7% dark pixels)
- **Result:** 23.8% of width lost, 14.9% of height lost

The `left: 0.22` value was likely set to trim the illustration border on the left side of the strip, but it overshoots and removes text.

### 2.3 The Rotation Tradeoff (Understanding, not a bug)

The `ocr_rotate_left_cw90` flag rotates the vertical text strip 90° CCW via `Image.Transpose.ROTATE_90`. This is a deliberate design choice with a tradeoff:

**With rotation:** Characters are sideways → PaddleX `textline_orientation` model detects and corrects each text region → recognition works but at ~90% character accuracy (政/攻, 里/裹 confusions). **Text ORDER is correct.**

**Without rotation:** Characters are upright in dense vertical columns → PaddleX text detection fragments text into tiny regions → reading order is scrambled (63 chars, 0.435 confidence). **Text ORDER is wrong.**

CCW rotation vs CW rotation: CCW (`ROTATE_90`) preserves right-to-left column reading order. CW (`ROTATE_270`) reverses it. The comment in `ocr_text.py` correctly explains this; the flag name `cw90` is misleading but the implementation is correct.

---

## 4. Fixes Applied

### 4.1 Fix 1: Enhancement Retry Garbage Guard

**File:** `ocr_text.py` (lines 581-588)

**Changes:**
```python
base_garbage = len(base_text) - base_cjk
enhanced_garbage = len(enhanced_text) - enhanced_cjk

# Was: if enhanced_conf > base_conf + 0.03:
if enhanced_conf > base_conf + 0.10 and enhanced_garbage <= base_garbage + 2:
    return enhanced_text
# Was: if enhanced_cjk > base_cjk + 8:
if enhanced_cjk > base_cjk + 8 and enhanced_garbage <= base_garbage + 2:
    return enhanced_text
return base_text
```

**Rationale:**
- Raised confidence threshold from `+0.03` to `+0.10` — a substantial gain is required to prefer enhanced
- Added garbage guard: non-CJK character count must increase by ≤2 — penalizes noise injection
- Safe for `jiang-yun-xing` (high confidence >0.88 triggers early return, never reaches enhancement)

### 4.2 Fix 2: Crop Margin Tuning

**File:** `configs/books/01-桃园结义.json`

```diff
- "top": 0.13,  "left": 0.22,  "right": 0.02
+ "top": 0.05,  "left": 0.18,  "right": 0.02
```

**Rationale:**
- `left: 0.22` (90px) cut into first text column at x=60-89 → `left: 0.18` (77px) trims only the border line (x=0-9, 41% dark) and gap
- `top: 0.13` (219px) cut into text starting at y=100 → `top: 0.05` (80px) preserves more text
- Image dimension change: 412×1469 → 433×1604 (+91px width, +135px height)

**Propagated to:**
| Book | old top | old left | new top | new left |
|---|---|---|---|---|
| `01-桃园结义` | 0.13 | 0.22 | 0.05 | 0.18 |
| `14-煮酒论英雄` | 0.13 | 0.22 | 0.05 | 0.18 |
| `41-定军山` | 0.17 | 0.27 | 0.07 | 0.20 |

---

## 5. Verification

### 5.1 Unit Tests

All 37 existing tests pass:
```
Ran 37 tests in 0.021s
OK
```

### 5.2 Golden Reference Test (page_010, CPU PP-OCRv3 mobile)

| Version | Chars | Conf | Quality |
|---|---|---|---|
| **Before fix** | 46 | 0.651 | ❌ Noisy — `張四個二了祀口口火O30中` |
| **After fix** | 41 | 0.593 | ✅ Cleaner — `回·政o里XO巾軍` |
| Expected | 42 | — | `四他們約定了一個日子各地一齊向官兵進攻因為他們以黃巾裹頭為標幟所以叫作黃巾軍` |

The after-fix output is **cleaner** but still has PaddleOCR character-level confusions (政/攻, 里/裹, 回/了一個). These are expected to improve on AutoDL GPU where `PP-OCRv5_server_rec` replaces the mobile model.

### 5.3 No-Rotation Comparison

Confirmed rotation is necessary:
| Approach | Chars | Conf | Order |
|---|---|---|---|
| No rotation (new margins) | 63 | 0.435 | ❌ Fragmented |
| With rotation (after fix) | 41 | 0.593 | ✅ Coherent |

---

## 6. Remaining Issues & Future Work

### 6.1 Character-Level Confusions on CPU

The remaining character errors (政/攻, 里/裹, 回/了一個) are PaddleOCR recognition-level issues caused by the rotation+orientation-correction path. These should be significantly better on AutoDL GPU with PP-OCRv5 server models.

**Action:** Upload the fixed `work/01-桃园结义/crops_left.zip` to AutoDL and run Phase 2. If results are acceptable, the fix is complete.

### 6.2 41-定军山 Margin Tuning

The margin values for `41-定军山.json` (top=0.07, left=0.20) were applied by proportional calculation without empirical testing. The original config had more aggressive margins (top=0.17, left=0.27, right=0.06), suggesting different border proportions.

**Action:** Visually inspect one crop page and adjust if needed.

### 6.3 Enhancement Retry May Still Be Wrong for Some Cases

The fix raises the threshold to `+0.10`, which is a heuristic. If future PDFs produce base confidence >0.90, the enhancement is skipped entirely (early return at `base_cjk >= 40 and base_conf >= 0.88`). For edge cases between 0.80-0.88, the enhanced version could still be selected.

**Action:** Monitor on AutoDL GPU results. If character-level errors persist, consider a more sophisticated selection (e.g., language model scoring).

---

## 7. Appendix: Key Commands

```bash
# Diagnostic: run golden test with rotation (tests ocr_image with all variants)
PYTHONPATH=. venv/bin/python tools/golden_ocr_loop.py --cpu --rotate \
  work/01-桃园结义/tmp_crops/page_010_left.jpg --max-iters 24

# Manual test: isolate base vs enhanced (bypasses ocr_image selection)
PYTHONPATH=. venv/bin/python << 'EOF'
# ... (see analysis scripts in session history)
EOF

# Reprocess with fixed margins
python main_local.py --book "01-桃园结义.PDF" --start-step 2

# Golden test after fix
PYTHONPATH=. venv/bin/python tools/golden_ocr_loop.py --cpu --rotate \
  work/01-桃园结义/tmp_crops/page_010_left.jpg --max-iters 2

# Run tests
PYTHONPATH=. venv/bin/python -m unittest discover tests/ -v
```
