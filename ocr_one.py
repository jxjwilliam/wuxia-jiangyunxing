#!/usr/bin/env python3
"""
Local / smoke test: run the same OCR path as AutoDL on a single image.

Examples:
  # Mac / no GPU: force CPU
  python ocr_one.py --cpu work/01-桃园结义/tmp_crops/page_008_left.jpg --rotate

  # Optional Simplified output (needs opencc-python-reimplemented)
  python ocr_one.py --cpu --simplified --rotate path/to/page_011_left.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="OCR one image (PaddleOCR, same as Phase 2).")
    p.add_argument("image", type=Path, help="Path to JPG/PNG left crop")
    p.add_argument(
        "--rotate",
        action="store_true",
        help="Simulate phase2_manifest ocr_rotate_left_cw90 (vertical strip → horizontal).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (use on Mac / machines without paddlepaddle-gpu).",
    )
    p.add_argument(
        "--simplified",
        action="store_true",
        help="Print Simplified Chinese (OpenCC t2s) after OCR.",
    )
    p.add_argument(
        "--doc-preprocessor",
        action="store_true",
        help="Enable PaddleX doc orientation + UVDoc unwarp (default: off for flat comics; see configs.config).",
    )
    args = p.parse_args()

    if not args.image.is_file():
        print(f"❌ Not a file: {args.image}", file=sys.stderr)
        sys.exit(2)

    import configs.config as cfg

    if args.cpu:
        cfg.OCR_USE_GPU = False
        # PP-OCRv5 server det on ~9k-wide CPU runs can spike RAM → macOS SIGKILL (“killed”).
        if getattr(cfg, "OCR_MAX_INPUT_LONG_SIDE", 0) == 0:
            cfg.OCR_MAX_INPUT_LONG_SIDE = 4480
        lite = getattr(cfg, "OCR_CPU_USE_LITE_MODELS", True)
        print(
            f"→ CPU: PP-OCRv3 mobile={'on' if lite else 'off'} (set OCR_CPU_USE_LITE_MODELS=false to match PP-OCRv5 server—needs RAM)",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"→ CPU: OCR_MAX_INPUT_LONG_SIDE={cfg.OCR_MAX_INPUT_LONG_SIDE} (0=all pixels; avoids OOM on server models)",
            file=sys.stderr,
            flush=True,
        )
    if args.doc_preprocessor:
        cfg.OCR_PADDLEX_USE_DOC_PREPROCESSOR = True

    from ocr_text import ocr_image, set_phase2_manifest

    set_phase2_manifest({"ocr_rotate_left_cw90": bool(args.rotate)})

    from PIL import Image

    print(
        f"→ image={args.image.resolve()} rotate={args.rotate} cpu={args.cpu}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "Running OCR (CPU; large images can take several minutes; no per-step logs) …",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.perf_counter()
    with Image.open(args.image) as img:
        text = ocr_image(img)
    print(
        f"→ OCR done in {time.perf_counter() - t0:.1f}s, {len(text)} chars",
        file=sys.stderr,
        flush=True,
    )

    if args.simplified:
        try:
            from opencc import OpenCC

            from configs.config import OPENCC_CONFIG

            text = OpenCC(OPENCC_CONFIG).convert(text)
        except Exception as e:
            print(f"⚠️ OpenCC skipped: {e}", file=sys.stderr)

    print(text)


if __name__ == "__main__":
    main()
