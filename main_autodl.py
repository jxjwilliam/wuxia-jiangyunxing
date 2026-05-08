"""
Phase 2: AutoDL GPU Orchestrator.
Run this on the AutoDL GPU instance after uploading split crops.
Processes: OCR on left crops only (ocr_only_v1).
"""
import json
import os
import sys
from pathlib import Path

# Must run before Paddle / NumPy threaded BLAS imports for stable GPU OCR.
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image

from configs.config import (
    AUTODL_OUTPUT_DIR,
    AUTODL_REMOTE_DIR,
    OCR_OUTPUT_SIMPLIFIED,
    OPENCC_CONFIG,
    PHASE2_MANIFEST_NAME,
)

CROPS_DIR = Path(AUTODL_REMOTE_DIR)
OUT_DIR = Path(AUTODL_OUTPUT_DIR)


def phase2_gpu():
    if not CROPS_DIR.exists():
        print(f"❌ Input crops directory not found: {CROPS_DIR}")
        print("Unzip crops_left.zip into this directory before running.")
        sys.exit(1)

    manifest: dict = {}
    manifest_path = CROPS_DIR / PHASE2_MANIFEST_NAME
    if manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                manifest = raw
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON in {manifest_path.name}; ignoring.")

    from ocr_text import ocr_image, set_phase2_manifest

    set_phase2_manifest(manifest)
    if manifest.get("ocr_rotate_left_cw90"):
        print(
            "  → phase2_manifest: ocr_rotate_left_cw90 "
            "(vertical left strip → rotated for horizontal OCR + reading order)"
        )

    opencc = None
    if OCR_OUTPUT_SIMPLIFIED:
        try:
            from opencc import OpenCC

            opencc = OpenCC(OPENCC_CONFIG)
            print("  → OpenCC enabled: OCR output will be converted to Simplified Chinese.")
        except Exception as e:
            print(
                "  ⚠️ OpenCC not available; OCR stays Traditional. Install:\n"
                "     python -m pip install opencc-python-reimplemented"
            )
            print(f"     ({type(e).__name__}: {e})")

    print("=" * 50)
    print("PHASE 2: AutoDL GPU Processing")
    print("=" * 50)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect left crop files by page number
    left_pages = {}
    for f in sorted(CROPS_DIR.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # Expected format: page_006_left.jpg
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        page_num = parts[1]
        side = parts[2]
        if side == "left":
            if page_num in left_pages:
                print(
                    f"⚠️  Duplicate left crop for page_{page_num}: "
                    f"{left_pages[page_num].name} → using {f.name}"
                )
            left_pages[page_num] = f

    if not left_pages:
        print(f"❌ No left crop images found under {CROPS_DIR}")
        print("Expected files like: page_006_left.jpg")
        sys.exit(1)

    print(f"\nProcessing OCR for {len(left_pages)} pages...")

    for page_num in sorted(left_pages.keys(), key=int):
        page_dir = OUT_DIR / f"page_{page_num}"
        page_dir.mkdir(exist_ok=True)

        left_path = left_pages[page_num]
        print(f"  OCR page {page_num}...", end=" ", flush=True)
        with Image.open(left_path) as left_img:
            text = ocr_image(left_img)
        if opencc is not None:
            text = opencc.convert(text)
        (page_dir / "ocr_text.txt").write_text(text, encoding="utf-8")
        print("done")

    print(f"\n✅ Phase 2 complete. Output in {OUT_DIR}/")
    cmds = (
        "Commands to download:\n"
        f"  cd {AUTODL_OUTPUT_DIR} && zip -r ../wuxia_output.zip .\n"
        "  # Then on local machine:\n"
        "  scp -P <port> root@<ip>:/root/wuxia_output.zip ./\n"
        "  unzip wuxia_output.zip -d tmp_results/\n"
        "  python main_local.py --book <same-pdf-as-phase1> --phase3"
    )
    print(cmds)


if __name__ == "__main__":
    phase2_gpu()
