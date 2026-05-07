"""
Phase 2: AutoDL GPU Orchestrator.
Run this on the AutoDL GPU instance after uploading split crops.
Processes: OCR on left crops only (ocr_only_v1).
"""
import sys
from pathlib import Path
from PIL import Image
from config import AUTODL_REMOTE_DIR, AUTODL_OUTPUT_DIR

CROPS_DIR = Path(AUTODL_REMOTE_DIR)
OUT_DIR = Path(AUTODL_OUTPUT_DIR)


def phase2_gpu():
    from ocr_text import ocr_image

    print("=" * 50)
    print("PHASE 2: AutoDL GPU Processing")
    print("=" * 50)

    if not CROPS_DIR.exists():
        print(f"❌ Input crops directory not found: {CROPS_DIR}")
        print("Unzip crops_left.zip into this directory before running.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect left crop files by page number
    left_pages = {}
    for f in sorted(CROPS_DIR.iterdir()):
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        # Expected format: page_006_left.jpg
        parts = f.stem.split('_')
        if len(parts) < 3:
            continue
        page_num = parts[1]
        side = parts[2]
        if side == "left":
            left_pages[page_num] = f

    if not left_pages:
        print(f"❌ No left crop images found under {CROPS_DIR}")
        print("Expected files like: page_006_left.jpg")
        sys.exit(1)

    print(f"\nProcessing OCR for {len(left_pages)} pages...")

    for page_num in sorted(left_pages.keys()):
        page_dir = OUT_DIR / f"page_{page_num}"
        page_dir.mkdir(exist_ok=True)

        left_path = left_pages[page_num]
        print(f"  OCR page {page_num}...", end=" ", flush=True)
        left_img = Image.open(left_path)
        text = ocr_image(left_img)
        (page_dir / "ocr_text.txt").write_text(text, encoding="utf-8")
        print("done")

    print(f"\n✅ Phase 2 complete. Output in {OUT_DIR}/")
    cmds = (
        "Commands to download:\n"
        f"  cd {AUTODL_OUTPUT_DIR} && zip -r ../wuxia_output.zip .\n"
        "  # Then on local machine:\n"
        "  scp -P <port> root@<ip>:/root/wuxia_output.zip ./\n"
        "  unzip wuxia_output.zip -d tmp_results/\n"
        "  python main_local.py --phase3"
    )
    print(cmds)


if __name__ == "__main__":
    phase2_gpu()
