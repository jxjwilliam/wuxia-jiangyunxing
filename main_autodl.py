"""
Phase 2: AutoDL GPU Orchestrator.
Run this on the AutoDL GPU instance after uploading split crops.
Processes: OCR (left crops) + optional upscale (right crops).
"""
from pathlib import Path
from PIL import Image
from config import AUTODL_REMOTE_DIR, AUTODL_OUTPUT_DIR, UPSCALE_ENABLED

CROPS_DIR = Path(AUTODL_REMOTE_DIR)
OUT_DIR = Path(AUTODL_OUTPUT_DIR)


def phase2_gpu():
    from ocr_text import ocr_image

    if UPSCALE_ENABLED:
        from upscale import upscale_image

    print("=" * 50)
    print("PHASE 2: AutoDL GPU Processing")
    print("=" * 50)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group crop files by page number
    pages = {}
    for f in sorted(CROPS_DIR.iterdir()):
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        # Expected format: page_006_left.jpg or page_006_right.jpg
        parts = f.stem.split('_')
        if len(parts) < 3:
            continue
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
            print(f"  OCR page {page_num}...", end=" ", flush=True)
            left_img = Image.open(left_path)
            text = ocr_image(left_img)
            (page_dir / "ocr_text.txt").write_text(text, encoding="utf-8")
            print("done")

        if right_path:
            right_img = Image.open(right_path)
            if UPSCALE_ENABLED:
                print(f"  Upscaling page {page_num}...", end=" ", flush=True)
                right_img = upscale_image(right_img)
                print("done")
            right_img.save(page_dir / "illustration.png", "PNG")

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
