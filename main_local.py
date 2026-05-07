#!/usr/bin/env python3
"""
Wuxia PDF Extraction — Local Orchestrator (Phase 1 + Phase 3).

Usage:
  python main_local.py              # Phase 1: extract + split + package
  python main_local.py --phase3     # Phase 3: translate + detect + organise
  python main_local.py --all        # Phase 1 + Phase 3 (if results already downloaded)
"""
import sys
from pathlib import Path
from config import AUTODL_CROPS_DIR, AUTODL_RESULTS_DIR

TMP_DIR = Path("tmp_pages")
CROPS_DIR = Path(AUTODL_CROPS_DIR)
RESULTS_DIR = Path(AUTODL_RESULTS_DIR)


def phase1_preprocess():
    """Extract PDF pages and split into left/right crops."""
    from extract_pages import extract_pages
    from split_page import split_page

    print("=" * 50)
    print("PHASE 1: Local Preprocessing")
    print("=" * 50)

    print("\nStep 1: Extracting page images from PDF...")
    pages = extract_pages(TMP_DIR)
    print(f"  → {len(pages)} pages extracted to {TMP_DIR}/")

    print("\nStep 2: Splitting pages into left/right crops...")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for page_num, img_path in pages:
        left_img, right_img = split_page(img_path)
        left_path = CROPS_DIR / f"page_{page_num:03d}_left.jpg"
        right_path = CROPS_DIR / f"page_{page_num:03d}_right.jpg"
        left_img.save(left_path, "JPEG", quality=95)
        right_img.save(right_path, "JPEG", quality=95)
    print(f"  → {len(pages)} pairs saved to {CROPS_DIR}/")

    print("\nStep 3: Packaging crops for AutoDL upload...")
    from prepare_upload import prepare_upload
    prepare_upload(CROPS_DIR)

    print("\n" + "=" * 50)
    print("✅ Phase 1 complete.")
    print()
    print("Next steps:")
    print("  1. Upload crops_left.zip and crops_right.zip to AutoDL")
    print("  2. On AutoDL, unzip to /root/wuxia_crops/")
    print("  3. Run: python main_autodl.py")
    print("  4. Download results to tmp_results/")
    print("  5. Run: python main_local.py --phase3")
    print("=" * 50)


def phase3_assemble():
    """After AutoDL results are downloaded, translate and organize."""
    from translate import translate
    from detect_title import detect_title
    from organise import save_pair
    from PIL import Image

    print("=" * 50)
    print("PHASE 3: Local Assembly")
    print("=" * 50)

    if not RESULTS_DIR.exists():
        print(f"❌ {RESULTS_DIR}/ not found. Download AutoDL results first.")
        sys.exit(1)

    page_dirs = sorted(RESULTS_DIR.iterdir())
    total = 0

    print(f"\nStep 4: Translating pages...")
    for page_dir in page_dirs:
        if not page_dir.is_dir():
            continue
        text_file = page_dir / "ocr_text.txt"
        if not text_file.exists():
            continue

        raw_text = text_file.read_text(encoding="utf-8")

        # Step 4: Translate
        simplified = translate(raw_text)

        # Step 5: Detect chapter title
        # Extract page number from directory name (page_006 → 6)
        try:
            page_num = int(page_dir.name.split("_")[1])
        except (IndexError, ValueError):
            page_num = 0
        folder_name = detect_title(simplified, page_num)

        # Step 7: Save files
        img_path = page_dir / "illustration.png"
        img = Image.open(img_path) if img_path.exists() else None
        save_pair(folder_name, simplified, img)

        total += 1
        print(f"  → {page_dir.name} → {folder_name}/")

    from pathlib import Path as PP
    out_count = len(list(PP("wuxia").iterdir())) if PP("wuxia").exists() else 0
    print(f"\n✅ Phase 3 complete. {total} pages processed.")
    print(f"   Output in wuxia/ ({out_count} folders)")


def run_all():
    """Run Phase 1 + Phase 3 (if results already exist)."""
    phase1_preprocess()
    if RESULTS_DIR.exists():
        print("\n" + "-" * 50)
        phase3_assemble()
    else:
        print("\nPhase 3 skipped — no results found in tmp_results/.")
        print("Run Phase 2 on AutoDL first, then `python main_local.py --phase3`")


if __name__ == "__main__":
    if "--all" in sys.argv:
        run_all()
    elif "--phase3" in sys.argv:
        phase3_assemble()
    else:
        phase1_preprocess()
