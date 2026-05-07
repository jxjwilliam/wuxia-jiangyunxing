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
from config import AUTODL_CROPS_DIR, AUTODL_RESULTS_DIR, OUTPUT_DIR

TMP_DIR = Path("tmp_pages")
CROPS_DIR = Path(AUTODL_CROPS_DIR)
RESULTS_DIR = Path(AUTODL_RESULTS_DIR)
OUT_DIR = Path(OUTPUT_DIR)


def _extract_page_num(name: str) -> int | None:
    parts = name.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _resolve_unique_folder_name(base_name: str, page_num: int, used_names: set[str]) -> str:
    """Disambiguate duplicate chapter names to avoid overwriting output."""
    candidate = base_name
    if candidate in used_names or (OUT_DIR / candidate).exists():
        candidate = f"{base_name}_p{page_num:03d}"
        suffix = 2
        while candidate in used_names or (OUT_DIR / candidate).exists():
            candidate = f"{base_name}_p{page_num:03d}_{suffix}"
            suffix += 1
    used_names.add(candidate)
    return candidate


def _find_local_right_crop(page_num: int) -> Path | None:
    for ext in ("jpg", "jpeg", "png"):
        candidate = CROPS_DIR / f"page_{page_num:03d}_right.{ext}"
        if candidate.exists():
            return candidate
    return None


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
    print("  1. Upload crops_left.zip to AutoDL")
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

    if not CROPS_DIR.exists():
        print(f"❌ {CROPS_DIR}/ not found. Run phase 1 first for local right-side crops.")
        sys.exit(1)

    expected_pages = set()
    for left_crop in sorted(CROPS_DIR.glob("page_*_left.*")):
        page_num = _extract_page_num(left_crop.stem)
        if page_num is not None:
            expected_pages.add(page_num)

    if not expected_pages:
        print("❌ No local left crops found in tmp_crops/. Run phase 1 first.")
        sys.exit(1)

    result_pages = {}
    for page_dir in sorted(RESULTS_DIR.iterdir()):
        if not page_dir.is_dir():
            continue
        page_num = _extract_page_num(page_dir.name)
        if page_num is not None:
            result_pages[page_num] = page_dir

    missing_dirs = sorted(expected_pages - set(result_pages))
    if missing_dirs:
        preview = ", ".join(f"page_{n:03d}" for n in missing_dirs[:10])
        more = f" (+{len(missing_dirs) - 10} more)" if len(missing_dirs) > 10 else ""
        print(f"❌ Missing AutoDL result directories for: {preview}{more}")
        sys.exit(1)

    missing_ocr = []
    missing_right = []
    for page_num in sorted(expected_pages):
        page_dir = result_pages[page_num]
        if not (page_dir / "ocr_text.txt").exists():
            missing_ocr.append(page_num)
        if _find_local_right_crop(page_num) is None:
            missing_right.append(page_num)

    if missing_ocr:
        preview = ", ".join(f"page_{n:03d}" for n in missing_ocr[:10])
        more = f" (+{len(missing_ocr) - 10} more)" if len(missing_ocr) > 10 else ""
        print(f"❌ Missing OCR text files for: {preview}{more}")
        sys.exit(1)

    if missing_right:
        preview = ", ".join(f"page_{n:03d}" for n in missing_right[:10])
        more = f" (+{len(missing_right) - 10} more)" if len(missing_right) > 10 else ""
        print(f"❌ Missing local right crops for: {preview}{more}")
        sys.exit(1)

    total = 0
    used_names: set[str] = set()

    print(f"\nStep 4: Translating pages...")
    for page_num in sorted(expected_pages):
        page_dir = result_pages[page_num]
        text_file = page_dir / "ocr_text.txt"

        raw_text = text_file.read_text(encoding="utf-8")

        # Step 4: Translate
        simplified = translate(raw_text)

        # Step 5: Detect chapter title
        folder_name = detect_title(simplified, page_num)
        unique_folder_name = _resolve_unique_folder_name(folder_name, page_num, used_names)

        # Step 7: Save files
        img_path = _find_local_right_crop(page_num)
        if img_path is None:
            print(f"❌ Missing local right crop for page_{page_num:03d}")
            sys.exit(1)
        img = Image.open(img_path)
        save_pair(unique_folder_name, simplified, img)

        total += 1
        print(f"  → page_{page_num:03d} → {unique_folder_name}/")

    out_count = len(list(OUT_DIR.iterdir())) if OUT_DIR.exists() else 0
    print(f"\n✅ Phase 3 complete. {total} pages processed.")
    print(f"   Output in {OUT_DIR}/ ({out_count} folders)")


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
