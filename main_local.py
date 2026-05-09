#!/usr/bin/env python3
"""
Wuxia PDF Extraction — Local Orchestrator (Phase 1 + Phase 3).

Usage:
  python main_local.py --book data/jiang-yun-xing.pdf
  python main_local.py --book jiang-yun-xing.pdf --phase3
  python main_local.py --book jiang-yun-xing.pdf --all
"""
import argparse
import sys
from pathlib import Path

import book_run
from configs.config import write_phase2_manifest


def _extract_page_num(name: str) -> int | None:
    parts = name.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _resolve_unique_folder_name(
    base_name: str, page_num: int, used_names: set[str], out_dir: Path
) -> str:
    """Disambiguate duplicate chapter names to avoid overwriting output."""
    candidate = base_name
    if candidate in used_names or (out_dir / candidate).exists():
        candidate = f"{base_name}_p{page_num:03d}"
        suffix = 2
        while candidate in used_names or (out_dir / candidate).exists():
            candidate = f"{base_name}_p{page_num:03d}_{suffix}"
            suffix += 1
    used_names.add(candidate)
    return candidate


def _find_local_right_crop(page_num: int, crops_dir: Path) -> Path | None:
    for ext in ("jpg", "jpeg", "png"):
        candidate = crops_dir / f"page_{page_num:03d}_right.{ext}"
        if candidate.exists():
            return candidate
    return None


def _existing_extracted_pages(tmp_pages_dir: Path) -> list[tuple[int, Path]]:
    pages: list[tuple[int, Path]] = []
    for img_path in sorted(tmp_pages_dir.glob("page_*.jpg")):
        page_num = _extract_page_num(img_path.stem)
        if page_num is not None:
            pages.append((page_num, img_path))
    return pages


def _phase1_pages(run: book_run.ResolvedRun, *, start_step: int):
    from extract_pages import extract_pages

    if start_step == 2:
        pages = _existing_extracted_pages(run.tmp_pages)
        if not pages:
            print(f"❌ start-step=2 but no extracted pages found in {run.tmp_pages}/.")
            print("Run with --start-step 1 first to generate tmp_pages.")
            sys.exit(1)
        print(f"\nStep 1 skipped (--start-step 2). Reusing {len(pages)} pages from {run.tmp_pages}/")
        return pages

    print("\nStep 1: Extracting page images from PDF...")
    pages = extract_pages(
        run.pdf_path,
        run.tmp_pages,
        start_page=run.start_page,
        end_page=run.end_page,
        pdf_extract_mode=run.pdf_extract_mode,
    )
    print(f"  → {len(pages)} pages extracted to {run.tmp_pages}/")
    return pages


def phase1_preprocess(run: book_run.ResolvedRun, *, book_arg: str, start_step: int = 1):
    """Extract PDF pages and split into left/right crops."""
    from split_page import split_page, apply_crop_margins

    print("=" * 50)
    print("PHASE 1: Local Preprocessing")
    print("=" * 50)
    print(f"Book: {run.pdf_path}  →  work/{run.slug}/")

    pages = _phase1_pages(run, start_step=start_step)

    print("\nStep 2: Splitting pages into left/right crops...")
    run.tmp_crops.mkdir(parents=True, exist_ok=True)
    for page_num, img_path in pages:
        left_img, right_img = split_page(img_path, split_ratio=run.split_ratio)
        left_img = apply_crop_margins(left_img, run.ocr_left_crop)
        left_path = run.tmp_crops / f"page_{page_num:03d}_left.jpg"
        right_path = run.tmp_crops / f"page_{page_num:03d}_right.jpg"
        left_img.save(left_path, "JPEG", quality=95)
        right_img.save(right_path, "JPEG", quality=95)
    print(f"  → {len(pages)} pairs saved to {run.tmp_crops}/")

    write_phase2_manifest(run.tmp_crops, ocr_rotate_left_cw90=run.ocr_rotate_left_cw90)
    if run.ocr_rotate_left_cw90:
        print("  → phase2_manifest.json: ocr_rotate_left_cw90 (for AutoDL vertical text)")

    print("\nStep 3: Packaging crops for AutoDL upload...")
    from prepare_upload import prepare_upload

    prepare_upload(run.tmp_crops, run.crops_zip)

    print("\n" + "=" * 50)
    print("✅ Phase 1 complete.")
    print()
    print("Next steps:")
    print(f"  1. Upload {run.crops_zip} to AutoDL")
    print("  2. On AutoDL, empty /root/wuxia_crops/ and /root/wuxia_output/ if reusing the instance, then unzip to /root/wuxia_crops/")
    print("  3. Run: python main_autodl.py")
    print(f"  4. Download results to {run.tmp_results}/")
    print(f"  5. Run: python main_local.py --book {book_arg} --phase3")
    print("=" * 50)


def phase3_assemble(run: book_run.ResolvedRun):
    """After AutoDL results are downloaded, translate and organize."""
    from translate import translate
    from detect_title import detect_title
    from organise import save_pair
    from PIL import Image

    print("=" * 50)
    print("PHASE 3: Local Assembly")
    print("=" * 50)

    if not run.tmp_results.exists():
        print("❌ Phase 3 requires Phase 2 output (AutoDL OCR). Missing directory:")
        print(f"   {run.tmp_results.resolve()}")
        print("   Create it by downloading/unzipping wuxia_output into tmp_results/")
        print("   (see download_results.py). Then re-run --phase3.")
        sys.exit(1)

    if not run.tmp_crops.exists():
        print(f"❌ {run.tmp_crops}/ not found. Run phase 1 first for local right-side crops.")
        sys.exit(1)

    expected_pages = set()
    for left_crop in sorted(run.tmp_crops.glob("page_*_left.*")):
        page_num = _extract_page_num(left_crop.stem)
        if page_num is not None:
            expected_pages.add(page_num)

    if not expected_pages:
        print(f"❌ No local left crops found in {run.tmp_crops}/. Run phase 1 first.")
        sys.exit(1)

    result_pages = {}
    for page_dir in sorted(run.tmp_results.iterdir()):
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
        if _find_local_right_crop(page_num, run.tmp_crops) is None:
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
    run.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 4: Translating pages...")
    for page_num in sorted(expected_pages):
        page_dir = result_pages[page_num]
        text_file = page_dir / "ocr_text.txt"

        raw_text = text_file.read_text(encoding="utf-8")

        simplified = translate(raw_text)

        folder_name = detect_title(simplified, page_num)
        unique_folder_name = _resolve_unique_folder_name(
            folder_name, page_num, used_names, run.output_dir
        )

        img_path = _find_local_right_crop(page_num, run.tmp_crops)
        if img_path is None:
            print(f"❌ Missing local right crop for page_{page_num:03d}")
            sys.exit(1)
        with Image.open(img_path) as img:
            save_pair(unique_folder_name, simplified, img, run.output_dir)

        total += 1
        print(f"  → page_{page_num:03d} → {unique_folder_name}/")

    out_count = len(list(run.output_dir.iterdir())) if run.output_dir.exists() else 0
    print(f"\n✅ Phase 3 complete. {total} pages processed.")
    print(f"   Output in {run.output_dir}/ ({out_count} folders)")


def run_all(run: book_run.ResolvedRun, *, book_arg: str, start_step: int = 1):
    """Run Phase 1 + Phase 3 (if results already exist)."""
    phase1_preprocess(run, book_arg=book_arg, start_step=start_step)
    if run.tmp_results.exists() and any(run.tmp_results.iterdir()):
        print("\n" + "-" * 50)
        phase3_assemble(run)
    else:
        print(f"\nPhase 3 skipped — no results found in {run.tmp_results}/.")
        print(f"Run Phase 2 on AutoDL first, then `python main_local.py --book {book_arg} --phase3`")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wuxia PDF hybrid extraction (local orchestrator).")
    parser.add_argument(
        "--book",
        required=True,
        help="PDF under data/ (e.g. jiang-yun-xing.pdf) or path to .pdf",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Run Phase 3 only (translate + assemble from this book's tmp_results/).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run Phase 1, then Phase 3 if tmp_results/ for this book exists.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        choices=(1, 2),
        default=1,
        help="Phase 1 start step: 1=extract then split, 2=skip extract and split from existing tmp_pages.",
    )
    args = parser.parse_args()
    try:
        pdf_path = book_run.resolve_book_argument(args.book)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(2)
    run = book_run.build_resolved_run(pdf_path)

    if args.all:
        run_all(run, book_arg=args.book, start_step=args.start_step)
    elif args.phase3:
        phase3_assemble(run)
    else:
        phase1_preprocess(run, book_arg=args.book, start_step=args.start_step)


if __name__ == "__main__":
    main()
