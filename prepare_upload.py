"""
Package split crops into zip files for upload to AutoDL.
Creates crops_left.zip for OCR upload (ocr_only_v1).
"""
from pathlib import Path
import zipfile
from config import AUTODL_CROPS_DIR


def prepare_upload(crops_dir: Path):
    left_zip = Path("crops_left.zip")

    with zipfile.ZipFile(left_zip, 'w', zipfile.ZIP_DEFLATED) as zl:
        for img_path in sorted(crops_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            if '_left' in img_path.name:
                zl.write(img_path, img_path.name)

    left_mb = left_zip.stat().st_size / 1024 / 1024
    print(f"  → {left_zip}: {left_mb:.1f} MB")
    print("Upload this file to AutoDL via scp or web UI.")
