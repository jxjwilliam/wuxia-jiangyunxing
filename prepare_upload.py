"""
Package split crops into zip files for upload to AutoDL.
Creates crops_left.zip for OCR upload (ocr_only_v1).
"""
from pathlib import Path
import zipfile

from configs.config import PHASE2_MANIFEST_NAME


def prepare_upload(crops_dir: Path, zip_path: Path):
    if zip_path.exists():
        print(f"  → Replacing existing {zip_path.name}")

    manifest = crops_dir / PHASE2_MANIFEST_NAME

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zl:
        for img_path in sorted(crops_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if "_left" in img_path.name:
                zl.write(img_path, img_path.name)
        if manifest.is_file():
            zl.write(manifest, manifest.name)

    left_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  → {zip_path}: {left_mb:.1f} MB")
    print("Upload this file to AutoDL via scp or web UI.")
