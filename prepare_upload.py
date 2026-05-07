"""
Package split crops into zip files for upload to AutoDL.
Creates crops_left.zip and crops_right.zip for separate upload.
"""
from pathlib import Path
import zipfile
from config import AUTODL_CROPS_DIR


def prepare_upload(crops_dir: Path):
    left_zip = Path("crops_left.zip")
    right_zip = Path("crops_right.zip")

    with (
        zipfile.ZipFile(left_zip, 'w', zipfile.ZIP_DEFLATED) as zl,
        zipfile.ZipFile(right_zip, 'w', zipfile.ZIP_DEFLATED) as zr,
    ):
        for img_path in sorted(crops_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            if '_left' in img_path.name:
                zl.write(img_path, img_path.name)
            elif '_right' in img_path.name:
                zr.write(img_path, img_path.name)

    left_mb = left_zip.stat().st_size / 1024 / 1024
    right_mb = right_zip.stat().st_size / 1024 / 1024
    print(f"  → {left_zip}: {left_mb:.1f} MB")
    print(f"  → {right_zip}: {right_mb:.1f} MB")
    print("Upload these to AutoDL via scp or web UI.")
