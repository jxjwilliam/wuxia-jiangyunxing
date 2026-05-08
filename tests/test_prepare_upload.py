"""Tests for AutoDL crops packaging."""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from configs.config import PHASE2_MANIFEST_NAME, write_phase2_manifest
from prepare_upload import prepare_upload


class TestPrepareUpload(unittest.TestCase):
    def test_zip_includes_left_images_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            crops = root / "crops"
            crops.mkdir(parents=True, exist_ok=True)
            (crops / "page_006_left.jpg").write_bytes(b"fakejpg")
            (crops / "page_006_right.jpg").write_bytes(b"x")
            write_phase2_manifest(crops, ocr_rotate_left_cw90=True)

            zp = root / "crops_left.zip"
            prepare_upload(crops, zp)

            with zipfile.ZipFile(zp, "r") as zf:
                names = set(zf.namelist())

            self.assertIn("page_006_left.jpg", names)
            self.assertNotIn("page_006_right.jpg", names)
            self.assertIn(PHASE2_MANIFEST_NAME, names)
            with zipfile.ZipFile(zp, "r") as zf:
                data = json.loads(zf.read(PHASE2_MANIFEST_NAME).decode("utf-8"))
            self.assertTrue(data.get("ocr_rotate_left_cw90"))

    def test_no_manifest_when_rotate_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            crops = root / "crops"
            crops.mkdir(parents=True, exist_ok=True)
            (crops / "page_007_left.jpg").write_bytes(b"fakejpg")
            write_phase2_manifest(crops, ocr_rotate_left_cw90=False)

            zp = root / "crops_left.zip"
            prepare_upload(crops, zp)

            with zipfile.ZipFile(zp, "r") as zf:
                names = set(zf.namelist())

            self.assertNotIn(PHASE2_MANIFEST_NAME, names)


if __name__ == "__main__":
    unittest.main()
