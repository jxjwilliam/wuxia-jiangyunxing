"""Unit tests for pipeline helpers (stdlib unittest, no pytest required)."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import detect_title
import main_local
import split_page


class TestExtractPageNum(unittest.TestCase):
    def test_parsed(self):
        self.assertEqual(main_local._extract_page_num("page_006_left"), 6)
        self.assertEqual(main_local._extract_page_num("page_006"), 6)

    def test_invalid(self):
        self.assertIsNone(main_local._extract_page_num("page_foo_left"))
        self.assertIsNone(main_local._extract_page_num("nope"))


class TestResolveUniqueFolderName(unittest.TestCase):
    def test_unique_adds_to_used(self):
        used: set[str] = set()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            name = main_local._resolve_unique_folder_name("第九回_foo", 9, used, out)
        self.assertEqual(name, "第九回_foo")
        self.assertIn("第九回_foo", used)

    def test_in_used_gets_page_suffix(self):
        used: set[str] = {"第九回_foo"}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            name = main_local._resolve_unique_folder_name("第九回_foo", 9, used, out)
        self.assertEqual(name, "第九回_foo_p009")
        self.assertIn(name, used)

    def test_existing_dir_gets_suffix(self):
        used: set[str] = set()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            base = out / "第九回_foo"
            base.mkdir()
            name = main_local._resolve_unique_folder_name("第九回_foo", 9, used, out)
        self.assertEqual(name, "第九回_foo_p009")


class TestDetectTitle(unittest.TestCase):
    def test_match(self):
        text = "第一行\n第九回 鐵槍破犁\n正文"
        self.assertEqual(detect_title.detect_title(text, 1), "第九回_鐵槍破犁")

    def test_fallback(self):
        text = "沒有章節標題"
        self.assertEqual(detect_title.detect_title(text, 3), "page_003")


class TestSplitPage(unittest.TestCase):
    def test_split_dimensions(self):
        img = Image.new("RGB", (100, 40), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        left, right = split_page.split_page(buf)
        self.assertEqual(left.size, (50, 40))
        self.assertEqual(right.size, (50, 40))


if __name__ == "__main__":
    unittest.main()
