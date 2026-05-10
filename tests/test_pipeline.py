"""Unit tests for pipeline helpers (stdlib unittest, no pytest required)."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import detect_title
import main_local
import split_page
from ocr_text import _choose_ocr_result


class TestExtractPageNum(unittest.TestCase):
    def test_parsed(self):
        self.assertEqual(main_local._extract_page_num("page_006_left"), 6)
        self.assertEqual(main_local._extract_page_num("page_006"), 6)

    def test_invalid(self):
        self.assertIsNone(main_local._extract_page_num("page_foo_left"))
        self.assertIsNone(main_local._extract_page_num("nope"))


class TestExistingExtractedPages(unittest.TestCase):
    def test_reads_existing_page_jpgs(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "page_007.jpg").write_bytes(b"x")
            (d / "page_008.jpg").write_bytes(b"x")
            (d / "not-a-page.jpg").write_bytes(b"x")
            pages = main_local._existing_extracted_pages(d)
        self.assertEqual([n for n, _ in pages], [7, 8])

    def test_returns_empty_when_no_pages_exist(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            pages = main_local._existing_extracted_pages(d)
        self.assertEqual(pages, [])


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

    def test_disables_pillow_pixel_guard_for_trusted_pdf_pages(self):
        with patch.object(split_page.Image, "MAX_IMAGE_PIXELS", 12345):
            img = Image.new("RGB", (100, 40), color=(255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            split_page.split_page(buf)
            self.assertIsNone(split_page.Image.MAX_IMAGE_PIXELS)

    def test_apply_crop_margins_ignores_bottom_key(self):
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        out = split_page.apply_crop_margins(
            img, {"top": 0.1, "left": 0.1, "right": 0.1, "bottom": 0.5}
        )
        # bottom is intentionally ignored now; only top/left/right apply.
        self.assertEqual(out.size, (80, 45))


class TestChooseOcrResult(unittest.TestCase):
    """Guard against the page_010 regression where high-confidence noisy enhanced output
    was selected over a cleaner, lower-confidence base output (see docs/opencode-review.md)."""

    BASE_CLEAN_TEXT = "他們約定了一個日子各地一齊向官兵進攻"  # 19 CJK, 0 garbage
    ENHANCED_NOISY_TEXT = "張他們約定了一個日子各地一齊向官兵進祀口口政"  # 22 CJK, 4 garbage (張祀口口 ≠ CJK count? — see test docs)

    def test_enhanced_wins_on_big_conf_jump_with_no_garbage_growth(self):
        # +0.15 conf, equal garbage → enhanced selected (legitimate quality win).
        out = _choose_ocr_result(
            base_text="他們約定了一個",
            base_conf=0.60,
            base_cjk=7,
            enhanced_text="他們約定了一個日子",
            enhanced_conf=0.75,
            enhanced_cjk=9,
        )
        self.assertEqual(out, "他們約定了一個日子")

    def test_enhanced_wins_on_big_cjk_growth_with_no_garbage_growth(self):
        # +0.05 conf (below threshold) but +9 CJK and equal garbage → enhanced selected.
        out = _choose_ocr_result(
            base_text="他們約定",
            base_conf=0.60,
            base_cjk=4,
            enhanced_text="他們約定了一個日子各地一",
            enhanced_conf=0.65,
            enhanced_cjk=13,
        )
        self.assertEqual(out, "他們約定了一個日子各地一")

    def test_enhanced_loses_when_conf_jump_brings_extra_garbage(self):
        # The page_010 regression case: enhanced has +0.15 conf but injects 5 non-CJK
        # noise chars (張, 祀口口, 火). Base must win.
        base = "他們約定了一個日子"  # 9 CJK, 0 garbage
        enhanced = "張他們約定了一個祀口口火日子"  # 10 CJK, 5 garbage (張祀口口火 are CJK actually!)
        # Use ASCII garbage to keep the test unambiguous about CJK vs non-CJK.
        base_ascii = "abc他們約定了一個日子xyz"  # base_cjk=9, base_len=15, base_garbage=6
        enhanced_ascii = (
            "abcXYZ123他們約定了一個日子xyz!@#"  # enhanced_cjk=9, enhanced_len=24, enhanced_garbage=15
        )
        out = _choose_ocr_result(
            base_text=base_ascii,
            base_conf=0.60,
            base_cjk=9,
            enhanced_text=enhanced_ascii,
            enhanced_conf=0.78,  # +0.18, comfortably above +0.10 threshold
            enhanced_cjk=9,
        )
        # garbage delta = 15 - 6 = 9 > 2 → enhanced rejected, base returned.
        self.assertEqual(out, base_ascii)
        # Reference unused literal so linters don't flag it; documents the original case shape.
        self.assertNotEqual(base, enhanced)

    def test_enhanced_loses_when_cjk_growth_brings_extra_garbage(self):
        # +12 CJK growth but garbage also grows by 5 → reject (above the +2 garbage budget).
        base = "他們約定"  # base_cjk=4, base_len=4, garbage=0
        enhanced = "他們約定了一個日子各地一齊向官!@#$%"  # enhanced_cjk=16, enhanced_len=21, garbage=5
        out = _choose_ocr_result(
            base_text=base,
            base_conf=0.60,
            base_cjk=4,
            enhanced_text=enhanced,
            enhanced_conf=0.62,  # below conf threshold; only the cjk path could trigger
            enhanced_cjk=16,
        )
        # cjk delta = +12 (>=8) but garbage delta = +5 (> +2) → reject.
        self.assertEqual(out, base)

    def test_enhanced_below_both_thresholds_loses_even_when_clean(self):
        # +0.05 conf, +3 CJK, no garbage growth — neither threshold met → base wins.
        out = _choose_ocr_result(
            base_text="他們約定了一個",
            base_conf=0.60,
            base_cjk=7,
            enhanced_text="他們約定了一個日子各",
            enhanced_conf=0.65,
            enhanced_cjk=10,
        )
        self.assertEqual(out, "他們約定了一個")

    def test_garbage_budget_exactly_two_still_allows_enhanced(self):
        # Boundary check: enhanced_garbage - base_garbage == 2 → still allowed.
        base = "他們約定了一個日子"  # base_cjk=9, len=9, garbage=0
        enhanced = "他們約定了一個日子AB"  # enhanced_cjk=9, len=11, garbage=2
        out = _choose_ocr_result(
            base_text=base,
            base_conf=0.60,
            base_cjk=9,
            enhanced_text=enhanced,
            enhanced_conf=0.75,  # +0.15 conf
            enhanced_cjk=9,
        )
        self.assertEqual(out, enhanced)


if __name__ == "__main__":
    unittest.main()
