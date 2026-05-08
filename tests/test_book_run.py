"""Tests for book_run resolution and ResolvedRun layout."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import book_run


class TestSanitizeSlug(unittest.TestCase):
    def test_preserves_unicode(self):
        self.assertEqual(book_run.sanitize_slug("江湖奇俠傳"), "江湖奇俠傳")

    def test_replaces_separators(self):
        self.assertEqual(book_run.sanitize_slug("a/b\\c"), "a_b_c")


class TestResolveBookArgument(unittest.TestCase):
    def test_basename_under_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            pdf = data / "demo.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            p = book_run.resolve_book_argument("demo.pdf", cwd=root)
            self.assertTrue(p.is_file())
            self.assertEqual(p.resolve(), pdf.resolve())

    def test_rejects_non_pdf(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            f = root / "data" / "x.txt"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("n")
            with self.assertRaises(ValueError):
                book_run.resolve_book_argument("data/x.txt", cwd=root)


class TestLoadBookOverrides(unittest.TestCase):
    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "data" / "only.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf.write_bytes(b"x")
            self.assertEqual(book_run.load_book_overrides(pdf), {})

    def test_partial_merge(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "data"
            d.mkdir()
            pdf = d / "m.pdf"
            pdf.write_bytes(b"x")
            side = d / "m.book.json"
            side.write_text(json.dumps({"start_page": 10, "split_ratio": 0.48}), encoding="utf-8")
            o = book_run.load_book_overrides(pdf)
            self.assertEqual(o["start_page"], 10)
            self.assertEqual(o["split_ratio"], 0.48)
            self.assertNotIn("end_page", o)


class TestBuildResolvedRunLayout(unittest.TestCase):
    def test_paths_relative_to_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            data = root / "data"
            data.mkdir()
            pdf = data / "jiang.pdf"
            pdf.write_bytes(b"%PDF")
            run = book_run.build_resolved_run(pdf, cwd=root)
            self.assertEqual(run.slug, "jiang")
            self.assertEqual(run.work_root, root / "work" / "jiang")
            self.assertEqual(run.tmp_pages, root / "work" / "jiang" / "tmp_pages")
            self.assertEqual(run.tmp_crops, root / "work" / "jiang" / "tmp_crops")
            self.assertEqual(run.tmp_results, root / "work" / "jiang" / "tmp_results")
            self.assertEqual(run.output_dir, root / "work" / "jiang" / "output")
            self.assertEqual(run.crops_zip, root / "work" / "jiang" / "crops_left.zip")


if __name__ == "__main__":
    unittest.main()
