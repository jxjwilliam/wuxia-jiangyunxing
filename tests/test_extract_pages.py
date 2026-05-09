"""Unit tests for PDF Step 1 embedded-vs-render policy."""
from __future__ import annotations

import unittest

from extract_pages import embedded_raster_effective_dpi, pdf_page_should_use_embedded_native


class TestEmbeddedRasterEffectiveDpi(unittest.TestCase):
    def test_square_page_matches_pt(self):
        # 657 px on 657 pt page → 72 DPI effective
        self.assertAlmostEqual(embedded_raster_effective_dpi(657, 657.0), 72.0, places=3)

    def test_low_res_in_wide_page(self):
        # tests/c1.md: 1440 px on ~4082 pt ≈ 25.4 DPI
        self.assertAlmostEqual(embedded_raster_effective_dpi(1440, 4082.0), 25.3993, places=3)

    def test_zero_width_safe(self):
        self.assertEqual(embedded_raster_effective_dpi(100, 0.0), 0.0)


class TestPdfPageShouldUseEmbeddedNative(unittest.TestCase):
    def test_auto_below_threshold_yes(self):
        self.assertTrue(
            pdf_page_should_use_embedded_native(
                "auto",
                has_embedded_image=True,
                effective_dpi=25.0,
                threshold=80.0,
            )
        )

    def test_auto_72dpi_uses_embedded(self):
        # 72 < 80 → native extract matches jiang-yun-style full-page bitmaps
        self.assertTrue(
            pdf_page_should_use_embedded_native(
                "auto",
                has_embedded_image=True,
                effective_dpi=72.0,
                threshold=80.0,
            )
        )

    def test_auto_high_dpi_render(self):
        self.assertFalse(
            pdf_page_should_use_embedded_native(
                "auto",
                has_embedded_image=True,
                effective_dpi=150.0,
                threshold=80.0,
            )
        )

    def test_auto_no_images(self):
        self.assertFalse(
            pdf_page_should_use_embedded_native(
                "auto",
                has_embedded_image=False,
                effective_dpi=0.0,
                threshold=80.0,
            )
        )

    def test_force_render(self):
        self.assertFalse(
            pdf_page_should_use_embedded_native(
                "render",
                has_embedded_image=True,
                effective_dpi=25.0,
                threshold=80.0,
            )
        )

    def test_force_embedded_native_requires_image(self):
        self.assertTrue(
            pdf_page_should_use_embedded_native(
                "embedded_native",
                has_embedded_image=True,
                effective_dpi=300.0,
                threshold=80.0,
            )
        )
        self.assertFalse(
            pdf_page_should_use_embedded_native(
                "embedded_native",
                has_embedded_image=False,
                effective_dpi=0.0,
                threshold=80.0,
            )
        )

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            pdf_page_should_use_embedded_native(
                "banana",
                has_embedded_image=True,
                effective_dpi=50.0,
                threshold=80.0,
            )


if __name__ == "__main__":
    unittest.main()
