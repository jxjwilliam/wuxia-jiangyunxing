"""Light-weight checks around the OCR golden transcript (heavy loop lives in tools/golden_ocr_loop.py)."""

from tools.golden_ocr_loop import golden_expected_nospace, strip_ws


def test_golden_traditional_core_phrases():
    t = golden_expected_nospace(simplified=False)
    assert "約定了一個日子" in t
    assert "黃巾裹頭" in t and "標幟" in t


def test_strip_ws_stable():
    assert strip_ws("a  b \n c") == "abc"


def test_cycle_grid_sizes():
    from tools.golden_ocr_loop import variant_grid

    bounded = sum(1 for _ in variant_grid(omit_uncapped_side=True))
    full = sum(1 for _ in variant_grid(omit_uncapped_side=False))
    assert bounded == 5 * 2 * 2  # CPU path omits uncapped side
    assert full == (1 + 5) * 2 * 2
