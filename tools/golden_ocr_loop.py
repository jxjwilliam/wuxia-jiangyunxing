#!/usr/bin/env python3
"""
Cycle OCR preprocess / predict knobs until stdout text matches the target reference.

Paddle is deterministic — repeating the **same** parameters never yields a different hash of recognition.
This tool **cycles a parameter grid** (max input side, prediction kwargs, enhancement retry); run it indefinitely
until your upstream OCR changes make the stripped text equal the Traditional or Simplified reference.

Examples:
  python tools/golden_ocr_loop.py --cpu --rotate work/01-桃园结义/tmp_crops/page_010_left.jpg

  python tools/golden_ocr_loop.py --cpu --rotate --simplified path/to/page_010_left.jpg --max-iters 200

Environment: run from repo root or set PYTHONPATH=. so `configs` and `ocr_text` resolve.
Ctrl+C exits with 130.
"""
from __future__ import annotations

import argparse
import itertools
import signal
import sys
import time
from pathlib import Path

_GOLD_TRAD_LINES = (
    "四 他們約定了一個日子，各地一齊向官兵進攻。"
    "因為他們以黃巾裹頭為標幟，所以叫作黃巾軍。"
)


def golden_expected_nospace(*, simplified: bool) -> str:
    spaced = "".join(_GOLD_TRAD_LINES.strip().splitlines())
    if simplified:
        from opencc import OpenCC

        from configs.config import OPENCC_CONFIG

        spaced = OpenCC(OPENCC_CONFIG).convert(spaced)
    return "".join(spaced.split())


def strip_ws(text: str) -> str:
    return "".join(text.split())


STOP = {"flag": False}


def _handle_sig(_signum, _frame):  # noqa: ARG001
    STOP["flag"] = True


def variant_grid(*, omit_uncapped_side: bool):
    sides = [3072, 3584, 4096, 4480, 5120]
    if not omit_uncapped_side:
        sides = [0, *sides]
    preds = [{}, {"use_textline_orientation": False}]
    retries = [True, False]
    yield from itertools.product(sides, preds, retries)


def apply_variant(cfg, side: int, pred_kw: dict, retry_enh: bool) -> None:
    cfg.OCR_RETRY_WITH_ENHANCEMENT = retry_enh
    if side <= 0:
        cfg.OCR_MAX_INPUT_LONG_SIDE = 0
    else:
        cfg.OCR_MAX_INPUT_LONG_SIDE = int(side)


def variant_summary(side: int, pred_kw: dict, retry_enh: bool) -> str:
    return (
        f"OCR_MAX_INPUT_LONG_SIDE={side or 0}; "
        f"predict_kwargs={pred_kw if pred_kw else '{}'}; "
        f"OCR_RETRY_WITH_ENHANCEMENT={retry_enh}"
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="Spin until OCR equals the canonical page-010 string.")
    p.add_argument("image", nargs="?", type=Path, default=None, help="JPG or PNG crop")
    p.add_argument("--cpu", action="store_true", help="Force CPU + lite Paddle stack.")
    p.add_argument("--rotate", action="store_true", help="Simulate manifest ocr_rotate_left_cw90.")
    p.add_argument(
        "--simplified",
        action="store_true",
        help="Expect OpenCC t2s variant of the reference.",
    )
    p.add_argument(
        "--every",
        type=float,
        default=0.05,
        help="Sleep seconds between variant attempts.",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Exit 1 after this many iterations without hitting the target.",
    )

    args = p.parse_args()

    img_arg = (
        args.image.resolve()
        if args.image is not None
        else (root / "work" / "01-桃园结义" / "tmp_crops" / "page_010_left.jpg").resolve()
    )
    if not img_arg.is_file():
        print(f"❌ Missing image: {img_arg}", file=sys.stderr)
        print("Provide a path, e.g. work/…/tmp_crops/page_010_left.jpg", file=sys.stderr)
        return 2

    golden = golden_expected_nospace(simplified=args.simplified)
    cycle_def = itertools.cycle(
        variant_grid(omit_uncapped_side=bool(args.cpu))
    )

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    import configs.config as cfg

    if args.cpu:
        cfg.OCR_USE_GPU = False
        cfg.OCR_CPU_USE_LITE_MODELS = True
        if int(getattr(cfg, "OCR_MAX_INPUT_LONG_SIDE", 0) or 0) == 0:
            cfg.OCR_MAX_INPUT_LONG_SIDE = 4480

    print(
        f"→ target length={len(golden)} stripped chars ({'简体' if args.simplified else '繁体'})",
        file=sys.stderr,
    )
    print(f"→ image={img_arg} rotate={args.rotate} cpu={args.cpu}", file=sys.stderr)

    from ocr_text import ocr_image, set_ocr_predict_kwargs, set_phase2_manifest
    from PIL import Image

    iter_idx = -1

    while not STOP["flag"]:
        iter_idx += 1

        if args.max_iters is not None and iter_idx >= args.max_iters:
            print(
                f"❌ max-iters={args.max_iters} — target not reachable with current OCR build.",
                file=sys.stderr,
            )
            return 1

        side, pred_kw, retry_enh = next(cycle_def)
        apply_variant(cfg, side, pred_kw, retry_enh)

        set_ocr_predict_kwargs(pred_kw)

        try:
            set_phase2_manifest({"ocr_rotate_left_cw90": args.rotate})

            with Image.open(img_arg) as pil_img:
                text = ocr_image(pil_img)
        except KeyboardInterrupt:
            STOP["flag"] = True
            break
        except Exception as exc:  # noqa: BLE001
            print(f"iter {iter_idx} ❌ raised {exc!r}", file=sys.stderr)
            time.sleep(args.every)
            continue

        token = strip_ws(text)
        if token == golden:
            print(text)
            print(
                f"✅ match iter={iter_idx} vars=({variant_summary(side, pred_kw, retry_enh)})",
                file=sys.stderr,
            )
            return 0

        if iter_idx == 0 or iter_idx % 12 == 0:
            snippet = token[:140] + ("…" if len(token) > 140 else "")
            print(
                f"iter {iter_idx} miss — {variant_summary(side, pred_kw, retry_enh)}",
                file=sys.stderr,
            )
            print(f"  got ({len(token)} stripped): {snippet!s}", file=sys.stderr)

        time.sleep(args.every)

    print("Stopped by Ctrl+C.", file=sys.stderr)
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
