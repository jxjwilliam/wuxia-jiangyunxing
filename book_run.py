"""Compatibility layer for runtime book configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from configs import config


def sanitize_slug(stem: str) -> str:
    return config.sanitize_slug(stem)


def resolve_book_argument(book: str, *, cwd: Path | None = None) -> Path:
    return config.resolve_book_argument(book, cwd=cwd)


def load_book_overrides(pdf_path: Path) -> dict:
    return config.load_book_overrides(pdf_path)


@dataclass(frozen=True)
class ResolvedRun:
    pdf_path: Path
    slug: str
    start_page: int
    end_page: int | None
    split_ratio: float
    work_root: Path
    tmp_pages: Path
    tmp_crops: Path
    tmp_results: Path
    output_dir: Path
    crops_zip: Path
    ocr_left_crop: dict[str, float | int] | None


def build_resolved_run(pdf_path: Path, *, cwd: Path | None = None) -> ResolvedRun:
    runtime = config.build_book_runtime_config(str(pdf_path), cwd=cwd)
    return ResolvedRun(
        pdf_path=runtime.pdf_path,
        slug=runtime.slug,
        start_page=runtime.start_page,
        end_page=runtime.end_page,
        split_ratio=runtime.split_ratio,
        work_root=runtime.work_root,
        tmp_pages=runtime.tmp_pages,
        tmp_crops=runtime.tmp_crops,
        tmp_results=runtime.tmp_results,
        output_dir=runtime.output_dir,
        crops_zip=runtime.crops_zip,
        ocr_left_crop=runtime.ocr_left_crop,
    )
