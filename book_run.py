"""Resolve --book PDF path, optional JSON overrides, and per-run workspace paths."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import config


def sanitize_slug(stem: str) -> str:
    stem = stem.replace("\0", "_").replace("/", "_").replace("\\", "_")
    stem = stem.strip()
    return stem or "book"


def resolve_book_argument(book: str, *, cwd: Path | None = None) -> Path:
    base = Path.cwd() if cwd is None else cwd
    p = Path(book)
    if len(p.parts) == 1 and not p.is_absolute():
        candidate = (base / "data" / book).resolve()
    else:
        candidate = p.resolve() if p.is_absolute() else (base / p).resolve()
    if not candidate.is_file():
        raise ValueError(f"PDF not found: {candidate}")
    if candidate.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file (.pdf): {candidate}")
    return candidate


def load_book_overrides(pdf_path: Path) -> dict:
    sidecar = pdf_path.parent / f"{pdf_path.stem}.book.json"
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {sidecar}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{sidecar} must be a JSON object")
    out: dict = {}
    if "start_page" in data:
        out["start_page"] = int(data["start_page"])
    if "end_page" in data:
        v = data["end_page"]
        out["end_page"] = None if v is None else int(v)
    if "split_ratio" in data:
        out["split_ratio"] = float(data["split_ratio"])
    return out


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


def build_resolved_run(pdf_path: Path, *, cwd: Path | None = None) -> ResolvedRun:
    root = Path.cwd() if cwd is None else cwd
    slug = sanitize_slug(pdf_path.stem)
    work_root = (root / "work" / slug).resolve()
    overrides = load_book_overrides(pdf_path)
    start_page = overrides.get("start_page", config.START_PAGE)
    end_page = overrides.get("end_page", config.END_PAGE)
    split_ratio = overrides.get("split_ratio", config.SPLIT_RATIO)
    return ResolvedRun(
        pdf_path=pdf_path.resolve(),
        slug=slug,
        start_page=start_page,
        end_page=end_page,
        split_ratio=split_ratio,
        work_root=work_root,
        tmp_pages=work_root / "tmp_pages",
        tmp_crops=work_root / "tmp_crops",
        tmp_results=work_root / "tmp_results",
        output_dir=work_root / "output",
        crops_zip=work_root / "crops_left.zip",
    )
