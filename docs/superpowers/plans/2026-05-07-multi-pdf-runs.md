# Multi-PDF run isolation — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require `--book` on every `main_local.py` run, resolve a per-PDF workspace under `work/<slug>/`, merge optional `data/<stem>.book.json` overrides into extraction/split settings, and thread explicit paths through `extract_pages`, `split_page`, `prepare_upload`, and `organise` without mutating `config` path globals.

**Architecture:** Add `book_run.py` containing `ResolvedRun` (frozen dataclass), PDF resolution (`resolve_book_argument`), minimal slug sanitization (`sanitize_slug`), stdlib JSON sidecar loading (`load_book_overrides`), and `build_resolved_run()` that merges `config` defaults with overrides and builds all `Path`s. Refactor `main_local.py` to construct `ResolvedRun` once in `main()` and pass it (and path parameters) through phase 1 and phase 3 helpers. AutoDL script and remote dirs stay unchanged; document clear-between-books in README and helper text.

**Tech stack:** Python 3.10+ (existing), `pathlib`, `json`, `dataclasses`, stdlib `unittest` (match repo).

---

## File map

| File | Action |
|------|--------|
| `book_run.py` | **Create** — resolution, merge, `ResolvedRun` |
| `extract_pages.py` | **Modify** — parameterized `pdf_path`, `start_page`, `end_page`; optional `image_dpi` |
| `split_page.py` | **Modify** — optional `split_ratio` argument |
| `prepare_upload.py` | **Modify** — `zip_path` parameter; drop unused `config` import |
| `organise.py` | **Modify** — required `output_dir: Path`; keep `IMAGE_FORMAT` from `config` |
| `main_local.py` | **Modify** — required `--book`; path-parameterized helpers; update messages |
| `tests/test_book_run.py` | **Create** — resolution, slug, merge, layout |
| `tests/test_pipeline.py` | **Modify** — `_resolve_unique_folder_name(..., out_dir)` signature |
| `.gitignore` | **Modify** — ignore `work/` |
| `README.md` | **Modify** — `--book`, layout, sidecar JSON, AutoDL hygiene |
| `download_results.py` | **Modify** — mention slug-specific `work/<slug>/tmp_results/` |

**Unchanged:** `main_autodl.py`, `config.py` (defaults only; no path renames required), `ocr_text.py`, `translate.py`, `detect_title.py`.

---

### Task 1: `book_run.py` core — tests first

**Files:**
- Create: `tests/test_book_run.py`
- Create: `book_run.py`

- [ ] **Step 1: Add failing tests for resolution, slug, merge, and layout**

Create `tests/test_book_run.py`:

```python
"""Tests for book_run resolution and ResolvedRun layout."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            root = Path(td)
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
```

- [ ] **Step 2: Run tests — expect failures**

Run:

```bash
cd /path/to/wuxia-jiangyunxing && python -m unittest tests.test_book_run -v
```

Expected: **FAIL** (ImportError or missing attributes).

- [ ] **Step 3: Implement `book_run.py`**

Create `book_run.py`:

```python
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
    out = {}
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
```

- [ ] **Step 4: Re-run tests**

Run:

```bash
python -m unittest tests.test_book_run -v
```

Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add book_run.py tests/test_book_run.py
git commit -m "feat(book_run): ResolvedRun, --book resolution, sidecar merge"
```

---

### Task 2: Parameterize `extract_pages`

**Files:**
- Modify: `extract_pages.py`
- Modify: `tests/test_pipeline.py` (only if anything calls `extract_pages` — skip if not)

- [ ] **Step 1: Change `extract_pages` signature and body**

Replace imports and function in `extract_pages.py` so it no longer reads `PDF_PATH`, `START_PAGE`, or `END_PAGE` from `config`. Use parameters and keep `IMAGE_DPI` from `config` unless passed.

```python
def extract_pages(
    pdf_path: Path,
    tmp_dir: Path,
    *,
    start_page: int,
    end_page: int | None,
    image_dpi: int | None = None,
):
    if image_dpi is None:
        image_dpi = IMAGE_DPI
    doc = fitz.open(str(pdf_path))
    ...
    # replace START_PAGE with start_page in range logic
    start_idx = start_page - 1
    # replace END_PAGE with end_page in error message text
```

- [ ] **Step 2: Commit**

```bash
git add extract_pages.py
git commit -m "refactor(extract_pages): parameterized pdf path and page range"
```

---

### Task 3: Parameterize `split_page`

**Files:**
- Modify: `split_page.py`

- [ ] **Step 1: Add optional `split_ratio`**

```python
def split_page(image_path, split_ratio: float | None = None) -> tuple:
    ratio = SPLIT_RATIO if split_ratio is None else split_ratio
    with Image.open(image_path) as img:
        w, h = img.size
        mid = int(w * ratio)
        ...
```

- [ ] **Step 2: Commit**

```bash
git add split_page.py
git commit -m "refactor(split_page): optional split_ratio argument"
```

---

### Task 4: Parameterize `prepare_upload` and `organise`

**Files:**
- Modify: `prepare_upload.py`
- Modify: `organise.py`

- [ ] **Step 1: `prepare_upload(crops_dir: Path, zip_path: Path)`**

Remove unused `from config import AUTODL_CROPS_DIR`. Use `zip_path` instead of `Path("crops_left.zip")`.

- [ ] **Step 2: `save_pair(folder_name, text, illustration_img, output_dir: Path)`**

Remove `OUTPUT_DIR` import; use `output_dir` argument for the parent of chapter folders.

- [ ] **Step 3: Commit**

```bash
git add prepare_upload.py organise.py
git commit -m "refactor: prepare_upload zip path; organise output_dir param"
```

---

### Task 5: Wire `main_local.py` to `ResolvedRun`

**Files:**
- Modify: `main_local.py`

- [ ] **Step 1: Remove module-level `TMP_DIR`, `CROPS_DIR`, `RESULTS_DIR`, `OUT_DIR`**

- [ ] **Step 2: Update helpers**

```python
def _resolve_unique_folder_name(
    base_name: str, page_num: int, used_names: set[str], out_dir: Path
) -> str:
    ...
    if candidate in used_names or (out_dir / candidate).exists():
        ...


def _find_local_right_crop(page_num: int, crops_dir: Path) -> Path | None:
    ...
```

- [ ] **Step 3: `phase1_preprocess(run: ResolvedRun)`**

- Call `extract_pages(run.pdf_path, run.tmp_pages, start_page=run.start_page, end_page=run.end_page)`
- `split_page(img_path, split_ratio=run.split_ratio)`
- `prepare_upload(run.tmp_crops, run.crops_zip)`
- Print next steps using `run.tmp_results` and literal `--book` reminder, e.g.  
  `python main_local.py --book <same-as-this-run> --phase3`

- [ ] **Step 4: `phase3_assemble(run: ResolvedRun)`**

Replace `RESULTS_DIR` → `run.tmp_results`, `CROPS_DIR` → `run.tmp_crops`, `OUT_DIR` → `run.output_dir` in all checks, globs, and `_find_local_right_crop(..., run.tmp_crops)`.

- [ ] **Step 5: `run_all(run)`** — pass `run`; update skip message to `run.tmp_results`.

- [ ] **Step 6: `main()`**

```python
parser.add_argument(
    "--book",
    required=True,
    help="PDF basename under data/ (e.g. jiang-yun-xing.pdf) or path to .pdf",
)
...
run = book_run.build_resolved_run(book_run.resolve_book_argument(args.book))
if args.all:
    run_all(run)
elif args.phase3:
    phase3_assemble(run)
else:
    phase1_preprocess(run)
```

- [ ] **Step 7: Docstring at top of `main_local.py`**

Update usage examples to include `--book ...` on every line.

- [ ] **Step 8: Commit**

```bash
git add main_local.py
git commit -m "feat(main_local): required --book and work/<slug> paths"
```

---

### Task 6: Fix `tests/test_pipeline.py` for new helper signatures

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Pass `out_dir` into `_resolve_unique_folder_name`**

Replace every `main_local._resolve_unique_folder_name("…", n, used)` with an added `Path(td)` or `out` as last argument. Remove `patch.object(main_local, "OUT_DIR", ...)` where obsolete.

- [ ] **Step 2: Run full suite**

```bash
python -m unittest discover -s tests -v
```

Expected: **all PASS**.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: update main_local helper signatures"
```

---

### Task 7: Ignore `work/`, update README and `download_results.py`

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `download_results.py`

- [ ] **Step 1: Add to `.gitignore`**

```
work/
```

(Keep existing `tmp_*` entries for legacy trees.)

- [ ] **Step 2: README — document**

- `--book` required; examples with `data/*.pdf`
- Directory layout `work/<slug>/…`
- Optional `data/<stem>.book.json` keys
- AutoDL: clear `/root/wuxia_crops` and `/root/wuxia_output` between books; unzip downloads to `work/<slug>/tmp_results/`

- [ ] **Step 3: `download_results.py`**

Adjust printed instructions to say unzip destination is `work/<slug>/tmp_results/` and phase 3 command includes `--book`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore README.md download_results.py
git commit -m "docs: multi-book workflow and work/ gitignore"
```

---

## Plan self-review

**Spec coverage:**
- §3 `--book` required + resolution → Task 1, 5
- §4 `work/<slug>/` layout + zip path → Task 1, 5, 7
- §5 sidecar JSON + merge → Task 1
- §6 phases + messages + phase 3 paths → Task 5, 7
- §7 avoid mutating `config` paths → architecture uses `ResolvedRun` only
- §8 tests → Tasks 1, 6
- §9 docs → Task 7

**Placeholder scan:** No TBD/TODO in tasks; code blocks are complete starting points.

**Consistency:** `ResolvedRun` field names match usage in tasks; `end_page` uses same `None`/negative semantics as `config.END_PAGE`.

**Resolver rule:** Basename-only means `Path(book)` has a single part and is relative → resolve **`cwd/data/<book>`** only (per spec). Relative multi-part paths resolve under `cwd`; absolute paths resolve as-is.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-multi-pdf-runs.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.  
2. **Inline execution** — Run tasks in this session using executing-plans, batch execution with checkpoints.

Which approach do you want?
