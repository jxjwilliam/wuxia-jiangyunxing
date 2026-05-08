# Multi-PDF run isolation — design spec

**Status:** Approved by stakeholder (2026-05-07).  
**Scope:** Make the existing hybrid PDF → crops → OCR → assemble pipeline **book-aware**: each run targets one PDF under `data/`, with **local** temp and output directories derived from that PDF’s filename. **AutoDL** remote paths stay fixed; the operator clears them between books.

---

## 1. Goals

- Run the **same pipeline** for **multiple** PDFs without mixing intermediate or final artifacts on disk.
- Require an **explicit dataset** on every local invocation (`--book`), so a command always denotes one PDF.
- Keep **global defaults** in `config.py`; allow **optional per-PDF overrides** with minimal machinery (stdlib only).
- Preserve **phase 2** behavior on the GPU host: **`/root/wuxia_crops`** and **`/root/wuxia_output`** unchanged; isolation is **local** plus **operational** (empty remote dirs between books).

## 2. Non-goals

- Namespacing AutoDL remote directories per book (out of scope).
- A central registry of all books (e.g. `datasets.yaml`) unless added later.
- Automatic migration of existing repo-root **`wuxia/`**, **`tmp_crops/`**, **`tmp_results/`** into the new layout.
- UI or interactive picker.

---

## 3. CLI and PDF resolution

### 3.1 `--book` (required)

- Every **`main_local.py`** invocation must pass **`--book <value>`**.
- **`<value>`** may be:
  - A **path** to a `.pdf` (relative or absolute), or
  - A **basename** only (no directory separators), in which case the resolver tries **`data/<value>`** first.

### 3.2 Validation

- File must **exist**.
- Must resolve to a path whose suffix is **`.pdf`** (case-insensitive compare is acceptable).
- On failure: clear error message (missing file, wrong type, not found under `data/`).

### 3.3 Backward compatibility

- **`config.PDF_PATH`** may remain as documentation or a convenience default for humans, but **must not** be required for execution once this design is implemented: the **effective** PDF path always comes from **`--book`** resolution.

---

## 4. Slug and workspace layout

### 4.1 Slug

- **`slug`**: derived from the PDF file **stem** (filename without `.pdf`), after **minimal sanitization**:
  - Remove or replace characters that are invalid or risky in path components (e.g. `/`, `\`, NUL).
  - Prefer **preserving Unicode** in stems on POSIX (macOS/Linux); document that unusual stems may need manual renaming on Windows.

### 4.2 Root directory

- All per-book **local** artifacts live under:

  **`work/<slug>/`**

  (`work/` name is fixed for this project; it exists to keep `.gitignore` simple.)

### 4.3 Subdirectories

| Role | Path |
|------|------|
| Rasterized full pages | `work/<slug>/tmp_pages/` |
| Left/right crops | `work/<slug>/tmp_crops/` |
| Downloaded OCR results (phase 3 input) | `work/<slug>/tmp_results/` |
| Final chapter folders (text + illustration) | `work/<slug>/output/` |

- **`crops_left.zip`** for upload: **`work/<slug>/crops_left.zip`** (same directory level as `tmp_crops/` or inside `tmp_crops/` — implementation chooses one and documents it; prefer **next to `tmp_crops`** as `work/<slug>/crops_left.zip` for visibility).

**Overwrite policy:** Re-running phase 1 or 3 for the same **slug** may **overwrite** files inside that book’s tree. No extra versioning.

### 4.4 Legacy paths

- Existing **`wuxia/`** at repo root (or current global **`tmp_*`**) are **not** deleted. New runs use **`work/<slug>/...`**. Users may copy or symlink if desired.

---

## 5. Configuration merge (hybrid overrides)

### 5.1 Global defaults

- **`config.py`** continues to hold defaults: e.g. **`START_PAGE`**, **`END_PAGE`**, **`SPLIT_RATIO`**, OCR/translation/image settings, and naming patterns unrelated to per-book paths.

### 5.2 Optional sidecar

- Optional file: **`data/<stem>.book.json`** (same **stem** as the PDF basename without `.pdf`).
- Format: **JSON**, **stdlib `json` only**.
- Semantics: **only keys present** override the corresponding global defaults. **Missing file** ⇒ no overrides.
- Example keys (exact set to match implemented `config` surface): `start_page`, `end_page`, `split_ratio` (extend only as needed).

### 5.3 Effective configuration

- At process start, build a **`ResolvedRun`** (or equivalent) object containing:
  - Resolved **PDF path**
  - **slug**
  - Merged **extraction/split** parameters
  - **Absolute paths** for `tmp_pages`, `tmp_crops`, `tmp_results`, `output`, and zip path

---

## 6. Phase behavior

### 6.1 Phase 1 (local — `main_local` without `--phase3`)

- Input: PDF from **`ResolvedRun`**.
- Write page images to **`work/<slug>/tmp_pages/`**.
- Write crops to **`work/<slug>/tmp_crops/`**.
- Build **`crops_left.zip`** under **`work/<slug>/`** (see 4.3).
- Printed instructions: upload zip; unzip to **`/root/wuxia_crops`**; run **`main_autodl.py`**; download; unzip to **`work/<slug>/tmp_results/`** (wording must include **slug-specific** local path).

### 6.2 Phase 2 (AutoDL — `main_autodl.py`)

- **Unchanged:** read crops from **`AUTODL_REMOTE_DIR`** (e.g. `/root/wuxia_crops`), write **`page_XXX/ocr_text.txt`** under **`AUTODL_OUTPUT_DIR`** (e.g. `/root/wuxia_output`).
- **Operational rule:** Before processing a **different** book, the operator **must** remove or empty **`/root/wuxia_crops`** and **`/root/wuxia_output`** so left crops and OCR outputs from two PDFs never mix.

### 6.3 Phase 3 (local — `main_local --phase3`)

- Expect **`work/<slug>/tmp_results/`** populated from the download step for **this** slug.
- Expect **`work/<slug>/tmp_crops/`** right-side images for pairing.
- Write final chapter folders under **`work/<slug>/output/`**.

---

## 7. Implementation architecture (recommended)

- **Run context / explicit parameters:** Introduce a small **`ResolvedRun`** (or **`BookRun`**) dataclass built once in **`main_local`** after argparse + merge.
- **Pass paths into** **`extract_pages`**, **`prepare_upload`**, **`organise`**, and phase 3 assembly — or thin wrappers that close over **`ResolvedRun`**. Avoid mutating **`config`** module globals for paths (reduces test flakiness).
- **`extract_pages`**, **`organise`**, and any module that currently imports path constants from **`config`** should accept **path parameters** where practical, with **`config`** retaining only **defaults** and **non-path** settings.

---

## 8. Testing

- Unit tests (no full PDF required where possible):
  - PDF path resolution (`--book` basename vs path, errors).
  - Slug sanitization edge cases (separators, normal stem).
  - JSON merge: missing file, partial keys, invalid JSON error behavior (fail fast with message).
  - Path construction: expected `work/<slug>/...` layout.

---

## 9. Documentation updates

- **README** (when implementing): document **`--book`**, **`work/<slug>/`**, sidecar JSON, and **AutoDL clear-between-books** rule.
- **`download_results.py`** or similar helpers: mention slug-specific **`tmp_results`** path.

---

## 10. Spec self-review (2026-05-07)

| Check | Result |
|--------|--------|
| Placeholders / TBD | None intentional; zip exact placement fixed in §4.3 as `work/<slug>/crops_left.zip`. |
| Internal consistency | Local isolation vs fixed AutoDL paths explicit; operator duty in §6.2. |
| Scope | Single implementation track; no registry, no remote namespacing. |
| Ambiguity | Basename resolution: **`data/<value>`** only when value has no dir component — explicit in §3.1. |

---

## 11. Approval gate

Implementation and **`writing-plans`** proceed only after the stakeholder confirms this document is satisfactory as written (or requests edits in the file above).
