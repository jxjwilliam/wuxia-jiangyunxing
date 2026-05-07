## Chat with Claude

```text
I have a pdf, each page includes: left side text, right side image, as attached. can you analyse and find a solution:

1. extract the left-side text into a small txt file
2. extract the right-side image into a png/jpg/... file
3. make the 2 files as pairs into a folder which name to be page number or more advanced: far-left chinese title if you can
4. there should be a lot of such folders, make them under 1 parent folder: wuxia ask me questions if have then to start
```

```text
1. the pdf is large, 39mb, each page of the pdf is like the above attached: left-side text, right-side chinese traditional comic image
2. yes. but with some exception: e.g. start from page 6. that means skip from page 1 to page 6. hope this makes things simpler.
3. 第九回_鐵槍破犁
4. I have no idea, let's start simpier - extract text directly. if possible translate the traditional chinese into simple chinese.
5. you recommend. important: the image will be used to re-process, image-to-video in different tools or AI.
```

## Resources

- pdf: data/jiang-yun-xing.pdf
- implementation plan: docs/wuxia_extraction_plan.md


## Folder/files are created so far

### by me

- docs/init.md
- data/jiang-yun-xing.pdf

### Cursor IDE

- docs/wuxia_extraction_plan.md
- docs/superpowers/specs/2026-05-07-wuxia-hybrid-ocr-only-v1-design.md

### Opencode

- all others
- '/skills' -> 'brainstorming' -> write plan -> ...

## Why AutoDL for text extraction?

The cloud GPU step is for **OCR** (reading Traditional Chinese from the **left** panel crops), not because PDF rasterisation or left/right splitting must run remotely.

This repo uses **PaddleOCR** for Traditional Chinese (including vertical text). That stack is built around **NVIDIA GPU + CUDA**. On an **Apple Silicon Mac** (e.g. M3) there is no practical Paddle GPU path, so local OCR would be **CPU-only Paddle**—feasible but **much slower** on ~200 high-resolution crops—or you’d need a **different OCR engine**, often with **weaker fit** for this layout and script.

So the split is intentional:

| Where | Role |
|--------|------|
| **Local** | PDF → page images → left/right crops, packaging, OpenCC 繁→简, chapter detection, `wuxia/` assembly |
| **AutoDL** | PaddleOCR on left crops (GPU), one `ocr_text.txt` per page |

AutoDL is **pay-per-hour GPU**; for a one-off book run it is often cheaper in **elapsed time** than long CPU OCR on a laptop.

If you switched to an OCR stack that runs fast locally on Apple Silicon without a cloud GPU, you could drop AutoDL—but you’d be trading **tooling fit** (PaddleOCR + GPU) for **local-only convenience**.
