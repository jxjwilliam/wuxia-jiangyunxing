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
| **Local** | PDF → page images → left/right crops, packaging, OpenCC 繁→简, chapter detection, `work/<slug>/output/` assembly |
| **AutoDL** | PaddleOCR on left crops (GPU), one `ocr_text.txt` per page |

AutoDL is **pay-per-hour GPU**; for a one-off book run it is often cheaper in **elapsed time** than long CPU OCR on a laptop.

If you switched to an OCR stack that runs fast locally on Apple Silicon without a cloud GPU, you could drop AutoDL—but you’d be trading **tooling fit** (PaddleOCR + GPU) for **local-only convenience**.

```text
scp -P 46840 work/01-桃园结义/crops_left.zip  root@connect.cqa1.seetacloud.com:/root/
scp -P 46840 work/14-煮酒论英雄/crops_left.zip  root@connect.cqa1.seetacloud.com:/root/
scp -P 46840 work/41-定军山/crops_left.zip  root@connect.cqa1.seetacloud.com:/root/
```

```
scp -P 46840 work/01-桃园结义/crops_left.zip root@connect.cqa1.seetacloud.com:/root/
```

### 术语解释

- DPI：Dots Per Inch（每英寸点数），图像清晰度的核心指标，数值越低画面越模糊。
- 置信度（confidence）：OCR/VLM 识别结果的可信度评分，分数越低代表识别结果越不可靠。
- 回退方案（fallback）：主流程失败时自动触发的备用处理逻辑，保证整体任务不中断。
- VLM：Vision-Language Model（视觉语言模型），兼具图像理解与文本生成能力的多模态大模型，对低质量文本的识别鲁棒性远强于传统 OCR。
- Real-ESRGAN：开源 AI 超分辨率模型，专门用于修复低分辨率、低质量图像。
PaddleOCR PP-OCRv5：百度开源的工业级 OCR 工具，对常规文本识别效率高，但对极低质量扫描件的容错能力有限。
- AutoDL：国内主流的 GPU 算力租赁平台，常被用于本地模型推理与 AI 任务部署。


### Low DPI

当处理流水线检测到低 DPI / 置信度低的页面时，以下哪类引擎适合作为回退方案（fallback）？**选项**

- (A) 纯本地方案：添加 Real-ESRGAN 超分辨率增强，然后重试 PaddleOCR PP-OCRv5 服务。全程不调用任何外部 API。（效果提升有限；当 DPI≤30 时仍可能失败。）
- (B) 调用 API 的云端 VLM 方案：例如通义千问 VL-Max（DashScope 平台）、GPT-4o 视觉版、Claude 视觉版。对低质量扫描件识别准确率最高；需要 API 密钥 + 按页计费；AutoDL 平台需开通出站 HTTPS 访问权限。
- (C) 分层混合方案：先执行本地超分辨率 + OCR 识别，仅当置信度仍低于阈值时，再调用 VLM（在保证准确率的前提下，成本最低）。
- (D) 本地 VLM 方案：在 AutoDL 的 GPU 上直接运行通义千问 VL 或 InternVL 模型（无按次调用成本；但需下载大体积模型文件，且需要额外的显存余量）。
- (E) 不新增引擎方案：仅检测 + 告警 + 跳过该页，生成占位符，由用户手动转录文本。
