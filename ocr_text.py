"""
Step 3: OCR the left-side text panel.
Uses PaddleOCR for Traditional Chinese (supports vertical text).
Runs on AutoDL with GPU, or locally on M3 (see config).
"""
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from config import OCR_LANG, OCR_USE_GPU

_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            lang=OCR_LANG,
            use_angle_cls=True,   # detect rotated/vertical text
            use_gpu=OCR_USE_GPU,
            show_log=False,
        )
    return _ocr


def ocr_image(pil_image: Image.Image) -> str:
    arr = np.array(pil_image)
    result = get_ocr().ocr(arr, cls=True)

    lines = []
    if result and result[0]:
        # For vertical Chinese text: sort by x descending (right-to-left columns)
        boxes = sorted(result[0], key=lambda r: -r[0][0][0])
        for box in boxes:
            text = box[1][0]
            lines.append(text)

    return "\n".join(lines)
