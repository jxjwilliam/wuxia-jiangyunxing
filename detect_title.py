"""
Step 5: Detect chapter title from OCR text.
Looks for the 第X回 pattern (e.g. 第九回_鐵槍破犁).
"""
import re
from config import FOLDER_FALLBACK

# Matches: 第九回, 第十二回, 第一百回, etc.
CHAPTER_PATTERN = re.compile(
    r"第[零一二三四五六七八九十百千]+回\s*([^\n]{2,12})"
)


def detect_title(ocr_text: str, page_num: int) -> str:
    """
    Returns folder name like '第九回_鐵槍破犁'.
    Falls back to 'page_009' if no match found.
    """
    match = CHAPTER_PATTERN.search(ocr_text)
    if match:
        full = match.group(0).strip()
        # Sanitise for filesystem: remove problematic chars
        safe = re.sub(r'[\s/\\:*?"<>|]', "_", full)
        return safe
    return FOLDER_FALLBACK.format(page_num=page_num)
