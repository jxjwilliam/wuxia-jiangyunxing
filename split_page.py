from PIL import Image
from config import SPLIT_RATIO


def split_page(image_path, split_ratio: float | None = None) -> tuple:
    ratio = SPLIT_RATIO if split_ratio is None else split_ratio
    with Image.open(image_path) as img:
        w, h = img.size
        mid = int(w * ratio)
        left = img.crop((0, 0, mid, h)).copy()
        right = img.crop((mid, 0, w, h)).copy()
    return left, right
