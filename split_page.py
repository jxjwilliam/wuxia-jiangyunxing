from PIL import Image
from config import SPLIT_RATIO


def split_page(image_path) -> tuple:
    img = Image.open(image_path)
    w, h = img.size
    mid = int(w * SPLIT_RATIO)

    left = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))
    return left, right
