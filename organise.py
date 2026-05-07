from pathlib import Path
from PIL import Image
from config import OUTPUT_DIR, IMAGE_FORMAT


def save_pair(folder_name: str, text: str, illustration_img: Image.Image | None):
    folder = Path(OUTPUT_DIR) / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    # Save text
    (folder / "text.txt").write_text(text, encoding="utf-8")

    # Save illustration (if provided)
    if illustration_img is not None:
        ext = IMAGE_FORMAT.lower()
        illustration_img.save(folder / f"illustration.{ext}", IMAGE_FORMAT)
