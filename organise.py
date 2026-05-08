from pathlib import Path
from PIL import Image
from config import IMAGE_FORMAT


def save_pair(folder_name: str, text: str, illustration_img: Image.Image | None, output_dir: Path):
    folder = output_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "text.txt").write_text(text, encoding="utf-8")

    if illustration_img is None:
        raise ValueError(f"Missing illustration for output folder: {folder}")
    ext = IMAGE_FORMAT.lower()
    illustration_img.save(folder / f"illustration.{ext}", IMAGE_FORMAT)
