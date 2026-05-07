"""
Step 6 (optional): Upscale illustration using Real-ESRGAN.
Designed for AutoDL GPU instances with CUDA.
Uses the anime-optimised model for comic-style illustrations.
"""
import numpy as np
import cv2
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from config import UPSCALE_FACTOR


def upscale_image(pil_image: Image.Image) -> Image.Image:
    """
    Upscale a PIL image using Real-ESRGAN with anime model.
    Returns upscaled PIL Image.
    """
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=6, num_grow_ch=32, scale=4,
    )
    upsampler = RealESRGANer(
        scale=UPSCALE_FACTOR,
        model_path="weights/RealESRGAN_x4plus_anime_6B.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )

    arr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    out, _ = upsampler.enhance(arr, outscale=UPSCALE_FACTOR)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
