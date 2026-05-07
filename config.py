"""Wuxia PDF Extraction — all settings in one place."""

# PDF source
PDF_PATH = "data/jiang-yun-xing.pdf"
OUTPUT_DIR = "wuxia"
START_PAGE = 6           # 1-indexed; pages 1-5 are skipped (cover + front matter)
END_PAGE = None          # None = process to end of PDF

# Page split: ratio of left/right (0.5 = exact midpoint)
# Adjust if text/illustration boundary is off-centre
SPLIT_RATIO = 0.5

# OCR (works on both local M3 and AutoDL GPU)
OCR_LANG = "chinese_cht"    # Traditional Chinese for PaddleOCR
OCR_USE_GPU = False         # Set True on AutoDL with CUDA; False on local M3

# Translation (Traditional → Simplified Chinese)
OPENCC_CONFIG = "t2s"

# Image output
IMAGE_FORMAT = "PNG"        # PNG = lossless, best for AI tools
IMAGE_DPI = 300             # DPI for page rasterisation (72 PPI native → upscale)

# Upscaling (optional — AutoDL only; requires realesrgan)
UPSCALE_ENABLED = False
UPSCALE_FACTOR = 4          # 2× or 4×

# Fallback folder name when no chapter title detected
FOLDER_FALLBACK = "page_{page_num:03d}"

# Hybrid workflow paths
AUTODL_CROPS_DIR = "tmp_crops"          # Where split crops are stored locally
AUTODL_REMOTE_DIR = "/root/wuxia_crops"  # Where crops land on AutoDL
AUTODL_OUTPUT_DIR = "/root/wuxia_output" # Where AutoDL writes results
AUTODL_RESULTS_DIR = "tmp_results"       # Where downloaded results go locally
