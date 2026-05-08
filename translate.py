"""
Step 4: Translate Traditional Chinese → Simplified Chinese using OpenCC.
Fast, offline, deterministic character mapping.
"""
import opencc
from configs.config import OPENCC_CONFIG

_converter = None


def get_converter():
    global _converter
    if _converter is None:
        _converter = opencc.OpenCC(OPENCC_CONFIG)
    return _converter


def translate(text: str) -> str:
    return get_converter().convert(text)
