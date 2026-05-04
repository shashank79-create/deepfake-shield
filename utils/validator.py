"""
utils/validator.py
Validate all inputs before touching GPU.
Catches bad input early — before wasting compute.
"""

import logging
import os
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from config.settings import Config


class ValidationError(Exception):
    """Invalid input provided by user."""


class InputValidator:
    """Validate all inputs before touching GPU."""

    MAX_FILE_MB   = 100
    MIN_DIMENSION = 32
    VALID_MODES   = {"RGB", "RGBA", "L", "P"}
    VALID_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, logger: logging.Logger):
        self.log = logger

    def validate_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        if not p.exists():
            raise ValidationError(f"File not found: {p}")
        if not p.is_file():
            raise ValidationError(f"Not a file: {p}")
        if not os.access(p, os.R_OK):
            raise ValidationError(f"File not readable: {p}")
        if p.suffix.lower() not in self.VALID_EXTS:
            raise ValidationError(
                f"Unsupported format '{p.suffix}'. Allowed: {self.VALID_EXTS}"
            )
        size_mb = p.stat().st_size / 1e6
        if size_mb > self.MAX_FILE_MB:
            raise ValidationError(
                f"File too large ({size_mb:.1f} MB > {self.MAX_FILE_MB} MB)"
            )
        return p

    def validate_image(self, img: Image.Image) -> Image.Image:
        if img is None:
            raise ValidationError("Image is None")
        if img.mode not in self.VALID_MODES:
            raise ValidationError(f"Unsupported mode: {img.mode}")
        w, h = img.size
        if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
            raise ValidationError(
                f"Image too small: {img.size} (min {self.MIN_DIMENSION}px)"
            )
        return img.convert("RGB")

    def validate_strength(self, strength: str) -> str:
        if strength not in Config.PRESETS:
            raise ValidationError(
                f"Unknown strength '{strength}'. "
                f"Choose from: {list(Config.PRESETS)}"
            )
        return strength

    def validate_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 4:
            raise ValidationError(
                f"Expected 4D tensor [B,C,H,W], got {t.dim()}D"
            )
        if t.shape[1] != 3:
            raise ValidationError(
                f"Expected 3 channels, got {t.shape[1]}"
            )
        if t.min() < -0.01 or t.max() > 1.01:
            self.log.warning(
                f"Tensor out of [0,1]: min={t.min():.3f} max={t.max():.3f}"
            )
        return t
