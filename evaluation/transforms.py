"""
evaluation/transforms.py
What happens to a photo after it is posted: recompression, resizing, light blur.
A protection that only works on the pristine PNG protects nothing.
"""

import io
from typing import Callable, Dict

from PIL import Image, ImageFilter


def jpeg(quality: int) -> Callable[[Image.Image], Image.Image]:
    def f(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return f


def resize_round_trip(scale: float) -> Callable[[Image.Image], Image.Image]:
    def f(img: Image.Image) -> Image.Image:
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    return f


def blur(radius: float) -> Callable[[Image.Image], Image.Image]:
    return lambda img: img.filter(ImageFilter.GaussianBlur(radius))


def social_upload(img: Image.Image, long_side: int = 1080, quality: int = 80) -> Image.Image:
    """Roughly what Instagram/WhatsApp do: cap the long side, then JPEG."""
    w, h = img.size
    if max(w, h) > long_side:
        s = long_side / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return jpeg(quality)(img)


def nl_means_denoise(strength: int) -> Callable[[Image.Image], Image.Image]:
    """Non-local-means denoising: a standard way to wipe adversarial noise off a photo."""
    def f(img: Image.Image) -> Image.Image:
        import cv2
        import numpy as np
        arr = cv2.fastNlMeansDenoisingColored(np.asarray(img.convert("RGB"))[:, :, ::-1].copy(),
                                              None, strength, strength, 7, 21)
        return Image.fromarray(arr[:, :, ::-1].copy())
    return f


def median(size: int) -> Callable[[Image.Image], Image.Image]:
    return lambda img: img.filter(ImageFilter.MedianFilter(size))


# What a determined attacker does BEFORE face-swapping: try to strip the noise.
ADAPTIVE: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "denoise":    nl_means_denoise(7),
    "median3":    median(3),
    "down0.25":   resize_round_trip(0.25),
    "jpeg30":     jpeg(30),
}

TRANSFORMS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "none":       lambda img: img,
    "jpeg95":     jpeg(95),
    "jpeg75":     jpeg(75),
    "jpeg50":     jpeg(50),
    "resize0.5":  resize_round_trip(0.5),
    "blur1":      blur(1.0),
    "social":     social_upload,
    **ADAPTIVE,
}
