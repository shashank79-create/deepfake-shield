"""
provenance/fingerprint.py
Content fingerprints stored (and signed) inside the manifest.

pixel_sha256  exact: any single changed pixel breaks it. Survives only lossless copies.
thumbnail     a small grayscale copy (THUMB x THUMB). Survives recompression and
              resizing, and lets us say WHERE a copy differs from what was signed
              (e.g. "the face region was changed"), block by block.

Limitation: crops, rotations and flips break block alignment, so they show up
as "modified" (which is honest: the image is not what was signed).
"""

import base64
import hashlib
import io
from typing import Dict

import numpy as np
from PIL import Image

THUMB = 128   # thumbnail side (pixels)
GRID = 16     # blocks per side -> 8x8-pixel blocks


def pixel_sha256(img: Image.Image) -> str:
    rgb = img.convert("RGB")
    h = hashlib.sha256()
    h.update(f"{rgb.width}x{rgb.height}:".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def thumbnail_array(img: Image.Image) -> np.ndarray:
    """Grayscale THUMB x THUMB float array in [0, 255] (aspect ratio deliberately ignored)."""
    return np.asarray(img.convert("L").resize((THUMB, THUMB), Image.BOX), dtype=np.float32)


def encode_thumbnail(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def decode_thumbnail(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("L"), dtype=np.float32)


def fingerprint(img: Image.Image) -> Dict:
    return {
        "width": img.width, "height": img.height,
        "pixel_sha256": pixel_sha256(img),
        "thumbnail": {"size": THUMB, "grid": GRID, "png_b64": encode_thumbnail(thumbnail_array(img))},
    }


def _soften(arr: np.ndarray) -> np.ndarray:
    """Light blur so resampling/JPEG ringing does not count as a content change."""
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2)
    k /= k.sum()
    p = np.pad(arr, 3, mode="edge")
    p = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, p)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, p).astype(np.float32)


def block_differences(signed_thumb: np.ndarray, candidate: Image.Image) -> np.ndarray:
    """Per-block mean absolute difference (0-255 scale), GRID x GRID.
    Global brightness/contrast drift (filters, re-encoding) is removed first."""
    a = _soften(signed_thumb)
    b = _soften(thumbnail_array(candidate))
    # Robust brightness/contrast match (b -> gain*b + offset): refit on the best-matching 70% of
    # pixels so a locally edited region (e.g. a swapped face) does not skew the whole fit.
    mask = np.ones_like(a, dtype=bool)
    for _ in range(3):
        gain, offset = np.polyfit(b[mask], a[mask], 1)
        resid = np.abs(a - (gain * b + offset))
        mask = resid <= np.percentile(resid, 70)
    d = np.abs(a - (gain * b + offset))
    s = THUMB // GRID
    return d.reshape(GRID, s, GRID, s).mean(axis=(1, 3))
