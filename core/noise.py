"""
core/noise.py
Noise reduction utilities — keeps perturbations invisible to the human eye.

Three techniques combined:
1. Gaussian smoothing  — pushes noise to low frequencies (less visible)
2. TV regularization   — penalizes sharp pixel jumps
3. Perceptual masking  — face gets less noise than background
"""

import io
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


def gaussian_blur_tensor(tensor: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian blur to a [1,C,H,W] tensor.

    WHY: Raw adversarial gradients create high-frequency (sharp) noise —
    the kind the human eye notices most. Blurring periodically pushes the
    noise into lower frequencies, far less visible while keeping attack effective.
    """
    if sigma <= 0:
        return tensor

    radius   = max(1, int(3 * sigma))
    size     = 2 * radius + 1
    coords   = torch.arange(size, dtype=torch.float32, device=tensor.device) - radius
    kernel1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel1d = kernel1d / kernel1d.sum()
    kernel2d = kernel1d.outer(kernel1d).view(1, 1, size, size)
    ch       = tensor.shape[1]
    kernel2d = kernel2d.repeat(ch, 1, 1, 1)
    return F.conv2d(tensor, kernel2d, padding=radius, groups=ch)


def total_variation_loss(tensor: torch.Tensor) -> torch.Tensor:
    """
    Total Variation regularization loss.

    WHY: Penalizes large differences between adjacent pixels.
    Forces optimizer to prefer smooth, low-frequency perturbations
    over sharp, spiky ones — reduces visible noise without weakening attack.
    """
    diff_h = tensor[:, :, 1:, :]  - tensor[:, :, :-1, :]
    diff_w = tensor[:, :, :, 1:]  - tensor[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


def jpeg_simulate(tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Round-trip JPEG compress/decompress a [1,C,H,W] tensor.

    WHY: Makes the perturbation survive Instagram/WhatsApp/Twitter
    recompression. Without this, the protection is stripped when uploaded.
    """
    to_pil    = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    results   = []
    for i in range(tensor.shape[0]):
        img = to_pil(tensor[i].cpu().clamp(0, 1))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        results.append(to_tensor(Image.open(buf).convert("RGB")))
    return torch.stack(results).to(tensor.device)


def detect_face_region(image: torch.Tensor) -> torch.Tensor:
    """
    Detect face region using skin color heuristics + center-of-image bias.

    Returns [1,C,H,W] float mask:
        1.0 = face/skin region
        0.0 = background/hair/clothes

    Used by build_face_epsilon_map() which INVERTS this:
        face gets LOW epsilon  → clean natural skin
        background gets HIGH epsilon → strong hidden attack
    """
    with torch.no_grad():
        h, w = image.shape[2], image.shape[3]
        r, g, b = image[:, 0], image[:, 1], image[:, 2]

        # Skin color detection in RGB space
        skin_r  = (r > 0.35) & (r < 1.0)
        skin_g  = (g > 0.18) & (g < 0.85)
        skin_b  = (b > 0.10) & (b < 0.75)
        rg_ratio = (r > g * 0.9) & (r < g * 1.8)
        rb_ratio = (r > b * 1.05)
        skin_mask = (skin_r & skin_g & skin_b & rg_ratio & rb_ratio).float().unsqueeze(1)

        # Center-of-image Gaussian bias (faces are usually centered/upper)
        y_coords = torch.linspace(-1, 1, h, device=image.device)
        x_coords = torch.linspace(-1, 1, w, device=image.device)
        yy, xx   = torch.meshgrid(y_coords, x_coords, indexing='ij')
        center_gauss = torch.exp(
            -((yy - (-0.1))**2 / (2 * 0.55**2) +
              (xx - 0.0)**2    / (2 * 0.45**2))
        ).unsqueeze(0).unsqueeze(0).to(image.device)

        face_map = skin_mask * center_gauss
        if face_map.max() > 0:
            face_map = face_map / face_map.max()
        if face_map.sum() < (h * w * 0.01):
            face_map = center_gauss  # fallback if skin detection fails

        face_map = gaussian_blur_tensor(face_map, sigma=12.0)
        face_map = face_map / (face_map.max() + 1e-8)
        return face_map.expand_as(image)


def build_face_epsilon_map(
    image: torch.Tensor,
    face_epsilon: float,
    bg_epsilon: float,
) -> torch.Tensor:
    """
    Build per-pixel epsilon map with face-protective strategy:
        Face region  → face_epsilon  (gentle — no visible artifacts on skin)
        Hair/bg      → bg_epsilon    (heavy  — attack hidden in texture)

    WHY THIS WORKS:
    Deepfake encoders process the FULL image context to build an identity
    embedding. Heavy perturbation in background poisons that embedding,
    causing the generator to reconstruct the wrong identity — while the
    face skin stays completely natural.
    """
    face_mask = detect_face_region(image)
    bg_mask   = 1.0 - face_mask
    return face_epsilon * face_mask + bg_epsilon * bg_mask
