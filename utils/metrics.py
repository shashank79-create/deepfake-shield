"""
utils/metrics.py
Quality metrics in pure PyTorch — zero extra dependencies.

SSIM (Structural Similarity Index):
    Measures how similar two images look to the human eye.
    Range [0,1]. 1.0 = identical. We need > 0.97 to ensure
    the protected image looks identical to the original.

PSNR (Peak Signal-to-Noise Ratio):
    Measures signal quality in dB. Higher = better.
    > 40dB means the noise is imperceptible.

L-inf (Maximum pixel change):
    The single largest pixel value change across the image.
    We keep this < 16/255 (barely visible to human eye).
"""

import numpy as np
import torch
import torch.nn.functional as F


class QualityMetrics:
    """All metrics in pure PyTorch. Zero extra dependencies."""

    @staticmethod
    def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g).view(1, 1, size, size)

    @classmethod
    def ssim(cls, a: torch.Tensor, b: torch.Tensor) -> float:
        """Structural Similarity Index. Range [0,1]. 1.0 = identical."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        kernel  = cls._gaussian_kernel().to(a.device)
        ch      = a.shape[1]
        kernel  = kernel.repeat(ch, 1, 1, 1)

        def conv(x):
            return F.conv2d(x, kernel, padding=5, groups=ch)

        mu1, mu2 = conv(a), conv(b)
        s1sq = conv(a * a) - mu1 ** 2
        s2sq = conv(b * b) - mu2 ** 2
        s12  = conv(a * b) - mu1 * mu2

        num   = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        denom = (mu1 ** 2 + mu2 ** 2 + C1) * (s1sq + s2sq + C2)
        return (num / denom).mean().item()

    @staticmethod
    def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
        """Peak Signal-to-Noise Ratio in dB. Higher = better."""
        mse = F.mse_loss(a, b).item()
        return 100.0 if mse < 1e-10 else 20 * np.log10(1.0 / np.sqrt(mse))

    @staticmethod
    def l_inf(a: torch.Tensor, b: torch.Tensor) -> float:
        """Max pixel change in [0,255] scale."""
        return (a - b).abs().max().item() * 255.0

    @staticmethod
    def l2(a: torch.Tensor, b: torch.Tensor) -> float:
        """L2 norm of perturbation."""
        return (a - b).pow(2).sum().sqrt().item()
