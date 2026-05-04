"""
core/losses.py
Loss functions for the adversarial attack.
"""

import torch
import torch.nn.functional as F


class LossFunctions:
    """
    Well-defined, numerically stable loss functions.

    We use a combination of cosine distance + MSE because:
    - Cosine handles DIRECTION in feature space (which identity it looks like)
    - MSE handles MAGNITUDE (how different the features are)
    Together they push features firmly toward a different identity.
    """

    @staticmethod
    def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Cosine distance in [0, 2].
        Best for high-dimensional feature spaces — handles scale variation.
        """
        xn = F.normalize(x.flatten(1), dim=-1)
        yn = F.normalize(y.flatten(1), dim=-1)
        return (1 - (xn * yn).sum(dim=-1)).mean()

    @staticmethod
    def mse_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Clamped MSE — handles magnitude, prevents gradient explosion."""
        return torch.clamp(F.mse_loss(x.flatten(1), y.flatten(1)), max=1.0)

    @staticmethod
    def combined(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """70% cosine + 30% MSE — best of both."""
        return (0.7 * LossFunctions.cosine_distance(x, y) +
                0.3 * LossFunctions.mse_distance(x, y))
