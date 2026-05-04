"""
core/attack.py
PGD (Projected Gradient Descent) adversarial attack.

Key design decisions:
1. Adam optimizer     — stable gradient flow
2. Random targets     — DIRECTED attack toward unknown identity
                        (much stronger than "push away from original")
3. JPEG simulation    — social media robustness baked in
4. L-inf projection   — clamp delta each step (standard PGD constraint)
5. Gradient clipping  — prevents explosion on ViT attention layers
6. Cosine + MSE loss  — better than MSE alone for feature-space distances
"""

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import Config
from core.losses import LossFunctions
from core.noise import (
    build_face_epsilon_map,
    gaussian_blur_tensor,
    jpeg_simulate,
    total_variation_loss,
)


class PGDAttack:
    """Projected Gradient Descent adversarial attack."""

    def __init__(
        self,
        ensemble: Dict[str, nn.Module],
        device: torch.device,
        logger: logging.Logger,
    ):
        self.ensemble = ensemble
        self.device   = device
        self.log      = logger

    def _run_model(self, name: str, model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass handling different input size requirements per model."""
        size = Config.FACE_SIZE if name == "facenet" else Config.MODEL_SIZE
        x    = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        out  = model(x)
        if out.dim() > 2:
            out = out.mean(dim=[2, 3])  # global average pool spatial dims
        return out

    def _extract_features(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from all ensemble models."""
        feats = {}
        for name, model in self.ensemble.items():
            try:
                feats[name] = self._run_model(name, model, tensor)
            except Exception as e:
                self.log.debug(f"Skipping {name}: {e}")
        return feats

    def _compute_random_targets(self, original: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Pre-compute random-noise target features.

        WHY RANDOM TARGETS?
        Pushing the feature vector TOWARD a fixed random point means the
        deepfake 'sees' a completely different person — causing the output
        face to change or crash. Pushing 'away from original' is undirected
        and much weaker.
        """
        targets: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, model in self.ensemble.items():
                try:
                    noise = torch.randn_like(original)
                    targets[name] = self._run_model(name, model, noise).detach()
                except Exception:
                    pass
        return targets

    def run(
        self,
        original: torch.Tensor,
        preset: Dict,
        callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Execute noise-reduced PGD attack and return protected tensor.

        Args:
            original : [1, 3, H, W] float tensor in [0, 1]
            preset   : one of Config.PRESETS values
            callback : fn(step, total_steps, loss_value) for progress

        Returns:
            Protected tensor, same shape as original.
        """
        face_epsilon = preset.get("face_epsilon", preset.get("epsilon", 8/255))
        bg_epsilon   = preset.get("bg_epsilon",   face_epsilon * 0.15)
        steps        = preset["steps"]
        lr           = Config.ADAM_LR * preset["lr_scale"]
        jpeg_freq    = preset["jpeg_freq"]

        # Per-pixel epsilon map: face=low, background=high
        if Config.PERCEPTUAL_WEIGHT:
            eps_map = build_face_epsilon_map(original, face_epsilon, bg_epsilon)
        else:
            eps_map = torch.full_like(original, face_epsilon)

        perturbed = original.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([perturbed], lr=lr)
        targets   = self._compute_random_targets(original)

        self.log.info(
            f"  PGD: {steps} steps | face_eps={face_epsilon*255:.1f}/255 | "
            f"bg_eps={bg_epsilon*255:.1f}/255 | lr={lr:.5f}"
        )

        for step in range(steps):
            optimizer.zero_grad()

            # JPEG blend for social-media robustness
            if step > 0 and step % jpeg_freq == 0:
                with torch.no_grad():
                    jpeg_approx = jpeg_simulate(perturbed.detach(), Config.JPEG_QUALITY)
                inp = ((1 - Config.JPEG_BLEND) * perturbed +
                       Config.JPEG_BLEND * jpeg_approx.detach())
            else:
                inp = perturbed

            # Forward through ensemble
            curr_feats = self._extract_features(inp)

            # Feature loss toward random targets
            feat_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            count = 0
            for name, feat in curr_feats.items():
                if name in targets:
                    feat_loss = feat_loss + LossFunctions.combined(feat, targets[name])
                    count += 1

            if count == 0:
                raise RuntimeError("All feature extractors failed simultaneously.")

            feat_loss = feat_loss / count
            tv_loss   = total_variation_loss(perturbed - original)
            total_loss = feat_loss + Config.TV_WEIGHT * tv_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([perturbed], Config.GRAD_CLIP)
            optimizer.step()

            # Project to perceptually-weighted epsilon ball
            with torch.no_grad():
                delta = perturbed - original
                delta = torch.clamp(delta, -eps_map, eps_map)

                # Gaussian smoothing — pushes noise to low frequencies
                if Config.SMOOTH_SIGMA > 0 and (step + 1) % Config.SMOOTH_FREQ == 0:
                    delta = gaussian_blur_tensor(delta, sigma=Config.SMOOTH_SIGMA)
                    delta = torch.clamp(delta, -eps_map, eps_map)

                perturbed.data = torch.clamp(original + delta, 0.0, 1.0)

            if callback and (step + 1) % max(1, steps // 20) == 0:
                callback(step + 1, steps, total_loss.item())

        return perturbed.detach()
