"""
models/loader.py
Load ensemble models with validated fallbacks.

We use 4 diverse architectures for maximum attack transferability:
- ResNet50     : general low-level features
- EfficientNet : efficient multi-scale features
- ViT-B/16     : attention-based global features
- ConvNeXt     : modern CNN features

Diversity matters: an attack that fools multiple architectures is far
more likely to fool unknown real-world deepfake models.
"""

import gc
import logging
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models


class DependencyManager:
    """Check optional packages and provide fallbacks instead of crashing."""

    OPTIONAL = {
        "facenet_pytorch": "Face-aware attack (better transferability to face deepfakes)",
    }

    def __init__(self, logger: logging.Logger):
        self.log       = logger
        self.available: Dict = {}
        self._probe()

    def _probe(self):
        for pkg, purpose in self.OPTIONAL.items():
            try:
                self.available[pkg] = __import__(pkg)
                self.log.info(f"  [OK] {pkg} — {purpose}")
            except ImportError:
                self.available[pkg] = None
                self.log.info(f"  [--] {pkg} not found — {purpose} disabled")

    def has(self, pkg: str) -> bool:
        return bool(self.available.get(pkg))


class ModelLoader:
    """Load and manage ensemble models."""

    def __init__(
        self,
        device: torch.device,
        deps: DependencyManager,
        logger: logging.Logger,
    ):
        self.device = device
        self.deps   = deps
        self.log    = logger
        self._pool  = []

    def _register(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device).eval()
        self._pool.append(model)
        return model

    def load_ensemble(self) -> Dict[str, nn.Module]:
        """Load 4 diverse architectures."""
        ensemble: Dict[str, nn.Module] = {}

        # 1. ResNet-50
        try:
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            ensemble["resnet50"] = self._register(nn.Sequential(*list(m.children())[:-2]))
            self.log.info("  [OK] ResNet-50")
        except Exception as e:
            raise RuntimeError(f"ResNet-50 failed (required): {e}")

        # 2. EfficientNet-B0
        try:
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            ensemble["efficientnet"] = self._register(nn.Sequential(*list(m.children())[:-2]))
            self.log.info("  [OK] EfficientNet-B0")
        except Exception as e:
            self.log.warning(f"  EfficientNet failed ({e}), skipping")

        # 3. Vision Transformer
        try:
            m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            ensemble["vit"] = self._register(m)
            self.log.info("  [OK] ViT-B/16")
        except Exception as e:
            self.log.warning(f"  ViT failed ({e}), skipping")

        # 4. ConvNeXt-Tiny
        try:
            m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            ensemble["convnext"] = self._register(nn.Sequential(*list(m.children())[:-2]))
            self.log.info("  [OK] ConvNeXt-Tiny")
        except Exception as e:
            self.log.warning(f"  ConvNeXt failed ({e}), skipping")

        # 5. FaceNet (optional)
        if self.deps.has("facenet_pytorch"):
            try:
                from facenet_pytorch import InceptionResnetV1
                m = InceptionResnetV1(pretrained="vggface2")
                ensemble["facenet"] = self._register(m)
                self.log.info("  [OK] FaceNet (VGGFace2) — face-aware attack enabled")
            except Exception as e:
                self.log.warning(f"  FaceNet failed ({e}), skipping")

        if len(ensemble) < 2:
            raise RuntimeError("Need at least 2 models. Check PyTorch/torchvision install.")

        return ensemble

    def cleanup(self):
        """Free GPU memory."""
        for m in self._pool:
            try:
                m.zero_grad()
            except Exception:
                pass
            del m
        self._pool.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
