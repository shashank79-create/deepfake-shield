"""
engine/shield.py
GoldStandardEngine — orchestrates everything.

This is the top-level API. It:
1. Validates input
2. Loads ensemble models
3. Runs the PGD attack
4. Checks output quality (SSIM gate)
5. Returns results with full metrics

All the heavy lifting is done by the modules it imports.
This file just connects them in the right order.
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from config.settings import Config, select_device
from core.attack import PGDAttack
from models.loader import DependencyManager, ModelLoader
from utils.metrics import QualityMetrics
from utils.progress import ProgressBar
from utils.validator import InputValidator, ValidationError


def _make_logger(name: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        )
        logger.addHandler(h)
    return logger


class GoldStandardEngine:
    """
    Top-level protection API.

    Usage:
        with GoldStandardEngine() as engine:
            result = engine.protect("photo.jpg", strength="strong")
            if result["success"]:
                result["image"].save("protected.png")
                print(result["metrics"])
    """

    def __init__(self, verbose: bool = False, device: str = "auto"):
        self.log    = _make_logger("GoldStandard", verbose)
        self.verbose = verbose

        if device != "auto" or Config.DEVICE.type == "cpu":
            self.device = select_device(device)
            Config.DEVICE = self.device
        else:
            self.device = Config.DEVICE

        self.log.info(f"DeepFake Shield v5.0 | Device: {self.device}")

        self.validator = InputValidator(self.log)
        self.deps      = DependencyManager(self.log)
        self.loader    = ModelLoader(self.device, self.deps, self.log)
        self.qm        = QualityMetrics()

        self.log.info("Loading ensemble...")
        self.ensemble = self.loader.load_ensemble()
        self.attacker = PGDAttack(self.ensemble, self.device, self.log)
        self.log.info(f"Ready. Models: {list(self.ensemble.keys())}")

    # ── preprocessing ────────────────────────────────────────────────────────

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor, scaling down large images for memory safety."""
        w, h    = img.size
        longest = max(w, h)
        if longest > Config.IMAGE_SIZE:
            scale = Config.IMAGE_SIZE / longest
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            self.log.info(f"  Scaled {w}x{h} -> {img.size[0]}x{img.size[1]}")
        return transforms.ToTensor()(img).unsqueeze(0).to(self.device)

    def _to_pil(
        self, tensor: torch.Tensor, original_size: tuple = None
    ) -> Image.Image:
        """Convert tensor back to PIL, restoring original dimensions."""
        img = transforms.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))
        if original_size and img.size != original_size:
            img = img.resize(original_size, Image.LANCZOS)
        return img

    # ── public API ────────────────────────────────────────────────────────────

    def protect(
        self,
        image_path:  Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        strength:    str = "standard",
        verbose:     bool = False,
    ) -> Dict:
        """
        Protect an image from deepfake generation.

        Returns dict with keys:
            success     : bool
            image       : PIL.Image  (protected image)
            metrics     : dict       (SSIM, PSNR, latency, etc.)
            output_path : str | None
            error       : str        (only on failure)
        """
        import time
        t_start = time.time()

        try:
            # 1. Validate
            valid_path = self.validator.validate_path(image_path)
            strength   = self.validator.validate_strength(strength)
            preset     = Config.PRESETS[strength]
            pil_img    = self.validator.validate_image(Image.open(valid_path))

            self.log.info(f"Input   : {valid_path.name}  ({pil_img.size[0]}x{pil_img.size[1]})")
            self.log.info(f"Strength: {strength}  ({preset['description']})")

            # 2. Preprocess
            original = self._to_tensor(pil_img)
            self.validator.validate_tensor(original)

            # 3. Attack
            bar       = ProgressBar(preset["steps"]) if verbose else None
            protected = self.attacker.run(
                original, preset, callback=bar.update if bar else None
            )
            if bar:
                bar.finish()

            # 4. Quality gate — auto-reduce epsilon if SSIM too low
            ssim_val = self.qm.ssim(original, protected)
            if ssim_val < preset["ssim_target"]:
                self.log.warning(
                    f"SSIM {ssim_val:.4f} < target {preset['ssim_target']:.3f}. "
                    f"Retrying with reduced epsilon."
                )
                reduced = dict(preset)
                for key in ("epsilon", "face_epsilon", "bg_epsilon"):
                    if key in reduced:
                        reduced[key] *= Config.EPSILON_FALLBACK_SCALE
                protected = self.attacker.run(
                    original, reduced, callback=bar.update if bar else None
                )
                ssim_val = self.qm.ssim(original, protected)

            # 5. Save
            protected_pil = self._to_pil(protected, original_size=pil_img.size)
            if output_path:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                protected_pil.save(out, format="PNG")
                self.log.info(f"Saved   : {out}")

            # 6. Metrics
            buf = io.BytesIO()
            protected_pil.save(buf, format="PNG")
            face_eps = preset.get("face_epsilon", preset.get("epsilon", 8/255))
            bg_eps   = preset.get("bg_epsilon",   preset.get("epsilon", 8/255))

            metrics = {
                "strength":         strength,
                "steps":            preset["steps"],
                "face_epsilon_255": round(face_eps * 255, 2),
                "bg_epsilon_255":   round(bg_eps   * 255, 2),
                "ssim":             round(ssim_val, 5),
                "psnr_db":          round(self.qm.psnr(original, protected), 2),
                "max_pixel_change": round(self.qm.l_inf(original, protected), 3),
                "l2_norm":          round(self.qm.l2(original, protected), 4),
                "invisible":        self.qm.l_inf(original, protected) <= 16.0,
                "quality_ok":       ssim_val >= preset["ssim_target"],
                "protected_kb":     round(len(buf.getvalue()) / 1024, 1),
                "time_s":           round(time.time() - t_start, 2),
                "timestamp":        datetime.now().isoformat(),
                "device":           str(self.device),
                "models_used":      list(self.ensemble.keys()),
            }

            return {
                "success":     True,
                "image":       protected_pil,
                "metrics":     metrics,
                "output_path": str(output_path) if output_path else None,
            }

        except (ValidationError, RuntimeError) as e:
            self.log.error(f"{type(e).__name__}: {e}")
            return {"success": False, "error": str(e), "error_type": type(e).__name__}
        except Exception as e:
            self.log.exception("Unexpected error")
            return {"success": False, "error": str(e), "error_type": "UnexpectedError"}

    def cleanup(self):
        self.loader.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()
