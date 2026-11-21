"""
engine/face_shield.py
FaceShieldEngine - v6 face-targeted protection at full resolution.

Same public API as the legacy GoldStandardEngine:
    with FaceShieldEngine() as engine:
        result = engine.protect("photo.jpg", output_path="out.png", strength="standard")
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from config.settings import Config, select_device
from core.face_attack import FaceIdentityAttack, load_surrogates
from engine.shield import _make_logger
from utils.metrics import QualityMetrics
from utils.progress import ProgressBar
from utils.validator import InputValidator, ValidationError


class FaceShieldEngine:
    def __init__(self, verbose: bool = False, device: str = "auto", surrogates=None):
        self.log = _make_logger("FaceShield", verbose)
        self.device = Config.DEVICE if device == "auto" else select_device(device)
        self.validator = InputValidator(self.log)
        self.qm = QualityMetrics()
        self.surrogate_names = list(surrogates or Config.FACE_SURROGATES)
        self.log.info(f"DeepFake Shield v6 (face attack) | Device: {self.device}")
        self.attacker = FaceIdentityAttack(
            load_surrogates(self.device, self.surrogate_names, self.log), self.device, self.log)

    def protect(self, image_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                strength: str = "standard", verbose: bool = False, seed: int = 0) -> Dict:
        t0 = time.time()
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)

            path = self.validator.validate_path(image_path)
            if strength not in Config.FACE_PRESETS:
                raise ValidationError(f"Unknown strength '{strength}'. Choose from {list(Config.FACE_PRESETS)}")
            cfg = Config.FACE_PRESETS[strength]
            img = self.validator.validate_image(Image.open(path))

            bar = ProgressBar(cfg["steps"]) if verbose else None
            r = self.attacker.run(img, cfg, callback=bar.update if bar else None)
            if bar:
                bar.finish()

            arr = (r["tensor"][0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            out_img = Image.fromarray(arr)
            if output_path:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out_img.save(out, format="PNG")

            # quality at FULL resolution, on the quantised output actually saved
            a = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float() / 255
            b = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255
            ssim = self.qm.ssim(a, b)
            metrics = {
                "attack": "face-v6", "strength": strength, "steps": cfg["steps"],
                "face_epsilon_255": round(cfg["epsilon"] * 255, 2),
                "ssim": round(ssim, 5),
                "psnr_db": round(float(self.qm.psnr(a, b)), 2),
                "max_pixel_change": round(self.qm.l_inf(a, b), 3),
                "l2_norm": round(self.qm.l2(a, b), 4),
                "invisible": self.qm.l_inf(a, b) <= 16.0,
                "quality_ok": ssim >= 0.95,
                "faces": r["faces"],
                "surrogate_similarity": r["surrogate_sims"],
                "time_s": round(time.time() - t0, 2),
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "models_used": self.surrogate_names,
            }
            return {"success": True, "image": out_img, "metrics": metrics,
                    "output_path": str(output_path) if output_path else None}
        except (ValidationError, RuntimeError) as e:
            self.log.error(f"{type(e).__name__}: {e}")
            return {"success": False, "error": str(e), "error_type": type(e).__name__}
        except Exception as e:
            self.log.exception("Unexpected error")
            return {"success": False, "error": str(e), "error_type": "UnexpectedError"}

    def cleanup(self):
        self.attacker = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()
