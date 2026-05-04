"""
config/settings.py
All tunable parameters live here. Never hardcode values elsewhere.
"""

import torch
from typing import Dict


def select_device(requested: str = "auto") -> torch.device:
    """
    Resolve compute device with clear diagnostics.
    requested: 'auto' | 'cuda' | 'cuda:0' | 'cpu'
    """
    if requested == "cpu":
        print("  [Device] Forced CPU mode.")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if not hasattr(torch.version, "cuda") or torch.version.cuda is None:
            reason = (
                "PyTorch was installed WITHOUT CUDA support.\n"
                "  Fix: pip uninstall torch torchvision -y\n"
                "       pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu118"
            )
        else:
            reason = (
                f"PyTorch has CUDA {torch.version.cuda} but no GPU detected.\n"
                "  Check: run nvidia-smi to verify your driver."
            )
        print(f"  [Device] WARNING: GPU unavailable. Falling back to CPU.\n  {reason}")
        return torch.device("cpu")

    device = torch.device(requested if requested.startswith("cuda") else "cuda:0")
    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    mem_gb = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
    print(f"  [Device] GPU: {name}  ({mem_gb:.1f} GB VRAM)  [{device}]")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    return device


class Config:
    """Single source of truth for all parameters."""

    DEVICE = select_device("auto")

    # Image sizes
    IMAGE_SIZE = 512    # Internal cap for memory safety
    MODEL_SIZE = 224    # All ensemble models expect 224x224
    FACE_SIZE  = 160    # FaceNet input size

    # Optimizer
    ADAM_LR    = 1e-3
    GRAD_CLIP  = 1.0

    # JPEG simulation (social media robustness)
    JPEG_QUALITY = 75
    JPEG_BLEND   = 0.3

    # Quality thresholds
    MIN_SSIM               = 0.97
    EPSILON_FALLBACK_SCALE = 0.70

    # Noise reduction
    SMOOTH_SIGMA      = 1.5
    SMOOTH_FREQ       = 5
    PERCEPTUAL_WEIGHT = True
    TV_WEIGHT         = 0.08

    # Protection presets
    # face_epsilon = perturbation ON face skin  (LOW  = clean natural look)
    # bg_epsilon   = perturbation on background (HIGH = strong hidden attack)
    PRESETS: Dict = {
        "subtle": {
            "steps": 100, "face_epsilon": 2/255, "bg_epsilon": 8/255,
            "lr_scale": 0.5, "jpeg_freq": 10, "ssim_target": 0.985,
            "description": "Invisible on face. Light context disruption.",
        },
        "standard": {
            "steps": 200, "face_epsilon": 3/255, "bg_epsilon": 12/255,
            "lr_scale": 1.0, "jpeg_freq": 20, "ssim_target": 0.980,
            "description": "Clean face. Strong background attack. Recommended.",
        },
        "strong": {
            "steps": 300, "face_epsilon": 4/255, "bg_epsilon": 16/255,
            "lr_scale": 1.4, "jpeg_freq": 15, "ssim_target": 0.975,
            "description": "Perfectly clean face. Very aggressive context disruption.",
        },
        "maximum": {
            "steps": 400, "face_epsilon": 5/255, "bg_epsilon": 20/255,
            "lr_scale": 1.8, "jpeg_freq": 10, "ssim_target": 0.968,
            "description": "Max strength. Face untouched. Deepfake context destroyed.",
        },
    }
