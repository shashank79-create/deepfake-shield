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


class _ConfigMeta(type):
    """Resolve Config.DEVICE on first use, not at import time, so importing the
    package (tests, the provenance CLI, --help) never probes CUDA or prints."""

    @property
    def DEVICE(cls) -> torch.device:
        if cls._device is None:
            cls._device = select_device("auto")
        return cls._device

    @DEVICE.setter
    def DEVICE(cls, value: torch.device) -> None:
        cls._device = value


class Config(metaclass=_ConfigMeta):
    """Single source of truth for all parameters."""

    _device = None   # see Config.DEVICE (resolved lazily)

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

    # ── v6 face-targeted attack (core/face_attack.py) ────────────────────────
    # Surrogate face-recognition models. ArcFace is NOT allowed here: it is the
    # held-out evaluator. Add "facenet-casia-webface" for a second surrogate.
    FACE_SURROGATES = [
        "facenet-vggface2",                          # InceptionResnetV1, VGGFace2
        "facenet-casia-webface",                     # InceptionResnetV1, CASIA-WebFace
        "onnx:weights/buffalo_s/w600k_mbf.onnx",     # MobileFaceNet, WebFace600K (insightface)
    ]
    _FACE_EOT = {
        "eot_samples": 1, "jitter": 0.05,
        "p_resize": 0.5, "resize_min": 0.5,
        "p_blur": 0.3,
        "p_jpeg": 0.5, "jpeg_min": 60,
        "momentum": 1.0, "alpha_div": 10,
        # perceptual controls (see evaluation/tune_visibility.py rounds 1-3)
        "optimizer": "adam", "lambda_ssim": 10.0,
        "texture_mask": True, "tex_ref": 0.05, "tex_min": 0.3,
        "chroma_ratio": 0.5,
    }
    # Measured on held-out 1920px portraits (ArcFace after a social upload, face SSIM):
    #   8/255  ~0.75, 0.984 (near invisible, no protection)
    #   12/255 ~0.51, 0.970 (faint, rarely protects)
    #   16/255 ~0.31, 0.960 (visible, protects ~2/3)
    #   20/255 ~0.20, 0.950 (clearly visible, protects ~2/3+)
    FACE_PRESETS: Dict = {
        "subtle":   {"steps": 200, "epsilon": 8/255,  **_FACE_EOT},
        "standard": {"steps": 300, "epsilon": 12/255, **_FACE_EOT},
        "strong":   {"steps": 300, "epsilon": 16/255, **_FACE_EOT},
        "maximum":  {"steps": 400, "epsilon": 20/255, **_FACE_EOT},
    }

    # ── legacy v5 ImageNet-ensemble attack ───────────────────────────────────
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
