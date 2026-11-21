"""
core/face_attack.py
Face-targeted identity attack (v6).

Why this replaces the legacy ImageNet-ensemble attack:
  * Face-swap / face-recognition pipelines detect, crop and align the face
    and throw the rest of the photo away. So the perturbation lives ON the
    face region (soft elliptical mask), not in the background.
  * Surrogates are face-recognition models fed exactly what they expect:
    a differentiable crop of the face, correctly normalised.
  * Untargeted identity loss: push the embedding away from the person's own
    clean embedding (cosine), instead of toward the features of a noise image.
  * Robustness via Expectation over Transformation: every step sees a random
    box jitter, down/up resize, blur and a REAL JPEG round-trip with a
    straight-through gradient. Momentum (MI-FGSM) improves transfer.
  * Full resolution: the perturbation is optimised at <=WORK_MAX px on the
    face region, upsampled, and added to the untouched full-res original.
"""

import io
import logging
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from core.noise import gaussian_blur_tensor

WORK_MAX = 512        # max side of the optimisation region
REGION_SCALE = 1.8    # region side = REGION_SCALE * face box side
MIN_FACE_PROB = 0.90


# ── surrogates ────────────────────────────────────────────────────────────────

@dataclass
class Surrogate:
    name: str
    model: nn.Module
    size: int
    preprocess: Callable[[torch.Tensor], torch.Tensor]   # [0,1] RGB -> model input
    align: str = "box"   # "box": MTCNN box crop (FaceNet) | "arcface": 5-landmark similarity alignment


# Canonical 112x112 ArcFace landmark template (insightface)
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def load_surrogates(device: torch.device, names: List[str], log: logging.Logger) -> List[Surrogate]:
    """Face-recognition surrogates.
      facenet-vggface2 / facenet-casia-webface : InceptionResnetV1 (facenet-pytorch)
      onnx:<path>                              : insightface-style recognizer (112x112 aligned),
                                                 converted to PyTorch with onnx2torch
    The held-out evaluator (buffalo_l w600k_r50) must never be listed here."""
    out = []
    for n in names:
        if n.startswith("facenet-"):
            from facenet_pytorch import InceptionResnetV1
            m = InceptionResnetV1(pretrained=n.split("-", 1)[1]).eval().to(device)
            # facenet-pytorch: fixed_image_standardization on 0..255 => (x*255-127.5)/128
            s = Surrogate(n, m, 160, lambda x: (x * 255.0 - 127.5) / 128.0, "box")
        elif n.startswith("onnx:"):
            path = Path(n.split(":", 1)[1])
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            if "w600k_r50" in path.name:
                raise ValueError("w600k_r50 is the held-out evaluator; attacking it invalidates the eval")
            if not Path(path).exists():
                log.warning(f"  [--] surrogate {n} not found, skipping")
                continue
            import onnx
            import onnx2torch
            m = onnx2torch.convert(onnx.load(str(path))).eval().to(device)
            s = Surrogate(path.stem, m, 112, lambda x: x * 2.0 - 1.0, "arcface")
        else:
            raise ValueError(f"Unknown surrogate '{n}'")
        for p in s.model.parameters():
            p.requires_grad_(False)
        out.append(s)
        log.info(f"  [OK] surrogate {s.name} ({s.align} crop)")
    if not out:
        raise RuntimeError("No surrogate face models could be loaded.")
    return out


# ── differentiable helpers ────────────────────────────────────────────────────

def crop_box(img: torch.Tensor, box: torch.Tensor, size: int) -> torch.Tensor:
    """Differentiably crop box [x1,y1,x2,y2] (pixel coords of img) and resize to size x size."""
    _, _, H, W = img.shape
    x1, y1, x2, y2 = box
    theta = torch.zeros(1, 2, 3, device=img.device, dtype=img.dtype)
    theta[0, 0, 0] = (x2 - x1) / W
    theta[0, 0, 2] = (x1 + x2) / W - 1
    theta[0, 1, 1] = (y2 - y1) / H
    theta[0, 1, 2] = (y1 + y2) / H - 1
    grid = F.affine_grid(theta, (1, 3, size, size), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def crop_aligned(img: torch.Tensor, landmarks: np.ndarray, size: int) -> torch.Tensor:
    """Differentiably warp the face onto the ArcFace template (similarity transform from 5 landmarks)."""
    import cv2
    _, _, H, W = img.shape
    tmpl = ARCFACE_TEMPLATE * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), tmpl, method=cv2.LMEDS)
    M3 = np.vstack([M, [0, 0, 1]])                       # input px -> output px
    p_out = np.array([[size / 2, 0, size / 2 - 0.5], [0, size / 2, size / 2 - 0.5], [0, 0, 1]])
    n_in = np.array([[2 / W, 0, 1 / W - 1], [0, 2 / H, 1 / H - 1], [0, 0, 1]])
    theta = (n_in @ np.linalg.inv(M3) @ p_out)[:2]       # output normalised -> input normalised
    theta = torch.tensor(theta, dtype=img.dtype, device=img.device).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 3, size, size), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def jpeg_straight_through(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Real JPEG in the forward pass, identity gradient in the backward pass (BPDA)."""
    with torch.no_grad():
        arr = (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        y = torch.from_numpy(np.array(Image.open(buf).convert("RGB"))).to(x.device)
        y = y.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x + (y - x).detach()


def eot(x: torch.Tensor, cfg: Dict) -> torch.Tensor:
    """One random draw of real-world degradations."""
    _, _, h, w = x.shape
    if random.random() < cfg["p_resize"]:
        s = random.uniform(cfg["resize_min"], 1.0)
        x = F.interpolate(x, size=(max(8, int(h * s)), max(8, int(w * s))), mode="bilinear",
                          align_corners=False, antialias=True)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    if random.random() < cfg["p_blur"]:
        x = gaussian_blur_tensor(x, sigma=random.uniform(0.4, 1.0))
    if random.random() < cfg["p_jpeg"]:
        x = jpeg_straight_through(x, random.randint(cfg["jpeg_min"], 95))
    return x


def elliptical_mask(h: int, w: int, box: np.ndarray, device) -> torch.Tensor:
    """1.0 on the face, fading smoothly to 0 just outside the box."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rx, ry = (box[2] - box[0]) / 2 * 1.15, (box[3] - box[1]) / 2 * 1.25
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    d = torch.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    return torch.clamp(1.6 - d, 0.0, 1.0).view(1, 1, h, w)  # 1 inside d<=0.6.. fades to 0 at d=1.6


# ── perceptual helpers ────────────────────────────────────────────────────────

def ycbcr_to_rgb(d: torch.Tensor) -> torch.Tensor:
    """Map a perturbation expressed as (Y, Cb, Cr) offsets to RGB offsets (BT.601)."""
    y, cb, cr = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    return torch.cat([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], dim=1)


def texture_map(img: torch.Tensor, size: tuple, ref: float, floor: float) -> torch.Tensor:
    """Visual masking: busy texture (hair, beard, edges) hides noise; smooth skin does not.
    Returns [1,1,h,w] in [floor, 1], from local luminance std (7x7)."""
    gray = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
    gray = F.interpolate(gray, size=size, mode="bilinear", align_corners=False, antialias=True)
    mu = F.avg_pool2d(gray, 7, 1, 3, count_include_pad=False)
    var = F.avg_pool2d(gray * gray, 7, 1, 3, count_include_pad=False) - mu * mu
    t = (var.clamp_min(0).sqrt() / ref).clamp(floor, 1.0)
    return gaussian_blur_tensor(t, sigma=1.5).clamp(floor, 1.0)


def ssim_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable SSIM (11x11 Gaussian, sigma 1.5)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    coords = torch.arange(11, dtype=a.dtype, device=a.device) - 5
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    k = g.outer(g).view(1, 1, 11, 11).repeat(a.shape[1], 1, 1, 1)
    conv = lambda x: F.conv2d(x, k, padding=5, groups=a.shape[1])
    mu1, mu2 = conv(a), conv(b)
    s1, s2, s12 = conv(a * a) - mu1 ** 2, conv(b * b) - mu2 ** 2, conv(a * b) - mu1 * mu2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean()


# ── attack ────────────────────────────────────────────────────────────────────

class FaceIdentityAttack:
    """Untargeted, EOT-robust MI-FGSM against an ensemble of face-recognition surrogates."""

    def __init__(self, surrogates: List[Surrogate], device: torch.device, logger: logging.Logger):
        self.surrogates = surrogates
        self.device = device
        self.log = logger
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, img: Image.Image) -> List[tuple]:
        """[(box [4], landmarks [5,2]), ...] for every confident face."""
        boxes, probs, lms = self.mtcnn.detect(img, landmarks=True)
        if boxes is None:
            return []
        return [(np.asarray(b, dtype=np.float32), np.asarray(l, dtype=np.float32))
                for b, p, l in zip(boxes, probs, lms) if p is not None and float(p) >= MIN_FACE_PROB]

    def _embed(self, region: torch.Tensor, box: torch.Tensor, lms: np.ndarray,
               cfg: Optional[Dict] = None) -> List[torch.Tensor]:
        side0 = float((box[2] - box[0] + box[3] - box[1]) / 2)
        embs = []
        for s in self.surrogates:
            # Anti-aliased pre-downscale so the face is ~1.25x the model's input size. Without
            # this, grid_sample on a large face only reads a sparse lattice of pixels, so most
            # of the perturbation is invisible to the surrogate (weak attack on high-res photos).
            f = min(1.0, 1.25 * s.size / side0)
            reg, bx, lm = region, box, lms
            if f < 1.0:
                h, w = region.shape[2:]
                reg = F.interpolate(region, size=(max(8, round(h * f)), max(8, round(w * f))),
                                    mode="bilinear", align_corners=False, antialias=True)
                fy, fx = reg.shape[2] / h, reg.shape[3] / w
                bx = box * torch.tensor([fx, fy, fx, fy], device=box.device, dtype=box.dtype)
                lm = lms * np.array([fx, fy], dtype=np.float32)
            side = side0 * min(f, 1.0)
            if s.align == "arcface":
                l = lm
                if cfg is not None:  # random landmark jitter (shift + scale about the centre)
                    sc = 1 + random.uniform(-cfg["jitter"], cfg["jitter"])
                    sh = np.random.uniform(-1, 1, 2) * cfg["jitter"] * side
                    c = lm.mean(0)
                    l = (lm - c) * sc + c + sh + np.random.normal(0, 0.01 * side, lm.shape)
                face = crop_aligned(reg, l, s.size)
            else:
                b = bx
                if cfg is not None:  # random box jitter: shift and scale a few %
                    sh = torch.tensor([random.uniform(-1, 1), random.uniform(-1, 1)], device=bx.device) * cfg["jitter"] * side
                    sc = 1 + random.uniform(-cfg["jitter"], cfg["jitter"])
                    c = torch.stack([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2]) + sh
                    half = torch.stack([(bx[2] - bx[0]) / 2, (bx[3] - bx[1]) / 2]) * sc
                    b = torch.cat([c - half, c + half])
                face = crop_box(reg, b, s.size)
            embs.append(F.normalize(s.model(s.preprocess(face)).flatten(1), dim=-1))
        return embs

    def attack_face(self, full: torch.Tensor, box: np.ndarray, lms: np.ndarray, cfg: Dict,
                    callback: Optional[Callable] = None) -> Dict:
        """Perturb one face in `full` ([1,3,H,W] in [0,1], full resolution).
        Returns {'image': new full tensor, 'adv_sims': per-surrogate similarity after attack}."""
        _, _, H, W = full.shape
        bw, bh = box[2] - box[0], box[3] - box[1]
        side = REGION_SCALE * max(bw, bh)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        region = full[:, :, y0:y1, x0:x1]
        rh, rw = region.shape[2:]
        s = min(1.0, WORK_MAX / max(rh, rw))
        wh, ww = max(8, round(rh * s)), max(8, round(rw * s))

        box_r = torch.tensor([box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0],
                             device=self.device, dtype=torch.float32)
        lms_r = lms - np.array([x0, y0], dtype=np.float32)
        # Perturbation parameter: optionally coarser than the work res (smoother noise),
        # in YCbCr so colour (chroma) noise can be limited separately from brightness noise.
        lowres = max(1, int(cfg.get("lowres", 1)))
        ph, pw = max(8, wh // lowres), max(8, ww // lowres)
        eps_map = cfg["epsilon"] * elliptical_mask(ph, pw, (box_r * s / lowres).cpu().numpy(), self.device)
        if cfg.get("texture_mask", False):
            eps_map = eps_map * texture_map(region, (ph, pw), cfg["tex_ref"], cfg["tex_min"])
        chroma = float(cfg.get("chroma_ratio", 1.0))
        bound = torch.cat([eps_map, eps_map * chroma, eps_map * chroma], dim=1)  # [1,3,ph,pw] Y,Cb,Cr
        alpha = cfg["epsilon"] / cfg["alpha_div"]
        lam_ssim = float(cfg.get("lambda_ssim", 0.0))
        lam_lf = float(cfg.get("lambda_lowfreq", 0.0))
        lf_sigma = max(1.0, (rw / 512.0) * 4.0)   # ~4px at 512px region, scales with resolution

        with torch.no_grad():
            clean = [e.detach() for e in self._embed(region, box_r, lms_r)]

        param = torch.zeros(1, 3, ph, pw, device=self.device, requires_grad=True)
        use_adam = cfg.get("optimizer", "mifgsm") == "adam"
        opt = torch.optim.Adam([param], lr=alpha) if use_adam else None
        momentum = torch.zeros_like(param)

        # Hard RGB cap: no pixel channel ever moves more than its epsilon, whatever the YCbCr mix.
        rgb_cap = F.interpolate(eps_map, size=(rh, rw), mode="bilinear", align_corners=False)

        def render(p):
            d = F.interpolate(ycbcr_to_rgb(p), size=(rh, rw), mode="bilinear", align_corners=False)
            d = torch.max(torch.min(d, rgb_cap), -rgb_cap)
            return (region + d).clamp(0, 1)

        for step in range(cfg["steps"]):
            adv = render(param)
            id_loss = 0.0
            for _ in range(cfg["eot_samples"]):
                embs = self._embed(eot(adv, cfg), box_r, lms_r, cfg)
                id_loss = id_loss + sum((e * c).sum() for e, c in zip(embs, clean)) / len(embs)
            id_loss = id_loss / cfg["eot_samples"]    # mean cosine similarity: minimise
            loss = id_loss + (lam_ssim * (1 - ssim_torch(adv, region)) if lam_ssim else 0.0)
            if lam_lf:
                # Penalise smooth, blotchy shading (low-frequency part of the change): it is the
                # most noticeable artefact on skin. Pushes the change into fine texture instead.
                diff = adv - region
                loss = loss + lam_lf * gaussian_blur_tensor(diff, sigma=lf_sigma).abs().mean() * 255.0 / 16.0
            grad, = torch.autograd.grad(loss, param)
            with torch.no_grad():
                if use_adam:
                    param.grad = grad
                    opt.step()
                else:
                    momentum = cfg["momentum"] * momentum + grad / (grad.abs().mean() + 1e-12)
                    param -= alpha * momentum.sign()
                param.copy_(torch.max(torch.min(param, bound), -bound))
            if callback and (step + 1) % max(1, cfg["steps"] // 20) == 0:
                callback(step + 1, cfg["steps"], id_loss.item())

        with torch.no_grad():
            adv_region = render(param)
            adv_sims = {s.name: round((e * c).sum().item(), 4)
                        for s, e, c in zip(self.surrogates, self._embed(adv_region, box_r, lms_r), clean)}
            out = full.clone()
            out[:, :, y0:y1, x0:x1] = adv_region
        return {"image": out, "adv_sims": adv_sims}

    def run(self, img: Image.Image, cfg: Dict, callback: Optional[Callable] = None) -> Dict:
        faces = self.detect(img)
        if not faces:
            raise RuntimeError("No face detected - nothing to protect.")
        full = torch.from_numpy(np.array(img.convert("RGB"))).to(self.device)
        full = full.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        sims = []
        for i, (b, l) in enumerate(faces):
            self.log.info(f"  Face {i+1}/{len(faces)}: box {np.round(b).astype(int).tolist()}")
            r = self.attack_face(full, b, l, cfg, callback)
            full = r["image"]
            sims.append(r["adv_sims"])
        return {"tensor": full, "faces": len(faces), "surrogate_sims": sims}
