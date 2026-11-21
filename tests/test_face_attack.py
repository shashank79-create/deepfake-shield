"""Fast tests for the v6 face attack helpers (no model weights needed)."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.face_attack import (ARCFACE_TEMPLATE, crop_aligned, crop_box, elliptical_mask, eot,
                              jpeg_straight_through)


def test_crop_aligned_template_is_identity():
    img = torch.rand(1, 3, 112, 112)
    out = crop_aligned(img, ARCFACE_TEMPLATE.copy(), 112)
    assert torch.allclose(out, img, atol=1e-3)


def test_crop_aligned_matches_cv2_warp():
    import cv2
    rng = np.random.default_rng(0)
    arr = rng.random((200, 240, 3)).astype(np.float32)
    lms = ARCFACE_TEMPLATE * 1.3 + np.array([40, 30], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(lms, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    ref = cv2.warpAffine(arr, M, (112, 112), flags=cv2.INTER_LINEAR)
    out = crop_aligned(torch.from_numpy(arr).permute(2, 0, 1)[None], lms, 112)[0].permute(1, 2, 0).numpy()
    assert np.abs(out - ref)[4:-4, 4:-4].mean() < 0.02


def test_crop_box_full_image_is_resize():
    img = torch.rand(1, 3, 64, 64)
    out = crop_box(img, torch.tensor([0.0, 0.0, 64.0, 64.0]), 64)
    assert torch.allclose(out, img, atol=1e-5)


def test_crop_box_picks_the_right_region():
    img = torch.zeros(1, 3, 100, 100)
    img[:, :, 20:60, 30:70] = 1.0
    out = crop_box(img, torch.tensor([30.0, 20.0, 70.0, 60.0]), 32)
    assert out[:, :, 2:-2, 2:-2].min() > 0.99


def test_crop_box_is_differentiable():
    img = torch.rand(1, 3, 50, 50, requires_grad=True)
    crop_box(img, torch.tensor([10.0, 10.0, 40.0, 40.0]), 16).sum().backward()
    assert img.grad is not None and img.grad.abs().sum() > 0


def test_jpeg_straight_through_forward_is_jpeg_backward_is_identity():
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    y = jpeg_straight_through(x, 50)
    assert (y - x).abs().max() > 0.01            # forward really compressed
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_eot_keeps_shape_and_gradient():
    cfg = dict(p_resize=1.0, resize_min=0.5, p_blur=1.0, p_jpeg=1.0, jpeg_min=60)
    x = torch.rand(1, 3, 40, 48, requires_grad=True)
    y = eot(x, cfg)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad.abs().sum() > 0


def test_elliptical_mask_is_one_on_face_zero_far_away():
    m = elliptical_mask(100, 100, np.array([40, 40, 60, 60], dtype=np.float32), "cpu")
    assert m[0, 0, 50, 50] == 1.0 and m[0, 0, 0, 0] == 0.0
    assert 0 <= m.min() and m.max() <= 1
