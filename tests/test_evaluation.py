"""Fast tests for the evaluation harness (no face photos or model downloads needed)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import Config
from evaluation.identity import ARCFACE_TEMPLATE, align_arcface
from evaluation.run_eval import l2_matched_noise, quality, render_summary, resample_only
from evaluation.transforms import TRANSFORMS, social_upload


def _img(w=640, h=480, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_transforms_keep_rgb(name):
    out = TRANSFORMS[name](_img())
    assert out.mode == "RGB"
    if name != "social":
        assert out.size == (640, 480)


def test_social_caps_long_side():
    assert max(social_upload(_img(2000, 1000)).size) == 1080
    assert social_upload(_img(800, 600)).size == (800, 600)


def test_resample_only_noop_for_small_and_lossy_for_large():
    small = _img(Config.IMAGE_SIZE, 300)
    assert np.array_equal(np.asarray(resample_only(small)), np.asarray(small))
    big = _img(2048, 1536)
    out = resample_only(big)
    assert out.size == big.size
    assert quality(big, out)["psnr_db"] < 40  # the resize round-trip is lossy


def test_l2_matched_noise_matches_energy():
    orig = _img(seed=1)
    a = np.asarray(orig, dtype=np.float32)
    prot = Image.fromarray(np.clip(a + np.random.default_rng(2).normal(0, 4, a.shape), 0, 255).astype(np.uint8))
    noise = l2_matched_noise(orig, prot, seed=3)
    rms = lambda x: np.sqrt(np.mean((np.asarray(x, np.float32) - a) ** 2))
    assert rms(noise) == pytest.approx(rms(prot), rel=0.1)  # clipping at 0/255 costs a little


def test_quality_identical_image():
    q = quality(_img(), _img())
    assert q["linf_255"] == 0 and q["ssim"] == pytest.approx(1.0, abs=1e-4)


def test_align_arcface_maps_template_to_itself():
    face = _img(112, 112)
    out = align_arcface(face, ARCFACE_TEMPLATE.copy())
    assert out.shape == (112, 112, 3)
    assert np.abs(out.astype(int) - np.asarray(face).astype(int)).mean() < 2


def test_render_summary_has_every_condition():
    rows = []
    for cond in ["clean", "resample_only", "random_noise_standard", "protected_standard"]:
        for t in TRANSFORMS:
            rows.append(dict(image="a.jpg", condition=cond, transform=t, evaluator="facenet-vggface2",
                             white_box=True, threshold=0.4, face_found=True,
                             similarity=0.9 if cond == "clean" else 0.3, protected=cond != "clean",
                             psnr_db=40.0, ssim=0.99, linf_255=3.0))
    md = render_summary(rows, SimpleNamespace(strengths=["standard"], seed=0))
    for cond in ["clean", "resample_only", "random_noise_standard", "protected_standard"]:
        assert f"| {cond} |" in md
    assert "WHITE-BOX" in md
    assert "| clean | 0% / 0.90" in md
