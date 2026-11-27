"""main.py CLI behaviour, with the protection engine stubbed out (no GPU or model weights)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as cli
from provenance.keys import generate_keypair, load_public
from provenance.manifest import verify_image


class FakeEngine:
    """Stands in for FaceShieldEngine: 'protects' by copying the image."""
    calls = []

    def __init__(self, verbose=False, device="auto"):
        pass

    def protect(self, image_path, output_path=None, strength="standard", verbose=False):
        FakeEngine.calls.append((str(image_path), str(output_path), strength))
        img = Image.open(image_path).convert("RGB")
        img.save(output_path, format="PNG")
        metrics = dict(attack="face-v6", strength=strength, ssim=0.99, psnr_db=40.0, max_pixel_change=12.0,
                       l2_norm=1.0, invisible=True, quality_ok=True, time_s=0.1, models_used=["stub"])
        return {"success": True, "image": img, "metrics": metrics, "output_path": str(output_path)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


@pytest.fixture()
def photo(tmp_path):
    p = tmp_path / "me.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (120, 100, 3), dtype=np.uint8)).save(p)
    return p


@pytest.fixture(autouse=True)
def stub_engine(monkeypatch):
    import engine.face_shield
    FakeEngine.calls = []
    monkeypatch.setattr(engine.face_shield, "FaceShieldEngine", FakeEngine)


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["main.py", *map(str, argv)])
    cli.main()


def test_defaults_are_v6_standard():
    a = cli.build_parser().parse_args(["x.jpg"])
    assert a.attack == "face" and a.strength == "standard" and a.sign_key is None


def test_default_output_name_next_to_input(monkeypatch, photo):
    run(monkeypatch, photo)
    _, out, strength = FakeEngine.calls[0]
    assert out == str(photo.parent / "me_protected.png") and strength == "standard"
    assert Path(out).exists()


def test_sign_key_produces_verifiable_output(monkeypatch, photo, tmp_path):
    priv, pub = generate_keypair(str(tmp_path / "k"), "me")
    out = tmp_path / "out.png"
    run(monkeypatch, photo, "--strength", "strong", "--output", out, "--sign-key", priv, "--signer-name", "Me")
    v = verify_image(str(out), trusted=[load_public(str(pub))])
    assert v.status == "AUTHENTIC"
    assert v.claims == {"protected_by": "DeepFake Shield (face-v6)", "strength": "strong"}


def test_bad_sign_key_fails_before_protecting(monkeypatch, photo, tmp_path):
    with pytest.raises(SystemExit):
        run(monkeypatch, photo, "--sign-key", tmp_path / "nope.key")
    assert FakeEngine.calls == []


def test_batch_protects_every_image(monkeypatch, photo, tmp_path):
    Image.open(photo).save(photo.parent / "second.png")
    out_dir = tmp_path / "prot"
    run(monkeypatch, "--batch", photo.parent, "--output-dir", out_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == ["me_protected.png", "second_protected.png"]
