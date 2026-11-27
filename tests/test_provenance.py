"""Provenance signing: the scenarios that matter (no network, no model weights)."""

import base64
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.transforms import TRANSFORMS
from provenance.cli import main as cli
from provenance.keys import generate_keypair, load_private, load_public
from provenance.manifest import SIDECAR_SUFFIX, sign_image, verify_image


def _photo(w=900, h=1100, seed=0) -> Image.Image:
    """Smooth 'portrait-like' test image: gradients, an ellipse 'face', some texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    base = np.stack([0.3 + 0.4 * xx, 0.4 + 0.3 * yy, 0.6 - 0.2 * xx], -1)
    base += rng.normal(0, 0.02, base.shape)
    im = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.ellipse([300, 250, 600, 650], fill=(200, 150, 120))
    d.ellipse([370, 380, 410, 410], fill=(40, 30, 30))
    d.ellipse([490, 380, 530, 410], fill=(40, 30, 30))
    d.rectangle([400, 520, 500, 545], fill=(150, 60, 60))
    return im.filter(ImageFilter.GaussianBlur(1.5))


@pytest.fixture()
def keys(tmp_path):
    priv, pub = generate_keypair(str(tmp_path / "k"), "me")
    other_priv, other_pub = generate_keypair(str(tmp_path / "k"), "other")
    return load_private(str(priv)), load_public(str(pub)), load_private(str(other_priv)), load_public(str(other_pub))


@pytest.fixture()
def signed_png(tmp_path, keys):
    sk = keys[0]
    src = tmp_path / "photo.png"
    _photo().save(src)
    out = tmp_path / "photo_signed.png"
    sign_image(str(src), sk, str(out), "Test User", {"protected_by": "unit-test"})
    return out


def test_exact_copy_is_authentic(signed_png, keys):
    v = verify_image(str(signed_png), trusted=[keys[1]])
    assert v.status == "AUTHENTIC"
    assert v.signer["name"] == "Test User" and v.claims["protected_by"] == "unit-test"
    assert v.source == "embedded"


def test_recompressed_copy_with_stripped_metadata_is_reencoded_via_registry(tmp_path, signed_png, keys):
    reg = tmp_path / "registry"
    reg.mkdir()
    shutil.copy(str(signed_png) + SIDECAR_SUFFIX, reg)
    up = tmp_path / "upload.jpg"
    TRANSFORMS["social"](Image.open(signed_png)).save(up, quality=85)   # metadata stripped
    assert verify_image(str(up), trusted=[keys[1]]).status == "UNSIGNED"
    v = verify_image(str(up), trusted=[keys[1]], registry=str(reg))
    assert v.status == "REENCODED", v.detail
    assert v.source.startswith("registry:")


def test_face_edit_is_modified_and_localised(tmp_path, signed_png, keys):
    reg = tmp_path / "registry"
    reg.mkdir()
    shutil.copy(str(signed_png) + SIDECAR_SUFFIX, reg)
    im = Image.open(signed_png).convert("RGB")
    ImageDraw.Draw(im).ellipse([340, 300, 560, 600], fill=(170, 120, 100))   # "new face"
    fake = tmp_path / "fake.jpg"
    TRANSFORMS["social"](im).save(fake, quality=85)
    v = verify_image(str(fake), trusted=[keys[1]], registry=str(reg))
    assert v.status == "MODIFIED"
    rows = [r for r, _ in v.changed_blocks]
    cols = [c for _, c in v.changed_blocks]
    # every changed block lies in the face area (image is 16x16 blocks; face ~ rows 3-9, cols 5-10)
    assert min(rows) >= 2 and max(rows) <= 10 and min(cols) >= 4 and max(cols) <= 11


def test_unrelated_image_does_not_match_registry(tmp_path, signed_png, keys):
    reg = tmp_path / "registry"
    reg.mkdir()
    shutil.copy(str(signed_png) + SIDECAR_SUFFIX, reg)
    other = tmp_path / "other.jpg"
    Image.fromarray(np.random.default_rng(5).integers(0, 255, (1100, 900, 3), dtype=np.uint8)).save(other)
    assert verify_image(str(other), registry=str(reg)).status == "UNSIGNED"


def test_wrong_trusted_key_is_untrusted_signer(signed_png, keys):
    assert verify_image(str(signed_png), trusted=[keys[3]]).status == "UNTRUSTED_SIGNER"


def test_forged_claim_breaks_signature(tmp_path, signed_png, keys):
    signed = json.loads(Path(str(signed_png) + SIDECAR_SUFFIX).read_text())
    signed["manifest"]["signer"]["name"] = "Someone Famous"
    jpg = tmp_path / "x.jpg"
    Image.open(signed_png).convert("RGB").save(jpg)
    Path(str(jpg) + SIDECAR_SUFFIX).write_text(json.dumps(signed))
    assert verify_image(str(jpg)).status == "INVALID_SIGNATURE"


def test_attacker_resigning_with_own_key_is_untrusted(tmp_path, signed_png, keys):
    fake = tmp_path / "fake.png"
    Image.open(signed_png).convert("RGB").save(fake)
    sign_image(str(fake), keys[2], str(fake), "Test User")        # attacker's key, victim's name
    assert verify_image(str(fake), trusted=[keys[1]]).status == "UNTRUSTED_SIGNER"


def test_swapping_in_attacker_public_key_is_detected(tmp_path, signed_png, keys):
    """Keep the victim's signature but claim the attacker's key: signature check must fail."""
    signed = json.loads(Path(str(signed_png) + SIDECAR_SUFFIX).read_text())
    from provenance.keys import raw_public
    signed["manifest"]["signer"]["public_key"] = base64.b64encode(raw_public(keys[3])).decode()
    jpg = tmp_path / "y.jpg"
    Image.open(signed_png).convert("RGB").save(jpg)
    Path(str(jpg) + SIDECAR_SUFFIX).write_text(json.dumps(signed))
    assert verify_image(str(jpg)).status == "INVALID_SIGNATURE"


def test_keygen_refuses_to_overwrite(tmp_path):
    generate_keypair(str(tmp_path), "me")
    with pytest.raises(FileExistsError):
        generate_keypair(str(tmp_path), "me")


def test_cli_verify_folder_reports_each_file_and_worst_exit(tmp_path, signed_png, keys):
    folder = tmp_path / "inbox"
    folder.mkdir()
    shutil.copy(signed_png, folder / "good.png")
    Image.open(signed_png).convert("RGB").save(folder / "unsigned.jpg")
    out_csv = tmp_path / "report.csv"
    code = cli(["verify", str(folder), "--csv", str(out_csv)])
    assert code == 1                                     # worst verdict: UNSIGNED
    report = out_csv.read_text()
    assert "good.png,AUTHENTIC" in report and "unsigned.jpg,UNSIGNED" in report


def test_cli_round_trip_exit_codes(tmp_path):
    assert cli(["keygen", "--out", str(tmp_path / "k"), "--name", "me"]) == 0
    src = tmp_path / "p.jpg"
    _photo().save(src, quality=95)
    assert cli(["sign", str(src), "--key", str(tmp_path / "k/me.key"), "--name", "Me"]) == 0
    signed = tmp_path / "p_signed.png"
    assert signed.exists()
    assert cli(["verify", str(signed), "--trust", str(tmp_path / "k/me.pub")]) == 0
    assert cli(["verify", str(src)]) == 1    # the unsigned original
