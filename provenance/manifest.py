"""
provenance/manifest.py
Sign and verify images.

A signed image carries a manifest (who signed it, when, a fingerprint of the
pixels, optional claims such as "protected by DeepFake Shield v6") plus an
Ed25519 signature over that manifest. The manifest is embedded in PNG files
(iTXt chunk) and can also be written as a sidecar <image>.provenance.json.

Social networks strip embedded metadata, so verification can also recover the
manifest from a *registry*: a folder of your sidecar files. That lets you check
"is this re-uploaded copy really my photo, and is anything changed in it?".

What verification can and cannot tell you:
  AUTHENTIC          signature valid, pixels identical to what was signed
  REENCODED          signature valid, content matches (recompressed/resized copy)
  MODIFIED           signature valid, but regions differ from what was signed
  UNTRUSTED_SIGNER   validly signed, but not by the key you trust
  INVALID_SIGNATURE  manifest was altered or forged
  UNSIGNED           no manifest found (most photos on the internet)
UNSIGNED does NOT mean fake; it only means there is no proof either way.
"""

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from provenance.fingerprint import block_differences, decode_thumbnail, fingerprint, pixel_sha256
from provenance.keys import key_id, public_from_b64, raw_public

FORMAT = "deepfake-shield-provenance/1"
PNG_KEY = "dfshield-provenance"
SIDECAR_SUFFIX = ".provenance.json"

# Block threshold (0-255 mean abs diff on the softened 128px thumbnail), calibrated with
# evaluation/calibrate_provenance.py on 12 real photos:
#   JPEG 95/75/50, social upload, half-size resize   max 1.9
#   blurred face / small mouth edit                  min 3.2 (mean ~8.5)
#   crude face swap                                  min 41
#   +10% brightness filter                           0.6 - 15.6 (clipping) -> often MODIFIED, it is an edit
BLOCK_CHANGED = 3.0


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


# ── signing ───────────────────────────────────────────────────────────────────

def build_manifest(img: Image.Image, sk: Ed25519PrivateKey, signer_name: str = "",
                   claims: Optional[Dict] = None) -> Dict:
    pk = sk.public_key()
    manifest = {
        "format": FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signer": {"name": signer_name, "key_id": key_id(pk),
                   "public_key": base64.b64encode(raw_public(pk)).decode()},
        "image": fingerprint(img),
        "claims": claims or {},
    }
    return {"manifest": manifest, "signature": base64.b64encode(sk.sign(_canonical(manifest))).decode()}


def sign_image(src: str, sk: Ed25519PrivateKey, out: Optional[str] = None, signer_name: str = "",
               claims: Optional[Dict] = None, sidecar: bool = True) -> Dict:
    """Sign `src`. PNG output gets the manifest embedded; a sidecar JSON is written too.
    Non-PNG inputs are re-saved as PNG when `out` ends in .png, else only the sidecar is written."""
    img = Image.open(src)
    img.load()
    rgb = img.convert("RGB")
    signed = build_manifest(rgb, sk, signer_name, claims)
    out_path = Path(out) if out else Path(src)
    if out_path.suffix.lower() == ".png":
        info = PngInfo()
        info.add_itxt(PNG_KEY, json.dumps(signed), zip=True)
        rgb.save(out_path, format="PNG", pnginfo=info)
    elif out and Path(out) != Path(src):
        raise ValueError("Signed output must be .png (JPEG re-encoding would change the pixels).")
    if sidecar:
        Path(str(out_path) + SIDECAR_SUFFIX).write_text(json.dumps(signed, indent=2))
    return signed


# ── verification ──────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    status: str
    detail: str
    signer: Dict = field(default_factory=dict)
    claims: Dict = field(default_factory=dict)
    created: str = ""
    source: str = ""                       # embedded | sidecar | registry:<file>
    changed_blocks: List[List[int]] = field(default_factory=list)   # [row, col] in a GRIDxGRID map
    block_map: Optional[np.ndarray] = None
    max_block_diff: float = 0.0


def read_embedded(path: str) -> Optional[Dict]:
    try:
        with Image.open(path) as im:
            raw = getattr(im, "text", {}).get(PNG_KEY) if hasattr(im, "text") else None
            raw = raw or im.info.get(PNG_KEY)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def read_sidecar(path: str) -> Optional[Dict]:
    p = Path(str(path) + SIDECAR_SUFFIX)
    return json.loads(p.read_text()) if p.exists() else None


def check_signature(signed: Dict) -> Optional[Ed25519PublicKey]:
    """Return the signer's public key if the signature over the manifest is valid, else None."""
    try:
        m = signed["manifest"]
        pk = public_from_b64(m["signer"]["public_key"])
        pk.verify(base64.b64decode(signed["signature"]), _canonical(m))
        if m.get("format") != FORMAT or key_id(pk) != m["signer"]["key_id"]:
            return None
        return pk
    except (InvalidSignature, KeyError, ValueError, TypeError):
        return None


def compare_content(signed: Dict, img: Image.Image) -> Dict:
    fp = signed["manifest"]["image"]
    if pixel_sha256(img) == fp["pixel_sha256"]:
        return {"exact": True, "block_map": None, "changed": [], "max": 0.0}
    bm = block_differences(decode_thumbnail(fp["thumbnail"]["png_b64"]), img)
    changed = np.argwhere(bm > BLOCK_CHANGED).tolist()
    return {"exact": False, "block_map": bm, "changed": changed, "max": float(bm.max())}


def verify_image(path: str, trusted: Optional[List[Ed25519PublicKey]] = None,
                 registry: Optional[str] = None) -> Verdict:
    img = Image.open(path).convert("RGB")
    signed, source = read_embedded(path), "embedded"
    if signed is None:
        signed, source = read_sidecar(path), "sidecar"
    if signed is None and registry:
        signed, source = match_registry(img, registry)
    if signed is None:
        return Verdict("UNSIGNED", "No provenance manifest found. This proves nothing either way: "
                                   "most genuine photos are unsigned.")

    pk = check_signature(signed)
    m = signed.get("manifest", {})
    base = dict(signer={k: v for k, v in m.get("signer", {}).items() if k != "public_key"},
                claims=m.get("claims", {}), created=m.get("created", ""), source=source)
    if pk is None:
        return Verdict("INVALID_SIGNATURE", "The manifest was altered or forged; the signature does not match.",
                       **base)
    if trusted is not None and key_id(pk) not in {key_id(t) for t in trusted}:
        return Verdict("UNTRUSTED_SIGNER", f"Validly signed, but by key {key_id(pk)}, which is not a key you trust.",
                       **base)

    c = compare_content(signed, img)
    if c["exact"]:
        return Verdict("AUTHENTIC", "Signature valid and every pixel matches what was signed.", **base)
    if len(c["changed"]) == 0:
        return Verdict("REENCODED", "Signature valid. Not bit-identical, but the content matches what was signed "
                                    "(consistent with recompression or resizing).",
                       block_map=c["block_map"], max_block_diff=c["max"], **base)
    return Verdict("MODIFIED", f"Signature valid, but {len(c['changed'])} region(s) differ from what was signed. "
                               "Treat this copy as edited.",
                   changed_blocks=c["changed"], block_map=c["block_map"], max_block_diff=c["max"], **base)


# ── registry (manifests recovered after metadata stripping) ───────────────────

def match_registry(img: Image.Image, registry: str, max_mean_diff: float = 12.0):
    """Find the registry manifest whose thumbnail best matches `img` (same aspect ratio)."""
    best, best_score, best_file = None, float("inf"), None
    for f in Path(registry).glob(f"*{SIDECAR_SUFFIX}"):
        try:
            signed = json.loads(f.read_text())
            fp = signed["manifest"]["image"]
        except Exception:
            continue
        if abs(fp["width"] / fp["height"] - img.width / img.height) > 0.02:
            continue
        score = float(np.median(block_differences(decode_thumbnail(fp["thumbnail"]["png_b64"]), img)))
        if score < best_score:
            best, best_score, best_file = signed, score, f.name
    if best is None or best_score > max_mean_diff:
        return None, ""
    return best, f"registry:{best_file}"


def render_block_map(img: Image.Image, v: Verdict, out: str) -> None:
    """Save `img` with changed regions outlined in red (for MODIFIED verdicts)."""
    from PIL import ImageDraw
    vis = img.convert("RGB").copy()
    d = ImageDraw.Draw(vis)
    if v.block_map is not None:
        g = v.block_map.shape[0]
        bw, bh = vis.width / g, vis.height / g
        for r, c in v.changed_blocks:
            d.rectangle([c * bw, r * bh, (c + 1) * bw, (r + 1) * bh], outline=(255, 0, 0),
                        width=max(2, vis.width // 400))
    vis.save(out)
