"""
scripts/download_weights.py
Fetch the ONNX face models used by DeepFake Shield and verify their SHA-256.

    python scripts/download_weights.py                 # surrogate + judge (recommended)
    python scripts/download_weights.py --only swap     # InSwapper, for the face-swap test
    python scripts/download_weights.py --check         # verify what is already on disk

The FaceNet surrogates download automatically through facenet-pytorch.
insightface models are released for NON-COMMERCIAL RESEARCH use only.
"""

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IF_RELEASE = "https://github.com/deepinsight/insightface/releases/download/v0.7"

WEIGHTS = {
    "surrogate": {   # third attack surrogate (MobileFaceNet)
        "dest": "weights/buffalo_s/w600k_mbf.onnx", "zip": f"{IF_RELEASE}/buffalo_s.zip",
        "member": "w600k_mbf.onnx",
        "sha256": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
    },
    "judge": {       # held-out ArcFace evaluator - never used by the attack
        "dest": "weights/buffalo_l/w600k_r50.onnx", "zip": f"{IF_RELEASE}/buffalo_l.zip",
        "member": "w600k_r50.onnx",
        "sha256": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    },
    "swap": {        # InSwapper, only for the face-swap evaluation (community mirror)
        "dest": "weights/inswapper/inswapper_128.onnx",
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    },
}
DEFAULT = ["surrogate", "judge"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, to: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "deepfake-shield-weights/1.0"})
    with urllib.request.urlopen(req) as r, open(to, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / total * 100:5.1f}%  {done >> 20} / {total >> 20} MB", end="", flush=True)
    print()


def fetch(name: str) -> bool:
    spec = WEIGHTS[name]
    dest = ROOT / spec["dest"]
    if dest.exists() and sha256(dest) == spec["sha256"]:
        print(f"[ok] {name}: {spec['dest']} already present and verified")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if "zip" in spec:
            print(f"[..] {name}: downloading {spec['zip']}")
            download(spec["zip"], tmp / "pack.zip")
            with zipfile.ZipFile(tmp / "pack.zip") as z:
                with z.open(spec["member"]) as src, open(tmp / "model.onnx", "wb") as out:
                    shutil.copyfileobj(src, out)
        else:
            print(f"[..] {name}: downloading {spec['url']}")
            download(spec["url"], tmp / "model.onnx")
        got = sha256(tmp / "model.onnx")
        if got != spec["sha256"]:
            print(f"[XX] {name}: checksum mismatch ({got}); file discarded")
            return False
        shutil.move(str(tmp / "model.onnx"), dest)
    print(f"[ok] {name}: saved {spec['dest']} (sha256 verified)")
    return True


def check() -> bool:
    ok = True
    for name, spec in WEIGHTS.items():
        dest = ROOT / spec["dest"]
        if not dest.exists():
            print(f"[--] {name}: missing ({spec['dest']})")
        elif sha256(dest) == spec["sha256"]:
            print(f"[ok] {name}: {spec['dest']}")
        else:
            print(f"[XX] {name}: CHECKSUM MISMATCH ({spec['dest']})")
            ok = False
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(WEIGHTS), help=f"default: {' '.join(DEFAULT)}")
    ap.add_argument("--check", action="store_true", help="Only verify files already on disk")
    a = ap.parse_args(argv)
    if a.check:
        return 0 if check() else 1
    return 0 if all(fetch(n) for n in (a.only or DEFAULT)) else 1


if __name__ == "__main__":
    sys.exit(main())
