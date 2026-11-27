"""
provenance/cli.py
    python -m provenance.cli keygen --out keys --name me
    python -m provenance.cli sign photo_protected.png --key keys/me.key --name "Your Name"
    python -m provenance.cli verify suspicious.jpg --trust keys/me.pub --registry signed/
"""

import argparse
import getpass
import json
import sys
from pathlib import Path

from provenance.keys import generate_keypair, key_id, load_private, load_public
from provenance.manifest import render_block_map, sign_image, verify_image

ICON = {"AUTHENTIC": "[OK]", "REENCODED": "[OK]", "MODIFIED": "[!!]", "UNTRUSTED_SIGNER": "[??]",
        "INVALID_SIGNATURE": "[XX]", "UNSIGNED": "[--]"}
EXIT = {"AUTHENTIC": 0, "REENCODED": 0, "UNSIGNED": 1, "UNTRUSTED_SIGNER": 2, "MODIFIED": 3,
        "INVALID_SIGNATURE": 4}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def verify_folder(a, trusted) -> int:
    """Verify every image in a folder: one line each, a count per verdict, optional CSV.
    --map is treated as an output folder for MODIFIED images. Exit code = the worst verdict."""
    from collections import Counter
    from PIL import Image
    files = sorted(p for p in Path(a.image).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images in {a.image}")
        return 1
    if a.map:
        Path(a.map).mkdir(parents=True, exist_ok=True)
    rows, counts, worst = [], Counter(), 0
    for f in files:
        v = verify_image(str(f), trusted, a.registry)
        counts[v.status] += 1
        worst = max(worst, EXIT[v.status])
        if a.map and v.status == "MODIFIED":
            render_block_map(Image.open(f), v, str(Path(a.map) / f"{f.stem}_changed.png"))
        rows.append({"file": f.name, "status": v.status, "signer": v.signer.get("name", ""),
                     "key_id": v.signer.get("key_id", ""), "signed": v.created, "source": v.source,
                     "changed_regions": len(v.changed_blocks)})
        if not a.json:
            print(f"{ICON[v.status]} {v.status:17s} {f.name}"
                  + (f"  ({len(v.changed_blocks)} regions changed)" if v.status == "MODIFIED" else ""))
    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        print("\n" + "  ".join(f"{k}: {n}" for k, n in sorted(counts.items())))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(prog="provenance", description="Sign and verify photos (Ed25519)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("keygen", help="Create a signing key pair")
    k.add_argument("--out", default="keys")
    k.add_argument("--name", default="signer")
    k.add_argument("--password", action="store_true", help="Encrypt the private key (prompted)")

    s = sub.add_parser("sign", help="Sign an image")
    s.add_argument("image")
    s.add_argument("--key", required=True)
    s.add_argument("--out", help="Output .png (default: sign in place if PNG, else <name>_signed.png)")
    s.add_argument("--name", default="", help="Signer name to put in the manifest")
    s.add_argument("--claim", action="append", default=[], help="key=value claim (repeatable)")
    s.add_argument("--password", action="store_true")

    v = sub.add_parser("verify", help="Verify an image, or every image in a folder")
    v.add_argument("image", help="Image file or folder")
    v.add_argument("--csv", help="Folder mode: also write results to this CSV")
    v.add_argument("--trust", action="append", default=[], help="Trusted public key .pub (repeatable)")
    v.add_argument("--registry", help="Folder of *.provenance.json sidecars (for stripped re-uploads)")
    v.add_argument("--map", help="Save a copy with changed regions outlined")
    v.add_argument("--json", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "keygen":
        pw = getpass.getpass("Password for the private key: ") if a.password else None
        priv, pub = generate_keypair(a.out, a.name, pw)
        print(f"Private key: {priv}   <- keep secret, back it up\nPublic key : {pub}   <- share this\n"
              f"Key ID     : {key_id(load_public(str(pub)))}")
        return 0

    if a.cmd == "sign":
        pw = getpass.getpass("Private key password: ") if a.password else None
        sk = load_private(a.key, pw)
        out = a.out or (a.image if a.image.lower().endswith(".png") else a.image.rsplit(".", 1)[0] + "_signed.png")
        claims = dict(c.split("=", 1) for c in a.claim)
        signed = sign_image(a.image, sk, out, a.name, claims)
        print(f"Signed -> {out}  (key {signed['manifest']['signer']['key_id']}, sidecar {out}.provenance.json)")
        return 0

    trusted = [load_public(p) for p in a.trust] or None
    if Path(a.image).is_dir():
        return verify_folder(a, trusted)
    verdict = verify_image(a.image, trusted, a.registry)
    if a.map and verdict.status == "MODIFIED":
        from PIL import Image
        render_block_map(Image.open(a.image), verdict, a.map)
    if a.json:
        print(json.dumps({k: v for k, v in verdict.__dict__.items() if k != "block_map"}, indent=2))
    else:
        print(f"{ICON[verdict.status]} {verdict.status}: {verdict.detail}")
        if verdict.signer:
            print(f"     signer : {verdict.signer.get('name') or '(no name)'}  key {verdict.signer.get('key_id')}")
            print(f"     signed : {verdict.created}   via {verdict.source}")
        if verdict.claims:
            print(f"     claims : {verdict.claims}")
        if verdict.status == "MODIFIED" and a.map:
            print(f"     map    : {a.map}")
        if trusted is None and verdict.status in ("AUTHENTIC", "REENCODED"):
            print("     note   : no --trust key given; this proves it was signed by key "
                  f"{verdict.signer.get('key_id')}, not WHO owns that key.")
    return EXIT[verdict.status]


if __name__ == "__main__":
    sys.exit(main())
