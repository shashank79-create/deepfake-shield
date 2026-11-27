"""
examples/demo.py
Protect one photo with v6 and save a side-by-side so you can judge the visibility yourself.

    python examples/demo.py path/to/photo.jpg
    python examples/demo.py path/to/photo.jpg --strength strong
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from config.settings import Config
from engine.face_shield import FaceShieldEngine


def side_by_side(original: Image.Image, protected: Image.Image, label: str) -> Image.Image:
    h = 512
    a = original.resize((round(original.width * h / original.height), h), Image.LANCZOS)
    b = protected.resize(a.size, Image.LANCZOS)
    sheet = Image.new("RGB", (a.width * 2 + 10, h + 24), "white")
    sheet.paste(a, (0, 24))
    sheet.paste(b, (a.width + 10, 24))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), "original", fill="black")
    d.text((a.width + 16, 6), label, fill="black")
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description="DeepFake Shield v6 demo")
    ap.add_argument("image")
    ap.add_argument("--strength", default="standard", choices=list(Config.FACE_PRESETS))
    ap.add_argument("--output", help="default: <image>_protected.png next to the input")
    a = ap.parse_args()

    src = Path(a.image)
    out = Path(a.output) if a.output else src.with_name(f"{src.stem}_protected.png")
    print(f"\nProtecting {src} at '{a.strength}' (about a minute per face on a laptop GPU)...\n")

    with FaceShieldEngine(verbose=True) as engine:
        result = engine.protect(src, output_path=out, strength=a.strength, verbose=True)

    if not result["success"]:
        print(f"\nFAILED: {result['error']}")
        return 1

    m = result["metrics"]
    compare = out.with_name(f"{out.stem}_compare.png")
    side_by_side(Image.open(src).convert("RGB"), result["image"], f"protected ({a.strength})").save(compare)
    print(f"\nDone.\n  Output       : {out}\n  Side by side : {compare}  <- look at the face before posting")
    print(f"  Faces        : {m['faces']}")
    print(f"  Max change   : {m['max_pixel_change']:.0f}/255 on the face, SSIM {m['ssim']:.4f}")
    for i, sims in enumerate(m["surrogate_similarity"], 1):
        pretty = ", ".join(f"{k} {v:+.2f}" for k, v in sims.items())
        print(f"  Face {i} identity similarity (surrogates, lower = better): {pretty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
