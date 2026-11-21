"""
evaluation/tune_visibility.py
Compare v6 attack variants on protection (held-out ArcFace) vs visibility.

Use faces that are NOT in the eval set, so presets are not tuned on test data.

    python -m evaluation.tune_visibility --images a.jpg b.jpg --out tune_results
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from config.settings import Config
from core.face_attack import FaceIdentityAttack, load_surrogates
from engine.shield import _make_logger
from evaluation.identity import build_evaluators
from evaluation.run_eval import DEFAULT_ARCFACE, quality
from evaluation.transforms import TRANSFORMS

BASE = dict(Config.FACE_PRESETS["strong"])
PERCEPTUAL = {"optimizer": "adam", "lambda_ssim": 10.0, "texture_mask": True,
              "tex_ref": 0.05, "tex_min": 0.3, "chroma_ratio": 0.5}
VARIANTS = {
    "A_signed_eps12":      {**BASE},
    "B_percep_eps12":      {**BASE, **PERCEPTUAL},
    "C_percep_eps12_lr2":  {**BASE, **PERCEPTUAL, "lowres": 2},
    "D_percep_eps16":      {**BASE, **PERCEPTUAL, "epsilon": 16 / 255},
    # round 2 (hard RGB cap in place)
    "P8":                  {**BASE, **PERCEPTUAL, "epsilon": 8 / 255},
    "P12":                 {**BASE, **PERCEPTUAL, "epsilon": 12 / 255},
    "P16":                 {**BASE, **PERCEPTUAL, "epsilon": 16 / 255},
    "P16_ssim30":          {**BASE, **PERCEPTUAL, "epsilon": 16 / 255, "lambda_ssim": 30.0},
    # round 3 (anti-aliased surrogate crops + low-frequency penalty)
    "R16":                 {**BASE, **PERCEPTUAL, "epsilon": 16 / 255},
    "R16_lf":              {**BASE, **PERCEPTUAL, "epsilon": 16 / 255, "lambda_lowfreq": 1.0},
    "R20_lf":              {**BASE, **PERCEPTUAL, "epsilon": 20 / 255, "lambda_lowfreq": 1.0},
    "R20_lf_500":          {**BASE, **PERCEPTUAL, "epsilon": 20 / 255, "lambda_lowfreq": 1.0, "steps": 500},
}


def face_ssim(orig: Image.Image, other: Image.Image, box) -> float:
    x1, y1, x2, y2 = [int(v) for v in box]
    return quality(orig.crop((x1, y1, x2, y2)), other.crop((x1, y1, x2, y2)))["ssim"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="tune_results")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log = _make_logger("Tune", False)
    dev = Config.DEVICE
    attacker = FaceIdentityAttack(load_surrogates(dev, Config.FACE_SURROGATES, log), dev, log)
    arc = [e for e in build_evaluators(dev, str(DEFAULT_ARCFACE)) if e.name == "arcface"][0]

    rows = []
    for path in args.images:
        orig = Image.open(path).convert("RGB")
        ref = arc.embed(orig)
        box = attacker.detect(orig)[0][0]
        tiles = [("original", orig)]
        for v in args.variants:
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            t0 = time.time()
            r = attacker.run(orig, VARIANTS[v])
            arr = (r["tensor"][0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            adv = Image.fromarray(arr)
            sims = {}
            for t in ("none", "jpeg75", "social"):
                e = arc.embed(TRANSFORMS[t](adv))
                sims[t] = None if e is None else round(float(ref @ e), 3)
            q = quality(orig, adv)
            row = {"image": Path(path).name, "variant": v, "arcface": sims,
                   "ssim": q["ssim"], "face_ssim": round(face_ssim(orig, adv, box), 4),
                   "psnr_db": q["psnr_db"], "linf": q["linf_255"], "time_s": round(time.time() - t0, 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            tiles.append((v, adv))

        # side-by-side crop around the face, labelled, 2x for inspection
        x1, y1, x2, y2 = box
        side = 1.6 * max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop = tuple(int(v) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
        S = 400
        sheet = Image.new("RGB", (S * len(tiles), S + 24), "white")
        d = ImageDraw.Draw(sheet)
        for i, (name, im) in enumerate(tiles):
            sheet.paste(im.crop(crop).resize((S, S), Image.LANCZOS), (i * S, 24))
            d.text((i * S + 6, 5), name, fill="black")
        sheet.save(out / f"{Path(path).stem}_compare.png")

    (out / "tune.json").write_text(json.dumps(rows, indent=2))
    print("\nvariant               arcface(none/jpeg75/social)   face_ssim  ssim   psnr")
    for v in args.variants:
        rs = [r for r in rows if r["variant"] == v]
        m = lambda k: np.mean([r["arcface"][k] for r in rs if r["arcface"][k] is not None])
        protected = np.mean([(r["arcface"]["social"] is None) or r["arcface"]["social"] < arc.threshold for r in rs])
        print(f"{v:20s}  {m('none'):.2f} / {m('jpeg75'):.2f} / {m('social'):.2f}  "
              f"({protected*100:.0f}% protected)   {np.mean([r['face_ssim'] for r in rs]):.3f}  "
              f"{np.mean([r['ssim'] for r in rs]):.3f}  {np.mean([r['psnr_db'] for r in rs]):.1f}")


if __name__ == "__main__":
    main()
