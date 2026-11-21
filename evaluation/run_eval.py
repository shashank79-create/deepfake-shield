"""
evaluation/run_eval.py
Does DeepFake Shield actually stop a face-recognition model from recognising you?

For every photo it builds these conditions:
    clean          the original photo (sanity check: should be ~1.0 similarity)
    resample_only  the engine's 512px down/up resize with NO attack. Shows how much
                   of the "protection" is just quality loss from the resize.
    random_noise   random +/- noise with the same L2 energy as the protected image.
                   A real attack must beat this, or it is no better than noise.
    protected_<s>  DeepFake Shield output at strength <s>

Each condition is passed through real-world degradations (JPEG, resize,
blur, a social-media upload chain) and compared with the clean original by
every identity evaluator. "Protected" means cosine similarity fell below
that model's same-person threshold.

Usage (from the repo root):
    python -m evaluation.run_eval --images ./faces
    python -m evaluation.run_eval --images ./faces --strengths subtle standard strong maximum
    python -m evaluation.run_eval --images ./faces --arcface-onnx models/w600k_r50.onnx
"""

import argparse
import csv
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from config.settings import Config
from evaluation.identity import build_evaluators
from evaluation.transforms import TRANSFORMS
from utils.metrics import QualityMetrics

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# insightface buffalo_l recognizer; used automatically when present
DEFAULT_ARCFACE = Path(__file__).resolve().parents[1] / "weights" / "buffalo_l" / "w600k_r50.onnx"


# ── conditions ────────────────────────────────────────────────────────────────

def resample_only(img: Image.Image) -> Image.Image:
    """Exactly the engine's resize round trip, without the attack."""
    w, h = img.size
    if max(w, h) <= Config.IMAGE_SIZE:
        return img.copy()
    s = Config.IMAGE_SIZE / max(w, h)
    return img.resize((int(w * s), int(h * s)), Image.LANCZOS).resize((w, h), Image.LANCZOS)


def l2_matched_noise(orig: Image.Image, protected: Image.Image, seed: int) -> Image.Image:
    """Random sign noise with the same total L2 energy as (protected - original)."""
    o = np.asarray(orig, dtype=np.float32)
    p = np.asarray(protected, dtype=np.float32)
    rms = np.sqrt(np.mean((p - o) ** 2))
    rng = np.random.default_rng(seed)
    noisy = o + rng.choice([-1.0, 1.0], size=o.shape).astype(np.float32) * rms
    return Image.fromarray(np.clip(np.round(noisy), 0, 255).astype(np.uint8))


def quality(orig: Image.Image, other: Image.Image) -> Dict[str, float]:
    a = T.ToTensor()(orig).unsqueeze(0)
    b = T.ToTensor()(other).unsqueeze(0)
    return {
        "psnr_db": round(QualityMetrics.psnr(a, b), 2),
        "ssim": round(QualityMetrics.ssim(a, b), 4),
        "linf_255": round(QualityMetrics.l_inf(a, b), 1),
    }


# ── main loop ─────────────────────────────────────────────────────────────────

# attack name -> (condition label, noise-control label, output sub-folder prefix)
ATTACKS = {
    "legacy": ("protected", "random_noise", ""),        # v5 ImageNet ensemble (GoldStandardEngine)
    "face":   ("faceattack", "noise-face", "face-"),     # v6 face-targeted (FaceShieldEngine)
}


def _make_engine(attack: str, device: str):
    if attack == "legacy":
        from engine.shield import GoldStandardEngine
        return GoldStandardEngine(verbose=False, device=device)
    from engine.face_shield import FaceShieldEngine
    return FaceShieldEngine(verbose=False, device=device)


def protect_all(paths: List[Path], attacks: List[str], strengths: List[str], out_dir: Path,
                reuse: bool, device: str) -> Dict[tuple, Dict[str, Path]]:
    """Run each attack (or reuse cached outputs). Returns {(attack, strength): {stem: path}}."""
    todo = {(a, s): {p.stem: out_dir / "protected" / f"{ATTACKS[a][2]}{s}" / f"{p.stem}.png" for p in paths}
            for a in attacks for s in strengths}
    for a in attacks:
        missing = [(s, p) for s in strengths for p in paths if not (reuse and todo[(a, s)][p.stem].exists())]
        if not missing:
            continue
        with _make_engine(a, device) as engine:
            engine.log.setLevel(logging.WARNING)
            for i, (s, p) in enumerate(missing, 1):
                t0 = time.time()
                r = engine.protect(p, output_path=todo[(a, s)][p.stem], strength=s)
                status = "ok" if r["success"] else f"FAILED: {r['error']}"
                print(f"  [{a} {i}/{len(missing)}] protect {p.name} @ {s}: {status} ({time.time()-t0:.0f}s)",
                      flush=True)
                if not r["success"]:
                    todo[(a, s)].pop(p.stem)
    return todo


def evaluate(args) -> Path:
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {args.images}")
    print(f"\n  {len(paths)} images | attacks: {args.attacks} | strengths: {args.strengths}\n")

    protected = protect_all(paths, args.attacks, args.strengths, out_dir, args.reuse, args.device)

    print("\n  Loading identity evaluators...")
    arcface = args.arcface_onnx or (str(DEFAULT_ARCFACE) if DEFAULT_ARCFACE.exists() else None)
    if not arcface:
        print("    (no ArcFace model found - only the white-box FaceNet will be used)")
    evaluators = build_evaluators(Config.DEVICE, arcface, args.facenet_weights)
    for ev in evaluators:
        print(f"    {ev.name:20s} threshold={ev.threshold}  {'(WHITE-BOX: in attack ensemble)' if ev.white_box else ''}")

    swapper, swap_targets = None, []
    if args.swap_targets:
        from evaluation.faceswap import DEFAULT_INSWAPPER, InSwapper
        arc_ev = next((e for e in evaluators if e.name == "arcface"), None)
        if arc_ev is None or not DEFAULT_INSWAPPER.exists():
            print("    (face-swap test skipped: needs ArcFace and weights/inswapper/inswapper_128.onnx)")
        else:
            swapper = InSwapper(str(DEFAULT_INSWAPPER), arc_ev, arc_ev.detector)
            swap_targets = [Image.open(p).convert("RGB") for p in sorted(Path(args.swap_targets).iterdir())
                            if p.suffix.lower() in IMAGE_EXTS]
            print(f"    face swap: InSwapper onto {len(swap_targets)} target face(s), "
                  f"source transforms {args.swap_transforms}")
    swap_rows, examples = [], out_dir / "swap_examples"

    rows = []
    for idx, path in enumerate(paths, 1):
        orig = Image.open(path).convert("RGB")
        refs = {ev.name: ev.embed(orig) for ev in evaluators}
        if all(r is None for r in refs.values()):
            print(f"  [{idx}/{len(paths)}] {path.name}: no face found in the ORIGINAL, skipped")
            continue

        conditions = {"clean": orig, "resample_only": resample_only(orig)}
        for (a, s), done in protected.items():
            if path.stem in done:
                label, noise_label, _ = ATTACKS[a]
                p_img = Image.open(done[path.stem]).convert("RGB")
                conditions[f"{label}_{s}"] = p_img
                conditions[f"{noise_label}_{s}"] = l2_matched_noise(orig, p_img, args.seed + idx)

        for cond, cimg in conditions.items():
            q = quality(orig, cimg)
            for tname, tf in TRANSFORMS.items():
                timg = tf(cimg)
                for ev in evaluators:
                    if refs[ev.name] is None:
                        continue
                    emb = ev.embed(timg)
                    sim = None if emb is None else float(torch.dot(refs[ev.name], emb))
                    rows.append({
                        "image": path.name, "condition": cond, "transform": tname,
                        "evaluator": ev.name, "white_box": ev.white_box,
                        "threshold": ev.threshold,
                        "face_found": emb is not None,
                        "similarity": None if sim is None else round(sim, 4),
                        "protected": (emb is None) or (sim < ev.threshold),
                        **q,
                    })

        if swapper and refs.get("arcface") is not None:
            arc_ev = swapper.arcface
            for cond, cimg in conditions.items():
                if cond == "resample_only" or cond.startswith(("random_noise", "noise-")):
                    continue
                for tname in args.swap_transforms:
                    src = TRANSFORMS[tname](cimg)
                    for ti, tgt in enumerate(swap_targets):
                        res = swapper.swap(src, tgt)
                        emb = None if res is None else arc_ev.embed(res)
                        sim = None if emb is None else float(torch.dot(refs["arcface"], emb))
                        swap_rows.append({"image": path.name, "condition": cond, "transform": tname,
                                          "target": ti, "swapped": res is not None,
                                          "similarity": None if sim is None else round(sim, 4),
                                          "protected": sim is None or sim < arc_ev.threshold})
                        if res is not None and ti == 0 and tname == args.swap_transforms[0]:
                            examples.mkdir(exist_ok=True)
                            res.save(examples / f"{path.stem}__{cond}.jpg", quality=90)
        print(f"  [{idx}/{len(paths)}] {path.name}: evaluated {len(conditions)} conditions", flush=True)

    if not rows:
        raise SystemExit("No faces were detected in any original photo, so nothing was evaluated.")

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = render_summary(rows, args)
    if swap_rows:
        with open(out_dir / "faceswap.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(swap_rows[0].keys()))
            w.writeheader()
            w.writerows(swap_rows)
        summary += "\n" + render_swap_summary(swap_rows, args.swap_transforms)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"  Full results: {csv_path}\n  Summary     : {out_dir / 'summary.md'}\n")
    return out_dir


# ── report ────────────────────────────────────────────────────────────────────

def _cond_order(c: str) -> tuple:
    base = ["clean", "resample_only"]
    if c in base:
        return (base.index(c), "", 0)
    kind, s = c.rsplit("_", 1)
    order = list(Config.PRESETS).index(s) if s in Config.PRESETS else 99
    kinds = [k for a in ATTACKS.values() for k in (a[1], a[0])]  # noise control, then its attack
    return (2, order, kinds.index(kind) if kind in kinds else 99)


def render_summary(rows: List[dict], args) -> str:
    evaluators = list(dict.fromkeys(r["evaluator"] for r in rows))
    conds = sorted(set(r["condition"] for r in rows), key=_cond_order)
    n_imgs = len(set(r["image"] for r in rows))
    wb = {r["evaluator"]: r["white_box"] for r in rows}

    g = defaultdict(list)
    for r in rows:
        g[(r["condition"], r["transform"], r["evaluator"])].append(r)

    out = [f"# DeepFake Shield evaluation\n",
           f"{n_imgs} image(s), attacks: {', '.join(getattr(args, 'attacks', ['legacy']))}, "
           f"strengths: {', '.join(args.strengths)}, seed {args.seed}.\n",
           "Conditions: `protected_*` = legacy v5 attack (control `random_noise_*`), "
           "`faceattack_*` = v6 face attack (control `noise-face_*`).\n",
           "Cell = **% protected** (face not detected, or similarity below threshold) "
           "/ mean cosine similarity to the clean original. Higher % is better protection.\n"]
    for ev in evaluators:
        note = " — WHITE-BOX: this model is inside the attack ensemble, so these numbers are optimistic" if wb[ev] else ""
        out.append(f"\n## {ev}{note}\n")
        out.append("| condition | " + " | ".join(TRANSFORMS) + " |")
        out.append("|---|" + "---:|" * len(TRANSFORMS))
        for c in conds:
            cells = []
            for t in TRANSFORMS:
                rs = g.get((c, t, ev), [])
                if not rs:
                    cells.append("–")
                    continue
                pct = 100 * np.mean([r["protected"] for r in rs])
                sims = [r["similarity"] for r in rs if r["similarity"] is not None]
                cells.append(f"{pct:.0f}% / {np.mean(sims):.2f}" if sims else f"{pct:.0f}% / no face")
            out.append(f"| {c} | " + " | ".join(cells) + " |")

    out.append("\n## Image quality vs original (full resolution, before any transform)\n")
    out.append("| condition | PSNR dB | SSIM | max change /255 |")
    out.append("|---|---:|---:|---:|")
    first_ev = evaluators[0]
    for c in conds:
        rs = g.get((c, "none", first_ev), [])
        if rs:
            out.append(f"| {c} | {np.mean([r['psnr_db'] for r in rs]):.2f} | "
                       f"{np.mean([r['ssim'] for r in rs]):.4f} | {np.mean([r['linf_255'] for r in rs]):.1f} |")

    out.append("\n**How to read this:** each attack must clearly beat its own noise control (same noise energy) "
               "and resample_only, on a non-white-box evaluator, after the `social` transform. "
               "Anything less is not protection.\n")
    return "\n".join(out)


def render_swap_summary(rows: List[dict], transforms: List[str]) -> str:
    conds = sorted(set(r["condition"] for r in rows), key=_cond_order)
    out = ["## Real face swap (InSwapper) — is the swapped result still recognised as the victim?\n",
           "The victim's photo (after the column's transform) is the identity source, pasted onto other "
           "people's faces. Cell = **% protected** (swap result below ArcFace's 0.30 same-person threshold) "
           "/ mean similarity to the victim. Higher % is better protection.\n",
           "| source | " + " | ".join(transforms) + " |", "|---|" + "---:|" * len(transforms)]
    for c in conds:
        cells = []
        for t in transforms:
            rs = [r for r in rows if r["condition"] == c and r["transform"] == t]
            sims = [r["similarity"] for r in rs if r["similarity"] is not None]
            cells.append(f"{100 * np.mean([r['protected'] for r in rs]):.0f}% / {np.mean(sims):.2f}"
                         if sims else "–")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Evaluate DeepFake Shield against face-recognition models")
    ap.add_argument("--images", required=True, help="Folder of face photos (one clear face each)")
    ap.add_argument("--attacks", nargs="+", default=["legacy", "face"], choices=list(ATTACKS))
    ap.add_argument("--strengths", nargs="+", default=["standard"], choices=list(Config.PRESETS))
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--arcface-onnx", help="Path to an ArcFace ONNX model "
                                           "(default: weights/buffalo_l/w600k_r50.onnx if present)")
    ap.add_argument("--facenet-weights", nargs="+", default=["vggface2"],
                    choices=["vggface2", "casia-webface"])
    ap.add_argument("--reuse", action="store_true", help="Reuse protected images already in --out")
    ap.add_argument("--swap-targets", help="Folder of OTHER people's faces to face-swap the victim onto "
                                           "(needs weights/inswapper/inswapper_128.onnx)")
    ap.add_argument("--swap-transforms", nargs="+", default=["none", "social", "denoise"],
                    choices=list(TRANSFORMS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
