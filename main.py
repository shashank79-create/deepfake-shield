"""
main.py
DeepFake Shield — Command Line Interface

Usage:
    python main.py photo.jpg
    python main.py photo.jpg --strength strong --output protected.png
    python main.py photo.jpg --strength maximum --verbose
    python main.py photo.jpg --test-all
    python main.py --batch ./photos/ --output-dir ./protected/ --strength strong
"""

import argparse
import json
import sys
from pathlib import Path

from config.settings import Config
from engine.shield import GoldStandardEngine


BANNER = """
╔══════════════════════════════════════════════════════════╗
║        DeepFake Shield v6.0  —  Protection Engine        ║
║   Face-identity cloaking  |  Ed25519 provenance signing   ║
╚══════════════════════════════════════════════════════════╝
"""


def print_metrics(m: dict):
    ssim_ok = "[OK]" if m["quality_ok"] else "[!!]"
    inv_ok  = "[OK]" if m["invisible"]  else "[!!]"
    print(f"""
  ┌─────────────────────────────────────┐
  │         Protection Metrics          │
  ├─────────────────────────────────────┤
  │  SSIM              {m['ssim']:.5f}  {ssim_ok}    │
  │  PSNR              {m['psnr_db']:>7.2f} dB         │
  │  Max pixel change  {m['max_pixel_change']:>7.2f} / 255      │
  │  L2 norm           {m['l2_norm']:>10.4f}         │
  │  Invisible         {str(m['invisible']):<21}│
  │  Processing time   {m['time_s']:>7.2f} s          │
  │  Models used       {len(m['models_used']):<21}│
  └─────────────────────────────────────┘""")


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="DeepFake Shield — Adversarial Image Protection"
    )
    p.add_argument("image",       nargs="?",  default=None,       help="Input image path")
    p.add_argument("--output",    "-o",       metavar="PATH",      help="Output path")
    p.add_argument("--strength",  "-s",       default="standard",
                   choices=list(Config.PRESETS),                   help="Protection strength")
    p.add_argument("--verbose",   "-v",       action="store_true", help="Show progress bar")
    p.add_argument("--test-all",              action="store_true", help="Run all 4 strengths")
    p.add_argument("--batch",                 metavar="DIR",       help="Protect all images in folder")
    p.add_argument("--output-dir",            metavar="DIR",       default="protected_output")
    p.add_argument("--save-metrics",          metavar="PATH",      help="Save metrics JSON")
    p.add_argument("--device",                default="auto",      help="cuda / cpu / auto")
    p.add_argument("--sign-key",              metavar="KEY",
                   help="Sign each protected PNG with this Ed25519 private key (see provenance.cli keygen)")
    p.add_argument("--signer-name",           default="",          help="Name to put in the signature manifest")
    p.add_argument("--attack",                default="face",      choices=["face", "legacy"],
                   help="face = v6 face-targeted full-res attack (default); "
                        "legacy = DEPRECATED v5 ImageNet ensemble (measured 0%% protection)")
    return p


def maybe_sign(args, out_path, metrics):
    """Sign a protected output if --sign-key was given."""
    if not args.sign_key:
        return
    from provenance.keys import load_private
    from provenance.manifest import sign_image
    if not hasattr(args, "_sk"):
        args._sk = load_private(args.sign_key)
    claims = {"protected_by": f"DeepFake Shield ({metrics.get('attack', 'legacy-v5')})",
              "strength": metrics.get("strength", "")}
    sign_image(str(out_path), args._sk, str(out_path), args.signer_name, claims)
    print(f"  Signed : {out_path}  (+ {out_path}.provenance.json)")


def cmd_protect(engine, args):
    if not args.image:
        print("  ERROR: Provide an image path.\n")
        sys.exit(1)
    output = args.output or str(
        Path(args.image).parent / f"{Path(args.image).stem}_protected.png"
    )
    result = engine.protect(args.image, output_path=output,
                             strength=args.strength, verbose=args.verbose)
    if result["success"]:
        print(f"\n  ✅ Protection complete!")
        print(f"  Output : {result['output_path']}")
        print_metrics(result["metrics"])
        maybe_sign(args, result["output_path"], result["metrics"])
        if args.save_metrics:
            with open(args.save_metrics, "w") as f:
                json.dump(result["metrics"], f, indent=2)
            print(f"\n  Metrics saved → {args.save_metrics}")
    else:
        print(f"\n  ❌ FAILED: {result['error']}")
        sys.exit(1)


def cmd_test_all(engine, args):
    if not args.image:
        print("  ERROR: Provide an image path.\n")
        sys.exit(1)
    out_dir = Path("test_all_results")
    out_dir.mkdir(exist_ok=True)
    all_metrics = {}
    print(f"\n  Testing all 4 strengths on: {args.image}\n")
    for strength in Config.PRESETS:
        out = out_dir / f"{strength}.png"
        print(f"\n  [{strength.upper()}] ─────────────────────────────")
        result = engine.protect(args.image, output_path=out,
                                strength=strength, verbose=args.verbose)
        if result["success"]:
            print_metrics(result["metrics"])
            all_metrics[strength] = result["metrics"]
        else:
            print(f"  FAILED: {result['error']}")
    comp = out_dir / "comparison.json"
    with open(comp, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  All done! Results in: {out_dir}/")


def cmd_batch(engine, args):
    src   = Path(args.batch)
    exts  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [p for p in src.iterdir() if p.suffix.lower() in exts]
    if not files:
        print(f"  ERROR: No images found in {args.batch}")
        sys.exit(1)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Batch: {len(files)} images → {out_dir}/\n")
    ok, fail = 0, 0
    for f in files:
        out    = out_dir / f"{f.stem}_protected.png"
        result = engine.protect(f, output_path=out,
                                strength=args.strength, verbose=False)
        if result["success"]:
            m = result["metrics"]
            print(f"  [OK] {f.name:<40} SSIM={m['ssim']:.4f}  {m['time_s']:.1f}s")
            maybe_sign(args, out, m)
            ok += 1
        else:
            print(f"  [!!] {f.name:<40} {result['error']}")
            fail += 1
    print(f"\n  Batch complete: {ok} protected, {fail} failed.\n")


def main():
    # Windows consoles/pipes often default to cp1252, which cannot print the banner or emoji.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args   = parser.parse_args()
    print(BANNER)

    if args.attack == "face":
        from engine.face_shield import FaceShieldEngine
        engine_cm = FaceShieldEngine(verbose=args.verbose, device=args.device)
    else:
        print("  WARNING: --attack legacy (v5) is deprecated. In evaluation it protected 0/14 faces\n"
              "           against a held-out face recognizer and degrades high-res photos. Use --attack face.\n")
        engine_cm = GoldStandardEngine(verbose=args.verbose, device=args.device)

    with engine_cm as engine:
        if args.batch:
            cmd_batch(engine, args)
        elif args.test_all:
            cmd_test_all(engine, args)
        else:
            cmd_protect(engine, args)


if __name__ == "__main__":
    main()
