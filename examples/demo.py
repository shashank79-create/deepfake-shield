"""
examples/demo.py
Quick demo — protect a single image and print results.

Run from project root:
    python examples/demo.py examples/sample.jpg
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.face_shield import FaceShieldEngine

def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/demo.py <image_path>")
        print("Example: python examples/demo.py examples/sample.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"\nProtecting: {image_path}")
    print("Strength: standard (recommended)")
    print("This will take 1-3 minutes depending on your hardware...\n")

    with FaceShieldEngine(verbose=True) as engine:
        result = engine.protect(
            image_path,
            output_path="protected_output.png",
            strength="standard",
            verbose=True
        )

    if result["success"]:
        m = result["metrics"]
        print(f"\n✅ Done!")
        print(f"   Output  : {result['output_path']}")
        print(f"   SSIM    : {m['ssim']:.4f} (1.0 = identical to original)")
        print(f"   PSNR    : {m['psnr_db']:.1f} dB")
        print(f"   Time    : {m['time_s']:.1f}s")
        print(f"   Invisible: {m['invisible']}")
        print(f"\n   Models used: {', '.join(m['models_used'])}")
    else:
        print(f"\n❌ Failed: {result['error']}")

if __name__ == "__main__":
    main()
