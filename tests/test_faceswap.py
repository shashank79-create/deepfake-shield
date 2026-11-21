"""Face-swap harness check. Skipped unless the InSwapper/ArcFace weights and LFW are present locally."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INSWAPPER = ROOT / "weights" / "inswapper" / "inswapper_128.onnx"
ARCFACE = ROOT / "weights" / "buffalo_l" / "w600k_r50.onnx"
LFW = ROOT.parent / "datasets" / "lfw"

pytestmark = pytest.mark.skipif(not (INSWAPPER.exists() and ARCFACE.exists() and LFW.exists()),
                                reason="needs local InSwapper, ArcFace weights and LFW")


def test_clean_swap_transfers_identity():
    from PIL import Image
    from config.settings import Config
    from evaluation.faceswap import InSwapper
    from evaluation.identity import build_evaluators

    arc = [e for e in build_evaluators(Config.DEVICE, str(ARCFACE)) if e.name == "arcface"][0]
    sw = InSwapper(str(INSWAPPER), arc, arc.detector)
    victim = Image.open(LFW / "Colin_Powell" / "Colin_Powell_0001.jpg").convert("RGB")
    target = Image.open(LFW / "Laura_Bush" / "Laura_Bush_0001.jpg").convert("RGB")
    ref = arc.embed(victim)
    swapped = sw.swap(victim, target)
    assert swapped is not None and swapped.size == target.size
    assert float(ref @ arc.embed(swapped)) > 0.5            # the swap really carries the identity
    assert float(ref @ arc.embed(target)) < 0.3              # and the target was someone else
