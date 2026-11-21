"""
evaluation/faceswap.py
A REAL face-swap attack, used only to measure protection.

InSwapper (insightface inswapper_128) takes the victim's photo as the identity
SOURCE and pastes that identity onto another person's face (the TARGET). If
protection works, the swapped result should NOT be recognised as the victim.

Score = ArcFace cosine similarity between the swapped result and the victim's
clean photo. A clean-source swap typically scores well above the 0.30
same-person threshold; protection succeeds when the score falls below it.

Note: InSwapper encodes the source with the same ArcFace (w600k_r50) we use as
the judge, so identity-encoder protection and this score are correlated; the
swap test checks that the effect really carries through a generator.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from evaluation.identity import ARCFACE_TEMPLATE, ArcFaceEvaluator, FaceDetector

DEFAULT_INSWAPPER = Path(__file__).resolve().parents[1] / "weights" / "inswapper" / "inswapper_128.onnx"


class InSwapper:
    def __init__(self, onnx_path: str, arcface: ArcFaceEvaluator, detector: FaceDetector):
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper
        self.emap = numpy_helper.to_array(onnx.load(onnx_path).graph.initializer[-1])
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        names = [i.name for i in self.sess.get_inputs()]
        self.in_target, self.in_source = names[0], names[1]
        self.arcface = arcface
        self.detector = detector

    def swap(self, source: Image.Image, target: Image.Image) -> Optional[Image.Image]:
        """Put the identity from `source` onto the face in `target`. None if a face is missing."""
        import cv2
        emb = self.arcface.embed(source)
        det = self.detector.detect(target)
        if emb is None or det is None:
            return None
        latent = emb.numpy().reshape(1, -1) @ self.emap
        latent = (latent / np.linalg.norm(latent)).astype(np.float32)

        tgt = np.asarray(target.convert("RGB"))
        M, _ = cv2.estimateAffinePartial2D(det.landmarks.astype(np.float32),
                                           ARCFACE_TEMPLATE * (128 / 112.0), method=cv2.LMEDS)
        aligned = cv2.warpAffine(tgt, M, (128, 128), borderValue=0)
        blob = (aligned.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.in_target: blob, self.in_source: latent})[0][0]
        face = (np.clip(out.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

        # paste back with a feathered mask
        IM = cv2.invertAffineTransform(M)
        h, w = tgt.shape[:2]
        back = cv2.warpAffine(face, IM, (w, h), borderValue=0)
        mask = np.zeros((128, 128), np.uint8)
        cv2.ellipse(mask, (64, 66), (50, 58), 0, 0, 360, 255, -1)
        mask = cv2.warpAffine(mask, IM, (w, h), borderValue=0)
        k = max(3, int(np.sqrt(mask.sum() / 255) * 0.12) | 1)
        mask = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(k / 2)),
                          dtype=np.float32)[:, :, None] / 255.0
        res = back * mask + tgt * (1 - mask)
        return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))
