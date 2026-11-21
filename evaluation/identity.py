"""
evaluation/identity.py
Identity evaluators: "would a face-recognition model still say this is the same person?"

Every evaluator detects the face on the image it is given (an attacker
would run detection on the protected photo too), so a failed detection is
a real outcome and is reported, not hidden.

FaceNetEvaluator  - facenet-pytorch InceptionResnetV1 (VGGFace2 or CASIA-WebFace).
                    NOTE: the VGGFace2 FaceNet is also inside the attack ensemble,
                    so scores from it are WHITE-BOX (optimistic). Use it as a sanity
                    check, never as proof of protection.
ArcFaceEvaluator  - an ArcFace ONNX model (e.g. insightface buffalo_l w600k_r50.onnx),
                    the identity encoder family used by SimSwap / InSwapper face swaps.
                    Not in the attack ensemble, so it measures real transfer.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class FaceDetection:
    box: np.ndarray        # [x1, y1, x2, y2]
    landmarks: np.ndarray  # [5, 2]: left eye, right eye, nose, mouth left, mouth right
    prob: float


class FaceDetector:
    """MTCNN wrapper that returns the most confident face, or None."""

    def __init__(self, device: torch.device):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, img: Image.Image) -> Optional[FaceDetection]:
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return None
        i = int(np.argmax(probs))
        return FaceDetection(boxes[i], landmarks[i], float(probs[i]))


class IdentityEvaluator:
    """Base class. Subclasses set name/threshold/white_box and implement _embed()."""

    name: str = "base"
    threshold: float = 0.4     # cosine >= threshold -> "same person"
    white_box: bool = False    # True if this model is part of the attack ensemble

    def __init__(self, detector: FaceDetector, device: torch.device):
        self.detector = detector
        self.device = device

    def embed(self, img: Image.Image) -> Optional[torch.Tensor]:
        """L2-normalised 1-D embedding, or None if no face was found."""
        det = self.detector.detect(img)
        if det is None:
            return None
        return F.normalize(self._embed(img, det).flatten(), dim=0).cpu()

    def _embed(self, img: Image.Image, det: FaceDetection) -> torch.Tensor:
        raise NotImplementedError


class FaceNetEvaluator(IdentityEvaluator):
    threshold = 0.40  # ~ euclidean distance 1.1 on unit vectors, the usual FaceNet cut-off

    def __init__(self, detector, device, weights: str = "vggface2"):
        super().__init__(detector, device)
        from facenet_pytorch import InceptionResnetV1
        from facenet_pytorch.models.mtcnn import extract_face
        self._extract = extract_face
        self.model = InceptionResnetV1(pretrained=weights).eval().to(device)
        self.name = f"facenet-{weights}"
        self.white_box = weights == "vggface2"  # the attack ensemble loads this one

    @torch.no_grad()
    def _embed(self, img, det):
        face = self._extract(img, det.box, image_size=160, margin=0)  # [3,160,160] in 0..255
        face = (face - 127.5) / 128.0                                   # FaceNet standardisation
        return self.model(face.unsqueeze(0).to(self.device))


# Canonical 112x112 ArcFace landmark template (insightface)
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def align_arcface(img: Image.Image, landmarks: np.ndarray) -> np.ndarray:
    """Similarity-transform the face onto the ArcFace template. Returns 112x112x3 uint8 RGB."""
    import cv2
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_TEMPLATE,
                                       method=cv2.LMEDS)
    return cv2.warpAffine(np.asarray(img.convert("RGB")), M, (112, 112), borderValue=0)


class ArcFaceEvaluator(IdentityEvaluator):
    threshold = 0.30  # typical ArcFace (w600k_r50) verification cut-off

    def __init__(self, detector, device, onnx_path: str):
        super().__init__(detector, device)
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.name = "arcface"

    def _embed(self, img, det):
        face = align_arcface(img, det.landmarks).astype(np.float32)
        x = ((face - 127.5) / 127.5).transpose(2, 0, 1)[None]  # NCHW, RGB, [-1, 1]
        return torch.from_numpy(self.sess.run(None, {self.input_name: x})[0])


def build_evaluators(device: torch.device, arcface_onnx: Optional[str] = None,
                     facenet_weights: List[str] = ("vggface2",)) -> List[IdentityEvaluator]:
    detector = FaceDetector(device)
    evals: List[IdentityEvaluator] = [FaceNetEvaluator(detector, device, w) for w in facenet_weights]
    if arcface_onnx:
        evals.append(ArcFaceEvaluator(detector, device, arcface_onnx))
    return evals
