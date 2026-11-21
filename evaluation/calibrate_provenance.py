"""
evaluation/calibrate_provenance.py
Calibrate provenance.manifest.BLOCK_CHANGED on real photos.

Benign   : JPEG 95/75/50, social upload, half-size resize, mild brightness filter
Malicious: crude face swap (another person's face pasted with a feathered mask),
           blurred face, small mouth-region edit

    python -m evaluation.calibrate_provenance --images a.jpg b.jpg ... --donors c.jpg d.jpg
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from evaluation.transforms import TRANSFORMS
from provenance.fingerprint import block_differences, thumbnail_array


def face_box(img, mtcnn):
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return None
    return [int(v) for v in boxes[int(np.argmax(probs))]]


def feather_mask(w, h):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).ellipse([w * 0.08, h * 0.05, w * 0.92, h * 0.95], fill=255)
    return m.filter(ImageFilter.GaussianBlur(max(2, w // 15)))


def face_swap(img, box, donor, dbox):
    x1, y1, x2, y2 = box
    face = donor.crop(dbox).resize((x2 - x1, y2 - y1), Image.LANCZOS)
    out = img.copy()
    out.paste(face, (x1, y1), feather_mask(x2 - x1, y2 - y1))
    return out


def blur_face(img, box):
    out = img.copy()
    region = out.crop(box).filter(ImageFilter.GaussianBlur(max(2, (box[2] - box[0]) // 25)))
    out.paste(region, box[:2], feather_mask(box[2] - box[0], box[3] - box[1]))
    return out


def mouth_edit(img, box):
    """Shift the lower-middle of the face a little: a 'small expression change'."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mb = (x1 + int(w * 0.25), y1 + int(h * 0.62), x1 + int(w * 0.75), y1 + int(h * 0.9))
    out = img.copy()
    patch = out.crop(mb).resize((mb[2] - mb[0], int((mb[3] - mb[1]) * 1.15)), Image.LANCZOS)
    out.paste(patch.crop((0, 0, mb[2] - mb[0], mb[3] - mb[1])), mb[:2],
              feather_mask(mb[2] - mb[0], mb[3] - mb[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--donors", nargs="+", required=True)
    args = ap.parse_args()
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=True)

    donors = []
    for d in args.donors:
        im = Image.open(d).convert("RGB")
        b = face_box(im, mtcnn)
        if b:
            donors.append((im, b))

    benign = {
        "jpeg95": TRANSFORMS["jpeg95"], "jpeg75": TRANSFORMS["jpeg75"], "jpeg50": TRANSFORMS["jpeg50"],
        "social": TRANSFORMS["social"], "resize0.5": TRANSFORMS["resize0.5"],
        "brighter": lambda im: TRANSFORMS["social"](ImageEnhance.Brightness(im).enhance(1.1)),
    }
    res = {}
    for path in args.images:
        img = Image.open(path).convert("RGB")
        box = face_box(img, mtcnn)
        if not box:
            continue
        signed = thumbnail_array(img)
        cases = {k: f(img) for k, f in benign.items()}
        for i, (dimg, dbox) in enumerate(donors[:2]):
            cases[f"faceswap{i}+social"] = TRANSFORMS["social"](face_swap(img, box, dimg, dbox))
        cases["blurface+social"] = TRANSFORMS["social"](blur_face(img, box))
        cases["mouth+social"] = TRANSFORMS["social"](mouth_edit(img, box))
        for k, cand in cases.items():
            bm = block_differences(signed, cand)
            res.setdefault(k, []).append((bm.max(), np.median(bm)))

    print(f"{'case':22s} {'max block diff (min / mean / max over images)':48s} median")
    for k, v in res.items():
        mx = np.array([a for a, _ in v])
        md = np.array([b for _, b in v])
        print(f"{k:22s} {mx.min():6.2f} / {mx.mean():6.2f} / {mx.max():6.2f}{'':26s} {md.mean():5.2f}")


if __name__ == "__main__":
    main()
