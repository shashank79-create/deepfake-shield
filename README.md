# DeepFake Shield
[![tests](https://github.com/shashank79-create/deepfake-shield/actions/workflows/tests.yml/badge.svg)](https://github.com/shashank79-create/deepfake-shield/actions/workflows/tests.yml)

**Face-identity cloaking + cryptographic provenance for your photos.**

Two layers, because neither is enough alone:

1. **Cloaking (v6):** adds a perturbation to the face that makes face-recognition models, and face-swap tools
   that rely on them, see a *different person*. It raises the cost of off-the-shelf face swaps.
2. **Provenance signing:** signs your genuine photos with an Ed25519 key, so anyone can check whether a copy
   is really yours, and *where* an edited copy was changed (e.g. a swapped face).

> **Honest limits.** Cloaking is not a guarantee. At the strengths that protect, the change on the face is
> visible on close inspection. A determined attacker can weaken it (see the purification results), future
> models may ignore it, and your unprotected photos and videos are still out there. Signing cannot stop a
> fake from being made; it makes "is this really me?" checkable.

---

## Results (measured, not promised)

14 faces (6 high-res public-domain portraits + 8 LFW benchmark photos), evaluated with
`evaluation/run_eval.py`. "Protected" = the held-out ArcFace recognizer (not used by the attack) no longer
matches the person (cosine < 0.30).

**Real face swap (InSwapper, the swapper behind many one-click deepfake apps).** The victim's photo is
the identity source; the swap result is pasted onto another person's face:

| Source photo | Untouched | After social-media upload | After attacker denoising |
|---|---:|---:|---:|
| Clean | 0% protected (similarity 0.75) | 0% (0.74) | 0% (0.70) |
| v6 standard | 57% (0.26) | 54% (0.26) | 36% (0.34) |
| **v6 strong** | **89% (0.10)** | **96% (0.09)** | **93% (0.17)** |
| v6 maximum | 93% (0.02) | 100% (0.00) | 100% (0.08) |

**Face recognition (ArcFace) on the photo itself:**

| Preset | Untouched | Social upload | JPEG 30 | Denoise | Median 3×3 | ¼-size resize | Face change (max /255) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random noise, same energy | 0% | 0% | 0% | 0% | 0% | 0% | – |
| v6 standard | 29% | 29% | 7% | 7% | 36% | 7% | 12 |
| v6 strong | 64% | 71% | 57% | 57% | 71% | 57% | 16 |
| v6 maximum | 79% | 100% | 79% | 79% | 100% | 71% | 20 |
| v5 (old attack) | 0% | 0% | – | – | – | – | up to 95 (resize bug) |

**Visibility.** Whole-image SSIM stays ≥ 0.985, but that hides the truth: the change is concentrated on the
face. `subtle` (8/255) is near-invisible and does not protect; `standard` shows faint unevenness; `strong`
and `maximum` give a visibly "retouched", mottled skin look. This trade-off is the main open problem.

**Caveats.** One of the three attack surrogates (MobileFaceNet) is trained on the same data as the ArcFace
judge, and InSwapper uses that same ArcFace as its identity encoder, so these numbers are likely optimistic
against unrelated face models. Adaptive attackers with stronger purification (e.g. diffusion-based) were
not tested.

---

## Installation

```bash
git clone https://github.com/shashank79-create/deepfake-shield
cd deepfake-shield
pip install -r requirements.txt
# or, to get the commands deepfake-shield / dfs-provenance / dfs-eval / dfs-weights:
pip install -e ".[test]"
```

Use an editable install (`-e`): model weights are looked up in the repo's `weights/` folder.

Surrogate weights: the two FaceNet models download automatically. The third surrogate (MobileFaceNet) is
optional but recommended:

```bash
python scripts/download_weights.py            # MobileFaceNet surrogate + ArcFace judge, SHA-256 verified
python scripts/download_weights.py --check    # verify what is on disk
```

The insightface models are released for non-commercial research use only.

---

## Usage

```bash
python main.py photo.jpg                                   # v6, standard strength -> photo_protected.png
python main.py photo.jpg --strength strong --output protected.png
python main.py --batch ./photos/ --output-dir ./protected/ --strength strong
python main.py photo.jpg --test-all                        # all four strengths, for comparison
```

| Preset | Max change on face | Steps | Protection (InSwapper, after upload) | Look |
|---|---:|---:|---:|---|
| subtle | 8/255 | 200 | not measured on the 14-face set; ~0% on tuning set | near invisible |
| standard | 12/255 | 300 | 54% | faint unevenness |
| strong | 16/255 | 300 | 96% | visible on close look |
| maximum | 20/255 | 400 | 100% | clearly visible |

About 1 minute per face on an RTX 3050 laptop GPU.

### How v6 works

- Detects every face (MTCNN) and perturbs **the face itself**, the only part a face-swap pipeline keeps,
  with a soft elliptical mask.
- Attacks an ensemble of face-recognition surrogates (FaceNet VGGFace2, FaceNet CASIA-WebFace,
  MobileFaceNet) on the exact crops they use (box crop / 5-point ArcFace alignment), with anti-aliased
  downscaling so high-res photos are attacked properly.
- Untargeted identity loss: pushes the face embedding away from the person's own clean embedding.
- Robust by construction: every step sees random box jitter, resizing, blur and a real JPEG round-trip
  (straight-through gradient).
- Perceptual controls: texture masking (noise goes into hair/brows/beard more than smooth skin), limited
  colour change, an SSIM penalty, Adam updates.
- Works at full resolution: the perturbation is added to the untouched original.

`--attack legacy` still runs the old v5 ImageNet-ensemble attack, but it is deprecated: it protected 0 of 14
faces and degrades high-res photos.

---

## Provenance signing

```bash
python -m provenance.cli keygen --out keys --name me          # keys/me.key (SECRET), keys/me.pub (share)
python main.py photo.jpg --strength strong --sign-key keys/me.key --signer-name "Your Name"
python -m provenance.cli verify suspicious.jpg --trust keys/me.pub --registry signed/ --map changed.png
```

Each signed PNG carries an Ed25519-signed manifest (signer, time, claims such as which protection was
applied, an exact pixel hash and a 128px fingerprint) and a `*.provenance.json` sidecar. Social networks
strip embedded metadata, so keep your sidecars in a folder and pass it as `--registry`: re-uploaded
copies are matched back to their manifest.

| Verdict | Meaning | Exit code |
|---|---|---|
| AUTHENTIC | Signed by a trusted key, pixels identical | 0 |
| REENCODED | Signed by a trusted key, same content after recompression/resizing | 0 |
| UNSIGNED | No manifest found. **Proves nothing either way** | 1 |
| UNTRUSTED_SIGNER | Valid signature, but not from a key you trust (e.g. an impostor's own key) | 2 |
| MODIFIED | Signed, but regions differ: `--map` outlines them (e.g. a swapped face) | 3 |
| INVALID_SIGNATURE | Manifest altered or forged | 4 |

Calibrated on 12 real photos (`evaluation/calibrate_provenance.py`): JPEG 95/75/50, social-style upload
and half-size resizing all stay REENCODED; face swaps, a blurred face and a small mouth edit are all
MODIFIED. Brightness/colour filters usually count as MODIFIED (they are edits). Crops, rotations and flips
break matching. A key proves *which key* signed, not *who* owns it: publish your public key somewhere people
already trust. This is a lightweight scheme in the spirit of C2PA Content Credentials, not C2PA itself.

---

## Evaluation

```bash
python scripts/download_weights.py --only judge   # held-out ArcFace - never used by the attack
python scripts/download_weights.py --only swap    # optional InSwapper for the face-swap test

python -m evaluation.run_eval --images ./faces --attacks face --strengths standard strong maximum \
       --swap-targets ./other_faces
python -m pytest tests -q
```

Every protected image is compared with controls (the clean photo, and random noise of equal energy) after
11 transforms: JPEG 95/75/50/30, resizing, blur, a social-media upload chain, and attacker purification
(non-local-means denoising, median filter, ¼-size resize). Results go to `eval_results/summary.md`,
`results.csv` and `faceswap.csv`. Tune presets with `evaluation/tune_visibility.py` on photos that are
**not** in your eval set.

---

## Project structure

```
deepfake-shield/
├── main.py                   CLI: protect (+ optional signing)
├── config/settings.py        presets and parameters
├── core/face_attack.py       v6 face-identity attack
├── engine/face_shield.py     v6 engine (full-resolution, metrics)
├── provenance/               Ed25519 keys, signed manifests, verification, CLI
├── evaluation/               eval harness, face-swap test, tuning, provenance calibration
├── tests/                    pytest suite
├── core/attack.py, engine/shield.py, models/   deprecated v5 attack
└── utils/                    validation, metrics, progress bar
```
