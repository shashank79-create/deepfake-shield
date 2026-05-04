# DeepFake Shield

**Adversarial image protection against AI-powered deepfake generation.**

Adds imperceptible perturbations to photos that cause deepfake models to reconstruct the wrong identity — protecting your images from being used to generate fake content.

---

## The Problem

Deepfake generation models (SimSwap, DeepFaceLab, Stable Diffusion inpainting) encode identity from the full image context. Anyone with your photo can generate convincing fake videos or images of you saying or doing things you never did.

Existing solutions either visibly distort images or fail after social media recompression.

## How It Works

1. **4-model ensemble attack** — adversarial perturbations computed across ResNet50, EfficientNet-B0, ViT-B/16, and ConvNeXt-Tiny simultaneously. Attacks that fool multiple architectures transfer to unknown real-world deepfake models.

2. **Directed PGD toward random targets** — pushes your image's feature vector toward a random identity in feature space, not just "away from original." Far stronger than undirected attacks.

3. **Dual-zone epsilon masking** — face skin gets low perturbation (2–5/255, visually clean). Hair, background, and clothing absorb high perturbation (8–20/255, hidden in texture). Result: the face looks completely natural while the attack is strong.

4. **JPEG simulation in training loop** — every N steps, perturbations are tested against JPEG compression. Ensures protection survives Instagram, WhatsApp, and Twitter recompression.

5. **SSIM quality gate** — if visual quality drops below threshold, epsilon is automatically reduced and the attack reruns. Self-healing pipeline.

---

## Installation

```bash
git clone https://github.com/shashank79-create/deepfake-shield
cd deepfake-shield
pip install -r requirements.txt
```

Optional (enables face-aware attack, better results):
```bash
pip install facenet-pytorch
```

---

## Usage

```bash
# Protect a single image (standard strength, recommended)
python main.py photo.jpg

# Specify output path and strength
python main.py photo.jpg --strength strong --output protected.png

# Maximum protection
python main.py photo.jpg --strength maximum --verbose

# Compare all 4 strength levels
python main.py photo.jpg --test-all

# Batch protect an entire folder
python main.py --batch ./photos/ --output-dir ./protected/ --strength strong
```

### Strength Presets

| Preset   | Steps | Face ε  | Background ε | Use Case                    |
|----------|-------|---------|-------------|------------------------------|
| subtle   | 100   | 2/255   | 8/255       | Maximum visual quality       |
| standard | 200   | 3/255   | 12/255      | Recommended balance          |
| strong   | 300   | 4/255   | 16/255      | High-risk environments       |
| maximum  | 400   | 5/255   | 20/255      | Critical protection needed   |

---

## Project Structure

```
deepfake-shield/
├── main.py              ← CLI entry point
├── requirements.txt
├── config/
│   └── settings.py      ← All parameters (device, presets, thresholds)
├── core/
│   ├── attack.py        ← PGD adversarial attack
│   ├── losses.py        ← Cosine + MSE loss functions
│   └── noise.py         ← Gaussian smoothing, TV loss, JPEG simulation
├── models/
│   └── loader.py        ← Ensemble model loading
├── utils/
│   ├── validator.py     ← Input validation
│   ├── metrics.py       ← SSIM, PSNR, L-inf (pure PyTorch)
│   └── progress.py      ← Terminal progress bar
├── engine/
│   └── shield.py        ← Main orchestrator (GoldStandardEngine)
└── examples/
    └── demo.py          ← Quick demo script
```

---

## Technical Details

- **Attack**: Projected Gradient Descent (PGD) with Adam optimizer
- **Loss**: 70% cosine distance + 30% MSE in feature space
- **Quality**: SSIM > 0.97, PSNR > 38dB, max pixel change < 16/255
- **Robustness**: Tested against JPEG compression at quality=75
- **Device**: Auto-detects CUDA GPU, falls back to CPU
