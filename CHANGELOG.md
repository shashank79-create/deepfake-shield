# Changelog

## 6.1.0

- `pip install -e .` provides `deepfake-shield`, `dfs-provenance`, `dfs-eval` and `dfs-weights` commands.
- `scripts/download_weights.py`: fetches the MobileFaceNet surrogate, the ArcFace judge and (optionally)
  InSwapper, verifying SHA-256; `--check` audits files already on disk.
- `provenance.cli verify <folder>`: one verdict per image, totals, `--csv` report, change maps, exit code
  of the worst verdict.
- Password-protected signing keys work with `main.py --sign-key` (prompted), and a bad key or password
  fails before the slow protection step.
- The compute device is chosen on first use, so imports, `--help` and the provenance CLI no longer probe CUDA.
- `examples/demo.py` rewritten for v6: `--strength`, output next to the input, a side-by-side comparison
  image, per-surrogate identity similarity, non-zero exit on failure.
- GitHub Actions runs the test suite on CPU for every push and pull request.
- Tests for the `main.py` command line (engine stubbed, no GPU needed). 42 tests in total.
- `.gitattributes` normalises line endings.

## 6.0.0

- **New face-identity attack (v6), now the default.** Perturbs the detected face at full resolution,
  attacks three face-recognition surrogates on their real crops, and is trained against JPEG, resizing and
  blur. Perceptual controls: texture masking, limited colour change, SSIM penalty.
- **Evaluation harness** with a held-out ArcFace judge, equal-energy random-noise controls, 11 real-world
  and purification transforms, and a real InSwapper face-swap test. On 14 faces, `strong` makes face swaps
  fail 96% of the time after a social-media upload (93% after attacker denoising). Visible at that strength.
- **Provenance signing**: Ed25519 keys, signed manifests (embedded in PNG and as sidecars), registry
  matching for re-uploads with stripped metadata, and localisation of edited regions.
- v5 ImageNet-ensemble attack deprecated: measured 0/14 faces protected.
- Fixed a Windows crash when output is redirected (console encoding).

## 5.0.0

- Initial release: PGD against an ImageNet model ensemble.
