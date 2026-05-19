# Refactor v2 Changelog

## Proposal method correction

The first cleaned proposal implementation was too simplified and produced weak robustness. This version restores the uploaded notebook's core math:

- Q4 candidate branch based on Hessenberg `Q` statistic:
  - `q4_stat = Q[1,1]^2 + Q[2,1]^2`.
  - Bit 1 if `q4_stat >= tau`, else bit 0.
- H-domain candidate branch:
  - uses Hessenberg coefficients `h21` and `h22`;
  - embeds bit 0 near `q/4` and bit 1 near `3q/4` modulo `q`.
- Candidate screening:
  - exact, rounded, positive drift, negative drift stress tests;
  - Bit Survival Score;
  - MSE limit and combined score.
- Structured repetition and majority voting.
- DWT mode switched to `orthonormal`, closer to `pywt.dwt2(..., 'haar')` used in the notebook.

## Optimizer control

Added explicit CLI option:

```bash
--proposal-use-optimizer
```

Optimizer is OFF by default. This is important for fair fixed-parameter comparison.

## Attack suite

Expanded `attack.py` with:

- JPEG and JPEG2000,
- Gaussian noise,
- salt-and-pepper noise,
- speckle noise,
- Poisson noise,
- median filter,
- average filter,
- lowpass filter,
- Gaussian blur,
- motion blur,
- sharpening,
- unsharp mask,
- histogram equalization,
- CLAHE-like autocontrast,
- rotation,
- translation,
- resize,
- crop-resize,
- random crop-resize,
- occlusion,
- gamma correction,
- brightness,
- contrast,
- bit-depth reduction.

## Tests

The package now has math tests for:

- DWT reconstruction and energy preservation,
- IWT reconstruction,
- multilevel IWT reconstruction,
- Arnold scrambling roundtrip,
- chaotic encrypt/decrypt roundtrip,
- SVD principal component reconstruction,
- Hessenberg reconstruction,
- PSNR/MSE/MAE/NC/NCC/BER known values,
- proposal QIM modulo math,
- proposal H-domain bit extraction,
- attack registration and deterministic stochastic attacks.

Current test status:

```text
23 passed
```
