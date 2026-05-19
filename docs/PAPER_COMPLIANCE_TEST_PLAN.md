# Paper-compliance and math test plan

This project separates two goals:

1. **Paper-compliance tests**: verify that each baseline follows the algorithmic contract described in the paper as closely as possible under the common benchmark input format: 512x512 RGB host + 64x64 binary watermark.
2. **Math-correctness tests**: verify that shared mathematical modules implement the intended equations, round-trip properties, and extraction inverses.

## Common benchmark adaptation rule

Some baselines originally use 256x256 grayscale watermarks. For the proposal-paper comparison, every method is adapted to consume the same 64x64 binary watermark. Tests therefore check both:

- the original paper mechanism, such as transform domain, channel, decomposition, and embedding equation;
- the common input contract used for fair comparison.

## Test files

| Test file | Purpose |
|---|---|
| `tests/test_common_paper_math.py` | Checks YCbCr, Haar-DWT, LWT/IWT, entropy, Arnold map, logistic map, SVD, Hessenberg, and alpha-blending equations. |
| `tests/test_baseline_paper_compliance.py` | Checks every baseline method against its paper-level algorithmic contract and the common 512x512/64x64 input interface. |
| `tests/test_math_core.py` | Checks lower-level chaos, SVD, Hessenberg, and proposal QIM/H-domain functions. |
| `tests/test_attack.py` | Checks attack shape, dtype, value range, deterministic seeding, and preset coverage. |
| `tests/test_methods_smoke.py` | Checks that all methods run end-to-end on real input. |

## Baseline contracts checked

### Roy 2018 DWT-SVD

- RGB -> YCbCr.
- Embed in Y component.
- Split Y into 32x32 blocks.
- Three-level DWT gives 4x4 LL block.
- Split 64x64 watermark into 4x4 blocks.
- Store SVD side information for non-blind extraction.

### Kumar 2021 LWT-Entropy

- RGB -> YCbCr.
- Embed in Y component.
- Use LWT/IWT, not ordinary DWT.
- Select the HH block by maximizing B_E = E_v - E_e.
- Use Arnold Cat Map with 50 iterations.
- Use alpha blending in the selected LWT-HH block.

### IWT-Hess-SVD 2024

- Use red channel of RGB cover image.
- Use 3-level IWT for cover image.
- Use 2-level IWT for watermark preprocessing.
- Use Hessenberg + SVD.
- Modify principal components rather than singular values.
- Keep green and blue channels unchanged.

### DWT-HD-SVD 2025

- RGB -> YCbCr.
- Embed in Y component.
- Use one-level Haar DWT.
- Use Hessenberg decomposition of LL.
- Use SVD on the Hessenberg matrix.
- Use logistic chaotic watermark encryption.
- Use paper Eq. (17): S'_H = alpha*S_w + (1-alpha)*S_H.
- Common-benchmark adaptation uses a 64x64 LL-ROI because the user's watermark is 64x64.

### Proposal method

- Optimizer is OFF by default for fair deterministic benchmark.
- Optimizer can be enabled explicitly with `use_optimizer=True` or CLI `--proposal-use-optimizer`.

## How to run

```bash
PYTHONPATH=src pytest -q
```

Expected result in this release:

```text
40 passed
```
