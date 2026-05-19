# Refactor Summary

## What changed

1. Reorganized the project into a clean Python package under `src/watermarklab`.
2. Added shared reusable modules:
   - `common/dwt.py` for Haar DWT/IDWT.
   - `common/iwt.py` for lifting-based IWT/IIWT.
   - `common/metrics.py` for PSNR, SSIM, NC, NCC, and BER.
   - `common/attack.py` for deterministic attack tests.
   - `common/io_utils.py` for strict 512x512 RGB host loading and 64x64 binary watermark loading.
3. Converted the proposal notebook idea into `methods/proposal_qh_dwt_hess.py`.
4. Added a common method interface: `embed(host_rgb, watermark_binary)` and `extract(image, key, host_rgb=None)`.
5. Added a unified benchmark runner in `src/watermarklab/benchmark.py` and `main.py`.
6. Added pytest tests for input loading, DWT/IWT reconstruction, metrics, attacks, and method smoke testing.
7. Added common-benchmark adaptation for methods whose original papers used 256x256 watermarks.

## Important note about paper faithfulness

The clean benchmark uses the user's requested common input format:

- Host: 512x512 RGB 24-bit images.
- Watermark: 64x64 binary image.

When an original paper used a different watermark size, the method includes an explicit adapter. This is necessary for fair comparison in the new proposal paper, but it is not identical to the original paper's native experiment setting.

## Included result table

`results/common_benchmark/compare_psnr_nc_ber_ncc_before_after_attack.csv` is a smoke benchmark table generated on the first real host image. To regenerate the full 11-image benchmark, run:

```bash
python main.py --host-dir data/host --watermark data/watermark/wm.png --output results/common_benchmark
```
