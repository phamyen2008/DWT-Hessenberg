# Changelog v5 — four uploaded baselines

## Added

- Added `Gaata2022DWTHessFWA` wrapper for the DWT + Hessenberg + FWA paper.
- Added `Mahto2022FireflyDual` wrapper for the firefly-optimized dual/multi watermarking paper.
- Vendored missing baseline support code under `src/watermarklab/vendor/`.
- Added `--attack-preset none` for clean/no-attack validation.
- Added paper-similarity tests in `tests/test_uploaded_baseline_paper_report_similarity.py`.
- Added `docs/BASELINE_VALIDATION_V5.md`.

## Verified

```text
52 passed, 1 skipped
```

## Note

The baselines are paper-guided reproductions.  They document assumptions where the original reports do not specify enough implementation details for byte-identical reproduction.
