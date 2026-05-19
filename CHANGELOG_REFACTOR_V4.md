# Refactor V4 - Proposal validation layer and expanded attacks

## New proposal validation layer

Added `src/watermarklab/proposal_validation.py` and `tools/validate_proposal.py`.

This layer validates the proposal method at four levels:

1. **Notebook contract constants**
   - 512x512 RGB host image
   - 64x64 binary watermark
   - 4x4 block size
   - Arnold iterations = 17
   - private key = `KB123`
   - DWT bands = `LL, HL, HH, LH`
   - Q4 statistic and H-domain candidate definitions

2. **Pure proposal math**
   - Q4 branch extraction rule
   - H-domain QIM rule for `h21` and `h22`
   - modulo target placement for bit 0 and bit 1

3. **Structured repetition schedule**
   - verifies payload-bit repetition count
   - verifies full 512x512 RGB capacity: 49,152 candidate blocks
   - verifies max auto repeat factor = 12 for a 64x64 watermark

4. **End-to-end proposal smoke validation**
   - embeds/extracts on a real 512x512 host and 64x64 watermark
   - checks PSNR, NC, BER, candidate modes, skip rate, and support counts

Run pure validation:

```bash
PYTHONPATH=src python tools/validate_proposal.py --no-end-to-end
```

Run validation including real image embedding/extraction:

```bash
PYTHONPATH=src python tools/validate_proposal.py \
  --host data/host/lenna.bmp \
  --watermark data/watermark/wm.png
```

## Expanded attack suite

Added new attacks to `src/watermarklab/common/attack.py`:

- affine shear
- row/column deletion and resize
- mosaic/pixelation
- posterization
- solarization
- saturation change
- channel dropout
- checkerboard cutout
- border crop/pad
- color quantization

Added `stress` preset:

```bash
PYTHONPATH=src python main.py --attack-preset stress --methods proposal --max-images 1
```

## New tests

Added:

- `tests/test_proposal_validation_layer.py`
- extra attack tests in `tests/test_attack.py`

Full result:

```text
47 passed
```

## Test policy

Default pytest run keeps the real proposal end-to-end validation skipped because it is slower and uses all 512x512 RGB real data path.

Default:

```text
46 passed, 1 skipped
```

Run the slow proposal end-to-end validation explicitly:

```bash
RUN_SLOW_PROPOSAL=1 PYTHONPATH=src pytest -q \
  tests/test_proposal_validation_layer.py::test_proposal_end_to_end_validation_layer_on_real_input
```

Expected:

```text
1 passed
```
