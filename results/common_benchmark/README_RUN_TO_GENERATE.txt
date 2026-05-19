This folder is intentionally not pre-filled with a full benchmark because the user requested code/test corrections rather than running all metrics.

Generate the full comparison table with:

PYTHONPATH=src python main.py --host-dir data/host --watermark data/watermark/wm.png --output results/common_benchmark --attack-preset lite --proposal-repeat 3

For the larger attack suite, use --attack-preset full.
For notebook-style full proposal repetition, use --proposal-repeat auto (much slower).
