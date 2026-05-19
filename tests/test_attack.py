import numpy as np
from watermarklab.common.attack import (
    default_attack_suite,
    full_attack_suite,
    stress_attack_suite,
    apply_attack,
    AttackConfig,
    available_attack_groups,
)


def test_lite_attacks_preserve_shape_dtype_and_range():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    for cfg in default_attack_suite(preset="lite"):
        out = apply_attack(img, cfg)
        assert out.shape == img.shape, cfg.name
        assert out.dtype == np.uint8, cfg.name
        assert out.min() >= 0 and out.max() <= 255, cfg.name


def test_full_attack_suite_has_many_attack_types_and_all_are_registered():
    suite = full_attack_suite(include_none=True)
    groups = {cfg.group for cfg in suite}
    assert len(suite) >= 55
    assert groups.issubset(set(available_attack_groups()))
    required = {
        "jpeg", "jpeg2000", "gaussian_noise", "salt_pepper", "speckle_noise", "poisson_noise",
        "median_filter", "average_filter", "gaussian_blur", "lowpass", "motion_blur",
        "sharpen", "unsharp_mask", "hist_equalization", "rotation", "translation",
        "resize", "crop_resize", "occlusion", "gamma", "brightness", "contrast", "bit_depth",
        "shear", "row_col_delete", "mosaic", "posterize", "solarize", "saturation",
        "channel_dropout", "checkerboard_cutout", "border_crop_pad", "color_quantization",
    }
    assert required.issubset(groups)


def test_representative_full_attacks_preserve_shape_dtype_and_range():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    representatives = [
        AttackConfig("jpeg", "jpeg", {"quality": 50}),
        AttackConfig("noise", "poisson_noise", {"peak": 64, "seed": 7}),
        AttackConfig("motion", "motion_blur", {"size": 7}),
        AttackConfig("translate", "translation", {"shift_x": 3, "shift_y": -2}),
        AttackConfig("bitdepth", "bit_depth", {"bits": 5}),
    ]
    for cfg in representatives:
        out = apply_attack(img, cfg)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255


def test_stochastic_attack_is_deterministic_with_seed():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cfg = AttackConfig("gaussian", "gaussian_noise", {"sigma": 5.0, "seed": 999})
    assert np.array_equal(apply_attack(img, cfg), apply_attack(img, cfg))


def test_new_attack_groups_preserve_shape_dtype_and_range():
    rng = np.random.default_rng(2026)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    representatives = [
        AttackConfig("shear", "shear", {"shear_x": 0.08}),
        AttackConfig("rowcol", "row_col_delete", {"rows": 3, "cols": 2, "seed": 5}),
        AttackConfig("mosaic", "mosaic", {"factor": 8}),
        AttackConfig("posterize", "posterize", {"bits": 4}),
        AttackConfig("solarize", "solarize", {"threshold": 128}),
        AttackConfig("saturation", "saturation", {"factor": 0.5}),
        AttackConfig("dropout", "channel_dropout", {"channel": 1, "value": 0}),
        AttackConfig("checker", "checkerboard_cutout", {"tile": 16, "value": 0}),
        AttackConfig("border", "border_crop_pad", {"pixels": 4, "value": 0}),
        AttackConfig("quant", "color_quantization", {"colors": 16}),
    ]
    for cfg in representatives:
        out = apply_attack(img, cfg)
        assert out.shape == img.shape, cfg.name
        assert out.dtype == np.uint8, cfg.name
        assert out.min() >= 0 and out.max() <= 255, cfg.name


def test_stress_attack_suite_is_compact_and_registered():
    suite = stress_attack_suite(include_none=True)
    assert 10 <= len(suite) <= 25
    groups = {cfg.group for cfg in suite}
    assert groups.issubset(set(available_attack_groups()))
    assert {"jpeg", "rotation", "shear", "row_col_delete", "mosaic", "posterize"}.issubset(groups)
    assert len(default_attack_suite(preset="stress")) == len(suite)
