"""Recompute a clip score from the published formulas and compare.

`bench/results/scoring_rules.md` states every constant in the scoring path. This
test exists so that document cannot silently drift from the code: it recomputes
the seven dimensions and the weighted overall by hand, using only the formulas
written there, and asserts the result equals what `evaluate_clip` returns.

If someone changes a threshold in `metrics.py` without updating the spec — or
the reverse — this test fails and names the dimension.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcp_servers.metrics import compare_to_reference, measure, score_dimensions
from mcp_servers.videobench_eval import DIMENSIONS, PASS_SCORE, evaluate_clip

# The constants the spec quotes. Written out here rather than imported on
# purpose: importing them would let the code define its own answer.
SHARPNESS_LO, SHARPNESS_HI = 8.0, 90.0
CONTRAST_LO, CONTRAST_HI = 18.0, 62.0
COLORFUL_LO, COLORFUL_HI = 12.0, 62.0
EXPOSURE_TARGET = 118.0
MOTION_BAND = (0.5, 8.0, 45.0, 70.0)
DELTA_STD_LO, DELTA_STD_HI = 8.0, 30.0
FLICKER_LO, FLICKER_HI = 12.0, 40.0
JITTER_LO, JITTER_HI = 6.0, 30.0

WEIGHTS = {
    "imaging_quality": 0.15,
    "aesthetic_quality": 0.15,
    "temporal_consistency": 0.20,
    "motion_effects": 0.15,
    "color_consistency": 0.10,
    "scene_consistency": 0.15,
    "object_consistency": 0.10,
}


def _squash(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return float(min(max((value - low) / (high - low), 0.0), 1.0))


def _band(value: float, low: float, ideal_low: float, ideal_high: float, high: float) -> float:
    if value < ideal_low:
        return _squash(value, low, ideal_low)
    if value > ideal_high:
        return 1.0 - _squash(value, ideal_high, high)
    return 1.0


def _frame(value: int, noise: int = 0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((96, 172, 3), value, dtype=np.float32)
    if noise:
        base = base + rng.normal(0, noise, base.shape).astype(np.float32)
    return np.clip(base, 0, 255)


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert set(WEIGHTS) == set(DIMENSIONS)
    for name, weight in WEIGHTS.items():
        assert DIMENSIONS[name][0] == weight


def test_hand_computed_dimensions_match_code():
    """Every dimension recomputed from the spec's formulas, not from the code."""
    frames = [_frame(60 + 12 * i, noise=14, seed=i) for i in range(6)]
    reference = _frame(70, noise=10, seed=99)

    raw = measure(frames)
    ref = compare_to_reference(frames, reference)
    produced = score_dimensions(raw, ref)

    sharp = _squash(raw["sharpness"], SHARPNESS_LO, SHARPNESS_HI)
    contrast = _squash(raw["contrast"], CONTRAST_LO, CONTRAST_HI)
    colorful = _squash(raw["colorfulness"], COLORFUL_LO, COLORFUL_HI)
    exposure = 1.0 - min(abs(raw["brightness"] - EXPOSURE_TARGET) / EXPOSURE_TARGET, 1.0)
    motion = _band(raw["mean_frame_delta"], *MOTION_BAND)
    smooth = 1.0 - _squash(raw["flicker"], FLICKER_LO, FLICKER_HI)
    stable = 1.0 - _squash(raw["brightness_jitter"], JITTER_LO, JITTER_HI)

    expected = {
        "imaging_quality": round(float(np.mean([sharp, contrast, exposure])), 3),
        "aesthetic_quality": round(float(np.mean([colorful, exposure, contrast])), 3),
        "temporal_consistency": round(float(np.mean([smooth, stable])), 3),
        "motion_effects": round(
            float(np.mean([motion, 1.0 - _squash(raw["delta_std"], DELTA_STD_LO, DELTA_STD_HI)])), 3
        ),
        "color_consistency": round(ref["color_consistency"], 3),
        "scene_consistency": round(ref["structure_consistency"], 3),
        "object_consistency": round(
            float(np.mean([ref["subject_consistency"], ref["subject_color_consistency"]])), 3
        ),
    }

    for name, value in expected.items():
        assert produced[name]["score"] == pytest.approx(value, abs=1e-3), name


def test_overall_is_weighted_mean_over_scored_dimensions(tmp_path):
    """The weighted average in the spec, recomputed against evaluate_clip."""
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    report = evaluate_clip(clip, shot={"shot_size": "中景"}, reference_frame=_frame(70, noise=10, seed=5))
    scored = {n: e for n, e in report["dimensions"].items() if e["score"] is not None}

    manual = sum(e["score"] * WEIGHTS[n] for n, e in scored.items()) / sum(WEIGHTS[n] for n in scored)
    assert report["score"] == pytest.approx(round(manual, 3), abs=1e-3)
    assert report["passed"] == (report["score"] >= PASS_SCORE)


def test_cross_framing_drops_scene_from_the_denominator(tmp_path):
    """A non-applicable dimension is removed from the denominator, not scored 0."""
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    report = evaluate_clip(
        clip,
        shot={"shot_size": "全景"},
        reference_frame=_frame(70, noise=10, seed=5),
        reference_shot_size="特写",
    )
    scene = report["dimensions"]["scene_consistency"]
    assert scene["applicable"] is False
    assert scene["score"] is None
    assert scene["measured_score"] is not None

    scored = [n for n, e in report["dimensions"].items() if e["score"] is not None]
    assert "scene_consistency" not in scored

    manual = (sum(e["score"] * WEIGHTS[n] for n, e in report["dimensions"].items()
                  if e["score"] is not None)
              / sum(WEIGHTS[n] for n in scored))
    assert report["score"] == pytest.approx(round(manual, 3), abs=1e-3)


def _write_clip(path) -> None:
    """A tiny real H.264 clip, because the scorer decodes frames from a file."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available")
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=192x342:rate=10:duration=2",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
