from __future__ import annotations

import numpy as np
import pytest

from mcp_servers.metrics import compare_to_reference, measure, score_dimensions


def _frame(value: int, noise: int = 0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((96, 172, 3), value, dtype=np.float32)
    if noise:
        base = base + rng.normal(0, noise, base.shape).astype(np.float32)
    return np.clip(base, 0, 255)


def test_identical_frames_have_no_drift():
    frame = _frame(120, noise=8, seed=1)
    result = compare_to_reference([frame, frame], frame)
    assert result["color_consistency"] > 0.99
    assert result["structure_consistency"] > 0.99


def test_different_frames_drift():
    reference = _frame(30, noise=5, seed=2)
    other = _frame(220, noise=40, seed=3)
    result = compare_to_reference([other], reference)
    assert result["structure_consistency"] < 0.9


def test_flat_frame_has_low_sharpness():
    flat = measure([_frame(120)] * 4)
    noisy = measure([_frame(120, noise=30, seed=i) for i in range(4)])
    assert flat["sharpness"] < noisy["sharpness"]


def test_motion_detected_between_frames():
    static = measure([_frame(80)] * 4)
    moving = measure([_frame(20 * i) for i in range(4)])
    assert moving["mean_frame_delta"] > static["mean_frame_delta"]


def test_scores_are_bounded():
    raw = measure([_frame(120, noise=12, seed=i) for i in range(4)])
    reference = compare_to_reference([_frame(120, noise=12, seed=0)], _frame(120, noise=12, seed=0))
    scores = score_dimensions(raw, reference)
    for name, entry in scores.items():
        assert 0.0 <= entry["score"] <= 1.0, name


def test_empty_input_returns_empty():
    assert measure([]) == {}
    assert score_dimensions({}) == {}
