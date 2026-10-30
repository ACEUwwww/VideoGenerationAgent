from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mcp_servers.metrics import center_crop, compare_to_reference
from mcp_servers.videobench_eval import DIMENSIONS, evaluate_clip

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None, reason="ffmpeg is required"
)


def _frame(value: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(np.full((342, 192, 3), value, dtype=np.float32) + rng.normal(0, 6, (342, 192, 3)), 0, 255)


def _clip(tmp_path: Path, frames: int = 6) -> Path:
    """Render a tiny real mp4 so the evaluator runs its full path."""
    import subprocess

    out = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size=192x342:rate=24:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


def test_center_crop_keeps_the_middle():
    frame = np.zeros((100, 100, 3), dtype=np.float32)
    frame[40:60, 40:60] = 255
    crop = center_crop(frame)
    assert crop.shape[:2] == (50, 50)
    assert crop.mean() > 0, "the bright centre must survive the crop"


def test_cross_framing_marks_scene_not_applicable(tmp_path):
    clip = _clip(tmp_path)
    shot = {"id": "shot_002", "shot_size": "特写", "duration_s": 2.0}
    report = evaluate_clip(clip, shot=shot, reference_frame=_frame(120), reference_shot_size="全景")
    scene = report["dimensions"]["scene_consistency"]
    assert scene["score"] is None
    assert scene["applicable"] is False
    assert scene["measured_score"] is not None, "the raw measurement is kept for diagnosis"
    assert report["score"] > 0


def test_same_framing_still_scores_scene(tmp_path):
    clip = _clip(tmp_path)
    shot = {"id": "shot_002", "shot_size": "特写", "duration_s": 2.0}
    report = evaluate_clip(clip, shot=shot, reference_frame=_frame(120), reference_shot_size="近景")
    scene = report["dimensions"]["scene_consistency"]
    assert scene["score"] is not None
    assert "applicable" not in scene or scene.get("applicable") is not False


def test_not_applicable_dimension_leaves_the_weighted_average(tmp_path):
    clip = _clip(tmp_path)
    comparable = evaluate_clip(clip, shot={"shot_size": "近景", "duration_s": 2.0}, reference_frame=_frame(120), reference_shot_size="特写")
    skipped = evaluate_clip(clip, shot={"shot_size": "全景", "duration_s": 2.0}, reference_frame=_frame(120), reference_shot_size="特写")
    scored = sum(1 for entry in skipped["dimensions"].values() if entry["score"] is not None)
    assert scored == len(DIMENSIONS) - 1
    assert comparable["dimensions"]["scene_consistency"]["score"] is not None


def test_framing_unknown_does_not_skip(tmp_path):
    clip = _clip(tmp_path)
    report = evaluate_clip(clip, shot={"shot_size": "中景", "duration_s": 2.0}, reference_frame=_frame(120), reference_shot_size="")
    assert report["dimensions"]["scene_consistency"]["score"] is not None


def test_subject_consistency_ignores_camera_move():
    reference = _frame(90, seed=1)
    same_subject = [reference]
    result = compare_to_reference(same_subject, reference, framing_comparable=True)
    assert result["subject_consistency"] > 0.95
    drifted = [np.clip(reference * 0.2 + 60, 0, 255)]
    assert compare_to_reference(drifted, reference)["subject_consistency"] < result["subject_consistency"]
