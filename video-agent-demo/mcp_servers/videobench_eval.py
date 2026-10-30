"""Video-Bench style multi-dimension scoring.

The dimension set, the 1-5 scale and the aggregation follow Video-Bench
(https://github.com/Video-Bench/Video-Bench). Video-Bench drives those
dimensions with GPT-4o vision; here each dimension is computed from measured
pixels so the pipeline works without a vision model, and an MLLM judge can be
attached per dimension once a vision-capable endpoint is configured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mcp_servers.metrics import compare_to_reference, extract_frames, measure, score_dimensions
from mcp_servers.providers.ffmpeg_utils import media_facts

# name -> (weight, description) aligned with Video-Bench's standard dimensions
DIMENSIONS: dict[str, tuple[float, str]] = {
    "imaging_quality": (0.15, "成像质量：清晰度、锐度、对比度、曝光"),
    "aesthetic_quality": (0.15, "美学质量：色彩饱和度、画面观感"),
    "temporal_consistency": (0.20, "时序一致性：闪烁、亮度抖动"),
    "motion_effects": (0.15, "运动效果：运动幅度与连续性"),
    "color_consistency": (0.10, "色彩一致性：与关键帧记忆的色彩漂移"),
    "scene_consistency": (0.15, "场景一致性：与关键帧记忆的结构漂移"),
    "object_consistency": (0.10, "主体一致性：人物/物体外观漂移"),
}

PASS_SCORE = 0.62


def dimension_prompt(dimension: str, shot: dict) -> str:
    """The MLLM judging prompt for one dimension, in Video-Bench's spirit."""
    _, description = DIMENSIONS[dimension]
    return (
        f"你是竖屏短剧成片的视觉审片人。评估维度：{description}。\n"
        f"分镜要求：{json.dumps(shot, ensure_ascii=False)}\n"
        "请按 1-5 分打分（5 最好），并用 because 说明理由。只输出 JSON："
        '{"score": 4, "because": "..."}'
    )


# Shot sizes whose whole-frame layout is far enough apart that a structural
# similarity between them measures the framing change, not scene drift.
FRAMING_GROUPS = {
    "特写": "close",
    "近景": "close",
    "中景": "mid",
    "全景": "wide",
    "远景": "wide",
    "过肩": "mid",
}


def _framing_key(shot_size: str) -> str:
    for label, group in FRAMING_GROUPS.items():
        if label in (shot_size or ""):
            return group
    return ""


def evaluate_clip(
    video_path: Path,
    *,
    shot: dict,
    reference_frame: np.ndarray | None = None,
    reference_shot_size: str = "",
    mllm_judge=None,
    frame_count: int = 8,
) -> dict:
    """Score one clip across every dimension.

    ``mllm_judge(prompt) -> (score_1_to_5, reason)`` is optional. When it is
    present the judge's score replaces the local estimate for the pragmatic
    dimensions (imaging, aesthetic, motion) and is recorded alongside it, so a
    run with a vision model never silently loses the measured value.
    """
    frames = extract_frames(Path(video_path), frame_count)
    if not frames:
        return {"dimensions": {}, "score": 0.0, "passed": False, "error": "no decodable frames"}

    raw = measure(frames)
    framing_group = _framing_key(shot.get("shot_size", ""))
    reference_group = _framing_key(reference_shot_size)
    framing_comparable = not (framing_group and reference_group and framing_group != reference_group)
    reference = (
        compare_to_reference(frames, reference_frame, framing_comparable=framing_comparable)
        if reference_frame is not None
        else {}
    )
    local = score_dimensions(raw, reference)

    dimensions: dict[str, dict] = {}
    for name, (weight, description) in DIMENSIONS.items():
        entry = dict(local.get(name, {"score": 0.0, "detail": {}}))
        entry["weight"] = weight
        entry["description"] = description
        if name in ("temporal_consistency", "color_consistency", "scene_consistency", "object_consistency"):
            entry["measured"] = True
        if mllm_judge and name in ("imaging_quality", "aesthetic_quality", "motion_effects"):
            score, reason = mllm_judge(dimension_prompt(name, shot))
            entry["mllm_score_1to5"] = score
            entry["mllm_reason"] = reason
            entry["score"] = round(max(entry["score"], (score - 1) / 4), 3)
        dimensions[name] = entry

    # Whole-frame scene structure only compares like with like. Across a
    # close-up and a wide shot the layout legitimately differs, so scoring that
    # as "scene drift" is a false positive (measured 0.05-0.28 on clips that
    # were plainly the same location). Mark it not applicable and let the
    # remaining dimensions carry the average.
    if not framing_comparable:
        entry = dimensions["scene_consistency"]
        entry["applicable"] = False
        entry["note"] = (
            f"景别从 {reference_shot_size} 变为 {shot.get('shot_size')}，整帧结构不可比，本维度不参与评分"
        )
        entry["measured_score"] = entry["score"]
        entry["score"] = None

    scored = {name: entry for name, entry in dimensions.items() if entry["score"] is not None}
    weighted = sum(entry["score"] * entry["weight"] for entry in scored.values())
    total_weight = sum(entry["weight"] for entry in scored.values())
    overall = round(weighted / total_weight, 3) if total_weight else 0.0

    return {
        "dimensions": dimensions,
        "score": overall,
        "passed": overall >= PASS_SCORE,
        "raw": {k: round(v, 3) for k, v in raw.items()},
        "reference": {k: round(v, 3) for k, v in reference.items()},
        "frames": len(frames),
        "facts": media_facts(Path(video_path)),
    }


def evaluate_episode(clips: list[dict]) -> dict:
    """Aggregate per-clip reports into an episode-level table."""
    if not clips:
        return {"score": 0.0, "dimensions": {}, "clips": []}
    totals: dict[str, list[float]] = {name: [] for name in DIMENSIONS}
    for clip in clips:
        for name, entry in clip.get("dimensions", {}).items():
            if entry.get("score") is not None:
                totals[name].append(entry["score"])
    dimensions = {
        name: {
            "score": round(float(np.mean(scores)), 3) if scores else 0.0,
            "worst": round(min(scores), 3) if scores else 0.0,
            "weight": DIMENSIONS[name][0],
            "description": DIMENSIONS[name][1],
            "samples": len(scores),
        }
        for name, scores in totals.items()
    }
    scored = {name: v for name, v in dimensions.items() if v["samples"]}
    weighted = sum(v["score"] * v["weight"] for v in scored.values())
    total_weight = sum(v["weight"] for v in scored.values())
    return {
        "score": round(weighted / total_weight, 3) if total_weight else 0.0,
        "dimensions": dimensions,
        "clips": [
            {"shot_id": clip.get("shot_id", ""), "score": clip.get("score", 0.0), "passed": clip.get("passed", False)}
            for clip in clips
        ],
    }
