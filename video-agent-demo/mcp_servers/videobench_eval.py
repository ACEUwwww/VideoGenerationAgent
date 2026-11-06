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


MLLM_DIMENSIONS = ("imaging_quality", "aesthetic_quality", "motion_effects", "object_consistency")


def dimension_prompt(dimensions: tuple[str, ...], shot: dict) -> str:
    """One judging prompt covering every MLLM dimension in a single call.

    Scoring each dimension separately costs an image upload each time; the
    vision model sees the same frame regardless, so one call returning all
    dimensions is both cheaper and more coherent.
    """
    lines = [
        "你是竖屏短剧成片的视觉审片人。请逐项评估这张关键帧，按 1-5 分打分（5 最好）。",
        "",
        "评估维度：",
    ]
    for name in dimensions:
        _, description = DIMENSIONS[name]
        lines.append(f"- {name}: {description}")
    lines += [
        "",
        f"分镜要求：{json.dumps(shot, ensure_ascii=False)}",
        "",
        "重点检查：人物面部是否变形、五官是否错位、肢体是否畸形、文字是否乱码。",
        "只输出 JSON，每个维度一个 1-5 的整数分数，另给一句理由：",
        '{"imaging_quality": 4, "aesthetic_quality": 4, "motion_effects": 4, '
        '"object_consistency": 4, "reason": "..."}',
    ]
    return "\n".join(lines)


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
    keyframe_path: Path | None = None,
    frame_count: int = 8,
) -> dict:
    """Score one clip across every dimension.

    ``mllm_judge(image_path, prompt) -> (scores_dict, reason)`` is optional.
    It receives the full-resolution keyframe, not the 192px thumbnails used for
    pixel measurement: at that size a vision model cannot see a deformed face,
    which is exactly what it is there to catch. The judge covers every
    dimension in one call and its 1-5 scores are max-merged with the measured
    estimate, so a run with a vision model never silently loses the measured
    value.
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

    judged: dict[str, int] = {}
    judge_reason = ""
    judge_error = ""
    if mllm_judge and keyframe_path is not None:
        scores, reason = mllm_judge(keyframe_path, dimension_prompt(MLLM_DIMENSIONS, shot))
        for name, value in (scores or {}).items():
            if name in MLLM_DIMENSIONS:
                try:
                    judged[name] = int(value)
                except (TypeError, ValueError):
                    continue
        judge_reason = reason or ""
        if not judged:
            judge_error = reason or "vision judge returned no usable scores"

    dimensions: dict[str, dict] = {}
    for name, (weight, description) in DIMENSIONS.items():
        entry = dict(local.get(name, {"score": 0.0, "detail": {}}))
        entry["weight"] = weight
        entry["description"] = description
        if name in ("temporal_consistency", "color_consistency", "scene_consistency", "object_consistency"):
            entry["measured"] = True
        if name in judged:
            score = judged[name]
            entry["mllm_score_1to5"] = score
            entry["mllm_reason"] = judge_reason
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
