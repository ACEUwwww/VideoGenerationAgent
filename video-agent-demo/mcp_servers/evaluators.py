"""Deterministic media QA.

Scores come from measured file facts (ffprobe) plus the prompt contract that
produced them, never from a model pretending to look at pixels it cannot see.
The LLM review step in the agent adds editorial feedback on top; this module is
the part that can actually catch a broken render.
"""

from __future__ import annotations

from pathlib import Path

from mcp_servers.providers.ffmpeg_utils import media_facts

IMAGE_PASS = 0.78
VIDEO_PASS = 0.78
# Vendor models return their own native sizes (Seedance 720p/9:16 gives 704x1248).
# The composer normalises everything to the canvas, so QA must judge aspect ratio
# and a usable pixel height instead of demanding an exact canvas match.
MIN_VIDEO_HEIGHT = 720
ASPECT_TOLERANCE = 0.03
# A mock render can never satisfy the real-art gate. It is scored on the parts
# that are genuinely valid (size, duration, decodability) but the verdict is
# "not a real take", which the retry logic treats as a retry-without-value:
# repairing the prompt cannot change what a placeholder draws.
MOCK_VERDICT_PASSED = False


def evaluate_image(
    image_path: Path,
    *,
    width: int,
    height: int,
    prompt: str,
    anchor: str,
    provider: str,
) -> dict:
    facts = media_facts(image_path)
    issues: list[str] = []
    score = 1.0

    if not facts["exists"]:
        return {"score": 0.0, "passed": False, "issues": ["关键帧文件不存在"], "feedback": "重新生成关键帧。", "facts": facts}

    actual_w, actual_h = facts.get("width", 0), facts.get("height", 0)
    if (actual_w, actual_h) != (width, height):
        issues.append(f"分辨率 {actual_w}x{actual_h}，需要 {width}x{height}")
        score -= 0.35
    if actual_h and abs((actual_w / actual_h) - (width / height)) > 0.02:
        issues.append("不是竖屏 9:16 构图")
        score -= 0.3
    if facts["bytes"] < 20_000:
        issues.append("图片体积异常小，疑似占位图")
        score -= 0.5
    if anchor and anchor.lower() not in prompt.lower():
        issues.append("提示词中缺少角色外观锚点，跨镜一致性会失效")
        score -= 0.2
    if len(prompt) < 120:
        issues.append("提示词过短，缺少构图/光线/镜头语言")
        score -= 0.15

    score = round(max(score, 0.0), 2)
    passed = score >= IMAGE_PASS and not any("不存在" in i for i in issues)
    if provider == "mock":
        return {
            "score": score,
            "passed": MOCK_VERDICT_PASSED,
            "mock": True,
            "issues": issues,
            "feedback": "当前为 mock 关键帧，非真实生成画面；跳过返工重试。",
            "facts": facts,
        }
    return {
        "score": score,
        "passed": passed,
        "mock": False,
        "issues": issues,
        "feedback": "；".join(issues) if issues else "关键帧通过质检。",
        "facts": facts,
    }


def evaluate_video(
    video_path: Path,
    *,
    duration_s: float,
    width: int,
    height: int,
    provider: str,
) -> dict:
    facts = media_facts(video_path)
    issues: list[str] = []
    score = 1.0

    if not facts["exists"]:
        return {"score": 0.0, "passed": False, "issues": ["视频文件不存在"], "feedback": "重新生成视频片段。", "facts": facts}

    actual_w, actual_h = facts.get("width", 0), facts.get("height", 0)
    target_aspect = width / height
    if actual_h and actual_w:
        aspect = actual_w / actual_h
        if abs(aspect - target_aspect) > ASPECT_TOLERANCE:
            issues.append(f"画幅 {actual_w}x{actual_h}（{aspect:.3f}）不是 {target_aspect:.3f} 的竖屏比")
            score -= 0.35
        elif actual_h < MIN_VIDEO_HEIGHT:
            issues.append(f"分辨率 {actual_w}x{actual_h} 低于可用下限 {MIN_VIDEO_HEIGHT}p")
            score -= 0.2
    actual_duration = facts.get("duration_s", 0.0)
    if actual_duration <= 0:
        issues.append("无法读取时长，文件可能损坏")
        score -= 0.6
    elif abs(actual_duration - duration_s) > max(0.5, duration_s * 0.25):
        issues.append(f"时长 {actual_duration:.2f}s 偏离目标 {duration_s:.2f}s")
        score -= 0.2
    if not facts.get("codec"):
        issues.append("没有可解码的视频流")
        score -= 0.6
    score = round(max(score, 0.0), 2)
    if provider == "mock":
        return {
            "score": score,
            "passed": MOCK_VERDICT_PASSED,
            "mock": True,
            "issues": issues,
            "feedback": "当前为 mock 视频片段，非真实生成画面；跳过返工重试。",
            "facts": facts,
        }
    return {
        "score": score,
        "passed": score >= VIDEO_PASS,
        "mock": False,
        "issues": issues,
        "feedback": "；".join(issues) if issues else "片段通过质检。",
        "facts": facts,
    }
