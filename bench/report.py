"""Turn a run's scores into a readable comparison, in Chinese like the README.

Only what the task needs: which timed region and material, which models, which
git/config identity, the per-dimension table, and an explicit statement of what
was NOT established. The Video-Bench dimensions do not all share a scale — the
four generated dimensions are 1-5 and the five video-text ones are 1-3 — so the
table prints the range next to the header rather than normalising, which would
hide a real difference between the groups.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import BenchConfig, DYNAMIC_DIMENSIONS, STATIC_DIMENSIONS, VIDEO_TEXT_DIMENSIONS

LABELS = {
    "imaging_quality": "成像质量",
    "aesthetic_quality": "美学质量",
    "temporal_consistency": "时序一致性",
    "motion_effects": "运动效果",
    "overall_consistency": "视频-文本一致性",
    "color": "色彩一致性",
    "object_class": "目标类别一致性",
    "scene": "场景一致性",
    "action": "动作一致性",
}

GROUPS = (
    ("生成质量（1-5 分）", STATIC_DIMENSIONS + DYNAMIC_DIMENSIONS, 5),
    ("视频-文本对齐（1-3 分）", VIDEO_TEXT_DIMENSIONS, 3),
)


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def write_report(run_dir: Path, summary: dict, config: BenchConfig) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    clips = json.loads((run_dir / "clips.json").read_text(encoding="utf-8"))
    models = manifest["models"]
    scored = [m for m in models if any(
        d["scores"].get(m) is not None for d in summary["dimensions"].values()
    )]
    lines: list[str] = []
    add = lines.append

    add(f"# 视频生成对比：{run_dir.name}")
    add("")
    add("## 评测身份")
    add("")
    add(f"- 裁判模型：`{manifest['judge']['model']}`（{manifest['judge']['base_url']}）")
    add(f"- 抽帧：{manifest['judge']['sample_fps']} fps，最长边 {manifest['judge']['max_edge']}px")
    add(f"- Video-Bench 版本：`{manifest['vendor_commit'][:12] or 'unknown'}`")
    add(f"- prompts：{len(manifest['prompts'])} 条，seeds {manifest['seeds']}，目标时长 {manifest['seconds']}s")
    add(f"- 硬件：{', '.join(manifest['hardware'].get('gpu') or ['未检测到 GPU'])}")
    add(f"- torch {manifest['hardware'].get('torch', '?')} / CUDA {manifest['hardware'].get('cuda', '?')}")
    add("")

    add("## 对比对象与生成方式")
    add("")
    add("| 模型 | 本轮出片 | 生成方式 | 说明 |")
    add("| --- | --- | --- | --- |")
    # Availability is probed when the report is written, but the clips were
    # produced earlier. A run generated with VIDEO_PROVIDER=ark and reported
    # afterwards with the provider back on mock said "unavailable" for a model
    # whose four scored clips were sitting in the same directory. What the
    # reader needs to know is whether this run has clips for the model, so that
    # is what the column reports; the probe's detail moves to a note.
    produced = {c["model"] for c in clips}
    for row in manifest["backends"]:
        if row["name"] not in models:
            continue
        shown = "是" if row["name"] in produced else "否"
        detail = row["detail"]
        if row["name"] in produced and not row["available"]:
            detail = f"片段已生成；当前环境探测：{detail}"
        add(f"| `{row['name']}` | {shown} | {row['modality']} | {detail} |")
    add("")
    add("生成方式不同、且已经分开记录：本项目走「关键帧 → 图生视频」，开源模型走「文生视频」。")
    add("两者不是同一个任务，所以下面的分数比较的是各自的产物质量，不是同一条流水线的高低。")
    add("")

    add("## 分数")
    add("")
    for title, dimensions, maximum in GROUPS:
        present = [d for d in dimensions if d in summary["dimensions"]]
        if not present:
            continue
        add(f"### {title}")
        add("")
        add("| 维度 | " + " | ".join(f"`{m}`" for m in models) + " |")
        add("| --- | " + " | ".join("---" for _ in models) + " |")
        for dimension in present:
            entry = summary["dimensions"][dimension]
            cells = [_fmt(entry["scores"].get(m)) for m in models]
            add(f"| {LABELS.get(dimension, dimension)} | " + " | ".join(cells) + " |")
            if entry.get("error"):
                add(f"| ↳ 错误 | " + " | ".join([entry["error"][:60]] * len(models)) + " |")
        add("")
        if any(summary["dimensions"][d]["scores"].get(m) is None for d in present for m in models):
            add(f"`—` 表示该维度没有取到分数（裁判未返回可解析的数字）。该维度满分 {maximum}。")
            add("")

    usage = summary.get("usage") or {}
    add("## 成本与耗时")
    add("")
    add(f"- 裁判调用：{usage.get('calls', 0)} 次")
    add(f"- tokens：prompt {usage.get('prompt_tokens', 0)} / completion {usage.get('completion_tokens', 0)}")
    total_judge = sum(d["elapsed_s"] for d in summary["dimensions"].values())
    add(f"- 裁判耗时合计：{total_judge:.0f}s")
    render = sum(c["elapsed_s"] for c in clips)
    add(f"- 渲染耗时合计：{render:.0f}s（{len(clips)} 条片段）")
    add("")

    add("## 本次结果没有证明的事")
    add("")
    add("- 样本量。这一轮是冒烟规模，每个模型的每个 prompt 只有 "
        f"{len(manifest['seeds'])} 条片段；单条分数不能当作模型的稳定水平，跑多 seed 才能看出方差。")
    add("- 时长与分辨率不一致。开源模型的帧率不同（CogVideoX 8fps、Wan 16fps），"
        "实际时长按各自的帧数取整，与目标时长有小幅偏差。")
    add("- 没有做人工对齐。Video-Bench 论文报告的是与人类偏好的相关性，"
        "这次只用它的裁判和维度，没有复现人工标注对比。")
    add("- 分数跨组不可加。生成质量是 1-5 分、视频-文本对齐是 1-3 分，两组不能直接相加或平均。")
    add("")

    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
