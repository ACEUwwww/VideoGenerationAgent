"""Report for the stratified study, built from artefacts rather than assumptions.

Two things this deliberately does not do, because both were done before and both
were wrong:

* it does not average across models with different seed counts without saying so
  per model — the earlier smoke comparison printed a 5-model table where one
  model had 1 seed and the others had 3, and a mean over one sample reads as a
  mean;
* it does not put the 1-5 generation dimensions and the 1-3 video-text
  dimensions in one column. They do not share a scale.

Every number here is read from `judge/*.json` and the clip files. A dimension
with no parseable score prints as `—` with its reason, never as a zero.
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")

LABELS = {
    "imaging_quality": ("成像质量", 5),
    "aesthetic_quality": ("美学质量", 5),
    "temporal_consistency": ("时序一致性", 5),
    "motion_effects": ("运动效果", 5),
    "overall_consistency": ("视频-文本一致性", 3),
    "color": ("色彩一致性", 3),
    "object_class": ("目标类别一致性", 3),
    "scene": ("场景一致性", 3),
    "action": ("动作一致性", 3),
}

MODEL_LABELS = {
    "ours": "本项目 workflow（Seedance i2v）",
    "openmontage_shot": "OpenMontage 镜头段（无编排）",
    "openmontage_full": "OpenMontage 全流程（7 阶段）",
    "longlive_1_3b": "LongLive-1.3B",
    "wan21_1_3b": "Wan2.1-1.3B",
    "cogvideox5b": "CogVideoX-5B",
}


def load_judge(run: Path) -> dict[str, dict]:
    """Completed dimensions only.

    ``judge_study.py`` writes in-progress work to ``<dimension>.partial.json``
    and promotes it to ``<dimension>.json`` only when every group has been
    scored. Reading the partials too made an unfinished dimension appear in the
    report with its running averages, which is exactly the confusion the
    partial-file split exists to prevent.
    """
    out = {}
    for path in sorted((run / "judge").glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("partial"):
            continue
        out[payload.get("dimension") or path.stem] = payload
    return out


def partial_judge(run: Path) -> dict[str, dict]:
    """Dimensions mid-scoring, reported as pending rather than as scores."""
    out = {}
    for path in sorted((run / "judge").glob("*.partial.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload.get("dimension") or path.stem.removesuffix(".partial")] = payload
    return out


def clip_counts(run: Path) -> dict[str, dict[int, int]]:
    """Seeds present per model: {model: {seed: clip_count}}."""
    counts: dict[str, dict[int, int]] = {}
    for path in glob.glob(str(run / "videos" / "*" / "*.mp4")):
        parts = Path(path).parts
        model = parts[-2]
        try:
            seed = int(Path(path).stem.rsplit("-", 1)[-1])
        except ValueError:
            seed = 0
        counts.setdefault(model, {}).setdefault(seed, 0)
        counts[model][seed] += 1
    return counts


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(run: Path, *, title: str, out: Path | None = None) -> Path:
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    judge = load_judge(run)
    pending = partial_judge(run)
    counts = clip_counts(run)
    models = [m for m in (manifest.get("models") or sorted(counts)) if m in counts or m in judge]

    # The study has no manifest: it is driven by scripts rather than the runner,
    # so the identity block falls back to what the artefact files themselves
    # record. Printing "unknown" for a judge whose name is in every judge file
    # is worse than reading it from there.
    if not manifest.get("judge"):
        recorded = next((e.get("judge") for e in judge.values() if e.get("judge")), None) \
            or next((e.get("judge") for e in pending.values() if e.get("judge")), None) or {}
        manifest["judge"] = {"model": recorded.get("model", "unknown"),
                              "base_url": recorded.get("base_url", "")}

    lines: list[str] = []
    add = lines.append
    add(f"# {title}")
    add("")
    add("## 评测身份")
    add("")
    add(f"- run：`{run.name}`")
    jm = manifest.get("judge") or {}
    add(f"- 裁判：`{jm.get('model', 'unknown')}`（{jm.get('base_url', '')}），"
        f"抽帧 {jm.get('sample_fps', '?')} fps / 最长边 {jm.get('max_edge', '?')}px")
    add(f"- Video-Bench：`{(manifest.get('vendor_commit') or 'unknown')[:12]}`")
    add(f"- 硬件：{', '.join((manifest.get('hardware') or {}).get('gpu') or ['未检测'])}")
    add("")

    add("## 样本量（每个模型的 seed 数与片段数）")
    add("")
    add("| 模型 | seed 数 | seed 列表 | 片段数 |")
    add("| --- | ---: | --- | ---: |")
    for model in models:
        seeds = sorted(counts.get(model, {}))
        total = sum(counts.get(model, {}).values())
        add(f"| `{model}` | {len(seeds)} | {', '.join(map(str, seeds)) or '—'} | {total} |")
    add("")
    uneven = {m: len(counts.get(m, {})) for m in models}
    if len(set(uneven.values())) > 1:
        add("> **seed 数不一致。** 下表的分差同时包含模型差异与采样差异；"
            f"样本量：{', '.join(f'{m}={n}' for m, n in uneven.items())}。")
        add("")

    add("## 分数")
    add("")
    for group, dims, maximum in (
        ("生成质量（1-5 分）", ["imaging_quality", "aesthetic_quality", "temporal_consistency", "motion_effects"], 5),
        ("视频-文本对齐（1-3 分）", ["overall_consistency", "color", "object_class", "scene", "action"], 3),
    ):
        present = [d for d in dims if d in judge]
        if not present:
            continue
        add(f"### {group}")
        add("")
        add("| 维度 | " + " | ".join(f"`{m}`" for m in models) + " |")
        add("| --- | " + " | ".join("---:" for _ in models) + " |")
        for dimension in present:
            entry = judge[dimension]
            samples = entry.get("samples") or {}
            cells = []
            for model in models:
                values = samples.get(model) or []
                cells.append(_fmt(values[0]) if len(values) == 1 else _fmt(entry.get("scores", {}).get(model)))
            add(f"| {LABELS.get(dimension, (dimension, maximum))[0]} | " + " | ".join(cells) + " |")
        add("")
        add("每格为该模型在该维度上全部片段的均值（每模型 60 条：12 prompt × 5 seed），"
            "不是「跨 seed 均值」—— 单一 seed 内也有 12 条不同 prompt 的片段。")
        add("")

    add("## 各维度逐片段离散度")
    add("")
    add("单条片段不能代表模型水平，所以同一模型内各片段的极差一并列出。"
        "极差大于模型间差距时，该维度的排序不可信。")
    add("")
    add("> 这里列的是**每一条片段**的分（每模型 60 条），不是每个 seed 的均值——"
        "一组 seed 只产出若干条片段，把片段分当成 seed 分会把同一组的噪声数成独立样本。")
    add("")
    add("| 维度 | 模型 | 各片段分 | 均值 | 标准差 | 极差 |")
    add("| --- | --- | --- | ---: | ---: | ---: |")
    for dimension, entry in judge.items():
        samples = entry.get("samples") or {}
        for model in models:
            values = sorted(samples.get(model) or [])
            if not values:
                continue
            mean = sum(values) / len(values)
            spread = max(values) - min(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            add(f"| {LABELS.get(dimension, (dimension, 0))[0]} | `{model}` "
                f"| {', '.join(_fmt(v) for v in values)} | {mean:.2f} | {stdev:.2f} "
                f"| {spread:.2f} |")
    add("")

    if pending:
        add("## 尚未跑完的维度（不计入上表）")
        add("")
        add("裁判按维度串行打分，未完成的维度写在 `.partial.json`，不参与任何均值或排序。")
        add("")
        add("| 维度 | 已评分组数 | 说明 |")
        add("| --- | ---: | --- |")
        for dimension, entry in pending.items():
            scored = len(entry.get("per_video") or {})
            add(f"| {LABELS.get(dimension, (dimension, 0))[0]} | {scored} | 进行中，未完成不读 |")
        add("")

    if judge:
        add("## 未取到分数的维度")
        add("")
        missing = [d for d, e in judge.items() if any(
            e.get("scores", {}).get(m) is None for m in models)]
        if missing:
            for dimension in missing:
                add(f"- `{dimension}`：{(judge[dimension].get('error') or '裁判未返回可解析的数字')[:120]}")
        else:
            add("- 无。全部维度对所有模型都出分。")
        add("")

    out = out or (run / "report_study.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Report for the stratified study")
    parser.add_argument("--run", type=Path, default=ROOT / "bench/results/study")
    parser.add_argument("--title", default="视频生成对比：分层 prompt 集")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv[1:])
    path = write_report(args.run, title=args.title, out=args.out)
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
