"""Figures for the stratified study, in the same visual system as bench/figures.py.

Different question from ``bench/figures.py``. That one plots a five-model smoke
run where each model had one to three samples, and it deliberately draws each
sample as a dot so a single-sample model looks like a single dot. This one plots
a stratified run with five seeds per model, so the honest chart is a
distribution: the spread is what a reader needs, and collapsing 60 samples into
one bar would throw away exactly the information this study was run to collect.

    study_quality     judged score per dimension, distribution over seeds
    study_identity    within-shot identity, per-clip distribution
    study_workflow    what each workflow delivered, as facts not scores

Only ``study_quality`` and ``study_identity`` use judged or measured scores.
``study_workflow`` is drawn from file facts (shots, duration, audio, subtitles)
because the two workflows' scores are not comparable at n=1 and pretending
otherwise is the failure mode this whole exercise is meant to avoid.

Colour slots are assigned by model and never by rank, so a model keeps its hue
when a figure drops one — the same rule bench/figures.py follows.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

_CJK = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
)
for _path in _CJK:
    if Path(_path).exists():
        font_manager.fontManager.addfont(_path)

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STUDY = ROOT / "bench/results/study"
WORKFLOWS = ROOT / "bench/results/compare_workflows/episodes.json"
OUT = ROOT / "docs/figures"

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#86847d"
GRID, SURFACE = "#e8e6e0", "#fcfcfb"

# Fixed order and hue per model, matching the README's existing figures so the
# two sets read as one system.
MODELS = ["cogvideox5b", "wan21_1_3b", "longlive_1_3b"]
LABELS = {
    "cogvideox5b": "CogVideoX-5B",
    "wan21_1_3b": "Wan2.1-1.3B",
    "longlive_1_3b": "LongLive-1.3B",
    "ours": "本项目 workflow",
    "openmontage_full": "OpenMontage 全流程",
}
COLORS = {
    "cogvideox5b": "#e34948",
    "wan21_1_3b": "#eda100",
    "longlive_1_3b": "#1baf7a",
    "ours": "#2a78d6",
    "openmontage_full": "#eb6834",
}
DIMS = {
    "imaging_quality": "成像质量",
    "aesthetic_quality": "美学质量",
    "temporal_consistency": "时序一致性",
    "motion_effects": "运动效果",
    # The five video-text dimensions use a 1-3 scale, so they are drawn by a
    # separate function with their own y-axis. Putting them in this dict would
    # plot a 1-3 score on a 0.5-5.5 axis and make them look artificially low.
}
TEXT_DIMS = {
    "overall_consistency": "整体一致性",
    "color": "色彩一致性",
    "object_class": "目标类别",
    "scene": "场景一致性",
    "action": "动作一致性",
}


def _style(ax, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"{path}  {path.stat().st_size // 1024} KB")
    return path


def load_study() -> dict[str, dict]:
    """Completed dimensions only; a partial pass is not a result."""
    out: dict[str, dict] = {}
    judge_dir = STUDY / "judge"
    if not judge_dir.is_dir():
        return out
    for path in sorted(judge_dir.glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("partial"):
            continue
        out[payload.get("dimension") or path.stem] = payload
    return out


def chart_quality() -> None:
    """Score distribution per dimension, five seeds per model.

    A bar alone would hide whether a gap between models is bigger than the
    spread within a model, which is the only question that matters here. Each
    model's individual samples are drawn as points beside its box.
    """
    study = load_study()
    dims = [d for d in DIMS if d in study]
    if not dims:
        print("skip study_quality: no completed dimension yet")
        return

    fig, axes = plt.subplots(1, len(dims), figsize=(3.4 * len(dims), 4.0), dpi=200,
                             facecolor=SURFACE, squeeze=False)
    rng = np.random.default_rng(0)

    for ax, dimension in zip(axes[0], dims):
        _style(ax)
        samples = study[dimension].get("samples") or {}
        present = [m for m in MODELS if samples.get(m)]
        positions = np.arange(len(present))
        for i, model in enumerate(present):
            values = np.asarray(samples[model], dtype=float)
            color = COLORS[model]
            # Jitter is deterministic so two runs of this script draw the same
            # figure, and a reader comparing two exports is not misled by dots
            # that moved on their own.
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(i + jitter, values, s=16, color=color, alpha=0.45,
                       linewidths=0, zorder=3)
            mean = float(values.mean())
            ax.plot([i - 0.26, i + 0.26], [mean, mean], color=color, linewidth=2.4, zorder=5)
            ax.plot([i, i], [values.min(), values.max()], color=color, linewidth=1.0,
                    alpha=0.55, zorder=2)
            ax.annotate(f"{mean:.2f}", (i, mean), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels([LABELS[m] for m in present], fontsize=8.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(DIMS[dimension], color=INK, fontsize=11, loc="left", pad=8)

    axes[0][0].set_ylabel("裁判分（1-5）", color=INK_2, fontsize=10)
    counts = {m: len((study[dims[0]].get("samples") or {}).get(m) or []) for m in MODELS}
    # Laid out before the title is added: tight_layout cannot enlarge the top
    # margin far enough for a suptitle placed outside the axes and warns on
    # every render, so the title goes on afterwards at a y the layout does not
    # have to reason about.
    fig.tight_layout()
    fig.suptitle(
        f"分层 prompt 集上的判分分布（每模型 {counts[MODELS[0]]} 个样本；粗线=均值，细线=极差）",
        color=INK, fontsize=12.5, x=0.01, ha="left", y=1.02)
    _save(fig, "study_quality.png")


def chart_identity() -> None:
    """Within-shot identity per clip, from the segmented evaluation."""
    seg_path = STUDY / "segments/segments.json"
    if not seg_path.exists():
        # Free and local, so it can be produced on demand from the clips rather
        # than waiting for the judge.
        print("skip study_identity: run the segment stage first")
        return
    payload = json.loads(seg_path.read_text(encoding="utf-8"))
    per_model: dict[str, list[float]] = {}
    for clip in payload.get("clips") or []:
        value = (clip.get("summary") or {}).get("identity_median")
        if value is not None:
            per_model.setdefault(clip.get("model") or "?", []).append(float(value))
    present = [m for m in MODELS if per_model.get(m)]
    if not present:
        print("skip study_identity: no segments")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200, facecolor=SURFACE)
    _style(ax)
    rng = np.random.default_rng(1)
    for i, model in enumerate(present):
        values = np.asarray(per_model[model], dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(i + jitter, values, s=22, color=COLORS[model], alpha=0.6,
                   linewidths=0, zorder=3, label=LABELS[model])
        median = float(np.median(values))
        ax.plot([i - 0.28, i + 0.28], [median, median], color=COLORS[model], linewidth=2.4, zorder=5)
        ax.annotate(f"{median:.3f}", (i, median), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0.75, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.02, 0.755, "漂移判定线 0.75", transform=ax.get_yaxis_transform(),
            fontsize=8, color=MUTED, va="bottom")
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([LABELS[m] for m in present])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("镜头内 identity 中位（0-1）", color=INK_2, fontsize=10)
    ax.set_title(f"单片段内部稳定性（每模型 {len(per_model[present[0]])} 条片段，1 秒切窗）",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    _save(fig, "study_identity.png")


def chart_workflow() -> None:
    """Delivered capability, side by side, as facts rather than scores.

    Laid out in figure coordinates on a fully hidden axes. An earlier version
    placed the text in data coordinates, which let matplotlib autoscale an empty
    axes to the text's own extent and rendered a 1170x5421 image of one table —
    the same trap as drawing a table inside a data plot.
    """
    if not WORKFLOWS.exists():
        print("skip study_workflow: run compare_workflows.py first")
        return
    payload = json.loads(WORKFLOWS.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate") or {}
    ours, om = aggregate.get("ours") or {}, aggregate.get("openmontage_full") or {}
    if not ours or not om:
        print("skip study_workflow: missing a side")
        return

    def fmt(value, spec: str = "{}") -> str:
        return "—" if value is None else spec.format(value)

    def score_label(side: dict) -> str:
        """Median per-episode character score, when the VLM pass has run."""
        return fmt(side.get("character_median"), "{:g}")

    rows = [
        ("成片数 n", fmt(ours.get("n")), fmt(om.get("n"))),
        ("交付时长中位（秒）", fmt(ours.get("duration_median"), "{:.2f}"),
         fmt(om.get("duration_median"), "{:.2f}")),
        ("交付镜数中位", fmt(ours.get("shots_median"), "{:g}"),
         fmt(om.get("shots_median"), "{:g}")),
        ("含音轨的成片", f"{fmt(ours.get('with_audio'))}/{fmt(ours.get('n'))}",
         f"{fmt(om.get('with_audio'))}/{fmt(om.get('n'))}"),
        ("角色一致性中位（VLM, 1-5）", score_label(ours), score_label(om)),
        ("镜头内构图中位", fmt(ours.get("structure_median"), "{:.3f}"),
         fmt(om.get("structure_median"), "{:.3f}")),
        ("镜头内色彩中位", fmt(ours.get("color_median"), "{:.3f}"),
         fmt(om.get("color_median"), "{:.3f}")),
        ("合成片运动量中位", fmt(ours.get("motion_median"), "{:.2f}"),
         fmt(om.get("motion_median"), "{:.2f}")),
    ]

    fig = plt.figure(figsize=(7.6, 3.9), dpi=200, facecolor=SURFACE)
    fig.text(0.03, 0.93, "两套 workflow 的交付事实", color=INK, fontsize=12.5, ha="left", va="center",
             fontweight="bold")
    fig.text(0.03, 0.855, "同一 brief、同一底模 doubao-seedance-1-0-pro-fast、同为 15 镜",
             color=MUTED, fontsize=8.5, ha="left", va="center")

    col_label, col_ours, col_om = 0.03, 0.56, 0.76
    fig.text(col_ours, 0.775, LABELS["ours"], color=COLORS["ours"], fontsize=9.5,
             ha="left", va="center", fontweight="bold")
    fig.text(col_om, 0.775, "OpenMontage 全流程", color=COLORS["openmontage_full"],
             fontsize=9.5, ha="left", va="center", fontweight="bold")
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.735, 0.735], color=GRID, linewidth=1.0))

    y = 0.675
    for label, left, right in rows:
        fig.text(col_label, y, label, color=INK_2, fontsize=9, ha="left", va="center")
        fig.text(col_ours, y, left, color=INK, fontsize=9.5, ha="left", va="center")
        fig.text(col_om, y, right, color=INK, fontsize=9.5, ha="left", va="center")
        y -= 0.075

    fig.text(col_label, y - 0.035,
             "identity 与运动量需一起读：运动量更低的成片在指纹指标上天然占优，单看 identity 会读反。",
             color=MUTED, fontsize=7.8, ha="left", va="top")
    _save(fig, "study_workflow.png")


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Study figures")
    parser.add_argument("--only", default="", help="quality|identity|workflow")
    args = parser.parse_args(argv[1:])
    wanted = {"quality": chart_quality, "identity": chart_identity, "workflow": chart_workflow}
    for name, fn in wanted.items():
        if args.only and name != args.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
