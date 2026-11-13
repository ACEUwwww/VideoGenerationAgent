"""Render the benchmark comparison as charts.

Three views of the same run, because "which model is better" is not one
question:

  time        how long each backend took to produce its clip
  quality     the judge's per-dimension scores
  consistency how much each clip drifts from its own first segment

Kept as one script rather than three so the palette, the ordering and the
model names are defined once — a chart that disagrees with its neighbour about
which colour is which model is worse than no chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

# matplotlib ships no CJK face, and its cached font list omits the system's
# Noto CJK collection. Register the file directly rather than relying on a
# rebuild, which would be a side effect on the user's cache. The family is
# reported as "Noto Sans CJK JP" but carries the full CJK glyph set, Chinese
# included, which is why every label on these charts renders.
_CJK_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
)
for _path in _CJK_CANDIDATES:
    if Path(_path).exists():
        font_manager.fontManager.addfont(_path)

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BENCH_ROOT = Path(__file__).resolve().parent

# Fixed order and one hue per model, assigned by identity so every panel agrees.
MODELS = ["ours", "openmontage_shot", "longlive_1_3b", "wan21_1_3b", "cogvideox5b"]
LABELS = {
    "ours": "本项目 workflow",
    "openmontage_shot": "OpenMontage（同底模）",
    "longlive_1_3b": "LongLive-1.3B",
    "wan21_1_3b": "Wan2.1-1.3B",
    "cogvideox5b": "CogVideoX-5B",
}
# Categorical slots in fixed order from the reference palette. Slot 4 (yellow)
# sits beside slot 2 (orange) under all-pairs checking, which only matters for
# scatter-like forms; these are grouped bars, where the adjacent pairlist
# applies and the order validates.
COLORS = {
    "ours": "#2a78d6",
    "openmontage_shot": "#eb6834",
    "longlive_1_3b": "#1baf7a",
    "wan21_1_3b": "#eda100",
    "cogvideox5b": "#e34948",
}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#86847d"
GRID = "#e8e6e0"
SURFACE = "#fcfcfb"

DIMENSION_LABELS = {
    "imaging_quality": "成像质量",
    "aesthetic_quality": "美学质量",
    "temporal_consistency": "时序一致性",
    "motion_effects": "运动效果",
    "overall_consistency": "视频-文本一致性",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(run_dir: Path) -> dict:
    """Judge scores, clip metadata and segment summaries for one run."""
    out = {"scores": {}, "elapsed": {}, "segments": {}}

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        out["scores"] = {
            dim: row["scores"] for dim, row in summary.get("dimensions", {}).items()
        }

    clips_path = run_dir / "clips.json"
    if clips_path.exists():
        for row in json.loads(clips_path.read_text(encoding="utf-8")):
            out["elapsed"][row["model"]] = row.get("elapsed_s") or 0.0

    seg_path = run_dir / "segments" / "segments.json"
    if seg_path.exists():
        for clip in json.loads(seg_path.read_text(encoding="utf-8"))["clips"]:
            out["segments"][clip["model"]] = clip["summary"]
    return out


def chart_scores(data: dict, out_path: Path) -> None:
    """Grouped bars: one group per dimension, one bar per model.

    Grouped rather than stacked because the dimensions are independent scores,
    not parts of a total — stacking would imply they add up.

    A model with no score for a dimension draws no bar at all. Drawing a
    zero-height bar would read as "scored 0", which is a different and much
    worse claim than "not measured" — and the missing scores here came from
    parse failures, not from a zero.
    """
    # A dimension where fewer than half the models have a score is not a
    # comparison; drop it rather than show a lone bar.
    dims = [
        d for d in DIMENSION_LABELS
        if d in data["scores"]
        and sum(1 for m in MODELS if data["scores"][d].get(m) is not None) >= 2
    ]
    if not dims:
        return

    # 1-3 and 1-5 dimensions share this axis, so the panel says which is which.
    low_scale = [d for d in dims if d == "overall_consistency"]
    title = "同一 prompt 的质量评分"
    if low_scale:
        title += "（视频-文本一致性为 1-3 分，其余 1-5 分）"

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200, facecolor=SURFACE)
    _style(ax)

    x = np.arange(len(dims))
    width = 0.17
    for i, model in enumerate(MODELS):
        offsets = []
        heights = []
        positions = []
        for j, dim in enumerate(dims):
            value = data["scores"].get(dim, {}).get(model)
            if value is None:
                continue
            offset = (i - (len(MODELS) - 1) / 2) * width
            positions.append(x[j] + offset)
            heights.append(value)
            offsets.append(x[j])
        bars = ax.bar(
            positions, heights, width * 0.92,
            label=LABELS[model], color=COLORS[model], zorder=3,
        )
        for bar, value in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.08,
                f"{value:g}", ha="center", va="bottom",
                fontsize=8.5, color=INK_2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in dims], color=INK_2, fontsize=10)
    ax.set_ylim(0, 5.6)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel("得分", color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=14)
    ax.legend(
        frameon=False, ncol=5, loc="upper left",
        bbox_to_anchor=(0, -0.18), fontsize=9, labelcolor=INK_2,
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def chart_consistency(data: dict, out_path: Path) -> None:
    """Cross-segment identity, lowest first — the drift story as one bar each."""
    rows = [
        (m, data["segments"][m]) for m in MODELS
        if m in data["segments"] and data["segments"][m].get("identity_min") is not None
    ]
    if not rows:
        return
    rows.sort(key=lambda r: r[1]["identity_min"])

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=200, facecolor=SURFACE)
    _style(ax)

    y = np.arange(len(rows))
    values = [r[1]["identity_min"] for r in rows]
    colors = [COLORS[r[0]] for r in rows]
    ax.barh(y, values, height=0.58, color=colors, zorder=3)

    for i, (model, summary) in enumerate(rows):
        onset = summary.get("drift_onset_segment")
        note = f"  {values[i]:.3f}"
        if onset is not None:
            note += f"  (第 {onset} 段起漂移)"
        ax.text(values[i] + 0.012, i, note, va="center", fontsize=9.5, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[r[0]] for r in rows], color=INK_2, fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("跨段结构一致性（与首段比，1.0 = 完全一致）", color=INK_2, fontsize=10)
    ax.set_title(
        "同一片段内部的稳定性：越高越不漂",
        color=INK, fontsize=13, loc="left", pad=14,
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def chart_time(data: dict, out_path: Path) -> None:
    """Wall-clock cost per clip, horizontal for readable labels."""
    rows = [(m, data["elapsed"].get(m, 0)) for m in MODELS if data["elapsed"].get(m)]
    if not rows:
        return
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=200, facecolor=SURFACE)
    _style(ax)

    y = np.arange(len(rows))
    values = [r[1] for r in rows]
    ax.barh(y, values, height=0.55, color=[COLORS[r[0]] for r in rows], zorder=3)
    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.015, i, f"{value:.0f}s",
                va="center", fontsize=9.5, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[r[0]] for r in rows], color=INK_2, fontsize=10)
    ax.set_xlabel("生成耗时（秒，5 秒片段）", color=INK_2, fontsize=10)
    ax.set_title("出片成本", color=INK, fontsize=13, loc="left", pad=14)
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def chart_profile(data: dict, out_path: Path) -> None:
    """One radar per model is four charts; this is the same data as grouped bars
    on a normalised 0-1 axis, which compares shapes without pretending the
    dimensions share a scale."""
    # Same floor as the score chart: a dimension only one model was measured on
    # is not a comparison.
    dims = [
        d for d in DIMENSION_LABELS
        if d in data["scores"]
        and sum(1 for m in MODELS if data["scores"][d].get(m) is not None) >= 2
    ]
    if not dims:
        return

    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=200, facecolor=SURFACE)
    _style(ax)

    # Normalise each dimension by its own maximum so a 1-3 dimension is not
    # dwarfed by a 1-5 one. The reader is comparing shape, not raw points.
    maxima = {d: max((data["scores"][d].get(m) or 0) for m in MODELS) or 1 for d in dims}
    x = np.arange(len(dims))
    width = 0.17
    for i, model in enumerate(MODELS):
        values = [((data["scores"].get(d, {}).get(model) or 0) / maxima[d]) for d in dims]
        offset = (i - (len(MODELS) - 1) / 2) * width
        ax.bar(x + offset, values, width * 0.9, label=LABELS[model],
               color=COLORS[model], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in dims], color=INK_2, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("相对该维度最高分", color=INK_2, fontsize=10)
    ax.set_title("各维度相对表现（每个维度按自身最高分归一）",
                 color=INK, fontsize=13, loc="left", pad=14)
    ax.legend(frameon=False, ncol=5, loc="upper left",
              bbox_to_anchor=(0, -0.20), fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]) if len(argv) > 1 else BENCH_ROOT / "results" / "compare"
    out_dir = Path(argv[2]) if len(argv) > 2 else run_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load(run_dir)
    if not data["scores"]:
        raise SystemExit(f"no summary.json in {run_dir}; run --stage judge first")

    chart_scores(data, out_dir / "scores.png")
    chart_consistency(data, out_dir / "consistency.png")
    chart_time(data, out_dir / "time.png")
    chart_profile(data, out_dir / "profile.png")
    for path in sorted(out_dir.glob("*.png")):
        print(f"{path}  {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
