"""The head-to-head figure: same brief, same checkpoints, both workflows end to end.

The two OpenMontage rows come from one episode (``omfix``, 12 clips). The 5-shot
row is the first five shots of that same episode, scored on their own — it is a
subsample, not a second run, and the figure says so because the difference
matters to how the bars should be read.

    bench/.venv/bin/python bench/study/om_control_figure.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

FIG = ROOT / "docs/figures"
FULL = ROOT / "bench/results/om_control/consistency.json"
FIVE = ROOT / "bench/results/om_control/om5.json"

OURS = "#2a78d6"
CONTROL = "#eb6834"
MUTED = "#8a8f98"
INK = "#14171a"
SURFACE = "#ffffff"


def _score(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["label"]: row for row in rows}


def main() -> Path:
    full = _score(FULL)
    five = _score(FIVE)

    om = full["openmontage:omfix"]
    ours = full["ours:project_b3a0cd04"]
    om5 = five["openmontage:omfix"]

    rows = [
        ("OpenMontage, full episode", om["shots"], om["score"], om["changed"], CONTROL),
        ("OpenMontage, first 5 shots", om5["shots"], om5["score"], om5["changed"], "#f2a78a"),
        ("Ours, same brief", ours["shots"], ours["score"], ours["changed"], OURS),
    ]

    fig, ax = plt.subplots(figsize=(9.4, 3.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d9de")
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color="#eef0f3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ypos = list(range(len(rows)))[::-1]
    labels = [f"{name}\n({shots} shots)" for name, shots, _, _, _ in rows]
    values = [score for _, _, score, _, _ in rows]
    colors = [color for *_, color in rows]

    bars = ax.barh(ypos, values, height=0.52, color=colors, zorder=3)
    for rect, (name, shots, score, changed, _) in zip(bars, rows):
        ax.annotate(f"{score:.1f}", (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    textcoords="offset points", xytext=(7, -1),
                    ha="left", va="center", fontsize=12, color=INK, weight="bold")
        ax.annotate(f"judge: {changed.lower()} drifted", (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    textcoords="offset points", xytext=(40, -1),
                    ha="left", va="center", fontsize=7.6, color=MUTED)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, 5.9)
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("Cross-shot character consistency (VLM, 1-5)", fontsize=10, color=INK)
    ax.set_title(
        "Same brief, same 1.3B-5B checkpoints, both workflows' own pipelines\n"
        "the difference at equal shot count is 2 points on a 5-point scale (n=1 per side)",
        fontsize=11.5, color=INK, loc="left", pad=14)

    fig.text(0.012, 0.015,
             "The 5-shot OpenMontage row is the first five shots of the 12-shot episode, scored\n"
             "separately - a subsample, not a second render. OpenMontage's reference-image path\n"
             "could not run on local weights, so both sides are text-to-video.\n"
             "Data: bench/results/om_control/",
             fontsize=7.0, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out = FIG / "tier3a_matched.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
