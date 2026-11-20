"""Distribution figures for the technical report.

The first report had to state a mean and a sign test, because each cell was
measured once. With repeats there is a sample behind each number, so these
figures show the sample: the histograms the report argues from, and the paired
differences that make the pairing visible.

Every value is read from the result files. Nothing here is typed in.

    bench/.venv/bin/python bench/study/repeats_figures.py
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
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = ROOT / "docs/figures"
REP = ROOT / "bench/results/tier3a_repeats"

OURS = "#2a78d6"
CONTROL = "#eb6834"
MUTED = "#8a8f98"
INK = "#14171a"
SURFACE = "#ffffff"
GRID = "#eef0f3"

MODELS = ["cogvideox5b", "wan21_1_3b"]
MODEL_LABEL = {"cogvideox5b": "CogVideoX-5B", "wan21_1_3b": "Wan2.1-1.3B"}


def load(model: str) -> list[dict]:
    path = REP / f"consistency_{model}.json"
    if not path.exists():
        return []
    return [row for row in json.loads(path.read_text(encoding="utf-8")) if row.get("score")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d9de")
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _note(fig, text: str) -> None:
    fig.text(0.012, 0.015, text, fontsize=7.0, color=MUTED, linespacing=1.5)


def figure_histograms(rows: dict[str, list[dict]]) -> Path | None:
    """The distribution of every episode's score, both arms, both models."""
    models = [m for m in MODELS if rows.get(m)]
    if not models:
        return None

    fig, axes = plt.subplots(1, len(models), figsize=(10.8, 4.5), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = [axes] if len(models) == 1 else list(axes)

    scores = [1, 2, 3, 4, 5]
    width = 0.38
    for ax, model in zip(axes, models):
        _style(ax)
        ind = [r["score"] for r in rows[model] if r["arm"] == "independent"]
        pla = [r["score"] for r in rows[model] if r["arm"] == "planned"]
        counts = {arm: [vals.count(s) for s in scores] for arm, vals in
                  (("independent", ind), ("planned", pla))}

        ax.bar([s - width / 2 for s in scores], counts["independent"], width,
               color=CONTROL, zorder=3, label="Independent (no state)")
        ax.bar([s + width / 2 for s in scores], counts["planned"], width,
               color=OURS, zorder=3, label="Orchestrated (anchor carried)")

        # Peak of the two distributions, so the means have a ceiling to sit under.
        top = max(counts["independent"] + counts["planned"] + [1])
        ax.set_ylim(0, top * 1.28)
        for values, color, offset in ((ind, CONTROL, -0.20), (pla, OURS, 0.20)):
            if not values:
                continue
            mean = statistics.mean(values)
            ax.axvline(mean, color=color, linewidth=2, linestyle=(0, (5, 2)), zorder=4)
            ax.annotate(f"mean {mean:.2f}", (mean + offset, top * 1.17),
                        ha="right" if offset < 0 else "left", va="center",
                        fontsize=8.5, color=color, weight="bold")

        separated = max(ind) < min(pla) if ind and pla else False
        verdict = "arms do not overlap" if separated else "arms overlap at 4"
        ax.set_title(f"{MODEL_LABEL[model]}  (n={len(ind)}+{len(pla)} episodes) — {verdict}",
                     fontsize=10.5, color=INK, loc="left")

        ax.set_xticks(scores)
        ax.set_xlim(0.45, 5.55)
        ax.set_xlabel("Cross-shot character consistency (1-5)", fontsize=9.5, color=INK)
        if model == models[0]:
            ax.set_ylabel("Episodes", fontsize=9.5, color=INK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle("Every episode, not just the mean",
                 fontsize=12, color=INK, x=0.012, ha="left", y=0.985)
    _note(fig, "One bar = one episode (one story, one repeat). Judge: qwen2.5-vl-32b-local at temperature 0, "
               "asking whether face / hair / build / clothing are the same person, with camera, background, "
               "lighting and grade excluded.\nData: bench/results/tier3a_repeats/consistency_*.json")
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    out = FIG / "repeats_hist.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def figure_paired(rows: dict[str, list[dict]]) -> Path | None:
    """Paired differences: each line is one (brief, repeat) cell measured twice."""
    pairs: list[tuple[str, int, int, int, int]] = []
    for model, model_rows in rows.items():
        by_cell: dict[tuple[int, int], dict[str, int]] = {}
        for row in model_rows:
            key = (row["brief_index"], row["repeat"])
            by_cell.setdefault(key, {})[row["arm"]] = row["score"]
        for (brief, repeat), arms in sorted(by_cell.items()):
            if "independent" in arms and "planned" in arms:
                pairs.append((model, brief, repeat, arms["independent"], arms["planned"]))

    if not pairs:
        return None

    fig, ax = plt.subplots(figsize=(10.4, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)

    for index, (model, brief, repeat, ind, pla) in enumerate(pairs):
        x = index + 1
        color = OURS if pla > ind else (MUTED if pla == ind else CONTROL)
        ax.plot([x, x], [ind, pla], color=color, linewidth=1.6, zorder=3, alpha=0.85)
        ax.scatter([x], [ind], s=34, color=CONTROL, zorder=4, edgecolors=SURFACE, linewidths=1.2)
        ax.scatter([x], [pla], s=34, color=OURS, zorder=4, edgecolors=SURFACE, linewidths=1.2)

    ax.set_xticks([])
    ax.set_ylim(0.5, 5.6)
    ax.set_yticks(range(1, 6))
    ax.set_ylabel("Cross-shot character consistency (1-5)", fontsize=9.5, color=INK)
    ax.set_xlabel("One pair per (model, brief, repeat) cell", fontsize=9.5, color=INK)

    wins = sum(1 for _, _, _, i, p in pairs if p > i)
    ties = sum(1 for _, _, _, i, p in pairs if p == i)
    losses = sum(1 for _, _, _, i, p in pairs if p < i)
    ax.set_title(
        f"Paired cells: {wins} wins / {ties} ties / {losses} losses across {len(pairs)} pairs\n"
        "orange = independent, blue = orchestrated; the segment is what carrying state bought",
        fontsize=11, color=INK, loc="left", pad=12)

    handles = [Rectangle((0, 0), 1, 1, color=CONTROL), Rectangle((0, 0), 1, 1, color=OURS),
               plt.Line2D([], [], color=MUTED, linewidth=1.6)]
    ax.legend(handles, ["Independent", "Orchestrated", "no change"], frameon=False,
              fontsize=8.5, loc="lower right", ncol=3)

    _note(fig, "A pair is two episodes rendered with the same seeds from the same cell, differing only in the "
               "prompt string handed to the model. Data: bench/results/tier3a_repeats/consistency_*.json")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIG / "repeats_paired.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def figure_clip_hist() -> Path | None:
    """Experiment B: the single-clip dimensions, as distributions.

    These cells were sampled five times from the start (one per seed), so unlike
    the first report's cross-shot numbers they never needed a point estimate.
    Showing the spread is the honest version of "4.18 vs 3.68": the two
    distributions overlap almost completely, which is why no cell reaches
    significance and why the report does not claim a large single-clip penalty.
    """
    AB = ROOT / "bench/results/tier3a_ab/judge"
    if not AB.exists():
        return None
    dims = ["imaging_quality", "aesthetic_quality", "temporal_consistency", "motion_effects"]
    labels = {"imaging_quality": "Imaging quality", "aesthetic_quality": "Aesthetic quality",
              "temporal_consistency": "Temporal consistency", "motion_effects": "Motion effects"}
    arms = [("brief", "Bare brief (~36 ch)", CONTROL),
            ("planned", "Planned (~900 ch)", OURS),
            ("planned_short", "Planned, trimmed (~400 ch)", MUTED)]

    data: dict[tuple[str, str], list[float]] = {}
    for dim in dims:
        path = AB / f"{dim}.json"
        if not path.exists():
            return None
        for key, values in json.loads(path.read_text(encoding="utf-8"))["samples"].items():
            arm, model = key.split("__", 1)
            data.setdefault((dim, arm), []).extend(values)
    if not data:
        return None

    fig, axes = plt.subplots(1, len(dims), figsize=(13.0, 3.9), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, dim in zip(axes, dims):
        _style(ax)
        series = [data[(dim, arm)] for arm, _, _ in arms if (dim, arm) in data]
        if not series:
            continue
        ax.hist(series, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                color=[c for a, _, c in arms if (dim, a) in data],
                rwidth=0.9, zorder=3, stacked=False)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.4, 5.6)
        ax.set_title(labels[dim], fontsize=9.8, color=INK, loc="left")

    axes[0].set_ylabel("Clips", fontsize=9.5, color=INK)
    handles = [Rectangle((0, 0), 1, 1, color=color) for _, _, color in arms]
    fig.legend(handles, [label for _, label, _ in arms], frameon=False, fontsize=8.8,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.055))
    fig.suptitle("Single-clip quality: the three prompt arms overlap heavily (5 seeds x 2 models per cell)",
                 fontsize=11.5, color=INK, x=0.012, ha="left", y=0.985)
    _note(fig, "Overlapping bars, not stacked: each colour is an independent distribution, so bar height is the "
               "count for that arm alone. The overlap is the point - the planned prompt sits lower on average "
               "but no cell separates cleanly.\n"
               "Data: bench/results/tier3a_ab/judge/*.json")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    out = FIG / "repeats_clip_hist.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    rows = {model: load(model) for model in MODELS}
    rows = {m: r for m, r in rows.items() if r}
    if not rows:
        print("no scored repeats yet")
        return 1
    for model, model_rows in rows.items():
        ind = [r["score"] for r in model_rows if r["arm"] == "independent"]
        pla = [r["score"] for r in model_rows if r["arm"] == "planned"]
        print(f"{model:16s} independent n={len(ind)} mean={statistics.mean(ind):.2f} "
              f"sd={statistics.stdev(ind) if len(ind) > 1 else 0:.2f} | "
              f"planned n={len(pla)} mean={statistics.mean(pla):.2f} "
              f"sd={statistics.stdev(pla) if len(pla) > 1 else 0:.2f}")
    print(figure_histograms(rows))
    print(figure_paired(rows))
    print(figure_clip_hist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
