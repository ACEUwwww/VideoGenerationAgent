"""Draw the orchestration topology as a figure.

The report describes the state machine in prose; a reader who wants to check
that the prose matches the code should be able to look at one picture. Every
box here is a node registered in ``agent/graph.py``, and every arrow is an edge
``build_graph`` adds — this file is a redrawing, not a second specification.
Keep it that way: if the graph changes, this figure is stale until it changes.

    bench/.venv/bin/python bench/study/workflow_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
OUT = ROOT / "docs/figures/workflow_topology.png"

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

SURFACE = "#ffffff"
INK = "#14171a"
MUTED = "#6b7280"
OURS = "#2a78d6"
CONTROL = "#eb6834"
EDGE = "#9aa1ab"
BAND = {"plan": "#eef4fd", "image": "#fdf1ec", "video": "#eef7f1", "compose": "#f1f2f4"}


def box(ax, cx, cy, w, h, title, sub="", fill=OURS, ink=SURFACE, dashed=False):
    """A rounded node. Returns (left, right, top, bottom) for arrow wiring."""
    if dashed:
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.06",
            linewidth=1.2, edgecolor=fill, facecolor=SURFACE, linestyle="--", zorder=3)

        def txt(y, s, size, weight):
            ax.text(cx, y, s, ha="center", va="center", fontsize=size,
                    color=fill, weight=weight, zorder=4)
    else:
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.06",
            linewidth=1.0, edgecolor=fill, facecolor=fill, zorder=3)

        def txt(y, s, size, weight):
            ax.text(cx, y, s, ha="center", va="center", fontsize=size,
                    color=ink, weight=weight, zorder=4)

    ax.add_patch(patch)
    if sub:
        txt(cy + 0.10, title, 8.6, "bold")
        txt(cy - 0.10, sub, 6.6, "normal")
    else:
        txt(cy, title, 8.6, "bold")
    return cx - w / 2, cx + w / 2, cy + h / 2, cy - h / 2


def arrow(ax, start, end, label="", rad=0.0, color=EDGE, lw=1.0, dashed=False,
          label_dy=0.075, label_dx=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, zorder=2,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=1,
        linestyle=(0, (4, 3)) if dashed else "solid",
    ))
    if label:
        mx = (start[0] + end[0]) / 2 + label_dx
        my = (start[1] + end[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=6.6, color=MUTED,
                zorder=5, bbox={"boxstyle": "round,pad=0.18", "facecolor": SURFACE,
                                "edgecolor": "none", "alpha": 0.94})


def band(ax, y0, y1, title, note, key):
    ax.add_patch(FancyBboxPatch(
        (0.10, y0), 12.80, y1 - y0,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        linewidth=0, facecolor=BAND[key], zorder=0))
    ax.text(1.15, y1 - 0.20, title, ha="left", va="top",
            fontsize=9.4, color=INK, weight="bold", zorder=1)
    ax.text(12.70, y1 - 0.20, note, ha="right", va="top",
            fontsize=6.9, color=MUTED, style="italic", zorder=1)


def main() -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 13.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13.4)
    ax.axis("off")

    # ------------------------------------------------------------- 1 planning
    band(ax, 10.70, 13.00, "1  Planning",
         "no pixels yet; every later prompt is derived from this stage", "plan")
    y = 11.80
    a = box(ax, 2.55, y, 2.7, 0.92, "analyze_user_goal", "brief -> genre, style, shot count")
    b = box(ax, 6.05, y, 2.7, 0.92, "plan_story", "logline, beats, CAST + appearance anchors")
    c = box(ax, 9.85, y, 3.1, 0.92, "plan_shots", "storyboard; rhythm checked, total re-timed")
    arrow(ax, (a[1], y), (b[0], y))
    arrow(ax, (b[1], y), (c[0], y))

    # ------------------------------------------------------------ 2 keyframes
    band(ax, 6.95, 10.55, "2  Keyframe phase — runs to completion before any clip",
         "skipped entirely when the video model declares frame_roles = empty set", "image")
    y = 8.80
    d = box(ax, 2.45, y, 2.55, 0.86, "generate_image_prompt", "anchor restated verbatim", CONTROL)
    e = box(ax, 5.45, y, 2.10, 0.86, "generate_image", "seed = shot index", CONTROL)
    f = box(ax, 8.35, y, 2.10, 0.86, "evaluate_image", "VLM vs prompt + anchor", CONTROL)
    g = box(ax, 11.45, y, 2.35, 0.86, "advance_keyframe", "next keyframe, or exit", CONTROL)
    arrow(ax, (d[1], y), (e[0], y))
    arrow(ax, (e[1], y), (f[0], y))
    arrow(ax, (f[1], y), (g[0], y), "passed")

    box(ax, 5.45, 7.62, 2.30, 0.62, "rewrite_image_prompt", fill=CONTROL, dashed=True)
    arrow(ax, (7.95, f[3]), (6.62, 7.62), "failed, retries left", rad=-0.22, color=CONTROL, lw=1.2)
    arrow(ax, (5.45, 7.93), (5.45, e[3]), "", color=CONTROL, lw=1.2)

    arrow(ax, (11.45, g[2]), (2.45, d[2]), "more keyframes", rad=-0.10)
    ax.text(8.45, 7.14, "every keyframe is drawn from keyframe[1], never from keyframe[N-1]",
            ha="center", va="center", fontsize=7.0, color=CONTROL, style="italic",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": SURFACE, "edgecolor": "none"})

    # ---------------------------------------------------------------- 3 video
    band(ax, 2.30, 6.80, "3  Video phase — one clip per shot, memory extracted after each",
         "shot N starts at keyframe[N] and, where the model accepts it, ends at keyframe[N+1]", "video")
    y = 5.55
    h = box(ax, 2.25, y, 2.40, 0.84, "start_video_phase", "rewind the cursor")
    i = box(ax, 5.25, y, 2.55, 0.84, "generate_video_prompt", "T2V or motion template")
    j = box(ax, 8.10, y, 2.10, 0.84, "generate_video", "first/last frame join")
    k = box(ax, 10.95, y, 2.20, 0.84, "evaluate_video", "clip QA")
    arrow(ax, (h[1], y), (i[0], y))
    arrow(ax, (i[1], y), (j[0], y))
    arrow(ax, (j[1], y), (k[0], y), "rendered")

    l = box(ax, 10.95, 4.10, 2.20, 0.80, "evaluate_clip", "memory + 4 dims + drift")
    box(ax, 8.10, 4.10, 2.35, 0.62, "rewrite_video_prompt", fill=OURS, dashed=True)
    arrow(ax, (10.95, k[3]), (10.95, l[2]))
    arrow(ax, (10.50, k[3]), (9.28, 4.10), "failed, retries left", rad=0.20, color=OURS, lw=1.2)
    arrow(ax, (8.10, 4.41), (8.10, j[3]), "", color=OURS, lw=1.2)

    n = box(ax, 10.95, 2.95, 2.40, 0.62, "synthesize_dialogue", fill=OURS)
    o = box(ax, 7.60, 2.95, 2.30, 0.62, "advance_shot", fill=OURS)
    arrow(ax, (10.95, l[3]), (10.95, n[2]))
    arrow(ax, (n[0], 2.95), (o[1], 2.95))
    arrow(ax, (o[0], 2.95), (5.25, i[3]), "more shots", rad=0.30, label_dx=-0.9, label_dy=-0.28)

    # ------------------------------------------------------------ 4 composition
    band(ax, 0.35, 2.05, "4  Composition",
         "sequence-level judgement on the ordered keyframes", "compose")
    p = box(ax, 4.20, 1.20, 2.90, 0.72, "compose_video", "subtitles burned, cover made", INK)
    q = box(ax, 8.70, 1.20, 2.00, 0.72, "finish", "state -> completed", INK)
    arrow(ax, (p[1], 1.20), (q[0], 1.20))
    arrow(ax, (7.60, o[3]), (4.20, p[2]), "all shots done", label_dx=1.5)

    # --- the two-to-video bypass: plan_shots -> start_video_phase, in the gutter
    ax.plot([9.85, 9.85, 0.55, 0.55], [11.34, 11.02, 11.02, 5.55],
            color=INK, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
    ax.add_patch(FancyArrowPatch((0.55, 5.55), (1.03, 5.55), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2, color=INK,
                                 linestyle=(0, (5, 3)), zorder=2))
    ax.text(5.10, 11.02, "text-to-video model: the whole keyframe phase is skipped",
            ha="center", va="center", fontsize=7.0, color=INK, style="italic",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": SURFACE, "edgecolor": "none"})

    fig.text(0.010, 0.010,
             "Redrawn from agent/graph.py: every box is a registered node and every arrow an edge build_graph adds. "
             "A clip rejected by output moderation bypasses QA and goes straight to advance_shot, so one rejected "
             "shot costs one shot rather than the episode.\n"
             "Loops are bounded: image and video retries stop at the configured retry count, and LangGraph runs the "
             "whole graph under a fixed recursion limit of 200.",
             fontsize=6.6, color=MUTED, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main())
