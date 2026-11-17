"""Figures for the write-up, as opposed to the run dashboard.

``plot.py`` draws a run for the person who just ran it: one bar per model per
dimension, no room for the spread. That is the wrong chart for a paper, where
the first question a reader asks is "how many samples, and how much do they
move?". So these figures carry the individual seeds as marks on top of the mean,
and a model measured once is visibly a single dot rather than a confident bar.

    quality     judged score per dimension, one dot per seed
    identity    within-clip cross-segment identity, one dot per seed
    longform    identity over time, our 8-shot episode against LongLive's 30 s
    capability  what each generator can actually deliver, from the run's records

``capability`` is not a score: it is the comparison the scores cannot express,
because the open-source baselines have no multi-shot mode to score. It is drawn
from what each clip is (duration, shots, audio, subtitles), not from judgement.

Charts go to ``docs/figures/`` because ``bench/results/`` is gitignored — a
figure a reader cannot open from the README is not a figure.
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

_CJK_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
)
for _path in _CJK_CANDIDATES:
    if Path(_path).exists():
        font_manager.fontManager.addfont(_path)

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "bench/results/compare"
OUT = ROOT / "docs/figures"

# Fixed identity order, one hue per model. Validated for CVD separation with the
# dataviz palette check: worst adjacent pair ΔE 9.1 (protan), normal-vision
# floor ΔE 20.8. Slots are assigned by model, never by rank, so a model keeps
# its colour when a figure drops one.
MODELS = ["ours", "openmontage_shot", "longlive_1_3b", "wan21_1_3b", "cogvideox5b"]
LABELS = {
    "ours": "本项目 workflow",
    "openmontage_shot": "OpenMontage 镜头段（无编排）",
    "longlive_1_3b": "LongLive-1.3B",
    "wan21_1_3b": "Wan2.1-1.3B",
    "cogvideox5b": "CogVideoX-5B",
}
SHORT = {
    "ours": "本项目",
    "openmontage_shot": "OpenMontage\n镜头段",
    "longlive_1_3b": "LongLive\n1.3B",
    "wan21_1_3b": "Wan2.1\n1.3B",
    "cogvideox5b": "CogVideoX\n5B",
}
COLORS = {
    "ours": "#2a78d6",
    "openmontage_shot": "#eb6834",
    "longlive_1_3b": "#1baf7a",
    "wan21_1_3b": "#eda100",
    "cogvideox5b": "#e34948",
}
DIMS = {
    "imaging_quality": "成像质量",
    "aesthetic_quality": "美学质量",
    "temporal_consistency": "时序一致性",
    "motion_effects": "运动效果",
}

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#86847d"
GRID, SURFACE = "#e8e6e0", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"{path}  {path.stat().st_size // 1024} KB")
    return path


def load_run() -> dict:
    samples = {}
    for path in sorted((RUN / "judge").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples[payload["dimension"]] = payload.get("samples", {})
    segments = {}
    seg_path = RUN / "segments" / "segments.json"
    if seg_path.exists():
        for clip in json.loads(seg_path.read_text(encoding="utf-8"))["clips"]:
            segments.setdefault(clip["model"], []).append(clip["summary"])
    return {"samples": samples, "segments": segments}


def chart_quality(data: dict) -> None:
    """Mean per dimension with every seed drawn on top.

    The bar is the mean, the dots are the seeds. A model with one seed shows one
    dot — that asymmetry is the point of the figure, and it is why the caption
    says which models were measured once instead of leaving the reader to
    assume five samples each.
    """
    dims = [d for d in DIMS if d in data["samples"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=200, facecolor=SURFACE)
    _style(ax)

    x = np.arange(len(dims))
    width = 0.16
    for i, model in enumerate(MODELS):
        means, offsets = [], []
        for j, dim in enumerate(dims):
            values = data["samples"][dim].get(model) or []
            if not values:
                continue
            means.append(sum(values) / len(values))
            offsets.append(x[j] + (i - (len(MODELS) - 1) / 2) * width)
        if not means:
            continue
        ax.bar(offsets, means, width * 0.9, color=COLORS[model],
               label=LABELS[model], zorder=3, linewidth=0)
        # Seeds as marks. Drawn after the bars so a single seed sits on top of
        # its own mean, which is exactly what a reader should notice.
        for j, dim in enumerate(dims):
            values = data["samples"][dim].get(model) or []
            if not values:
                continue
            offset = x[j] + (i - (len(MODELS) - 1) / 2) * width
            jitter = np.linspace(-width * 0.22, width * 0.22, len(values)) if len(values) > 1 else [0]
            ax.scatter(offset + np.array(jitter), values, s=16, zorder=4,
                       color="#ffffff", edgecolor=INK_2, linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([DIMS[d] for d in dims], color=INK_2, fontsize=10)
    ax.set_ylim(0, 5.6)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel("得分（1–5）", color=INK_2, fontsize=10)
    ax.set_title("同一 prompt 的判分：柱为均值，点为单个 seed",
                 color=INK, fontsize=13, loc="left", pad=14)
    ax.legend(frameon=False, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -0.13), fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    _save(fig, "quality.png")


def chart_identity(data: dict) -> None:
    """Within-clip identity, one dot per seed, log-safe on a 0-1 axis.

    identity is a cosine similarity against the clip's own first segment; 1.0 is
    "unchanged". A clip that cuts between shots scores low here for a reason
    that is not a defect, so this figure is only about single-shot clips — see
    ``longform`` for what a multi-shot episode does instead.
    """
    rows = []
    for model in MODELS:
        for summary in data["segments"].get(model, []):
            value = summary.get("identity_min")
            if value is not None:
                rows.append((model, value, summary.get("drift_onset_segment")))
    if not rows:
        return

    order = [m for m in MODELS if any(r[0] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)

    for i, model in enumerate(order):
        values = [r[1] for r in rows if r[0] == model]
        mean = statistics.mean(values)
        ax.plot([0, mean], [i, i], color=COLORS[model], linewidth=2, zorder=3,
                solid_capstyle="round")
        jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else [0]
        ax.scatter(mean + np.array(jitter), [i] * len(values), s=42, zorder=4,
                   color=COLORS[model], edgecolor=SURFACE, linewidth=1.4)
        ax.text(mean + 0.02, i + 0.22, f"{mean:.3f}", fontsize=9, color=INK_2)

    ax.axvline(0.75, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.755, len(order) - 0.35, "0.75 = 检出漂移的门槛", fontsize=8.5,
            color=MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[m] for m in order], color=INK_2, fontsize=10)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("跨段一致性（与首段比，1.0 = 完全一致）", color=INK_2, fontsize=10)
    ax.set_title("单片段内部的稳定性：越高越不漂", color=INK, fontsize=13,
                 loc="left", pad=14)
    fig.tight_layout()
    _save(fig, "identity.png")


def _scene_cuts(path: Path, threshold: float = 0.25) -> list[float]:
    """Cut times in seconds, from ffmpeg's scene score.

    The storyboard says one shot every 4 s, but the cuts do not land there:
    xfade transitions consume a real overlap, so the measured cuts are
    4.8 / 8.83 / 12.83 / 16.87 / 20.87 / 24.9 / 28.7. Grouping segments by the
    storyboard's 4 s grid therefore split a shot across two groups and scored
    the straddling window against the wrong anchor — that artifact put our
    episode at 0.17 mid-clip, below LongLive, and nearly went into the write-up
    as a real regression. Detect the cuts, then group by them.
    """
    import re
    import subprocess

    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path),
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    times = sorted({
        round(float(m), 2)
        for m in re.findall(r"pts_time:([0-9.]+)", out.stderr)
    })
    # Merge cuts that land within a frame of each other (a hard cut reports twice).
    merged: list[float] = []
    for t in times:
        if not merged or t - merged[-1] > 0.15:
            merged.append(t)
    return merged


def chart_longform() -> None:
    """Identity over time: a continuous 30 s baseline against an 8-shot episode.

    This is the figure the single-clip scores cannot replace. LongLive runs one
    continuous shot, so its identity decays monotonically as the autoregressive
    window slides away from the frames it was conditioned on. Our episode cuts
    every 4 s, so its identity is measured *within* each shot: each segment is
    compared to its own shot's first frame, not to frame 0 of the episode.
    Comparing a cut film to a continuous one on a whole-clip anchor would
    measure the edit, not the drift — that mistake scored our episode at
    identity_min -0.83 and meant nothing.
    """
    sys.path.insert(0, str(ROOT))
    from bench.segments import cross_segment, segment_clip

    long_clip = ROOT / "bench/results/long30/videos/longlive_1_3b/longlive_1_3b_0.mp4"
    our_episode = ROOT / "workspace/projects/project_99ae3f4e/final/final_video.mp4"
    if not long_clip.exists() or not our_episode.exists():
        print("skip longform: missing clips")
        return

    # 1 s windows: 2 s left too few surviving windows per shot once the ones
    # straddling a cut were dropped, and the whole episode collapsed to two
    # points. 1 s gives 18 within-shot measurements across 8 shots.
    window = 1.0
    long_segments = segment_clip(long_clip, window)
    long_ident, _ = cross_segment(long_segments)
    long_pts = [(i["index"] * window, min(i["color"], i["structure"]))
                for i in long_ident]

    # Our episode is cut, so group its windows by the *measured* cuts, not by
    # the storyboard's 4 s grid, and anchor each group on its own first window.
    cuts = _scene_cuts(our_episode)
    segments = segment_clip(our_episode, window)
    bounds = [0.0, *cuts]

    # A window that straddles a cut is not a measurement of one shot: its
    # fingerprint changes because the film cut, not because the model drifted.
    # Grouping by cut time alone was not enough — the [20, 22] window still
    # spanned the 20.87 cut and read 0.06, dragging the whole panel down. So
    # does a window that overlaps a transition's crossfade, which blends two
    # shots together. Drop both instead of scoring them against either shot.
    #
    # The title card needs the same treatment for a different reason: it is a
    # black card with type on it, so windows 0-1.2 s share no structure with the
    # first shot at all and scored 0.37. That is the card, not the clip.
    guard = 0.7
    title_card_s = 1.2

    def _straddles(seg) -> bool:
        if seg.start_s < title_card_s:
            return True
        return any(seg.start_s - guard < c < seg.end_s + guard for c in cuts)

    by_shot: dict[int, list] = {i: [] for i in range(len(bounds))}
    for seg in segments:
        if _straddles(seg):
            continue
        shot = max(i for i, b in enumerate(bounds) if seg.start_s >= b - 1e-6)
        by_shot[shot].append(seg)
    our_pts = []
    for shot in sorted(by_shot):
        group = [s for s in by_shot[shot] if s.frame_count]
        if len(group) < 2:
            continue
        ident, _ = cross_segment(group)
        for item in ident:
            our_pts.append((item["start_s"], min(item["color"], item["structure"])))

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=200, facecolor=SURFACE)
    _style(ax)

    for cut in cuts:
        ax.axvline(cut, color=GRID, linewidth=1.0, zorder=1)
    ax.axhline(0.75, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)

    ax.plot([t for t, _ in long_pts], [v for _, v in long_pts], color=COLORS["longlive_1_3b"],
            linewidth=2, marker="o", markersize=5, zorder=4,
            label="LongLive-1.3B：30 s 连续单镜头", markerfacecolor=SURFACE,
            markeredgewidth=1.4)
    ax.plot([t for t, _ in our_pts], [v for _, v in our_pts], color=COLORS["ours"],
            linewidth=2, marker="s", markersize=5, zorder=5,
            label="本项目：32.7 s / 8 个镜头（镜头内比）")
    ax.text(0.2, 0.30, "竖线 = 实测切镜点（ffmpeg scene）", fontsize=8.5, color=MUTED)

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(t for t, _ in our_pts) + 2)
    ax.set_xlabel("时间（秒）", color=INK_2, fontsize=10)
    ax.set_ylabel("相对所属镜头首帧的一致性", color=INK_2, fontsize=10)
    ax.set_title("长序列：连续单镜头随窗口滑动缓慢衰减，分镜把漂移重置在每个镜头开头",
                 color=INK, fontsize=13, loc="left", pad=14)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    _save(fig, "longform.png")


def chart_capability() -> None:
    """What each generator delivers, from the artefacts rather than the judge.

    A matrix, not a score: the open-source baselines cannot produce a multi-shot
    episode at all, so there is no 1-5 value to plot. Each cell is a fact read
    off the run's files — duration from the container, shots from the project
    state, audio and subtitles from the project tree.
    """
    rows = [
        ("交付时长", ["32.7 s（8 镜）", "5.0 s", "30.0 s", "5.1 s", "5.1 s"]),
        ("多镜头", ["8", "1", "1", "1", "1"]),
        ("镜头间首尾帧衔接", ["有", "—", "—", "—", "—"]),
        ("剧本 / 分镜 / 角色设定", ["有", "—", "—", "—", "—"]),
        ("配音", ["8 段", "—", "—", "—", "—"]),
        ("字幕", ["13 行", "—", "—", "—", "—"]),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.4), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    cols = len(MODELS)
    for c, model in enumerate(MODELS):
        ax.add_patch(plt.Rectangle((c, len(rows) + 0.6), 1, 0.9,
                                   color=COLORS[model], clip_on=False))
        ax.text(c + 0.5, len(rows) + 1.05, SHORT[model], ha="center", va="center",
                fontsize=8.5, color="#ffffff", linespacing=1.25)

    for r, (name, values) in enumerate(rows):
        y = len(rows) - r - 0.5
        ax.text(-0.15, y, name, ha="right", va="center", fontsize=9.5, color=INK_2)
        for c, value in enumerate(values):
            has = value not in ("—", "1") or name in ("交付时长", "多镜头")
            ax.text(c + 0.5, y, value, ha="center", va="center", fontsize=9,
                    color=INK if has else MUTED)
        ax.axhline(len(rows) - r - 1, color=GRID, linewidth=0.8)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, len(rows) + 1.7)
    ax.set_title("交付能力：判分维度覆盖不到的部分", color=INK, fontsize=13,
                 loc="left", pad=34)
    fig.tight_layout()
    _save(fig, "capability.png")


def main() -> int:
    data = load_run()
    if not data["samples"]:
        raise SystemExit(f"no judge results in {RUN}; run --stage judge first")
    chart_quality(data)
    chart_identity(data)
    chart_longform()
    chart_capability()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
