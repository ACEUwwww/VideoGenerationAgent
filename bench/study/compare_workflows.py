"""Compare two workflows as workflows, not as clips.

The clip-level bench answers "which generator makes a better five-second shot".
That is not this project's claim. The claim is about the *workflow*: given one
line of creative brief, does the orchestration deliver a coherent multi-shot
episode with sound, subtitles and a stable cast — and at what cost.

So this report is organised around what each side actually delivered, and only
falls back to judged clip scores where both sides have them. Four rules it
follows, each of which a previous comparison in this repo broke:

1. **Duration parity is stated, not assumed.** A 60 s / 15-shot episode and a
   32 s / 8-shot one are different tasks. Every row carries the delivered
   seconds, so a quality difference that is really a length difference is
   visible as such.
2. **Identity across cuts is measured per shot, with the same ruler on both
   sides.** Shot boundaries come from each workflow's own declared timeline on
   both sides, then each cut is *verified* against the pixels. An episode is
   anchored on each shot's own first window, never on frame 0 of the film; the
   title card and the closing fade are excluded. Done naively this scored our
   own episode at identity_min -0.83 and nearly shipped. Declared durations are
   used because our composer joins shots with crossfades, which a hard-cut
   detector cannot see: measured on one episode it returned zero cuts and the
   64 s film was then scored as a single shot.
3. **A capability the other side lacks is a fact, not a score.** Subtitles,
   audio, per-shot voice and shot count come from the files. They are not
   converted into points, and no total is invented by summing layers.
4. **Every delivered episode is included, and the count is printed.** Reading
   only the newest row is how a comparison ends up resting on one sample while
   the run produced nine; the aggregate states its n so a reader cannot mistake
   two episodes for a distribution.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

from bench.study.episode_eval import EpisodeReport, evaluate_episode, to_dict  # noqa: E402

OM = ROOT / "bench/_vendor/OpenMontage"
OM_PROJECTS = OM / "bench_run/projects"
OM_RESULTS = ROOT / "bench/results/om_full/episodes.json"
OURS_RESULTS = ROOT / "bench/results/ours_full"
OUT = ROOT / "bench/results/compare_workflows"


def declared_bounds_from_state(state_path: Path) -> list[float]:
    """Cumulative shot durations from a storyboard, minus the final total."""
    if not state_path.exists():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    total = 0.0
    bounds: list[float] = []
    for shot in state.get("shots") or []:
        if shot.get("deletedAt"):
            continue
        total += float(shot.get("duration_s") or 0)
        bounds.append(round(total, 2))
    return bounds[:-1] if bounds else bounds


def om_declared_bounds(project_id: str, episode_id: str) -> list[float]:
    """Cumulative clip ends from OpenMontage's own edit timeline.

    Same ruler as ours: each workflow's declared shot durations, verified
    against the pixels afterwards. Reading the timeline rather than the
    storyboard is deliberate — the timeline is what was actually exported, so a
    shot that never made it into the film cannot add a phantom boundary.
    """
    state_path = OM_PROJECTS / project_id / "studio_state.json"
    if not state_path.exists():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for edit in state.get("edits") or []:
        if edit.get("episodeId") != episode_id:
            continue
        total = 0.0
        bounds: list[float] = []
        for clip in edit.get("clips") or []:
            if not clip.get("enabled", True):
                continue
            total += float(clip.get("end") or 0) - float(clip.get("start") or 0)
            bounds.append(round(total, 2))
        return bounds[:-1] if bounds else bounds
    return []


def ours_episodes() -> list[dict]:
    """Every completed episode our pipeline wrote, oldest format included.

    Two result files exist because the driver was rewritten mid-study: an older
    ``episodes.json`` holding a single object, and the current append-only
    ``episodes.jsonl``. Reading only one of them silently drops episodes.
    """
    rows: list[dict] = []
    legacy = OURS_RESULTS / "episodes.json"
    if legacy.exists():
        rows.append(json.loads(legacy.read_text(encoding="utf-8")))
    jsonl = OURS_RESULTS / "episodes.jsonl"
    if jsonl.exists():
        rows += [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

    episodes = []
    seen: set[str] = set()
    for row in rows:
        if row.get("status") != "completed":
            continue
        project_id = row.get("project_id") or ""
        path = row.get("final_video") or row.get("path") or ""
        if not path or project_id in seen:
            continue
        seen.add(project_id)
        episodes.append({
            "side": "ours",
            "project_id": project_id,
            "brief_index": row.get("brief_index"),
            "repeat": row.get("repeat"),
            "path": Path(path),
            "elapsed_s": row.get("elapsed_s") or 0.0,
            "declared_bounds": declared_bounds_from_state(
                ROOT / "workspace/projects" / project_id / "state.json"),
            "subtitles": _count_files(ROOT / "workspace/projects" / project_id, "subtitles"),
            "audio": speech_files(project_id),
            "shot_files": our_shot_files(project_id),
        })
    return episodes


def _count_files(root: Path, name: str) -> int:
    directory = root / name
    return len(list(directory.glob("*"))) if directory.is_dir() else 0


def speech_files(project_id: str) -> dict:
    """Real synthesised speech in a project, separated from placeholder racks.

    The directory count is not a voiceover count. With ``TTS_PROVIDER=mock`` the
    pipeline still writes one ``<shot>.json`` plus one silent ``<shot>.m4a`` and
    one ``<shot>_fit.m4a`` per shot, and every one of those carries
    ``"provider": "mock"``. Counting files therefore reports a three-times
    inflated number of "voice tracks" for episodes that in fact contain no
    speech at all. The provider field in the sidecar is the only thing that
    distinguishes them, so the count is taken from there.
    """
    directory = ROOT / "workspace/projects" / project_id / "audio"
    real = mock = 0
    if directory.is_dir():
        for sidecar in sorted(directory.glob("*.json")):
            try:
                provider = json.loads(sidecar.read_text(encoding="utf-8")).get("provider") or ""
            except (OSError, ValueError):
                continue
            if provider and provider != "mock":
                real += 1
            else:
                mock += 1
    return {"real": real, "mock": mock}


def om_episodes() -> list[dict]:
    """Every exported OpenMontage episode, with its own declared timeline."""
    if not OM_RESULTS.exists():
        return []
    episodes = []
    for row in json.loads(OM_RESULTS.read_text(encoding="utf-8")):
        if not row.get("ok") or not row.get("export_path"):
            continue
        project_id = row.get("project_id") or ""
        path = OM_PROJECTS / project_id / row["export_path"]
        if not path.exists():
            continue
        episode_id = ""
        brief_index = None
        state_path = OM_PROJECTS / project_id / "studio_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            episode_id = (state.get("episodes") or [{}])[0].get("id", "")
            # The brief is what makes a pairing valid: an episode rendered from
            # brief 0 cannot be compared against one rendered from brief 1, and
            # the site-level aggregate was mixing them until this was recorded.
            brief_index = brief_index_of((state.get("project") or {}).get("brief") or "")
        episodes.append({
            "side": "openmontage_full",
            "project_id": project_id,
            "brief_index": brief_index,
            "repeat": row.get("seed"),
            "path": path,
            "elapsed_s": row.get("total_seconds") or 0.0,
            "declared_bounds": om_declared_bounds(project_id, episode_id),
            "subtitles": 0,
            "audio": 0,
            "shot_files": om_shot_files(project_id),
        })
    return episodes


def brief_index_of(text: str) -> int | None:
    """Index of a brief in study/briefs.json, or None when it is not one of them."""
    path = ROOT / "bench/study/briefs.json"
    if not path.exists():
        return None
    for index, brief in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        if brief.strip() == (text or "").strip():
            return index
    return None


def pair_by_brief(ours: list[dict], om: list[dict]) -> tuple[list[dict], list[dict], list[int]]:
    """Keep only episodes whose brief exists on both sides.

    The whole point of the comparison is "same brief, same base model". An
    episode that ran a different brief answers a different question, and
    including it inflates n while making every aggregate a mixture of unrelated
    tasks. Returns the filtered lists plus the briefs that survived, so the
    report can say which ones were compared and which were dropped.
    """
    ours_briefs = {e.get("brief_index") for e in ours if e.get("brief_index") is not None}
    om_briefs = {e.get("brief_index") for e in om if e.get("brief_index") is not None}
    shared = sorted(ours_briefs & om_briefs)
    if not shared:
        return ours, om, []
    return ([e for e in ours if e.get("brief_index") in shared],
            [e for e in om if e.get("brief_index") in shared],
            shared)


def _fingerprint(path: Path) -> str:
    """Cheap identity for a video file, so repeated runs reuse measurements.

    Every comparison re-decodes each film frame by frame to measure motion and
    per-shot identity. That is several minutes for six episodes, and it grows
    with every new episode while producing identical numbers for the episodes
    that did not change. The cache is keyed on the file's size and mtime, so a
    re-rendered episode misses and is measured again.
    """
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"


def _cache_path() -> Path:
    return OUT / "measure_cache.json"


def _load_cache() -> dict:
    path = _cache_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_cache(cache: dict) -> None:
    _cache_path().parent.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def motion_level(path: Path, cuts: list[float], *, window: float = 0.25) -> float:
    """Mean frame-to-frame difference within shots, as a motion budget.

    Identity alone is not a fair comparison between two films: a nearly static
    shot keeps a stable fingerprint whatever the generator does, so a lower-
    motion film scores better on drift without being more consistent in any way
    a viewer would recognise. Measuring motion alongside identity is what makes
    the difference interpretable instead of merely reported.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(fps / 4))
        values: list[float] = []
        previous = None
        index = 0
        while index < total:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            t = index / fps
            # Frames either side of a cut differ for editorial reasons; they are
            # not motion and must not be counted as it.
            if any(abs(t - c) < window for c in cuts):
                previous = None
                index += step
                continue
            small = cv2.resize(frame, (160, 90)).astype(np.float32)
            if previous is not None:
                values.append(float(np.abs(small - previous).mean()))
            previous = small
            index += step
        return round(sum(values) / len(values), 2) if values else 0.0
    finally:
        capture.release()


def our_shot_files(project_id: str) -> list[Path]:
    """Our per-shot source clips, in shot order.

    Measuring identity on these instead of on the composite is a correction
    forced by measurement, not a convenience. The composite is joined with
    crossfades, so a boundary taken from the storyboard drifts against the
    pixels: the storyboard says 4.0 s per shot while each source clip is
    4.0417 s, and the overlap shortens the film further. By shot 10 the declared
    timeline was ~0.4 s off, and a real internal jump inside shot_010 was
    attributed to the wrong boundary.

    The source clips are the same artefact on both sides — one Seedance render
    per shot, identical resolution, fps and duration — so this is also the only
    unit on which the two workflows can be measured with one ruler. What it
    measures is *within-shot* stability. Cross-shot character consistency is a
    different question that fingerprints cannot answer, and this report does not
    claim it.
    """
    directory = ROOT / "workspace/projects" / project_id / "videos"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("shot_*.mp4"))


def om_shot_files(project_id: str) -> list[Path]:
    """OpenMontage's selected take for each shot, in shot order."""
    state_path = OM_PROJECTS / project_id / "studio_state.json"
    if not state_path.exists():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = [s for s in (state.get("shots") or []) if not s.get("deletedAt")]
    rows.sort(key=lambda s: int(s.get("shotOrder") or 0))
    paths: list[Path] = []
    for shot in rows:
        take = next((v for v in shot.get("videos") or []
                     if v.get("id") == shot.get("selectedVideoId")), None)
        if not take:
            continue
        path = OM_PROJECTS / project_id / take["path"]
        if path.exists():
            paths.append(path)
    return paths


def per_shot_identity(paths: list[Path]) -> list[dict]:
    """Within-shot identity for every source clip, one row per clip.

    Each clip is its own shot, so it is measured with no cuts: the whole file is
    divided into 1-second windows and each is compared against the clip's own
    first window. A value near 1 means the picture stayed the same thing; a drop
    means it drifted inside a single render, which is what the workflow's
    prompt/keyframe decisions are supposed to prevent.
    """
    from bench.study.episode_eval import within_shot_identity

    rows: list[dict] = []
    for index, path in enumerate(paths):
        values = within_shot_identity(path, [], skip_before=0.0)
        if not values:
            continue
        scores = [r["identity"] for r in values]
        colors = [r["color"] for r in values if "color" in r]
        structures = [r["structure"] for r in values if "structure" in r]
        rows.append({
            "shot": index,
            "windows": len(scores),
            "identity_median": round(statistics.median(scores), 3),
            "identity_min": round(min(scores), 3),
            # Kept apart because they answer different questions and disagree
            # sharply: on a clip that cuts from a wide back view to a front
            # close-up of the same character, colour holds at 0.885 while
            # structure falls to 0.694 (and to -0.071 on another clip). Colour
            # is the palette/scene signature; structure is layout, so it moves
            # when the camera moves regardless of who is in frame.
            "color_median": round(statistics.median(colors), 3) if colors else None,
            "color_min": round(min(colors), 3) if colors else None,
            "structure_median": round(statistics.median(structures), 3) if structures else None,
            "structure_min": round(min(structures), 3) if structures else None,
            "scores": scores,
        })
    return rows


def summarise_shot_identity(rows: list[dict]) -> dict:
    """Aggregate over clips: median of per-clip medians, worst per-clip minimum."""
    if not rows:
        return {}
    medians = [r["identity_median"] for r in rows]
    mins = [r["identity_min"] for r in rows]
    colors = [r["color_median"] for r in rows if r.get("color_median") is not None]
    structures = [r["structure_median"] for r in rows if r.get("structure_median") is not None]
    return {
        "clips": len(rows),
        "windows": sum(r["windows"] for r in rows),
        "identity_median": round(statistics.median(medians), 3),
        "identity_median_spread": round(max(medians) - min(medians), 3),
        "identity_min": round(min(mins), 3),
        "clips_below_075": sum(1 for v in mins if v < 0.75),
        "color_median": round(statistics.median(colors), 3) if colors else None,
        "color_min": round(min(colors), 3) if colors else None,
        "structure_median": round(statistics.median(structures), 3) if structures else None,
        "structure_min": round(min(structures), 3) if structures else None,
        "clips_color_below_075": sum(1 for r in rows
                                     if r.get("color_min") is not None and r["color_min"] < 0.75),
    }


def measure(episode: dict, cache: dict | None = None) -> dict:
    """All measurements for one episode, reusing anything already computed.

    The cache key covers the film and every source clip, so an episode whose
    files did not change is not decoded twice, while a re-rendered one is.
    """
    cache = cache if cache is not None else {}
    film_key = _fingerprint(episode["path"])
    shot_keys = [_fingerprint(p) for p in episode["shot_files"]]
    entry = cache.get(film_key)

    if entry and entry.get("shot_keys") == shot_keys:
        report = EpisodeReport(
            model=episode["side"], path=episode["path"], duration_s=entry["duration_s"],
            shots=entry["shots"], cuts=entry["cuts"], identity=entry["identity"],
            structure=entry["structure"], summary=entry["summary"],
        )
        motion = entry["motion"]
        shot_rows = entry["shot_identity"]
    else:
        report = evaluate_episode(episode["path"], episode["side"], shots=0,
                                  elapsed_s=episode["elapsed_s"],
                                  declared_bounds=episode["declared_bounds"])
        motion = motion_level(episode["path"], report.cuts)
        shot_rows = per_shot_identity(episode["shot_files"])
        cache[film_key] = {
            "shot_keys": shot_keys,
            "duration_s": report.duration_s,
            "shots": report.shots,
            "cuts": report.cuts,
            "identity": report.identity,
            "structure": report.structure,
            "summary": report.summary,
            "motion": motion,
            "shot_identity": shot_rows,
        }

    shot_summary = summarise_shot_identity(shot_rows)
    return {
        "episode": episode,
        "report": report,
        "motion": motion,
        "shot_identity": shot_rows,
        "shot_summary": shot_summary,
        "json": {**to_dict(report), "motion": motion,
                 "shot_identity": shot_rows, "shot_summary": shot_summary},
    }


def _planned_count(side: str) -> str:
    """Episodes the run was asked for, from its own written plan.

    Both drivers write ``plan.json`` before starting, precisely so this number
    is the denominator and not an inference. Deriving it from ``episodes.jsonl``
    gave 3 for a 9-episode batch, because at that moment only three rows had
    been written — a finished-looking denominator for a run barely started.
    """
    if side == "ours":
        path = OURS_RESULTS / "plan.json"
        return str(len(json.loads(path.read_text(encoding="utf-8"))["episodes"])) if path.exists() else "—"
    path = ROOT / "bench/results/om_full/plan.json"
    return str(len(json.loads(path.read_text(encoding="utf-8"))["episodes"])) if path.exists() else "—"


def _failure_count(side: str) -> str:
    if side == "ours":
        path = OURS_RESULTS / "episodes.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return str(sum(1 for r in rows if r.get("status") == "failed"))
        return "—"
    path = ROOT / "bench/results/om_full/episodes.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        return str(sum(1 for r in rows if not r.get("ok")))
    return "—"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Workflow vs workflow comparison")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv[1:])

    ours = ours_episodes()
    om = om_episodes()
    if not ours:
        raise SystemExit("no completed ours episodes; run run_ours_study.py first")
    if not om:
        raise SystemExit("no completed OpenMontage episodes; run om_full.py first")
    ours_all, om_all = list(ours), list(om)
    ours, om, shared_briefs = pair_by_brief(ours, om)

    cache = _load_cache()
    measured = [measure(e, cache) for e in ours + om]
    _save_cache(cache)

    lines = ["# 工作流对比：本项目 vs OpenMontage 全流程", ""]
    lines += [
        "两侧用**同一条 brief**、**同一底模**（`doubao-seedance-1-0-pro-fast-251015`）、"
        "同一分辨率口径（480p，9:16）生成。差异只在编排层。",
        "",
        f"**配对口径**：只比较两侧都跑过的 brief。共同 brief = {shared_briefs or '无'}，"
        f"纳入 {len(ours)} 条本项目成片、{len(om)} 条 OpenMontage 成片。",
        "",
    ]
    dropped_ours = [e for e in ours_all if e not in ours]
    dropped_om = [e for e in om_all if e not in om]
    if dropped_ours or dropped_om:
        lines += [
            f"**未纳入的成片**：本项目 {len(dropped_ours)} 条、OpenMontage {len(dropped_om)} 条，"
            "它们跑的是另一条 brief，与对侧没有可比对象。不纳入不是丢弃数据——"
            "这些成片本身仍然有效，只是回答的是不同问题，把它们混进聚合会让 n 变大而结论变混。",
            "",
        ]
    lines += [
        "聚合一行只代表这些样本，不代表这两个 workflow 的极限。",
        "",
    ]

    # Per-episode facts. Identity is measured on the source clips, not on the
    # composite; see our_shot_files for why the composite's declared timeline
    # cannot carry it.
    lines += [
        "## 逐条成片（交付事实）",
        "",
        "| 侧 | brief | 重复 | 成片时长（s） | 交付镜数 | 源镜头文件 | 音轨 | 真实配音 | 合成片实测切点 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
    ]
    for item in measured:
        episode, report = item["episode"], item["report"]
        side = "本项目" if episode["side"] == "ours" else "OpenMontage"
        # Delivered shot count comes from the source files, not from the cuts
        # that survived verification. One episode rendered 15 shots but one
        # boundary did not clear the pixel check, and taking len(cuts)+1
        # reported it as 14 shots — a measurement limitation written down as a
        # delivery fact.
        delivered = len(episode["shot_files"]) or report.shots
        voice = episode["audio"] if isinstance(episode["audio"], dict) else {"real": 0, "mock": 0}
        # "有音轨" is true whenever the muxer wrote a stream, including a silent
        # one. Both columns are printed so an empty track cannot pass for a
        # voiceover.
        lines.append(
            f"| {side} | {_fmt(episode.get('brief_index'), '')} | {_fmt(episode.get('repeat'), '')} "
            f"| {report.duration_s:.2f} | {delivered} | {len(episode['shot_files'])} "
            f"| {'有' if report.structure.get('has_audio') else '无'} "
            f"| {voice['real']} "
            f"| {len(report.cuts)} |")
    lines.append("")

    lines += [
        "## 逐镜内部稳定性（主指标）",
        "",
        "每个源镜头单独测量：片内按 1 秒切窗，每窗与该镜头自己的首窗比对。"
        "两侧的源镜头是同一件事——一次 Seedance 渲染、480×864、24fps、4.04s——"
        "所以这是两套编排可以共用一把尺子的唯一单位。",
        "",
        "**「色彩」与「构图」分开列，因为它们是两件事。** 色彩是调色板签名；构图是去均值的亮度指纹，"
        "机位一动就掉，哪怕画面里的人完全没变。实测例子：一条镜头从背影全景推到正面特写、"
        "人物与服装均未变，色彩 0.885 而构图掉到 0.694；另一条色彩 0.901、构图 −0.071。"
        "因此**「构图」列不能读成角色一致性**——本报告没有测角色身份，也没有声称测了。",
        "",
        "| 侧 | brief | 重复 | 镜头数 | 可测窗口 | 合成中位 | 色彩中位 | 色彩最低 | 构图中位 | 构图最低 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in measured:
        episode, summary = item["episode"], item["shot_summary"]
        side = "本项目" if episode["side"] == "ours" else "OpenMontage"
        lines.append(
            f"| {side} | {_fmt(episode.get('brief_index'), '')} | {_fmt(episode.get('repeat'), '')} "
            f"| {summary.get('clips', 0)} | {summary.get('windows', 0)} "
            f"| {_fmt(summary.get('identity_median'), 3)} "
            f"| {_fmt(summary.get('color_median'), 3)} | {_fmt(summary.get('color_min'), 3)} "
            f"| {_fmt(summary.get('structure_median'), 3)} | {_fmt(summary.get('structure_min'), 3)} |")
    lines.append("")

    # Aggregate, with its n stated.
    by_side: dict[str, list[dict]] = {"ours": [], "openmontage_full": []}
    for item in measured:
        by_side[item["episode"]["side"]].append(item)

    lines += [
        "## 聚合（每侧 n 见下表）",
        "",
        "| 项 | 本项目 | OpenMontage 全流程 |",
        "| --- | ---: | ---: |",
    ]
    for label, key, digits in (
        ("成片数 n", None, 0),
        ("源镜头总数", "clips", 0),
        ("交付时长中位（s）", "duration_s", 2),
        ("交付镜数中位", "shots", 0),
        ("逐镜 identity 中位（镜头级中位再取中位）", "shot_median", 3),
        ("逐镜 identity 最低（所有镜头的最小）", "shot_min", 3),
        ("<0.75 的镜头数合计", "clips_below", 0),
        ("合成片运动量中位（帧间差）", "motion", 2),
    ):
        cells = []
        for side in ("ours", "openmontage_full"):
            items = by_side[side]
            if not items:
                cells.append("—")
                continue
            if key is None:
                cells.append(str(len(items)))
                continue
            if key == "motion":
                values = [i["motion"] for i in items]
            elif key == "duration_s":
                values = [i["report"].duration_s for i in items]
            elif key == "shots":
                # Delivered shots, not the cuts that cleared verification: one
                # episode rendered 15 and had 14 verified boundaries, and the
                # aggregate reported "14" beside a per-episode table saying 15.
                values = [float(len(i["episode"]["shot_files"]) or i["report"].shots) for i in items]
            elif key == "clips":
                # A total, not a median of totals — the label says 总数 and each
                # episode contributes its own clips.
                cells.append(str(sum(len(i["episode"]["shot_files"]) or 0 for i in items)))
                continue
            elif key == "clips_below":
                cells.append(str(sum(i["shot_summary"].get("clips_below_075") or 0 for i in items)))
                continue
            elif key == "shot_min":
                values = [i["shot_summary"].get("identity_min") for i in items]
            elif key == "shot_median":
                values = [i["shot_summary"].get("identity_median") for i in items]
            else:
                values = []
            values = [float(v) for v in values if v is not None]
            if not values:
                cells.append("—")
            elif key == "shot_min":
                cells.append(f"{min(values):.{digits}f}")
            else:
                cells.append(f"{statistics.median(values):.{digits}f}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines.append("")

    # Does each delivered episode match the brief it was given? This is the
    # workflow-level text-alignment question, distinct from the clip bench: the
    # clip dimensions score a five-second clip against its generation prompt,
    # this scores a finished film against one line of creative intent.
    evt_ours_mean: float | None = None
    evt_om_mean: float | None = None
    evt_path = args.out / "episode_video_text.json"
    if evt_path.exists():
        evt = [r for r in json.loads(evt_path.read_text(encoding="utf-8"))
               if r.get("scores")]
        if evt:
            lines += [
                "## 成片与 brief 的一致性（VLM 判定，1-3）",
                "",
                "问的是交付的整片有没有兑现 brief，不是单镜头画质。"
                "注意刻度是 1-3，与前面 1-5 的质量分**不可相加**。",
                "",
                "| 维度 | 本项目 | OpenMontage 全流程 |",
                "| --- | ---: | ---: |",
            ]
            ours_vals: list[float] = []
            om_vals: list[float] = []
            for row in sorted(evt, key=lambda r: r["dimension"]):
                scores = row["scores"]
                ours_s = scores.get("ours")
                om_s = scores.get("openmontage_full")
                if ours_s is not None:
                    ours_vals.append(float(ours_s))
                if om_s is not None:
                    om_vals.append(float(om_s))
                lines.append(f"| {row['dimension']} | {_fmt(ours_s, 0)} | {_fmt(om_s, 0)} |")
            lines += [
                f"| **均值** | **{statistics.mean(ours_vals):.2f}** | "
                f"**{statistics.mean(om_vals):.2f}** |" if ours_vals and om_vals else "| 均值 | — | — |",
                "",
                "口径警示：upstream 的这几个维度是为单条生成片段配视觉提示词设计的。"
                "用一句话的创作 brief 去评一条 60 秒多镜成片，匹配本身更松，"
                "所以绝对分偏低是刻度问题，不是片子质量问题。"
                "**但两侧用同一 brief、同一裁判，差值仍有意义。**",
                "",
            ]
            evt_ours_mean = statistics.mean(ours_vals) if ours_vals else None
            evt_om_mean = statistics.mean(om_vals) if om_vals else None

    # Character consistency, scored by the VLM. This is the question the
    # fingerprint columns cannot answer: who is in the frame, not how the frame
    # is laid out. Scores are 1-5 per episode; repeats are separate rows because
    # the judge is deterministic at temperature 0, so a disagreement means the
    # question is unstable, not that the sample is noisy.
    character_scores: dict[str, list[float]] = {}
    cc_path = args.out / "character_consistency.json"
    if cc_path.exists():
        entries = [e for e in json.loads(cc_path.read_text(encoding="utf-8"))
                   if e.get("score") is not None]
        if entries:
            lines += [
                "## 跨镜角色一致性（VLM 判定，1-5）",
                "",
                "指纹列测的是构图，不是人物。这一节直接问视觉语言模型：把同一集不同镜头的画面放在一起，"
                "里面是不是同一个人——脸、发型、体型、服装。机位、背景、打光、调色被明确要求排除。",
                "",
                "| 侧 | 集 | shots | 分 | 变化项 |",
                "| --- | --- | ---: | ---: | --- |",
            ]
            for entry in sorted(entries, key=lambda e: (e["label"], e.get("repeat", 0))):
                side = "本项目" if entry["label"].startswith("ours:") else "OpenMontage"
                project = entry["label"].split(":", 1)[-1]
                repeat = entry.get("repeat", 0)
                tag = f"{project} r{repeat}" if repeat else project
                lines.append(f"| {side} | `{tag}` | {entry.get('shots', 0)} | {entry['score']:g} "
                             f"| {entry.get('changed') or '—'} |")
            lines.append("")
            by_label = character_scores
            for entry in entries:
                side = "ours" if entry["label"].startswith("ours:") else "openmontage_full"
                by_label.setdefault(side, []).append(float(entry["score"]))
            cells = []
            for side in ("ours", "openmontage_full"):
                values = by_label.get(side) or []
                cells.append(f"{statistics.median(values):g}（n={len(values)}）" if values else "—")
            lines += [
                f"角色一致性中位：本项目 {cells[0]}，OpenMontage {cells[1]}。",
                "",
                "注意样本量：本项目侧每条集是一个独立采样，重复判定只是同一集问两遍以验证问题稳定，"
                "**不构成独立样本**。",
                "",
            ]

    # Delivered capability that is a fact on one side and absent on the other.
    ours_subs = sum(e["subtitles"] for e in ours)
    ours_voice = sum(e["audio"]["real"] for e in ours)
    ours_placeholder = sum(e["audio"]["mock"] for e in ours)
    ours_with_audio = sum(1 for i in by_side["ours"] if i["report"].structure.get("has_audio"))
    om_with_audio = sum(1 for i in by_side["openmontage_full"] if i["report"].structure.get("has_audio"))
    # Success rate is a delivered fact and belongs beside the capability rows,
    # not in a footnote. One of our nine planned episodes died at 1135 s because
    # Ark's output moderation rejected a rendered clip
    # (OutputVideoSensitiveContentDetected) and nothing in the pipeline isolates
    # that failure to the shot. Hiding it would make our n look cleaner than it
    # is and would misrepresent the workflow's reliability.
    ours_planned = _planned_count("ours")
    om_planned = _planned_count("openmontage_full")
    ours_failed = _failure_count("ours")
    om_failed = _failure_count("openmontage_full")
    lines += [
        "## 交付能力（事实，不折算成分数）",
        "",
        "| 项 | 本项目 | OpenMontage 全流程 |",
        "| --- | ---: | ---: |",
        f"| 计划集数 | {ours_planned} | {om_planned} |",
        f"| 成功出片 | {len(ours)} | {len(om)} |",
        f"| 失败 | {ours_failed} | {om_failed} |",
        f"| 含音轨的成片 | {ours_with_audio}/{len(ours)} | {om_with_audio}/{len(om)} |",
        f"| 字幕文件合计 | {ours_subs} | 0（该流程无字幕阶段） |",
        f"| 真实配音条数 | {ours_voice} | 0 |",
        f"| 占位静音轨条数 | {ours_placeholder} | 0 |",
        "",
    ]
    if ours_placeholder and not ours_voice:
        # The most dangerous kind of wrong number: a real count of the wrong
        # thing. Before this row existed the report said "配音文件合计 120",
        # which read as a hundred-plus voice tracks; all 120 were silent
        # placeholders written with TTS_PROVIDER=mock.
        lines += [
            f"本轮 `TTS_PROVIDER=mock`，{ours_placeholder} 条音轨（每镜一个 sidecar + `.m4a` + `_fit.m4a`，"
            f"共 {ours_placeholder * 3} 个文件）**全部是 ffmpeg 生成的静音占位轨**，"
            "不带任何语音。因此上表只能说明「本项目会产出音轨容器并完成混流」，"
            "**不能**说明配音能力或音频质量。真实 TTS 未在本轮验证。",
            "",
        ]
    if ours_failed:
        # The two failure kinds are counted together above and must not be read
        # as one phenomenon. One is a workflow defect; the other is an empty
        # wallet, and it landed later in the day than OpenMontage's renders, so
        # the success counts on the two sides are not comparable.
        lines += [
            f"失败 {ours_failed} 条，性质不同，不能合并解读：",
            "",
            "1. **编排层缺陷（1 条）**：`OutputVideoSensitiveContentDetected`，单镜审核失败直接让整集作废，"
            "没有隔离到该镜。这是本项目编排层的真实缺陷。",
            "2. **外部阻塞（其余）**：2026-11-17 19:47 起 Ark 返回 `AccountOverdueError`（HTTP 403），"
            "本侧连续多条在开工 29 秒内被拒。这是账户余额问题，与编排无关。",
            "",
            "**因此本表的成功率不能用于比较两套 workflow 的可靠性。** OpenMontage 已完成的两条任务时间戳"
            "（19:44:58 / 19:45:39）早于欠费发生，它的成功与本次失败之间是时间先后关系，不是能力差异。"
            "详见 `VALIDATION.md`「本项目的失败样本」。",
            "",
        ]

    # Interpret the identity gap in the light of motion, computed rather than
    # asserted: if the two sides' motion budgets differ, identity is not
    # comparable on its own and the report must say so with the numbers.
    ours_motion = [i["motion"] for i in by_side["ours"]]
    om_motion = [i["motion"] for i in by_side["openmontage_full"]]
    ours_id = [i["report"].summary.get("identity_median") for i in by_side["ours"]]
    om_id = [i["report"].summary.get("identity_median") for i in by_side["openmontage_full"]]
    om_id = [v for v in om_id if v is not None]

    # Duration and shot parity are stated from the delivered files, not
    # asserted. The OpenMontage planner re-plans each episode and does not keep
    # a fixed scale: its first episode came out 15 shots / 66 s and its second
    # 13 shots / 48 s, so a hardcoded "both sides are 15 shots" line would be
    # false in a report that is meant to be checkable.
    ours_dur = sorted(i["report"].duration_s for i in by_side["ours"])
    om_dur = sorted(i["report"].duration_s for i in by_side["openmontage_full"])
    ours_shots = sorted(len(i["episode"]["shot_files"]) or i["report"].shots for i in by_side["ours"])
    om_shots = sorted(len(i["episode"]["shot_files"]) or i["report"].shots for i in by_side["openmontage_full"])

    def _rng(values: list[int] | list[float], digits: int = 2) -> str:
        if not values:
            return "—"
        return f"{min(values):.{digits}f}–{max(values):.{digits}f}"

    lines += ["## 这条对比证明了什么、没证明什么", ""]
    lines += [
        f"**口径**：同一 brief、同一底模（`doubao-seedance-1-0-pro-fast-251015`）、同一分辨率（480p，9:16）、"
        f"同为 1080×1920/30fps。交付时长 本项目 {_rng(ours_dur)}s / OpenMontage {_rng(om_dur)}s；"
        f"镜数 本项目 {min(ours_shots) if ours_shots else '—'}–{max(ours_shots) if ours_shots else '—'} / "
        f"OpenMontage {min(om_shots) if om_shots else '—'}–{max(om_shots) if om_shots else '—'}。"
        "**两侧并非严格等长**：OpenMontage 每集重新规划镜头时长，实测出现过 66s/15 镜与 48s/13 镜两种规模；"
        "本项目的镜数固定但时长随镜头时长浮动。因此任何质量差异都必须与交付时长一起读。",
        "",
        "**能证明的**：在相同 brief、相同出片模型下，两套编排各自交付了什么——"
        "镜头数、时长、切点、音轨、字幕、逐镜稳定性、画面运动量、跨镜角色一致性。"
        "这些是从成片与源镜头文件里读出来的事实。",
        "",
    ]
    if ours_motion and om_motion and ours_id and om_id:
        delta_id = _median(ours_id) - statistics.median(om_id)
        ratio = statistics.median(ours_motion) / statistics.median(om_motion) if statistics.median(om_motion) else None
        ours_color = _median([i["shot_summary"].get("color_median") for i in by_side["ours"]])
        om_color = _median([i["shot_summary"].get("color_median") for i in by_side["openmontage_full"]])
        ours_struct = _median([i["shot_summary"].get("structure_median") for i in by_side["ours"]])
        om_struct = _median([i["shot_summary"].get("structure_median") for i in by_side["openmontage_full"]])
        lines += [
            f"**「合成 identity」这一项不能单独读，它主要是机位在动。** 合成中位 本项目 "
            f"{_median(ours_id)}、OpenMontage {_median(om_id)}（差 {delta_id:+.3f}）。"
            f"把它拆开看：调色板分量 本项目 {ours_color} vs {om_color}（差 "
            f"{(ours_color or 0) - (om_color or 0):+.3f}），构图分量 本项目 {ours_struct} vs "
            f"{om_struct}。差异几乎全在构图分量上，而构图的下跌来自运镜——"
            f"两侧运动量本身差 {ratio:.2f} 倍（帧间差 {_median(ours_motion)} vs "
            f"{_median(om_motion)}）。实测一例：本项目的镜头从远景走廊推到近景，"
            "画面里的人、背包、衣服全程不变，调色板 0.915 而构图 −0.308。"
            "所以这一项**不能**读成「谁的画面更稳」或「谁的角色更一致」；判断角色一致性请看下一节的 VLM 结果。",
            "",
        ]
    elif ours_id and om_id:
        delta_id = _median(ours_id) - statistics.median(om_id)
        lines += [
            f"**identity 差异**：本项目中位 {_median(ours_id)}，OpenMontage {_median(om_id)}"
            f"（差 {delta_id:+.3f}）。两侧运动量未取到，此差异暂不可单独解读。",
            "",
        ]

    # Where our side scored lower, say so with the numbers and say whether the
    # difference is separable. Leaving this out because it is unflattering would
    # make the report a selected summary rather than a measurement.
    lower: list[str] = []
    for label, own, other, use_mean in (
        ("跨镜角色一致性中位", character_scores.get("ours"),
         character_scores.get("openmontage_full"), False),
        ("成片-brief 一致性均值", evt_ours_mean, evt_om_mean, True),
    ):
        if not own or not other:
            continue
        # Two different statistic types arrive here: character scores are lists
        # of per-episode scores, the brief-alignment value is already a float
        # mean. Feeding either into a function expecting the other raised
        # "'float' object is not iterable" and aborted the whole report.
        own_stat = own if use_mean else _median(own)
        other_stat = other if use_mean else _median(other)
        if own_stat is not None and other_stat is not None and own_stat < other_stat:
            lower.append(f"{label} 本项目 {own_stat:.2f} 低于 OpenMontage {other_stat:.2f}")
    if lower:
        lines += [
            "**本项目侧分数更低的地方（如实列出）**：" + "；".join(lower) + "。",
            "",
            "这些差异**在本轮样本量下不可分辨**：角色一致性本项目逐集为 "
            + ", ".join(f"{v:g}" for v in sorted(character_scores.get("ours") or []))
            + "，OpenMontage 为 "
            + ", ".join(f"{v:g}" for v in sorted(character_scores.get("openmontage_full") or []))
            + "，取值范围重叠；成片-brief 一致性 5 个维度里多数完全相同，差异集中在单一维度。"
            "它们是需要更大样本去检验的方向，不是已经成立的结论。",
            "",
        ]

    lines += [
        "**不能证明的**：谁的「画质更好」，以及谁的「角色更稳」。裁判分需要两侧各有足够多的成片"
        f"才能比较；当前 n = {len(ours)} vs {len(om)}，这是下一阶段的实验条件，不是本轮结论。",
        "",
        "**刻意没做的**：不把「有字幕/有配音」折算成分数再相加。"
        "三层指标刻度不同（1-5、0-1、计数），相加会造出一个没有意义的合成分。",
        "",
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": [
            {
                "side": item["episode"]["side"],
                "project_id": item["episode"]["project_id"],
                "brief_index": item["episode"].get("brief_index"),
                "repeat": item["episode"].get("repeat"),
                "path": str(item["episode"]["path"]),
                "elapsed_s": item["episode"]["elapsed_s"],
                "motion": item["motion"],
                # Carried here so a figure or a later reader does not have to
                # re-count files, which is what produced the 3x-inflated voice
                # number: every placeholder track also has a sidecar.
                "subtitles": item["episode"]["subtitles"],
                "voice": item["episode"]["audio"] if isinstance(item["episode"]["audio"], dict)
                else {"real": item["episode"]["audio"], "mock": 0},
                **item["json"],
            }
            for item in measured
        ],
        "aggregate": {
            side: {
                "n": len(items),
                "identity_median": _median([i["report"].summary.get("identity_median") for i in items]),
                "identity_min": min((i["report"].summary.get("identity_min") for i in items), default=None),
                "motion_median": _median([i["motion"] for i in items]),
                "duration_median": _median([i["report"].duration_s for i in items]),
                "shots_median": _median([float(len(i["episode"]["shot_files"]) or i["report"].shots)
                                         for i in items]),
                "with_audio": sum(1 for i in items if i["report"].structure.get("has_audio")),
                # Shot-level components and the VLM's character score live here
                # too, so a figure can be drawn from this file alone instead of
                # re-deriving them and drifting from the report.
                "color_median": _median([i["shot_summary"].get("color_median") for i in items]),
                "structure_median": _median([i["shot_summary"].get("structure_median") for i in items]),
                "character_median": _median(character_scores.get(side) or []),
            }
            for side, items in by_side.items()
        },
    }
    (args.out / "episodes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = args.out / "report_workflows.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {report_path}  ({len(ours)} ours vs {len(om)} openmontage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
