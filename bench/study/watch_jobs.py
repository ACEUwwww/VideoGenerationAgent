#!/usr/bin/env python3
"""Print the study's job status, one line per line of output, for a watcher.

Run it in a loop and each invocation prints exactly one summary line, so a
monitor emits a notification per poll instead of a wall of text. It exits
non-zero when every job has stopped, which is what lets the watcher end.

The failure this exists to catch is real: the judge process died mid-run once
and looked healthy for minutes because its log had merely stopped growing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
STUDY = ROOT / "bench/results/study"
JUDGE_LOG = ROOT / "bench/logs/study_judge.log"
OURS_ROWS = ROOT / "bench/results/ours_full/episodes.jsonl"
OM_STATE = ROOT / "bench/_vendor/OpenMontage/bench_run/projects"
DIMENSIONS = ("imaging_quality", "aesthetic_quality", "temporal_consistency", "motion_effects")


def alive(pattern: str) -> bool:
    return bool(subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.strip())


def judge_groups() -> int:
    if not JUDGE_LOG.exists():
        return 0
    return sum(1 for line in JUDGE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
               if "seed=" in line and "{" in line)


def done_dimensions() -> list[str]:
    finished = []
    for dimension in DIMENSIONS:
        path = STUDY / "judge" / f"{dimension}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not payload.get("partial"):
            finished.append(dimension)
    return finished


def ours_finished() -> int:
    if not OURS_ROWS.exists():
        return 0
    return sum(1 for line in OURS_ROWS.read_text(encoding="utf-8").splitlines() if line.strip())


def current_episode_clips() -> int:
    """Clips rendered in the newest project only.

    Scoped to the newest project because a finished episode keeps its clips: an
    earlier version counted the newest directory's videos without noticing the
    run had moved on, and printed "30/15" once two episodes were done.
    """
    projects = sorted((ROOT / "workspace/projects").glob("project_*"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not projects:
        return 0
    return len(list((projects[0] / "videos").glob("*.mp4")))


def om_progress() -> tuple[str, int, int]:
    dirs = [d for d in OM_STATE.glob("benchs*") if (d / "studio_state.json").exists()]
    if not dirs:
        return "", 0, 0
    newest = max(dirs, key=lambda d: (d / "studio_state.json").stat().st_mtime)
    try:
        state = json.loads((newest / "studio_state.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return newest.name, 0, 0
    shots = [s for s in (state.get("shots") or []) if not s.get("deletedAt")]
    counts = Counter(str(s.get("status")) for s in shots)
    return newest.name, counts.get("success", 0), len(shots)


def main() -> int:
    judge_alive = alive("study/judge_study.py")
    ours_alive = alive("study/run_ours_study.py")
    om_alive = alive("study/om_full.py")

    groups = judge_groups()
    dims = done_dimensions()
    finished = ours_finished()
    clips = current_episode_clips()
    om_project, om_done, om_total = om_progress()

    parts = [
        f"judge={'up' if judge_alive else 'DOWN'} {groups}/240 dims={len(dims)}/4",
        f"ours={'up' if ours_alive else 'DOWN'} {finished}/9 episodes (current {clips}/15 clips)",
        f"om={'up' if om_alive else 'DOWN'} {om_project or '-'} {om_done}/{om_total} shots",
    ]
    if dims:
        parts.append("done=" + ",".join(dims))
    print(" | ".join(parts), flush=True)
    # Non-zero exit ends the watcher loop: everything has stopped.
    return 0 if (judge_alive or ours_alive or om_alive) else 1


if __name__ == "__main__":
    sys.exit(main())
