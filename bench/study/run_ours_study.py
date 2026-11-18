"""Render our workflow's episodes for the paired comparison.

One brief × one repeat = one episode, matched to OpenMontage's delivered scale
(15 shots / ~60 s) so the two sides are compared at the same length rather than
at whatever each happens to produce.

A note on "seed" that the report has to carry: Seedance has no seed parameter in
either workflow's request, so a repeat here is a *fresh sample* of the whole
orchestration — the LLM writes a new script, the image model redraws the
keyframes, Seedance renders new clips. That is variation of the pipeline, not a
controlled diffusion seed, and it is the same kind of variation OpenMontage
exhibits. Calling it a seed without saying so would overstate the design.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

BRIEFS = ROOT / "bench/study/briefs.json"
OUT = ROOT / "bench/results/ours_full"
SHOTS = 15
TARGET_SECONDS = 60


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render our workflow's study episodes")
    parser.add_argument("--briefs", type=Path, default=BRIEFS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-start", type=int, default=0,
                        help="first repeat index; skip ones already rendered")
    parser.add_argument("--brief-index", type=int, default=-1, help="one brief only (-1 = all)")
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--target-seconds", type=int, default=TARGET_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv[1:])

    briefs = json.loads(args.briefs.read_text(encoding="utf-8"))
    # Keep the original indices before slicing. Slicing to one brief and then
    # enumerating renumbers it from 0, so `--brief-index 1` wrote brief_index=0
    # into the results — two different briefs would have shared one id in the
    # report and paired against the wrong OpenMontage episodes.
    indexed = list(enumerate(briefs))
    if args.brief_index >= 0:
        indexed = [indexed[args.brief_index]]
    plan = [(i, brief, rep) for i, brief in indexed
            for rep in range(args.repeat_start, args.repeats)]
    print(f"{len(indexed)} briefs x repeats {args.repeat_start}..{args.repeats - 1} = {len(plan)} episodes "
          f"({args.shots} shots / {args.target_seconds}s each)")
    if args.dry_run:
        for i, brief, rep in plan:
            print(f"  brief[{i}] rep={rep} {brief[:50]}")
        return 0

    os.environ.update(
        VIDEO_PROVIDER="ark", IMAGE_PROVIDER="ark", TTS_PROVIDER="mock",
        ARK_ALLOW_PAID="1",
    )
    # Load the project .env before printing the media config: agent.settings
    # reads it at import, and printing first reported ARK_VIDEO_MODEL as "?"
    # for a run that was in fact using the pinned model.
    from agent.settings import reset_settings
    reset_settings()

    print("env: VIDEO_PROVIDER=%s IMAGE_PROVIDER=%s ARK_VIDEO_MODEL=%s ARK_VIDEO_RESOLUTION=%s"
          % (os.environ["VIDEO_PROVIDER"], os.environ["IMAGE_PROVIDER"],
             os.environ.get("ARK_VIDEO_MODEL", "?"), os.environ.get("ARK_VIDEO_RESOLUTION", "?")))

    from agent.graph import run_agent
    from backend.services import project_store

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / "episodes.jsonl"
    # Plan first, results after. The report needs the denominator — how many
    # episodes were asked for — and cannot get it from a row count, which only
    # says how far the batch has got.
    (OUT / "plan.json").write_text(json.dumps(
        {"briefs": [b[:60] for b in briefs], "repeats": args.repeats,
         "episodes": [{"brief_index": i, "repeat": r} for i, _, r in plan]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    for index, (brief_index, brief, repeat) in enumerate(plan, start=1):
        print(f"\n[{index}/{len(plan)}] brief[{brief_index}] repeat={repeat} {brief[:44]!r}", flush=True)
        started = time.monotonic()
        row: dict = {
            "workflow": "ours", "brief_index": brief_index, "brief": brief, "repeat": repeat,
            "shots_requested": args.shots, "target_seconds": args.target_seconds,
            "seed_semantics": "fresh sample of the orchestration (no diffusion seed in Seedance)",
        }
        try:
            project = project_store.create_project(
                brief, "电影感竖屏短剧", args.shots, genre="悬疑",
                target_seconds=args.target_seconds,
            )
            state = project_store.load_state(project["project_id"])
            state["status"] = "running"
            final = run_agent(state)
            root = project_store.project_dir(project["project_id"])
            row.update(
                project_id=project["project_id"],
                status=final["status"],
                final_video=str(root / final["final_video_path"]) if final.get("final_video_path") else "",
                episode_score=(final.get("episode_report") or {}).get("score"),
                shot_count=len(final.get("shots") or []),
                durations=[s.get("duration_s") for s in (final.get("shots") or [])],
            )
        except Exception as exc:  # noqa: BLE001 - one episode must not sink the batch
            row.update(status="failed", error=str(exc)[:400])
        row["elapsed_s"] = round(time.monotonic() - started, 1)
        print(f"  -> status={row.get('status')} shots={row.get('shot_count')} "
              f"{row['elapsed_s']:.0f}s err={str(row.get('error', ''))[:100]!r}", flush=True)
        with out_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
