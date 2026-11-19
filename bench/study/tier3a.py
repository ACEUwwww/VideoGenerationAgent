"""Tier 3a: both sides at the shot stage, on the same open-weight model.

Scope decision, recorded because it bounds every claim this layer can make.
OpenMontage's full studio needs Postgres, Redis and MinIO, and its dispatch
layer does not route to the local video tools at all — reaching its full
workflow would mean editing the vendored tree, which this round forbids. So
both sides are compared at the *shot* stage: one brief in, one five-second clip
out.

That makes the measured difference small and specific, and the report must say
so:

* Our side keeps its planning layer. The brief goes through script and shot
  planning, and the motion prompt handed to the video model is written by the
  LLM from that plan. This is our contribution at this stage.
* OpenMontage's shot stage takes the brief verbatim.

So the variable is "does writing a shot prompt from a story plan help the clip",
not "which workflow produces a better film". Anything about editing, pacing,
multi-shot coherence or cast consistency is *not* measured here.

Cost: the video models and the judge are local. The only billed item is the
LLM call our planner makes, measured at about 0.086 CNY per episode.
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
OUT = ROOT / "bench/results/tier3a"

# ~5 s per shot is what every local model renders, so the episode length is a
# consequence of the model, not a choice. Recorded so the report can state it
# rather than let a reader assume 60 s.
SHOT_SECONDS = 5.0
# One shot per side. Two would compare stitching as well, and stitching on a
# fixed-length renderer means gluing identical-length clips — a seam exactly
# where a stability metric would be measuring.
SHOTS = 1


def episode_plan(briefs: list[str], repeats: int, brief_index: int) -> list[tuple[int, str, int]]:
    indexed = list(enumerate(briefs))
    if brief_index >= 0:
        indexed = [indexed[brief_index]]
    return [(i, brief, rep) for i, brief in indexed for rep in range(repeats)]


def render_ours(brief: str, brief_index: int, repeat: int, model: str) -> dict:
    """One episode through our orchestration on a local text-to-video model."""
    from agent.settings import reset_settings
    from agent.graph import run_agent
    from backend.services import project_store

    from mcp_servers.providers.local_media import LOCAL_HEIGHT, LOCAL_WIDTH, MODELS, _frames_for

    spec = MODELS[model]
    fps = spec["fps"]
    frames = _frames_for(spec.get("frame_rule", "4k+1"), fps, SHOT_SECONDS)

    os.environ.update(
        VIDEO_PROVIDER="local", VIDEO_MODEL=model,
        # No keyframe stage runs at all under t2v, so the image provider is only
        # ever consulted by code paths that are skipped. It stays mock so that
        # an accidental image call costs nothing.
        IMAGE_PROVIDER="mock", TTS_PROVIDER="mock",
        # Belt and braces: this layer must never reach a paid API.
        ARK_ALLOW_PAID="0",
        # The canvas must be the model's own output. Left at the project
        # default (1080x1920 portrait) the composer takes a 832x480 landscape
        # render, scales it 4x to 3328x1920 and crops 68% of the width away.
        # The judge then scores that crop, not the picture the model drew, and
        # the whole comparison measures the composer instead of the prompt.
        VIDEO_WIDTH=str(LOCAL_WIDTH), VIDEO_HEIGHT=str(LOCAL_HEIGHT), VIDEO_FPS=str(fps),
    )
    reset_settings()

    project = project_store.create_project(
        brief, "电影感短剧", SHOTS, genre="悬疑",
        target_seconds=int(SHOTS * SHOT_SECONDS),
    )
    state = project_store.load_state(project["project_id"])
    state["status"] = "running"
    final = run_agent(state)
    root = project_store.project_dir(project["project_id"])
    return {
        "side": "ours",
        "model": model,
        "brief_index": brief_index,
        "repeat": repeat,
        "project_id": project["project_id"],
        "status": final["status"],
        "final_video": str(root / final["final_video_path"]) if final.get("final_video_path") else "",
        "shot_count": len(final.get("shots") or []),
        "durations": [s.get("duration_s") for s in (final.get("shots") or [])],
    }


def render_openmontage(brief: str, brief_index: int, repeat: int, model: str) -> dict:
    """The same brief through OpenMontage's local shot stage.

    Its shot stage takes a prompt per clip, so the brief is used as that prompt
    verbatim — the same input our planner starts from. Anything the workflow
    adds on top of the bare brief (script, storyboard, prompt expansion) is
    exactly the variable this layer measures, so it must not be supplied here.
    """
    from bench.backends import get_backend
    from bench.config import BenchConfig

    backend = get_backend("openmontage_local", BenchConfig())
    backend.model = model
    out_dir = OUT / "clips" / f"om_{model}_b{brief_index}_r{repeat}"
    generation = backend.generate(brief, seed=repeat, seconds=SHOT_SECONDS, out_dir=out_dir)
    return {
        "side": "openmontage_full",
        "model": model,
        "brief_index": brief_index,
        "repeat": repeat,
        "project_id": out_dir.name,
        "status": "completed",
        "final_video": str(generation.path),
        "shot_count": 1,
        "durations": [generation.seconds],
        "elapsed_s": round(generation.elapsed_s, 1),
    }


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tier 3a: both workflows, same local model")
    parser.add_argument("--models", nargs="+", default=["cogvideox5b", "wan21_1_3b", "longlive_1_3b"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--brief-index", type=int, default=1,
                        help="brief 1 only by default: it is the one both sides rendered in Tier 1, "
                             "so this layer stays comparable to the paid one")
    parser.add_argument("--shot-seconds", type=float, default=SHOT_SECONDS)
    parser.add_argument("--shots", type=int, default=SHOTS,
                        help="shots per episode; this layer is shot-stage only, so 1")
    parser.add_argument("--side", choices=["ours", "openmontage", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv[1:])

    globals()["SHOT_SECONDS"] = args.shot_seconds
    globals()["SHOTS"] = args.shots

    briefs = json.loads(BRIEFS.read_text(encoding="utf-8"))
    plan = episode_plan(briefs, args.repeats, args.brief_index)
    total = len(plan) * len(args.models) * (1 if args.side != "both" else 2)
    print(f"{len(plan)} brief/repeat pairs x {len(args.models)} models x "
          f"{args.side} = {total} renders ({args.shots} shots / {args.shot_seconds}s)")
    if args.dry_run:
        for model in args.models:
            for index, brief, rep in plan:
                print(f"  {model:15s} brief[{index}] rep={rep} {brief[:40]}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    # The denominator, written before the first render: a row count only says
    # how far a batch got, never how far it was meant to go.
    (OUT / "plan.json").write_text(json.dumps({
        "repeats": args.repeats, "models": args.models, "shots": args.shots,
        "shot_seconds": args.shot_seconds, "side": args.side,
        "episodes": [{"model": m, "brief_index": i, "repeat": r}
                     for m in args.models for i, _, r in plan],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    results = OUT / "episodes.jsonl"
    done: set[tuple[str, str, int, int]] = set()
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("status") == "completed":
                done.add((row["side"], row["model"], row["brief_index"], row["repeat"]))

    for model in args.models:
        for brief_index, brief, repeat in plan:
            for side, fn in (("ours", render_ours), ("openmontage_full", render_openmontage)):
                if args.side != "both" and not (
                    (args.side == "ours" and side == "ours")
                    or (args.side == "openmontage" and side != "ours")
                ):
                    continue
                key = (side, model, brief_index, repeat)
                if key in done:
                    print(f"  reuse {side} {model} b{brief_index} r{repeat}", flush=True)
                    continue
                print(f"\n[{side}] {model} brief[{brief_index}] rep={repeat}", flush=True)
                started = time.monotonic()
                try:
                    if side == "ours":
                        row = fn(brief, brief_index, repeat, model)
                    else:
                        row = fn(brief, brief_index, repeat, model)
                except Exception as exc:  # noqa: BLE001 - one render must not sink the batch
                    row = {"side": side, "model": model, "brief_index": brief_index,
                           "repeat": repeat, "status": "failed", "error": str(exc)[:400]}
                row.setdefault("elapsed_s", round(time.monotonic() - started, 1))
                print(f"  -> {row.get('status')} {row.get('elapsed_s')}s "
                      f"{str(row.get('error', ''))[:110]!r}", flush=True)
                with results.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
