"""Do the delivered episodes match what the brief asked for?

Different question from the clip bench. The local baseline scores five-second
clips against their generation prompt; this scores a finished multi-shot episode
against the *creative brief* the workflow was given. That is the workflow-level
claim: given one line of intent, did the orchestration deliver a film that is
about that intent.

Caveat this script states in its output rather than hiding: upstream's
video-text dimensions were designed for a single generated clip and a visual
prompt. A 60 s, 15-shot episode against a one-line brief is a harder and
looser match — a score of 2 here does not mean the clip is bad, it means the
brief is abstract enough that "consistency" is doing less work than usual. The
number is still useful *comparatively*: both sides got the same brief through
the same judge, so the difference between them is meaningful even if the
absolute level is not.

Free: local Qwen2.5-VL on 127.0.0.1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

from bench.frames import build_payload  # noqa: E402
from bench.study.judge_study import local_judge  # noqa: E402

DIMS = ["overall_consistency", "color", "object_class", "scene", "action"]
OUT = ROOT / "bench/results/compare_workflows/episode_video_text.json"
WORKFLOWS = ROOT / "bench/results/compare_workflows/episodes.json"
BRIEFS = ROOT / "bench/study/briefs.json"


def pairs() -> list[dict]:
    """One entry per brief, with our episode and the OpenMontage one side by side."""
    payload = json.loads(WORKFLOWS.read_text(encoding="utf-8"))
    briefs = json.loads(BRIEFS.read_text(encoding="utf-8"))
    by_brief: dict[int, dict] = {}
    for episode in payload.get("episodes") or []:
        index = episode.get("brief_index")
        if index is None:
            continue
        by_brief.setdefault(index, {})[episode["side"]] = episode
    out = []
    for index, sides in sorted(by_brief.items()):
        if "ours" not in sides or "openmontage_full" not in sides:
            continue
        out.append({"brief_index": index, "brief": briefs[index], "sides": sides})
    return out


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score delivered episodes against their brief")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv[1:])

    groups = pairs()
    if args.limit:
        groups = groups[: args.limit]
    if not groups:
        raise SystemExit("no brief with both sides present; run compare_workflows.py first")
    print(f"{len(groups)} brief(s) with both sides", flush=True)

    judge = local_judge()
    results = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else []

    for group in groups:
        brief = group["brief"]
        for dimension in DIMS:
            already = next((r for r in results
                            if r["brief_index"] == group["brief_index"] and r["dimension"] == dimension), None)
            if already:
                print(f"  reuse brief{group['brief_index']} {dimension}", flush=True)
                continue
            payloads = {side: build_payload(Path(ep["path"]), judge.config)
                        for side, ep in sorted(group["sides"].items())}
            started = time.monotonic()
            try:
                reply = judge.score(dimension, payloads, brief)
            except Exception as exc:  # noqa: BLE001 - one dimension must not sink the run
                print(f"  brief{group['brief_index']} {dimension} FAILED: {str(exc)[:120]}", flush=True)
                reply = None
            row = {
                "brief_index": group["brief_index"],
                "brief": brief[:80],
                "dimension": dimension,
                "scores": dict(reply.scores) if reply else {},
                "error": (reply.error if reply else "exception")[:300],
                "elapsed_s": round(time.monotonic() - started, 1),
            }
            results = [r for r in results
                       if not (r["brief_index"] == row["brief_index"] and r["dimension"] == dimension)]
            results.append(row)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  brief{group['brief_index']} {dimension:20s} {row['scores']}", flush=True)

    print(f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
