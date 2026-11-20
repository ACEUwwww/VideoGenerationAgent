"""Our side of the matched comparison, on exactly the brief OpenMontage ran.

The control arm (project ``omfix``) was driven with a specific brief. Comparing
its score against Tier 3a's three study briefs would compare different stories,
which measures nothing. This script drives our orchestration on that same brief
and renders the planner's prompts with the same checkpoint class, so the only
difference left is orchestration.

Scope, stated because it bounds the claim: OpenMontage produced 12 clips for its
episode; this run produces ``--shots`` (5 by default) because each local clip is
~35 s of GPU and the point here is the cross-shot metric, which is a relation
between frames rather than a function of shot count. The count difference is
recorded in the report next to the two numbers.

Reference-image mode is not used. The vendor's reference path cannot run on
local weights (0 of 5 characters cleared its entity gate), and our Tier 3a
layer is text-to-video, so both sides are text-to-video here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/study"))

BRIEF = "深夜的便利店，一个外卖骑手发现收银员不是人。"
OUT = ROOT / "bench/results/om_control/ours_matched"


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cogvideox5b")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--tag", default="ours")
    args = parser.parse_args(argv[1:])

    import tier3a_story as story

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"driving our pipeline on the matched brief ({args.model}, {args.shots} shots)", flush=True)
    started = time.monotonic()
    prompts = story.story_prompts(BRIEF, args.model, 0, shots=args.shots)
    prompts = [p for p in prompts if p][: args.shots]
    print(f"  planner produced {len(prompts)} prompts in {time.monotonic()-started:.0f}s", flush=True)
    if len(prompts) < 2:
        print("  planner returned too few prompts", flush=True)
        return 1

    paths: list[str] = []
    for index, text in enumerate(prompts):
        t0 = time.monotonic()
        path = story.render(text, index, args.model, args.tag, OUT, index)
        if path:
            paths.append(str(path))
        print(f"  shot {index}: {time.monotonic()-t0:.0f}s", flush=True)

    row = {
        "arm": "planned", "model": args.model, "brief": BRIEF,
        "shots": paths, "prompt_chars": [len(t) for t in prompts],
    }
    (OUT / "clips.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(paths)} clips -> {OUT / 'clips.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
