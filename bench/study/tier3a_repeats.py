"""Repeated-measure runs of the cross-shot comparison.

Why this exists: the first Tier 3a pass measured each story once, so its
"2.00 -> 4.17" was six paired observations — enough for a sign test, not enough
to show a distribution. A report that quotes a mean without a spread is quoting
the only number it happened to see. This script re-runs the whole comparison R
times so every metric has a sample behind it.

Design, stated because it bounds what the numbers mean:

* One repeat is a full re-execution: a fresh planner call for the planned arm
  (the planner is not seeded, so its prompts genuinely vary across repeats) and
  a fresh seed band for both arms.
* Both arms share the SAME seeds within a repeat (``seed = repeat*100 + shot``),
  so a pair differs only in the prompt string it was given.
* The independent arm's prompt is the brief verbatim, identical in every repeat.
  Its spread across repeats is therefore render noise alone, which is what makes
  it a control rather than a second treatment.
* The planner's output is cached per (brief, repeat) and shared between models:
  it is a text artifact, and the same story is told with the same prompt on
  either checkpoint. Planning is the expensive part per cell (~3-4 min against
  ~6-11 min of rendering for ten clips) and duplicating it would buy nothing.

Repeat 0 uses seeds 0-4, the same band the earlier single-pass run used.

Usage:
    python bench/study/tier3a_repeats.py --models cogvideox5b wan21_1_3b --repeats 3
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/study"))

from tier3a_story import render, story_prompts  # noqa: E402

OUT = ROOT / "bench/results/tier3a_repeats"
BRIEFS = ROOT / "bench/study/briefs.json"
SHOTS = 5


def plan_path(brief_index: int, repeat: int) -> Path:
    return OUT / "plans" / f"b{brief_index}_r{repeat}.json"


def get_plan(brief: str, brief_index: int, repeat: int, shots: int) -> dict:
    """The planner's prompts for one (brief, repeat), cached on disk."""
    path = plan_path(brief_index, repeat)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    started = time.monotonic()
    prompts = [p for p in story_prompts(brief, "cogvideox5b", repeat, shots=shots) if p][:shots]
    row = {"brief_index": brief_index, "repeat": repeat, "prompts": prompts,
           "plan_seconds": round(time.monotonic() - started, 1)}
    if len(prompts) >= 2:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def seed_for(repeat: int, shot: int) -> int:
    """A disjoint seed band per repeat, shared by both arms."""
    return repeat * 100 + shot


def load_done(log: Path) -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["arm"], row["brief_index"], row["repeat"]))
    return done


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Repeated cross-shot runs")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--briefs", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shots", type=int, default=SHOTS)
    args = parser.parse_args(argv[1:])

    briefs = json.loads(BRIEFS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        log = OUT / f"clips_{model}.jsonl"
        done = load_done(log)

        for brief_index in args.briefs:
            brief = briefs[brief_index]
            for repeat in range(args.repeats):
                plan = get_plan(brief, brief_index, repeat, args.shots)
                prompts = plan["prompts"]
                print(f"[{model} b{brief_index} r{repeat}] plan {plan['plan_seconds']}s "
                      f"-> {len(prompts)} prompts", flush=True)
                if len(prompts) < 2:
                    print("    planner returned too few prompts; skipping cell", flush=True)
                    continue

                for arm in ("independent", "planned"):
                    if (arm, brief_index, repeat) in done:
                        print(f"    {arm}: already recorded", flush=True)
                        continue
                    texts = [brief] * len(prompts) if arm == "independent" else prompts
                    paths: list[str] = []
                    seconds: list[float] = []
                    for index, text in enumerate(texts):
                        started = time.monotonic()
                        path = render(text, seed_for(repeat, index), model,
                                      f"{arm}_{model}_b{brief_index}_r{repeat}", OUT, index)
                        seconds.append(round(time.monotonic() - started, 1))
                        if path:
                            paths.append(str(path))
                        print(f"    {arm} shot {index}: {seconds[-1]}s", flush=True)

                    row = {
                        "arm": arm, "model": model, "brief_index": brief_index,
                        "repeat": repeat, "shots": paths,
                        "prompt_chars": [len(t) for t in texts],
                        "seeds": [seed_for(repeat, i) for i in range(len(texts))],
                        "clip_seconds": seconds,
                        "plan_seconds": plan["plan_seconds"],
                    }
                    with log.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done.add((arm, brief_index, repeat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
