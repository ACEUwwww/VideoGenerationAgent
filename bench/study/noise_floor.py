"""How much does the cross-shot consistency score move when nothing changes?

Every comparison in this repo quotes a 1-5 score from one vision-language judge
over one frame per shot. Before any difference between arms can be called an
effect, the spread of that score under a *fixed* prompt has to be known — the
score is the judge's opinion of five sampled frames, and resampling the render
changes both the frames and the opinion.

The repeated-measure study is the source of that number. Its ``independent`` arm
gives the brief to the model verbatim in every repeat, so across repeats the
prompt string is identical and everything else — seed, sampled frames, render
noise, judge sampling — varies. The spread across repeats within one
(model, brief) cell is therefore the noise floor of the metric itself.

This script reports it. It does not add a new row to any result table; it tells
a reader how large a difference has to be before it means anything.

Usage:
    python bench/study/noise_floor.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "bench/results/tier3a_repeats"
MODELS = ("cogvideox5b", "wan21_1_3b")
BRIEFS = 3


def cells(arm: str) -> dict[tuple[str, int], list[float]]:
    """Scores per (model, brief) for one arm, across repeats."""
    out: dict[tuple[str, int], list[float]] = {}
    for model in MODELS:
        path = REP / f"consistency_{model}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("arm") != arm or row.get("score") is None:
                continue
            out.setdefault((model, row["brief_index"]), []).append(float(row["score"]))
    return out


def report(arm: str) -> dict:
    data = cells(arm)
    ranges, spreads = [], []
    print(f"\n{arm}: 5-point score across repeats, identical prompt per cell")
    for (model, brief), values in sorted(data.items()):
        if len(values) < 2:
            continue
        rng = max(values) - min(values)
        sd = statistics.stdev(values)
        ranges.append(rng)
        spreads.append(sd)
        print(f"  {model:<12} b{brief}  {values}  range={rng:.1f} sd={sd:.2f}")
    if not ranges:
        return {}
    return {
        "cells": len(ranges),
        "mean_range": round(statistics.mean(ranges), 2),
        "max_range": round(max(ranges), 2),
        "mean_sd": round(statistics.mean(spreads), 2),
    }


def main() -> int:
    independent = report("independent")
    planned = report("planned")

    print("\nnoise floor (independent arm — same prompt every repeat)")
    print(f"  within-cell range: mean={independent.get('mean_range')} "
          f"max={independent.get('max_range')} over {independent.get('cells')} cells")
    print(f"  within-cell sd:    mean={independent.get('mean_sd')}")
    print("\nplanned arm (prompt is re-written each repeat, so this is signal + noise)")
    print(f"  within-cell range: mean={planned.get('mean_range')} "
          f"max={planned.get('max_range')} over {planned.get('cells')} cells")

    floor = independent.get("mean_range")
    if floor:
        print(
            f"\nRead this before quoting any arm difference:\n"
            f"  the judge moves about {floor:.1f} points (and up to "
            f"{independent.get('max_range')} points) on the SAME prompt.\n"
            f"  A difference smaller than that is not evidence of an effect, and a\n"
            f"  single repeat per cell cannot separate the two."
        )

    out = ROOT / "bench/results/noise_floor.json"
    out.write_text(json.dumps({"independent": independent, "planned": planned}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
