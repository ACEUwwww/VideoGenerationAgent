"""Every number the technical report quotes, computed from the result files.

One script, one output file. The report should not contain a figure a reader
cannot recompute, so this writes ``summary.json`` next to the raw scores and
prints a human-readable digest.

Two experiments feed it:

  A  cross-shot character consistency, repeated R times per (model, brief).
     dispersion, paired difference, sign test, exact sign-flip permutation,
     bootstrap CI, and the judge's named trait.
  B  single-clip quality on four dimensions. Already has five seeds per
     (arm, model) cell, so it was a sample from the start and is reported as
     one — mean, sd, and the per-seed paired difference against the brief.

Usage:
    python bench/study/repeats_analysis.py
"""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
REP = ROOT / "bench/results/tier3a_repeats"
AB = ROOT / "bench/results/tier3a_ab/judge"
OUT = REP / "summary.json"

MODELS = ["cogvideox5b", "wan21_1_3b"]
DIMS = ["imaging_quality", "aesthetic_quality", "temporal_consistency", "motion_effects"]


# --------------------------------------------------------------------- helpers
def sign_test(diffs: list[float]) -> dict:
    """Exact binomial sign test, both tails.

    ``p_greater`` tests H1: the treatment is higher than the baseline;
    ``p_less`` tests H1: it is lower. Which one is reported depends on the
    direction the experiment actually hypothesised — quoting the wrong tail
    turns a clear effect into p ≈ 0.99, which is the mistake this replaced.
    Ties are excluded from n, as is standard.
    """
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties = len(diffs) - wins - losses
    n = wins + losses
    if n == 0:
        return {"wins": wins, "ties": ties, "losses": losses, "n": 0,
                "p_greater": 1.0, "p_less": 1.0, "p_two_sided": 1.0}
    p_greater = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    p_less = sum(math.comb(n, k) for k in range(0, wins + 1)) / (2 ** n)
    return {
        "wins": wins, "ties": ties, "losses": losses, "n": n,
        "p_greater": round(p_greater, 8),
        "p_less": round(p_less, 8),
        "p_two_sided": round(min(1.0, 2 * min(p_greater, p_less)), 8),
    }


def sign_flip(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation test on the mean difference."""
    n = len(diffs)
    if n == 0 or n > 20:
        return float("nan")
    observed = abs(statistics.mean(diffs))
    hits = sum(
        1 for signs in itertools.product((1, -1), repeat=n)
        if abs(statistics.mean([s * d for s, d in zip(signs, diffs)])) >= observed - 1e-12
    )
    return round(hits / (2 ** n), 6)


def bootstrap(diffs: list[float], trials: int = 5000, seed: int = 20261120) -> list[float]:
    if not diffs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(trials))
    return [round(means[int(0.025 * trials)], 3), round(means[int(0.975 * trials)], 3)]


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "histogram": {str(k): values.count(k) for k in range(1, 6)},
    }


def wilson(wins: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson interval for a win proportion — stable at small n."""
    if n == 0:
        return [0.0, 1.0]
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


# ----------------------------------------------------------------- experiment A
def experiment_a() -> dict:
    per_model: dict[str, dict] = {}
    all_diffs: list[float] = []

    for model in MODELS:
        path = REP / f"consistency_{model}.json"
        if not path.exists():
            continue
        rows = [r for r in json.loads(path.read_text(encoding="utf-8")) if r.get("score")]
        if not rows:
            continue

        arms = {}
        for arm in ("independent", "planned"):
            values = [r["score"] for r in rows if r["arm"] == arm]
            arms[arm] = describe(values)

        cells: dict[tuple[int, int], dict[str, float]] = {}
        for row in rows:
            cells.setdefault((row["brief_index"], row["repeat"]), {})[row["arm"]] = row["score"]
        diffs = [c["planned"] - c["independent"] for c in cells.values()
                 if "planned" in c and "independent" in c]
        all_diffs.extend(diffs)

        traits: dict[str, int] = {}
        for row in rows:
            if row["arm"] == "planned":
                traits[row.get("changed") or "NONE"] = traits.get(row.get("changed") or "NONE", 0) + 1

        clip_s = [s for r in rows for s in (r.get("clip_seconds") or [])]
        plan_s = sorted({r["plan_seconds"] for r in rows if r.get("plan_seconds")})
        per_model[model] = {
            "arms": arms,
            "pairs": {
                "n": len(diffs),
                "mean_diff": round(statistics.mean(diffs), 3) if diffs else 0.0,
                "sd_diff": round(statistics.stdev(diffs), 3) if len(diffs) > 1 else 0.0,
                "ci95": bootstrap(diffs),
                "sign_test": sign_test(diffs),
                "sign_flip_p": sign_flip(diffs),
                "wilson_win_rate": wilson(sum(1 for d in diffs if d > 0), len(diffs)),
                "mean_of_pair_means": round(
                    statistics.mean([(c["planned"] + c["independent"]) / 2 for c in cells.values()
                                     if "planned" in c and "independent" in c]), 3) if cells else 0.0,
            },
            "planned_traits": traits,
            "render_seconds": describe([round(s, 1) for s in clip_s]),
            "plan_seconds": plan_s,
        }

    pooled = {"pairs": {}}
    if all_diffs:
        pooled["pairs"] = {
            "n": len(all_diffs),
            "mean_diff": round(statistics.mean(all_diffs), 3),
            "sd_diff": round(statistics.stdev(all_diffs), 3) if len(all_diffs) > 1 else 0.0,
            "ci95": bootstrap(all_diffs),
            "sign_test": sign_test(all_diffs),
            "sign_flip_p": sign_flip(all_diffs),
            "wilson_win_rate": wilson(sum(1 for d in all_diffs if d > 0), len(all_diffs)),
        }
    per_model["pooled"] = pooled
    return per_model


# ----------------------------------------------------------------- experiment B
def experiment_b() -> dict:
    """Per-dimension distributions, and the per-seed paired difference."""
    arms = ("brief", "planned", "planned_short")
    out: dict[str, dict] = {}

    raw: dict[tuple[str, str], dict[str, list[float]]] = {}
    for dim in DIMS:
        path = AB / f"{dim}.json"
        if not path.exists():
            continue
        samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
        for key, values in samples.items():
            arm, model = key.split("__", 1)
            raw.setdefault((model, arm), {})[dim] = values

    for (model, arm), dims in raw.items():
        entry = out.setdefault(model, {}).setdefault(arm, {})
        flattened = [v for values in dims.values() for v in values]
        entry["four_dim_mean"] = round(statistics.mean(flattened), 3)
        entry["four_dim_sd"] = round(statistics.stdev(flattened), 3) if len(flattened) > 1 else 0.0
        entry["dimensions"] = {
            dim: {"mean": round(statistics.mean(vals), 3),
                  "sd": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                  "values": vals}
            for dim, vals in dims.items()
        }

    # Paired at the level of the clip, not the (clip, dimension) cell.
    #
    # The four dimensions are scored on the SAME render, so they are not four
    # independent observations: a clip that is sharp tends to also score well on
    # aesthetics. Treating 5 seeds x 4 dimensions as n=20 inflates the sample and
    # manufactures significance. The independent unit is the clip, so each seed's
    # four dimensions are averaged first and the test runs on those 5 differences.
    paired: dict[str, dict] = {}
    for model in MODELS:
        if model not in out or "brief" not in out[model]:
            continue
        for arm in ("planned", "planned_short"):
            if arm not in out[model]:
                continue
            diffs = []
            for dim in DIMS:
                a = out[model]["brief"]["dimensions"].get(dim, {}).get("values")
                b = out[model][arm]["dimensions"].get(dim, {}).get("values")
                if a and b and len(a) == len(b):
                    diffs.append([y - x for x, y in zip(a, b)])
            if not diffs or not diffs[0]:
                continue
            per_clip = [statistics.mean(values) for values in zip(*diffs)]
            paired.setdefault(model, {})[f"{arm}_minus_brief"] = {
                "n": len(per_clip),
                "mean": round(statistics.mean(per_clip), 3),
                "sd": round(statistics.stdev(per_clip), 3) if len(per_clip) > 1 else 0.0,
                "ci95": bootstrap(per_clip),
                "sign_test": sign_test(per_clip),
                "sign_flip_p": sign_flip(per_clip),
                "per_clip": [round(v, 3) for v in per_clip],
            }

    # Win count in the 8 (dimension, model) cells, the way the earlier report counted.
    cells = {"brief": 0, "planned": 0, "planned_short": 0}
    comparisons = 0
    for model in MODELS:
        if model not in out:
            continue
        for dim in DIMS:
            scores = {arm: out[model].get(arm, {}).get("dimensions", {}).get(dim, {}).get("mean")
                      for arm in arms}
            if any(v is None for v in scores.values()):
                continue
            comparisons += 1
            best = max(scores, key=lambda a: scores[a])
            cells[best] += 1
    return {"models": out, "paired": paired, "best_arm_per_cell": cells, "cells": comparisons}


def main() -> int:
    summary = {
        "experiment_a": experiment_a(),
        "experiment_b": experiment_b(),
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    a = summary["experiment_a"]
    print("=== A. cross-shot consistency ===")
    for model in MODELS + ["pooled"]:
        entry = a.get(model)
        if not entry:
            continue
        if model == "pooled":
            p = entry["pairs"]
            print(f"POOLED  n={p['n']:2d}  diff={p['mean_diff']:+.2f}  CI={p['ci95']}  "
                  f"W/T/L={p['sign_test']['wins']}/{p['sign_test']['ties']}/{p['sign_test']['losses']}  "
                  f"sign p={p['sign_test']['p_greater']:.2e}  flip p={p['sign_flip_p']}")
            continue
        ind, pla = entry["arms"]["independent"], entry["arms"]["planned"]
        p = entry["pairs"]
        print(f"{model:14s} independent {ind['mean']:.2f}±{ind['sd']:.2f} (n={ind['n']})  "
              f"planned {pla['mean']:.2f}±{pla['sd']:.2f} (n={pla['n']})  "
              f"diff {p['mean_diff']:+.2f} CI={p['ci95']}  W/T/L="
              f"{p['sign_test']['wins']}/{p['sign_test']['ties']}/{p['sign_test']['losses']}  "
              f"p={p['sign_test']['p_greater']:.2e}")
        print(f"{'':14s} clip {entry['render_seconds']['mean']:.1f}s "
              f"(sd {entry['render_seconds']['sd']:.1f})  traits {entry['planned_traits']}")

    b = summary["experiment_b"]
    print("\n=== B. single-clip four-dimension mean ===")
    for model in MODELS:
        if model not in b["models"]:
            continue
        for arm in ("brief", "planned", "planned_short"):
            e = b["models"][model].get(arm)
            if e:
                print(f"{model:14s} {arm:14s} {e['four_dim_mean']:.3f} ± {e['four_dim_sd']:.3f}")
        for key, v in b["paired"].get(model, {}).items():
            print(f"{'':14s} {key:20s} {v['mean']:+.3f} CI={v['ci95']} p={v['sign_test']['p_less']:.2e}")
    print(f"\nbest arm per (dimension, model) cell: {b['best_arm_per_cell']} of {b['cells']}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
