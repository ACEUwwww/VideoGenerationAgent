"""Statistics for the repeated-measure comparison.

Computes what the report quotes, and prints it in the shape the report needs:
per-arm means with dispersion, the paired difference with its confidence
interval, a sign test, and a permutation test on the means. Every method is
exact or enumerated rather than sampled where the sample allows it, so a reader
can reproduce a p-value without trusting a random seed.

Usage:
    python bench/study/repeats_stats.py
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
REP = ROOT / "bench/results/tier3a_repeats"
MODELS = ["cogvideox5b", "wan21_1_3b"]


def load(model: str) -> list[dict]:
    path = REP / f"consistency_{model}.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text(encoding="utf-8")) if r.get("score")]


def sign_test(diffs: list[float]) -> tuple[int, int, int, float]:
    """Exact one-sided sign test on non-zero paired differences."""
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return wins, 0, losses, 1.0
    # P(X >= wins) under Binomial(n, 0.5)
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return wins, n - wins - losses, losses, p


def paired_permutation(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation test on the mean paired difference."""
    n = len(diffs)
    observed = abs(statistics.mean(diffs))
    count = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        if abs(statistics.mean([s * d for s, d in zip(signs, diffs)])) >= observed - 1e-12:
            count += 1
    return count / total


def bootstrap_ci(diffs: list[float], trials: int = 5000, seed: int = 0) -> tuple[float, float]:
    import random

    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(trials):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(0.025 * trials)], means[int(0.975 * trials)]


def main() -> int:
    everything: list[dict] = []
    for model in MODELS:
        everything.extend(load(model))
    if not everything:
        print("no scored repeats yet")
        return 1

    print(f"{'model':14s}{'arm':14s}{'n':>4s}{'mean':>8s}{'sd':>7s}{'min':>5s}{'max':>5s}")
    print("-" * 57)
    for model in MODELS + ["ALL"]:
        rows = everything if model == "ALL" else load(model)
        if not rows:
            continue
        for arm in ("independent", "planned"):
            values = [r["score"] for r in rows if r["arm"] == arm]
            if not values:
                continue
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"{model:14s}{arm:14s}{len(values):4d}{statistics.mean(values):8.2f}"
                  f"{sd:7.2f}{min(values):5.0f}{max(values):5.0f}")

    print()
    for scope in ("per-model", "pooled"):
        diffs: list[float] = []
        pairs = 0
        for model in MODELS:
            rows = load(model)
            cells: dict[tuple[int, int], dict[str, float]] = {}
            for row in rows:
                cells.setdefault((row["brief_index"], row["repeat"]), {})[row["arm"]] = row["score"]
            model_diffs = [c["planned"] - c["independent"] for c in cells.values()
                           if "planned" in c and "independent" in c]
            if scope == "per-model" and model_diffs:
                w, t, l, p = sign_test(model_diffs)
                lo, hi = bootstrap_ci(model_diffs)
                print(f"{model:14s} pairs={len(model_diffs):2d} mean diff={statistics.mean(model_diffs):+.2f} "
                      f"CI95=[{lo:+.2f},{hi:+.2f}] W/T/L={w}/{t}/{l} sign p={p:.4f} "
                      f"perm p={paired_permutation(model_diffs):.4f}")
            diffs.extend(model_diffs)
            pairs += len(model_diffs)
        if scope == "pooled" and diffs:
            w, t, l, p = sign_test(diffs)
            lo, hi = bootstrap_ci(diffs)
            print(f"{'POOLED':14s} pairs={pairs:2d} mean diff={statistics.mean(diffs):+.2f} "
                  f"CI95=[{lo:+.2f},{hi:+.2f}] W/T/L={w}/{t}/{l} sign p={p:.5f} "
                  f"perm p={paired_permutation(diffs):.4f}")

    print()
    print("render time per clip (s):")
    for model in MODELS:
        secs = [s for r in load(model) for s in r.get("clip_seconds") or []]
        if secs:
            print(f"  {model:14s} n={len(secs):3d} mean={statistics.mean(secs):6.1f} "
                  f"sd={statistics.stdev(secs) if len(secs) > 1 else 0:5.1f} "
                  f"min={min(secs):5.1f} max={max(secs):5.1f}")

    print()
    print("judge 'changed' trait, planned arm:")
    for model in MODELS:
        counts: dict[str, int] = {}
        for row in load(model):
            if row["arm"] == "planned":
                counts[row.get("changed") or "NONE"] = counts.get(row.get("changed") or "NONE", 0) + 1
        if counts:
            print(f"  {model:14s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
