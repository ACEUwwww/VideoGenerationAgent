"""Statistics for the controlled prompt ablation.

Reads `bench/results/prompt_ablation/clips.jsonl` and reports the factorial
contrasts the design was built for. Each contrast is paired within
(model, brief, repeat), so it answers one question and not three:

| contrast | question |
| --- | --- |
| `brief_en` vs `scene` | does plan-derived scene/action content help on its own? |
| `scene` vs `planned` | does adding the character anchor help, given scene content? |
| `brief_en` vs `brief_en_anchor` | does the anchor help when there is no plan structure? |
| `planned` vs `planned_trunc` | is the effect just prompt length? |

The two anchor contrasts are complementary: one holds scene content present, the
other holds it absent. Reporting only one of them is what made the original
``planned`` vs ``independent`` comparison uninterpretable.

Pairing is by (model, brief, repeat). Every arm in a cell is rendered with the
same seeds, so a pair differs only in the prompt string.

Usage:
    python bench/study/ablation_stats.py
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "bench/results/prompt_ablation/clips.jsonl"

CONTRASTS = (
    ("brief_en", "scene", "plan content, anchor absent"),
    ("scene", "planned", "anchor effect, scene content present"),
    ("brief_en", "brief_en_anchor", "anchor effect, scene content absent"),
    ("planned", "planned_trunc", "prompt length"),
)

_SCORE = re.compile(r"SCORE:\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(reply: str) -> float | None:
    """The 1-5 verdict out of the judge's raw text."""
    if not reply:
        return None
    match = _SCORE.search(reply.replace("**", ""))
    if match is None:
        return None
    value = float(match.group(1))
    return value if 1.0 <= value <= 5.0 else None


def load() -> list[dict]:
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["score"] = parse_score(row.get("reply", ""))
        if row["score"] is not None:
            rows.append(row)
    return rows


def sign_test(diffs: list[float]) -> tuple[int, int, int, float]:
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return wins, 0, losses, 1.0
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return wins, n - wins - losses, losses, round(p, 6)


def bootstrap_ci(diffs: list[float], trials: int = 5000, seed: int = 0) -> tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(trials))
    return round(means[int(0.025 * trials)], 3), round(means[int(0.975 * trials)], 3)


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def main() -> int:
    rows = load()
    if not rows:
        print(f"no scored rows in {LOG}")
        return 1

    cells: dict[tuple[str, int, int], dict[str, float]] = {}
    chars: dict[str, list[int]] = {}
    for row in rows:
        cells.setdefault((row["model"], row["brief_index"], row["repeat"]), {})[row["arm"]] = row["score"]
        chars.setdefault(row["arm"], []).extend(row.get("prompt_chars") or [])

    arms = sorted({row["arm"] for row in rows})
    print("arm means (1-5, higher is better)")
    for arm in arms:
        values = [row["score"] for row in rows if row["arm"] == arm]
        stats = describe(values)
        mean_chars = round(statistics.mean(chars[arm])) if chars.get(arm) else None
        print(f"  {arm:<18} mean={stats.get('mean')} sd={stats.get('sd')} "
              f"n={stats.get('n')} chars≈{mean_chars}")

    print("\nfactorial contrasts (paired within model/brief/repeat)")
    results = {}
    for better, worse, question in CONTRASTS:
        diffs = [
            cell[better] - cell[worse]
            for cell in cells.values()
            if better in cell and worse in cell
        ]
        if not diffs:
            print(f"  {better} - {worse}: no complete pairs")
            continue
        wins, ties, losses, p = sign_test(diffs)
        lo, hi = bootstrap_ci(diffs)
        results[f"{better}-{worse}"] = {
            "question": question, "n": len(diffs),
            "mean_diff": round(statistics.mean(diffs), 3),
            "ci95": [lo, hi], "wins": wins, "ties": ties, "losses": losses, "p": p,
        }
        print(f"  {better} - {worse}: {statistics.mean(diffs):+.3f} "
              f"CI95=[{lo:+}, {hi:+}] w/t/l={wins}/{ties}/{losses} p={p}  ({question})")

    out = ROOT / "bench/results/prompt_ablation/summary.json"
    out.write_text(json.dumps({"arms": arms, "contrasts": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
