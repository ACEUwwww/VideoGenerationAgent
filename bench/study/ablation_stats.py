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
    """Scored rows, one per (arm, model, brief, repeat).

    The log is append-only and a cell can be written twice — a pilot pass
    followed by a full pass, or a resume after an interrupt. Keeping the last
    row per key makes the analysis idempotent instead of letting a duplicated
    cell double-count in every mean and every paired contrast.
    """
    if not LOG.exists():
        return []
    latest: dict[tuple[str, str, int, int], dict] = {}
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["score"] = parse_score(row.get("reply", ""))
        if row["score"] is None:
            continue
        latest[(row["arm"], row["model"], row["brief_index"], row["repeat"])] = row
    return list(latest.values())


def load_noise_floor() -> float | None:
    """How much the judge moves on an unchanged prompt, from the repeat study.

    A contrast smaller than this is not evidence of anything, so it is loaded
    and printed next to every contrast rather than left in another document.
    """
    path = ROOT / "bench/results/noise_floor.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data.get("independent") or {}).get("mean_range")


def sign_test(diffs: list[float]) -> tuple[int, int, int, float]:
    """One-sided sign test in the direction the hypothesis predicts.

    ``diffs`` is defined as ``better - worse``, so a positive difference is the
    predicted direction and the tail is P(X >= wins). The earlier version of
    this file reported P(X >= wins) while the contrast was written as
    ``better - worse`` in a way that produced negative means — the same
    wrong-sided tail bug the repeated-measure report had already corrected. The
    direction is now asserted here instead of assumed at the call site.
    """
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

    repeats = len({key[2] for key in cells})
    noise = load_noise_floor()
    print(f"cells={len(cells)} repeats={repeats} noise_floor={noise}")

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
        mean_diff = round(statistics.mean(diffs), 3)
        # The tail above is one-sided in the predicted direction. When the
        # difference runs the other way it returns a p near 1, which reads as
        # "no result" while the opposite effect may be the real finding. Report
        # that tail too, so a reversed effect cannot hide behind a p of 1.
        reversed_p = sign_test([-d for d in diffs])[3] if mean_diff < 0 else None
        effect_p = reversed_p if reversed_p is not None else p
        direction = "as predicted" if mean_diff > 0 else ("reversed" if mean_diff < 0 else "none")
        # Decision rule: does the paired confidence interval exclude zero?
        #
        # An earlier version compared |mean_diff| to the within-cell range from
        # noise_floor.json. That was wrong twice over: the range is a property
        # of ONE score across repeats, not of a paired DIFFERENCE, and pairing
        # removes most of the render-to-render variation the range measures. It
        # suppressed a contrast with 13 of 14 concordant pairs and p = 0.0009
        # because its mean (0.78) fell just under the range (1.0). The interval
        # is the honest test; the noise floor is context about one observation.
        excludes_zero = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
        verdict = "paired CI excludes zero" if excludes_zero else "paired CI includes zero"
        extra = f" p_effect={effect_p} ({direction})" if reversed_p is not None else ""
        results[f"{better}-{worse}"] = {
            "question": question, "n": len(diffs),
            "mean_diff": mean_diff,
            "ci95": [lo, hi], "wins": wins, "ties": ties, "losses": losses,
            "p_predicted_direction": p, "p_reversed": reversed_p,
            "p_effect": effect_p, "direction": direction,
            "single_observation_noise_range": noise,
            "verdict": verdict,
        }
        extra = f" p_effect={effect_p} ({direction})" if reversed_p is not None else ""
        print(f"  {better} - {worse}: {mean_diff:+.3f} "
              f"CI95=[{lo:+}, {hi:+}] w/t/l={wins}/{ties}/{losses} p={p}{extra}  "
              f"({question}) {verdict}")

    if repeats < 2:
        print("\nWARNING: one repeat per cell. The judge moves "
              f"~{noise} points on an unchanged prompt (bench/results/noise_floor.json),"
              " so these contrasts cannot separate arm effect from render noise.")

    out = ROOT / "bench/results/prompt_ablation/summary.json"
    out.write_text(json.dumps({"arms": arms, "contrasts": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
