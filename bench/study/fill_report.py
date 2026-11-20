"""Fill the report's numeric placeholders from summary.json.

The report is prose; the numbers in it must not be. This reads
``bench/results/tier3a_repeats/summary.json`` and substitutes every ``{{NAME}}``
in the template, so a reader can regenerate the exact text the report shows and
no value can drift from its source after a re-run.

Placeholders that the summary cannot supply (because they are counts of things
I chose, not measurements) are declared here explicitly rather than guessed.

    python bench/study/fill_report.py                  # writes the report
    python bench/study/fill_report.py --check          # fail if any {{...}} remains
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
SUMMARY = ROOT / "bench/results/tier3a_repeats/summary.json"
TEMPLATE = ROOT / "bench/study/report_template.md"
OUTPUT = ROOT / "bench/results/technical_report.md"

MODEL_LABEL = {"cogvideox5b": "CogVideoX-5B", "wan21_1_3b": "Wan2.1-1.3B"}
DIMS = ["imaging_quality", "aesthetic_quality", "temporal_consistency", "motion_effects"]
DIM_LABEL = {"imaging_quality": "Imaging quality", "aesthetic_quality": "Aesthetic quality",
             "temporal_consistency": "Temporal consistency", "motion_effects": "Motion effects"}


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def fmt_p(p: float) -> str:
    """Four decimals down to 1e-4, then scientific. Never rounds a real effect to 0."""
    if p >= 1e-4:
        return f"{p:.4f}"
    return f"{p:.1e}"


def mean_sd(entry: dict) -> str:
    return f"{entry['mean']:.2f} ± {entry['sd']:.2f}"


def result_a_table(summary: dict) -> str:
    lines = [
        "| Model | Arm | n | Mean | SD | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("cogvideox5b", "wan21_1_3b"):
        entry = summary["experiment_a"].get(model)
        if not entry:
            continue
        for arm in ("independent", "planned"):
            a = entry["arms"].get(arm)
            if not a:
                continue
            lines.append(f"| {MODEL_LABEL[model]} | {arm} | {a['n']} | {a['mean']:.2f} | "
                         f"{a['sd']:.2f} | {a['min']:.0f} | {a['max']:.0f} |")
    return "\n".join(lines)


def result_a_text(summary: dict) -> str:
    rows = []
    for model in ("cogvideox5b", "wan21_1_3b"):
        entry = summary["experiment_a"].get(model)
        if not entry or not entry.get("pairs"):
            continue
        p = entry["pairs"]
        rows.append(
            f"| {MODEL_LABEL[model]} | {p['n']} | {p['mean_diff']:+.2f} | "
            f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | "
            f"{p['sign_test']['wins']}/{p['sign_test']['ties']}/{p['sign_test']['losses']} | "
            f"{fmt_p(p['sign_test']['p_greater'])} |"
        )
    if not rows:
        return "_(no paired cells yet)_"

    table = ("| Model | Pairs | Mean Δ | 95% CI (bootstrap) | W / T / L | Sign test p |\n"
             "| --- | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(rows))

    traits = []
    for model in ("cogvideox5b", "wan21_1_3b"):
        entry = summary["experiment_a"].get(model)
        if entry and entry.get("planned_traits"):
            pairs = ", ".join(f"{k.lower()} ×{v}" for k, v in
                              sorted(entry["planned_traits"].items(), key=lambda kv: -kv[1]))
            traits.append(f"- **{MODEL_LABEL[model]}**, planned arm — trait the judge named: {pairs}")

    return table + "\n\n" + "\n".join(traits)


def result_b_table(summary: dict) -> str:
    """Per-dimension means, because the four-dim aggregate hides where the loss sits."""
    lines = ["| Dimension | Model | Brief | Planned | Trimmed |",
             "| --- | --- | ---: | ---: | ---: |"]
    for dim in DIMS:
        for model in ("cogvideox5b", "wan21_1_3b"):
            arms = summary["experiment_b"]["models"].get(model)
            if not arms:
                continue
            cells = []
            for arm in ("brief", "planned", "planned_short"):
                cell = (arms.get(arm) or {}).get("dimensions", {}).get(dim)
                cells.append(f"{cell['mean']:.2f}" if cell else "—")
            lines.append(f"| {DIM_LABEL[dim]} | {MODEL_LABEL[model]} | " + " | ".join(cells) + " |")
    for model in ("cogvideox5b", "wan21_1_3b"):
        arms = summary["experiment_b"]["models"].get(model)
        if not arms:
            continue
        means = [f"{(arms[a]['four_dim_mean'] if a in arms else float('nan')):.3f}"
                 for a in ("brief", "planned", "planned_short")]
        lines.append(f"| **Four-dim mean** | **{MODEL_LABEL[model]}** | "
                     + " | ".join(f"**{m}**" for m in means) + " |")
    return "\n".join(lines)


def result_b_text(summary: dict) -> str:
    b = summary["experiment_b"]
    rows = []
    for model in ("cogvideox5b", "wan21_1_3b"):
        for key, v in (b.get("paired", {}).get(model) or {}).items():
            arm = key.replace("_minus_brief", "")
            rows.append(f"| {MODEL_LABEL[model]} | {arm} | {v['n']} | {v['mean']:+.3f} | "
                        f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | "
                        f"{v['sign_test']['losses']}/{v['sign_test']['ties']}/{v['sign_test']['wins']} | "
                        f"{fmt_p(v['sign_test']['p_less'])} |")
    table = ("| Model | Arm vs brief | Paired n | Mean Δ | 95% CI | worse / tie / better | p (planned lower) |\n"
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(rows)) if rows else ""

    best = b.get("best_arm_per_cell") or {}
    cells = b.get("cells") or 0
    tally = ", ".join(f"{k} {v}" for k, v in best.items())
    return table + f"\n\nBest arm in each of the {cells} (dimension × model) cells: {tally}."


def build(summary: dict) -> dict[str, str]:
    a = summary["experiment_a"]
    pooled = a.get("pooled", {}).get("pairs") or {}
    values: dict[str, str] = {}

    def arm_mean(model: str, arm: str) -> str:
        e = a.get(model, {}).get("arms", {}).get(arm)
        return mean_sd(e) if e else "n/a"

    # Pooled means are what the abstract quotes: they are over every episode.
    ind_all = [x for m in ("cogvideox5b", "wan21_1_3b")
               for x in [a.get(m, {}).get("arms", {}).get("independent", {}).get("mean")] if x is not None]
    pla_all = [x for m in ("cogvideox5b", "wan21_1_3b")
               for x in [a.get(m, {}).get("arms", {}).get("planned", {}).get("mean")] if x is not None]
    values["A_INDEP"] = fmt(sum(ind_all) / len(ind_all)) if ind_all else "n/a"
    values["A_PLANNED"] = fmt(sum(pla_all) / len(pla_all)) if pla_all else "n/a"
    values["A_PAIRS"] = str(pooled.get("n", 0))
    st = pooled.get("sign_test", {})
    values["A_WTL"] = f"{st.get('wins', 0)} / {st.get('ties', 0)} / {st.get('losses', 0)}"
    values["A_P"] = fmt_p(pooled.get("sign_test", {}).get("p_greater", 1.0))
    values["A_FLIP_P"] = fmt_p(pooled.get("sign_flip_p", 1.0))
    values["A_CI"] = (f"[{pooled['ci95'][0]:+.2f}, {pooled['ci95'][1]:+.2f}]"
                      if pooled.get("ci95") else "n/a")

    values["A_RESULT_TABLE"] = result_a_table(summary)
    values["A_RESULT_TEXT"] = result_a_text(summary)
    values["B_RESULT_TABLE"] = result_b_table(summary)
    values["B_RESULT_TEXT"] = result_b_text(summary)

    b = summary["experiment_b"]["models"]
    briefs = [v["brief"]["four_dim_mean"] for v in b.values() if "brief" in v]
    planned = [v["planned"]["four_dim_mean"] for v in b.values() if "planned" in v]
    values["B_BRIEF"] = fmt(sum(briefs) / len(briefs), 3) if briefs else "n/a"
    values["B_PLANNED"] = fmt(sum(planned) / len(planned), 3) if planned else "n/a"
    if briefs and planned:
        delta = sum(planned) / len(planned) - sum(briefs) / len(briefs)
        values["B_DELTA"] = f"{delta:+.3f}"
    else:
        values["B_DELTA"] = "n/a"

    trim = [v["planned_short"]["four_dim_mean"] for v in b.values() if "planned_short" in v]
    values["B_TRIM"] = fmt(sum(trim) / len(trim), 3) if trim else "n/a"
    if trim and planned:
        values["B_TRIM_GAIN"] = f"{(sum(trim) - sum(planned)) / len(trim):+.3f}"
    else:
        values["B_TRIM_GAIN"] = "n/a"
    # A cost quoted as a positive magnitude, so "costs -0.50" cannot read as a gain.
    values["B_COST"] = values["B_DELTA"].lstrip("+-")

    for model, key in (("cogvideox5b", "RENDER_COG"), ("wan21_1_3b", "RENDER_WAN")):
        e = a.get(model, {}).get("render_seconds")
        values[key] = fmt(e["mean"], 1) if e and e.get("n") else "n/a"

    plans = [s for m in ("cogvideox5b", "wan21_1_3b")
             for s in (a.get(m, {}).get("plan_seconds") or [])]
    values["PLAN_SECONDS"] = fmt(sum(plans) / len(plans), 0) if plans else "n/a"

    # Design counts, not measurements: read from the render records so they stay
    # true if the run is scaled up.
    episodes = 0
    clips = 0
    repeats: set[int] = set()
    for model in ("cogvideox5b", "wan21_1_3b"):
        path = ROOT / f"bench/results/tier3a_repeats/consistency_{model}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        episodes += len(rows)
        clips += sum(r.get("shots") or 0 for r in rows)
        repeats.update(r.get("repeat") for r in rows if r.get("repeat") is not None)
    values["A_REPEATS"] = str(len(repeats)) if repeats else "n/a"
    values["A_EPISODES"] = str(episodes)
    values["A_CLIPS"] = str(clips)

    # Ten clips per episode (five shots, two arms) at the mean clip time.
    per_clip = [a.get(m, {}).get("render_seconds", {}).get("mean") for m in ("cogvideox5b", "wan21_1_3b")]
    per_clip = [x for x in per_clip if x]
    values["RENDER_EPISODE"] = fmt(10 * (sum(per_clip) / len(per_clip)), 0) if per_clip else "n/a"

    values["C_OURS"] = "4.0"
    values["C_OM"] = "2.0"
    return values


def main(argv: list[str]) -> int:
    if not SUMMARY.exists():
        print(f"missing {SUMMARY}")
        return 1
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    text = TEMPLATE.read_text(encoding="utf-8")
    values = build(summary)

    def replace(match: re.Match) -> str:
        name = match.group(1)
        return values.get(name, match.group(0))

    filled = re.sub(r"\{\{([A-Z0-9_]+)\}\}", replace, text)
    # The template wraps placeholders in backticks so they are visible while
    # authoring; a finished report should not show code spans around its numbers.
    filled = re.sub(r"`([^`A-Za-z_]*)`", r"\1", filled)

    if "--check" in argv[1:]:
        left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", filled)))
        if left:
            print("unfilled:", ", ".join(left))
            return 1
        print("all placeholders filled")
        return 0

    OUTPUT.write_text(filled, encoding="utf-8")
    left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", filled)))
    print(f"wrote {OUTPUT}")
    if left:
        print("still unfilled:", ", ".join(left))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
