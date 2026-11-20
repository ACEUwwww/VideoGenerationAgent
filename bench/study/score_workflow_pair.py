"""Score the two full-workflow episodes with the same cross-shot VLM metric.

The existing scorer reads its episode list from `compare_workflows/episodes.json`,
which pairs the paid Tier 1 runs. The OpenMontage control was added later and is
not in that table, so this script calls `score_episode` directly with an
explicit list. It reuses the scorer's own `om_shots` / `our_shots` and its
prompt unchanged — the metric must be the same one the report's §2 reports,
otherwise the two numbers are not comparable.

Usage:
    python bench/study/score_workflow_pair.py --om omt2vb --ours <project_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/study"))

from character_consistency import om_shots, our_shots, score_episode  # noqa: E402

OUT = ROOT / "bench/results/om_control/consistency.json"


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--om", nargs="+", default=[], help="OpenMontage project ids")
    parser.add_argument("--ours", nargs="+", default=[], help="our project ids")
    parser.add_argument("--base", default="http://127.0.0.1:8791/v1")
    parser.add_argument("--model", default="qwen2.5-vl-32b-local")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-shots", type=int, default=12,
                        help="same cap the report used, so shot counts stay comparable")
    args = parser.parse_args(argv[1:])

    tasks = [(f"openmontage:{p}", om_shots(p)) for p in args.om]
    tasks += [(f"ours:{p}", our_shots(p)) for p in args.ours]

    results = []
    for label, shots in tasks:
        if not shots:
            print(f"  {label:30s} NO CLIPS", flush=True)
            results.append({"label": label, "score": None, "error": "no clips", "shots": 0})
            continue
        try:
            row = score_episode(label, shots, base=args.base, model=args.model,
                                max_shots=args.max_shots)
        except Exception as exc:  # noqa: BLE001 - one episode must not sink the batch
            row = {"label": label, "score": None, "error": str(exc)[:300], "shots": 0}
        results.append(row)
        print(f"  {label:30s} shots={row.get('shots', 0):2d} score={row.get('score')} "
              f"changed={str(row.get('changed', ''))[:40]!r}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
