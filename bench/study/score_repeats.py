"""Score every repeated episode with the same cross-shot VLM metric.

Reads the render records written by ``tier3a_repeats.py`` and asks the judge the
same question the single-pass study asked, so the two are on one scale. Each row
is one episode: arm, brief, repeat, score, and the trait the judge named as
having changed.

The judge is deterministic at temperature 0, so a row is scored once. Re-scoring
the same frames would only measure the server's stability, not the workflow's.

Usage:
    python bench/study/score_repeats.py --model wan21_1_3b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/study"))

from character_consistency import score_episode  # noqa: E402

OUT = ROOT / "bench/results/tier3a_repeats"


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score repeated episodes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8791/v1")
    parser.add_argument("--model-name", default="qwen2.5-vl-32b-local")
    parser.add_argument("--max-shots", type=int, default=5)
    args = parser.parse_args(argv[1:])

    src = OUT / f"clips_{args.model}.jsonl"
    dst = OUT / f"consistency_{args.model}.json"
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]

    scored: list[dict] = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else []
    seen = {(r["arm"], r["brief_index"], r["repeat"]) for r in scored}

    for row in rows:
        key = (row["arm"], row["brief_index"], row["repeat"])
        if key in seen:
            continue
        shots = [Path(p) for p in row["shots"]]
        if len(shots) < 2:
            print(f"  {key} SKIP (only {len(shots)} clips)", flush=True)
            continue
        label = f"{row['arm']}:{args.model}:b{row['brief_index']}:r{row['repeat']}"
        try:
            verdict = score_episode(label, shots, base=args.base, model=args.model_name,
                                    max_shots=args.max_shots)
        except Exception as exc:  # noqa: BLE001 - one episode must not sink the batch
            verdict = {"label": label, "score": None, "error": str(exc)[:300], "shots": 0}
        verdict.update(arm=row["arm"], model=args.model,
                       brief_index=row["brief_index"], repeat=row["repeat"],
                       prompt_chars=row["prompt_chars"], clip_seconds=row["clip_seconds"],
                       plan_seconds=row["plan_seconds"])
        scored.append(verdict)
        print(f"  {label:40s} {verdict.get('score')} ({verdict.get('changed')})", flush=True)
        dst.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(scored)} episodes -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
