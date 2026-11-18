"""Segment the study's local baseline clips, for the identity layer.

The judge answers "how good does this look". This answers a different question
the judge cannot: "does the picture stay the same thing from the start of the
clip to its end". It is deterministic, local and free, so it runs on the CPU
while the GPU judge works.

Window choice is not the library default here. These clips are ~5 s, and a
5-second window would give every clip exactly one segment — which is not a
degradation of the measurement, it is the absence of one. A 1-second window
yields five windows per clip, the minimum that lets "drift" mean anything, and
it is stated in the output so a reader can see the ruler that was used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

from bench.segments import evaluate_clip, to_dict  # noqa: E402

MODELS = ["cogvideox5b", "wan21_1_3b", "longlive_1_3b"]


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Segment the study clips")
    parser.add_argument("--run", type=Path, default=ROOT / "bench/results/study")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    args = parser.parse_args(argv[1:])

    rows = []
    for model in args.models:
        for path in sorted((args.run / "videos" / model).glob("*.mp4")):
            report = evaluate_clip(path, model, args.segment_seconds)
            rows.append(to_dict(report))
            summary = report.summary
            print(f"  {model:14s} {path.name:34s} {summary['segments']:2d} segs  "
                  f"identity_min={summary.get('identity_min')}  "
                  f"median={summary.get('identity_median')}", flush=True)

    out = args.run / "segments"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"segment_seconds": args.segment_seconds, "clips": rows}
    (out / "segments.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} clips -> {out / 'segments.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
