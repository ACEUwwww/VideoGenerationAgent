"""Score the matched-v2 comparison: OpenMontage vs ours, same base model, same config.

Why v2 exists: the earlier comparison (``bench/results/om_control``) put a
12-shot OpenMontage episode against a 5-shot run of ours. That was not a
controlled comparison — the shot count differed, and OpenMontage's side also
used a different seed policy and, on the first attempt, an image-to-video
checkpoint. This run fixes the configuration on both sides:

| knob | value on both sides |
| --- | --- |
| checkpoint | ``cogvideox5b`` (text-to-video) |
| resolution / fps | 832x480 @ 8 fps |
| seconds per shot | 5 |
| seed | the shot's 0-based position (0..N-1) |
| shots | 12 |
| brief | ``bench/study/briefs_matched.json`` |
| judge | the local 32B VLM, same prompt, one frame at 60% per shot |

What still differs, and must be said: OpenMontage's storyboard stage has no
shot-count parameter (its runtime skill sets min 10 / max 14 / target 12, and
``skills_root()`` is fixed inside the vendor tree). Its natural plan for this
brief was 56 shots; the 12 scored here are its own first 12 after it deleted
the rest through its own API. So the two sides agree on shot count, but
OpenMontage's *internal* planning granularity was coarser than the 12 it kept.

Usage:
    python bench/study/score_matched_v2.py \
        --om-project benchs0-4 \
        --ours-dir bench/results/om_matched_v2/ours/clips/ours/videos
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/study"))

from character_consistency import om_shots, score_episode  # noqa: E402

OUT = ROOT / "bench/results/om_matched_v2/consistency.json"


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--om-project", required=True)
    parser.add_argument("--ours-dir", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8791/v1")
    parser.add_argument("--model", default="qwen2.5-vl-32b-local")
    parser.add_argument("--max-shots", type=int, default=12)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv[1:])

    ours = sorted(args.ours_dir.glob("shot_*.mp4"))
    tasks = [(f"openmontage:{args.om_project}", om_shots(args.om_project)),
             ("ours:matched_v2", ours)]

    results = []
    for label, shots in tasks:
        if not shots:
            print(f"  {label:32s} NO CLIPS", flush=True)
            results.append({"label": label, "score": None, "error": "no clips", "shots": 0})
            continue
        row = score_episode(label, shots, base=args.base, model=args.model,
                            max_shots=args.max_shots)
        results.append(row)
        print(f"  {label:32s} shots={row.get('shots', 0):2d} score={row.get('score')} "
              f"changed={str(row.get('changed', ''))[:40]!r}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
