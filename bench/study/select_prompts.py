"""Pick a stratified, reproducible prompt set from Video-Bench's own suite.

The smoke run used one prompt ("A person is marching"), which exercises one
dimension family and supports no statistical claim. This selects a fixed,
dimension-balanced subset instead, and records exactly which indices it took so
the set is reproducible without trusting this script's ordering later.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.prompts import Prompt, _slug, write_manifest

SUITE = ROOT / "bench/_vendor/Video-Bench/videobench/VideoBench_full.json"

# Quotas by the dimension a prompt was written for. overall_consistency gets the
# most weight because those are the only prompts describing a multi-element
# *scene* rather than a single object or action, which is what a short drama
# actually asks a generator to do.
QUOTA = {"overall_consistency": 4, "action": 2, "color": 2, "scene": 2, "object_class": 2}


def primary_dimension(entry) -> str:
    dims = entry["dimension"] if isinstance(entry["dimension"], list) else [entry["dimension"]]
    for name in QUOTA:
        if name in dims:
            return name
    return dims[0]


def main(count_scale: int = 1) -> int:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    picked: list[dict] = []
    for dimension, quota in QUOTA.items():
        want = quota * count_scale
        for index, entry in enumerate(suite):
            if len([p for p in picked if p["primary"] == dimension]) >= want:
                break
            if primary_dimension(entry) != dimension:
                continue
            picked.append({
                "index": index, "primary": dimension,
                "text": entry["prompt"],
                "dimensions": entry["dimension"] if isinstance(entry["dimension"], list) else [entry["dimension"]],
            })
    prompts = [Prompt(text=p["text"], dimensions=tuple(p["dimensions"]), slug=_slug(p["text"])) for p in picked]
    out = ROOT / "bench/study/prompts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(picked, indent=2, ensure_ascii=False), encoding="utf-8")
    for p in picked:
        print(f"{p['index']:4d} {p['primary']:20s} {p['text'][:70]}")
    print(f"\n{len(picked)} prompts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
