"""Controlled A/B: does our planning layer's prompt beat the raw brief?

Every earlier comparison in this repo let two things vary at once. In Tier 3a
our side re-sampled its prompt on every repeat while its video seed stayed
fixed at 1, and OpenMontage's side fixed the brief and stepped its seed — so
"repeat" meant different things on the two sides and no difference could be
attributed.

This experiment removes that: both arms render the SAME model, the SAME seed
list, the SAME resolution and frame count. The only thing that differs is the
string handed to the model.

  arm "brief"   — the creative brief, verbatim. This is what OpenMontage's shot
                  stage sends: one sentence of intent.
  arm "planned" — the prompt our planning layer writes from the same brief: a
                  standalone visual description carrying the character's
                  appearance, wardrobe, setting, lighting, framing and motion.

If the planning layer contributes anything at the shot stage, it shows up here
and nowhere else. If the two arms score the same, then at this stage our layer
adds cost without adding output quality, and the report has to say so.

Prompts are read from the Tier 3a run that produced them, so this compares real
planner output rather than prompts written for the experiment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "bench/results/tier3a_ab"
TIER3A = ROOT / "bench/results/tier3a/episodes.jsonl"
BRIEFS = ROOT / "bench/study/briefs.json"
SEEDS = [0, 1, 2, 3, 4]


def planned_prompts(model: str) -> dict[int, str]:
    """Our planner's prompts, one per Tier 3a repeat, keyed by that repeat index.

    Each repeat re-ran the planner, so each is a genuine sample of what our
    orchestration writes for the same brief.
    """
    out: dict[int, str] = {}
    for line in TIER3A.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("side") != "ours" or row.get("model") != model:
            continue
        state_path = ROOT / "workspace/projects" / row["project_id"] / "state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        prompt = (state.get("shots") or [{}])[0].get("motion_prompt") or ""
        if prompt:
            out[row["repeat"]] = prompt
    return out


def render(prompt: str, seed: int, model: str, tag: str, root: Path) -> dict:
    from mcp_servers.providers.base import VideoRequest
    from mcp_servers.providers.local_media import (
        LOCAL_HEIGHT, LOCAL_WIDTH, MODELS, LocalVideoProvider, _frames_for,
    )

    spec = MODELS[model]
    fps = spec["fps"]
    frames = _frames_for(spec.get("frame_rule", "4k+1"), fps, 5.0)
    shot_id = f"{tag}_s{seed}"
    out_dir = root / "clips" / f"{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = LocalVideoProvider(model)
    request = VideoRequest(
        image_path=None,
        motion_prompt=prompt,
        shot_id=shot_id,
        project_dir=out_dir,
        duration_s=frames / fps,
        width=LOCAL_WIDTH,
        height=LOCAL_HEIGHT,
        fps=fps,
        seed=seed,
    )
    result = provider.generate(request)
    return {
        "arm": tag,
        "model": model,
        "seed": seed,
        "path": str(out_dir / result["video_path"]),
        "seconds": result["duration_s"],
        "frames": result["frames"],
        "elapsed_s": result["elapsed_s"],
        "prompt_chars": len(prompt),
    }


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="A/B the prompt, holding the seed fixed")
    parser.add_argument("--models", nargs="+", default=["cogvideox5b", "wan21_1_3b"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--brief-index", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv[1:])

    brief = json.loads(BRIEFS.read_text(encoding="utf-8"))[args.brief_index]
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"brief[{args.brief_index}]: {brief[:60]}")
    print(f"{len(args.models)} models x {len(args.seeds)} seeds x 2 arms "
          f"= {2 * len(args.models) * len(args.seeds)} renders")

    results = OUT / "clips.jsonl"
    done: set[tuple[str, str, int]] = set()
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            done.add((row["arm"], row["model"], row["seed"]))

    for model in args.models:
        planned = planned_prompts(model)
        if not planned:
            print(f"  no planned prompts for {model}; run tier3a.py first")
            continue
        for seed in args.seeds:
            # Pair each seed with a different planner sample, so the "planned"
            # arm samples the planner's variability instead of reusing one text.
            prompt = planned[seed % len(planned)]
            import re as _re
            # Ablation arm: the same planned prompt cut to its opening
            # sentences. If the full prompt loses to the bare brief because a
            # 1.3B model cannot attend to 900 characters, the short version
            # should recover the gap; if it does not, length is not the cause.
            _sents = _re.split(r"(?<=[.!?])\s+", prompt)
            arms = [
                ("brief", brief),
                ("planned", prompt),
                ("planned_short", " ".join(_sents[:2])[:400]),
            ]
            for tag, text in arms:
                key = (tag, model, seed)
                if key in done:
                    print(f"  reuse {tag} {model} seed={seed}", flush=True)
                    continue
                if args.dry_run:
                    print(f"  {tag:8s} {model:15s} seed={seed} ({len(text)} chars)")
                    continue
                print(f"\n[{tag}] {model} seed={seed}", flush=True)
                started = time.monotonic()
                try:
                    row = render(text, seed, model, tag, OUT)
                except Exception as exc:  # noqa: BLE001 - one render must not sink the batch
                    row = {"arm": tag, "model": model, "seed": seed,
                           "error": str(exc)[:300], "elapsed_s": round(time.monotonic() - started, 1)}
                print(f"  -> {row.get('elapsed_s')}s {str(row.get('error', ''))[:90]!r}", flush=True)
                with results.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
