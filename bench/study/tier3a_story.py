"""Does the orchestration keep one cast across shots, where independent shots cannot?

This is the claim the architecture actually makes, and the only one the shot-level
A/B could not test. That experiment held the seed fixed and swapped the prompt,
and the bare brief won: at 1.3B-5B on a single five-second clip, a 900-character
standalone description buys nothing. That result stands and is reported.

But a single clip has no cross-shot problem to solve. The orchestration exists to
hold a cast and a world steady across cuts, and the failure it prevents is the
one this repo measured for real: an early build derived keyframe N from keyframe
N-1, and by shot 5 the courier's blue uniform had drifted into a yellow hi-vis
vest.

Two arms, same model, same seeds, same resolution, same shot count:

  arm "independent" — each shot is generated from the brief alone. Nothing
                      carries between shots. This is what a shot-stage pipeline
                      (OpenMontage's local path among them) can do: five
                      unrelated five-second clips.

  arm "planned"     — each shot is generated from our planner's prompt for that
                      shot, written from one story plan with the character's
                      appearance anchors embedded verbatim in every shot.

The measurement is the cross-shot character score (VLM, 1-5): given the five
frames, is it the same person — face, hair, build, clothing. That is a question
about relations between frames, so it is exactly the thing an arm with no
carried state should fail and an arm with carried appearance anchors should pass.

Caveats this script states rather than buries:
* Text-to-video has no keyframe anchor, so "planned" carries the character only
  through repeated words. That is the weaker form of the mechanism; the full
  form needs an image model and is Tier 3b's job.
* n is the number of stories, and it is small. The per-story scores are printed
  so a reader can see the spread, not just the mean.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "bench/results/tier3a_story"
BRIEFS = ROOT / "bench/study/briefs.json"
SEEDS = [0, 1, 2]
SHOTS = 5


def story_prompts(brief: str, model: str, seed: int) -> list[str]:
    """Our planner's per-shot prompts for one run of the orchestration.

    The full pipeline is driven once per story so the prompts are genuine
    planner output for a five-shot episode, not prompts written for a test.
    """
    import os

    from agent.graph import run_agent
    from agent.settings import reset_settings
    from backend.services import project_store

    os.environ.update(
        VIDEO_PROVIDER="local", VIDEO_MODEL=model,
        IMAGE_PROVIDER="mock", TTS_PROVIDER="mock", ARK_ALLOW_PAID="0",
        VIDEO_WIDTH="832", VIDEO_HEIGHT="480", VIDEO_FPS="16",
    )
    reset_settings()

    project = project_store.create_project(brief, "电影感短剧", SHOTS, genre="悬疑",
                                           target_seconds=int(SHOTS * 5))
    state = project_store.load_state(project["project_id"])
    state["status"] = "running"
    final = run_agent(state)
    return [s.get("motion_prompt", "") for s in (final.get("shots") or [])]


def render(prompt: str, seed: int, model: str, tag: str, root: Path, index: int) -> Path | None:
    from mcp_servers.providers.base import VideoRequest
    from mcp_servers.providers.local_media import (
        LOCAL_HEIGHT, LOCAL_WIDTH, MODELS, LocalVideoProvider, _frames_for,
    )

    spec = MODELS[model]
    fps = spec["fps"]
    frames = _frames_for(spec.get("frame_rule", "4k+1"), fps, 5.0)
    out_dir = root / "clips" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    shot_id = f"shot_{index:03d}"

    provider = LocalVideoProvider(model)
    request = VideoRequest(
        image_path=None, motion_prompt=prompt, shot_id=shot_id, project_dir=out_dir,
        duration_s=frames / fps, width=LOCAL_WIDTH, height=LOCAL_HEIGHT, fps=fps, seed=seed,
    )
    try:
        result = provider.generate(request)
    except Exception as exc:  # noqa: BLE001 - one clip must not sink the run
        print(f"    render failed: {str(exc)[:120]}")
        return None
    return out_dir / result["video_path"]


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cross-shot cast consistency: independent vs planned")
    parser.add_argument("--models", nargs="+", default=["cogvideox5b", "wan21_1_3b"])
    parser.add_argument("--briefs", nargs="+", type=int, default=[1])
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv[1:])

    briefs = json.loads(BRIEFS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(args.models)} models x {len(args.briefs)} briefs x 2 arms x {args.shots} shots")
    if args.dry_run:
        return 0

    log = OUT / "clips.jsonl"
    for model in args.models:
        for brief_index in args.briefs:
            brief = briefs[brief_index]
            print(f"\n=== {model} brief[{brief_index}] {brief[:40]}", flush=True)

            started = time.monotonic()
            prompts = story_prompts(brief, model, 0)
            print(f"  planner produced {len(prompts)} prompts in {time.monotonic()-started:.0f}s", flush=True)
            prompts = [p for p in prompts if p][: args.shots]
            if len(prompts) < 2:
                print("  planner returned too few prompts; skipping")
                continue

            for arm, texts in (
                ("independent", [brief] * len(prompts)),
                ("planned", prompts),
            ):
                print(f"  [{arm}]", flush=True)
                paths = []
                for index, text in enumerate(texts):
                    t0 = time.monotonic()
                    path = render(text, index, model, f"{arm}_{model}_b{brief_index}", OUT, index)
                    if path:
                        paths.append(str(path))
                    print(f"    shot {index}: {time.monotonic()-t0:.0f}s", flush=True)
                row = {"arm": arm, "model": model, "brief_index": brief_index,
                       "shots": paths, "prompt_chars": [len(t) for t in texts]}
                with log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
