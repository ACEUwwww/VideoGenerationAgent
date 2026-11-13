"""Run one benchmark: generate clips, judge them, write results.

The two stages are separate on purpose. Generation is expensive and slow (a 5B
diffusion model takes minutes per clip); judging is a paid API call per video
per dimension. Splitting them means a scoring mistake does not force a
re-render, and existing clips can be re-judged after a prompt or judge change.

Layout under ``bench/results/<run>/``::

    manifest.json      prompts, seeds, backends, hardware, config
    videos/<model>/<slug>-<seed>.mp4
    judge/<dimension>.json
    summary.json
    report.md
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .backends import Generation, describe_backends, get_backend
from .config import ALL_DIMENSIONS, BenchConfig, RESULT_ROOT, load_config
from .frames import FramePayload, build_payload
from .judge import Judge, JudgeReply
from .prompts import Prompt, load_prompt_file, select_prompts, write_manifest


def run_name(tag: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{tag}" if tag else stamp


def _hardware() -> dict:
    info: dict = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        info["gpu"] = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    except Exception:  # noqa: BLE001 - the report must survive a missing GPU
        info["gpu"] = []
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return info


def estimate(config: BenchConfig, prompts: int, models: int, seeds: int, dimensions: list[str]) -> dict:
    """Calls and rough cost before anything is spent.

    Counted from the actual call pattern, not a guess: the static dimensions
    make one call per model per video, the dynamic ones one call per video, and
    the video-text ones five calls per model per video (describe, two
    questions, answer, summarise).
    """
    from .config import DYNAMIC_DIMENSIONS, STATIC_DIMENSIONS, VIDEO_TEXT_DIMENSIONS

    videos = prompts * models * seeds
    per_dimension = {
        "static": models, "dynamic": 1, "video_text": 5 * models,
    }
    calls = 0
    for dimension in dimensions:
        if dimension in STATIC_DIMENSIONS:
            calls += per_dimension["static"]
        elif dimension in DYNAMIC_DIMENSIONS:
            calls += per_dimension["dynamic"]
        elif dimension in VIDEO_TEXT_DIMENSIONS:
            calls += per_dimension["video_text"]
    return {"videos": videos, "judge_calls": videos * calls, "per_video_calls": calls}


def generate(
    config: BenchConfig,
    models: list[str],
    prompts: list[Prompt],
    seeds: list[int],
    seconds: float,
    run_dir: Path,
) -> list[Generation]:
    """Render every (model, prompt, seed) clip that is not already on disk.

    Existing files are reused so an interrupted run resumes instead of paying
    for the same render twice.
    """
    made: list[Generation] = []
    for model in models:
        backend = get_backend(model, config)
        ok, detail = backend.available()
        if not ok:
            print(f"skip {model}: {detail}")
            continue
        for prompt in prompts:
            for seed in seeds:
                target = run_dir / "videos" / model / f"{prompt.slug}-{seed}.mp4"
                if target.exists() and target.stat().st_size > 1024:
                    print(f"  reuse {model} {prompt.slug} seed={seed}")
                    made.append(
                        Generation(
                            model=model, prompt=prompt.text, seed=seed, path=target,
                            seconds=seconds, width=0, height=0, fps=0,
                            modality=backend.modality, elapsed_s=0.0, note="reused",
                        )
                    )
                    continue
                print(f"  render {model} {prompt.slug} seed={seed} ...", flush=True)
                started = time.monotonic()
                try:
                    generation = backend.generate(prompt.text, seed, seconds, target.parent)
                except Exception as exc:  # noqa: BLE001 - one clip must not sink the run
                    print(f"  FAILED {model} {prompt.slug} seed={seed}: {exc}")
                    continue
                print(f"  done {model} {prompt.slug} seed={seed} in {generation.elapsed_s:.0f}s")
                made.append(generation)
    return made


def judge_all(
    config: BenchConfig,
    generations: list[Generation],
    dimensions: list[str],
    run_dir: Path,
) -> dict[str, JudgeReply]:
    """Score every clip on every dimension, caching each dimension's result.

    Grouped by ``(prompt, seed)``, not by prompt alone: a prompt rendered at two
    seeds is two different videos, and keying by prompt would keep only the last
    one while the budget still counted both. Each group holds every model, so the
    dynamic dimensions — which compare models inside a single call — see all of
    them together.
    """
    judge = Judge(config.judge)
    groups: dict[tuple[str, int], dict[str, Generation]] = {}
    for generation in generations:
        groups.setdefault((generation.prompt, generation.seed), {})[generation.model] = generation

    payloads: dict[tuple[str, int], dict[str, FramePayload]] = {
        key: {model: build_payload(gen.path, config.judge) for model, gen in per_model.items()}
        for key, per_model in groups.items()
    }

    replies: dict[str, JudgeReply] = {}
    out_dir = run_dir / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dimension in dimensions:
        cache = out_dir / f"{dimension}.json"
        if cache.exists():
            print(f"  reuse judge/{dimension}.json")
            payload = json.loads(cache.read_text(encoding="utf-8"))
            replies[dimension] = JudgeReply(
                dimension=dimension,
                scores=payload["scores"],
                raw={},
                usage=payload.get("usage", {}),
                elapsed_s=payload.get("elapsed_s", 0.0),
                error=payload.get("error", ""),
            )
            continue
        print(f"  judge {dimension} ...", flush=True)
        scores: dict[str, list[float]] = {}
        per_group: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        started = time.monotonic()
        for (prompt_text, seed), per_model in payloads.items():
            # The video-text dimensions score how well each clip matches the
            # text it was generated from, so the prompt travels with the call.
            reply = judge.score(dimension, per_model, prompt_text)
            if reply.error:
                errors.append(f"{prompt_text[:40]} seed={seed}: {reply.error}")
            per_group[f"{prompt_text} | seed={seed}"] = {
                "scores": reply.scores, "error": reply.error,
                # The raw replies are what a parse failure is diagnosed from.
                # Without them a None score is indistinguishable from a call
                # that never returned a number, and the run has to be paid for
                # again to find out. Truncated: a full reply is prose, and what
                # matters is whether it ended in a parseable line.
                "raw": {name: text[-400:] for name, text in reply.raw.items()},
            }
            for model, score in reply.scores.items():
                if score is not None:
                    scores.setdefault(model, []).append(float(score))
        combined = JudgeReply(
            dimension=dimension,
            scores={m: (sum(v) / len(v) if v else None) for m, v in scores.items()},
            raw={},
            usage=dict(judge.usage),
            elapsed_s=time.monotonic() - started,
            error="; ".join(errors),
        )
        cache.write_text(
            json.dumps(
                {
                    "dimension": dimension,
                    "scores": combined.scores,
                    "samples": scores,
                    "per_video": per_group,
                    "usage": combined.usage,
                    "elapsed_s": round(combined.elapsed_s, 1),
                    "error": combined.error,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        replies[dimension] = combined
        print(f"  {dimension}: {combined.scores}" + (f"  [{combined.error}]" if combined.error else ""))
    return replies


def write_run(
    config: BenchConfig,
    run_dir: Path,
    prompts: list[Prompt],
    models: list[str],
    seeds: list[int],
    seconds: float,
    dimensions: list[str],
    generations: list[Generation],
    replies: dict[str, JudgeReply],
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(prompts, run_dir / "manifest_prompts.json")
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "prompts": [p.to_dict() for p in prompts],
        "models": models,
        "backends": describe_backends(config),
        "seeds": seeds,
        "seconds": seconds,
        "dimensions": dimensions,
        "hardware": _hardware(),
        "judge": {
            "model": config.judge.model,
            "base_url": config.judge.base_url,
            "sample_fps": config.judge.sample_fps,
            "max_edge": config.judge.max_edge,
        },
        "vendor_commit": _vendor_commit(config),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    clips = [
        {
            "model": g.model, "prompt": g.prompt, "seed": g.seed,
            # A clip can live outside run_dir when a run reuses an earlier
            # run's videos (--run-dir) or when a backend writes elsewhere.
            "file": str(g.path.relative_to(run_dir)) if _inside(g.path, run_dir) else str(g.path),
            "seconds": g.seconds,
            "modality": g.modality, "elapsed_s": round(g.elapsed_s, 1), "note": g.note,
        }
        for g in generations
    ]
    (run_dir / "clips.json").write_text(json.dumps(clips, indent=2, ensure_ascii=False), encoding="utf-8")

    # Merge with whatever dimensions are already cached. A run is often judged
    # in several passes — one dimension re-scored after a fix, the rest left
    # alone — and writing only this pass's dimensions erased the others from
    # summary.json, so the report silently lost them while their per-dimension
    # cache files were still on disk.
    merged: dict[str, dict] = {}
    for cache_file in sorted((run_dir / "judge").glob("*.json")):
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        dimension = cached.get("dimension") or cache_file.stem
        merged[dimension] = {
            "scores": cached.get("scores", {}),
            "elapsed_s": cached.get("elapsed_s", 0.0),
            "error": cached.get("error", ""),
        }
    for dimension, reply in replies.items():
        merged[dimension] = {
            "scores": reply.scores,
            "elapsed_s": round(reply.elapsed_s, 1),
            "error": reply.error,
        }

    summary = {
        "run": run_dir.name,
        "dimensions": merged,
        "usage": replies[list(replies)[0]].usage if replies else {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_segments(generations: list[Generation], run_dir: Path, segment_seconds: float) -> dict:
    """Layers 1-3 for every clip, written as one JSON per run.

    No judge call: these are the deterministic cross-segment measurements that
    stay meaningful even when the paid dimensions are unavailable. Kept in its
    own file so a report can be regenerated from segments without re-decoding
    video.
    """
    from .segments import evaluate_clip, to_dict

    out = run_dir / "segments"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for generation in generations:
        if not generation.path.exists():
            continue
        report = evaluate_clip(generation.path, generation.model, segment_seconds)
        rows.append(to_dict(report))
        summary = report.summary
        print(
            f"  segment {generation.model:14s} {generation.path.name:28s} "
            f"{summary['segments']:3d} segs  identity_min={summary['identity_min']:.3f} "
            f"onset={summary['drift_onset_segment']}"
        )
    payload = {"segment_seconds": segment_seconds, "clips": rows}
    (out / "segments.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _vendor_commit(config: BenchConfig) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(config.vendor), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Video-Bench comparison for this pipeline")
    parser.add_argument("--models", nargs="+", default=["ours", "cogvideox5b", "wan21_1_3b"])
    parser.add_argument("--prompts", type=int, default=1, help="how many suite prompts; 0 = all")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--dimension-filter", default="", help="only prompts tagged with this dimension")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--dimensions", nargs="+", default=["imaging_quality", "temporal_consistency"])
    parser.add_argument("--tag", default="")
    parser.add_argument("--stage", choices=["all", "generate", "segment", "judge"], default="all")
    parser.add_argument("--segment-seconds", type=float, default=5.0,
                        help="fixed window size for the segmented evaluation")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--estimate", action="store_true", help="print the work budget and exit")
    args = parser.parse_args(argv)

    config = load_config()
    prompts = (
        load_prompt_file(args.prompt_file) if args.prompt_file
        else select_prompts(args.prompts, args.dimension_filter)
    )
    unknown = [d for d in args.dimensions if d not in ALL_DIMENSIONS]
    if unknown:
        raise SystemExit(f"unknown dimensions {unknown}; expected from {ALL_DIMENSIONS}")

    run_dir = args.run_dir or (RESULT_ROOT / run_name(args.tag))
    budget = estimate(config, len(prompts), len(args.models), len(args.seeds), args.dimensions)
    print(f"run: {run_dir}")
    print(f"models: {args.models}")
    print(f"prompts: {len(prompts)}  seeds: {args.seeds}  seconds: {args.seconds}")
    print(f"dimensions: {args.dimensions}")
    print(f"budget: {budget['videos']} videos, {budget['judge_calls']} judge calls")
    for row in describe_backends(config):
        print(f"  {row['name']:14s} {'OK' if row['available'] else 'unavailable':12s} {row['detail']}")
    if args.estimate:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    generations: list[Generation] = []
    if args.stage in ("all", "generate"):
        generations = generate(config, args.models, prompts, args.seeds, args.seconds, run_dir)
    if args.stage in ("all", "segment", "judge") and not generations:
        generations = _load_existing(run_dir, args.models)
    # Segmentation is local and free, so it runs before judging and regardless
    # of whether the run is allowed to spend money. It is also what makes a
    # long clip legible: one number for two minutes hides where it breaks.
    if args.stage in ("all", "segment"):
        if not generations:
            raise SystemExit("nothing to segment: no clips on disk")
        write_segments(generations, run_dir, args.segment_seconds)
    if args.stage in ("all", "judge"):
        if not generations:
            raise SystemExit("nothing to judge: no clips on disk")
        if not config.allow_paid:
            raise SystemExit(
                "judging costs money; set BENCH_ALLOW_PAID=1 and ARK_ALLOW_PAID=1 to proceed"
            )
        replies = judge_all(config, generations, args.dimensions, run_dir)
        summary = write_run(config, run_dir, prompts, args.models, args.seeds, args.seconds,
                            args.dimensions, generations, replies)
        from .report import write_report

        write_report(run_dir, summary, config)
        print(f"\nreport: {run_dir / 'report.md'}")
    return 0


def _load_existing(run_dir: Path, models: list[str]) -> list[Generation]:
    """Clips for this run, from the manifest if present, else from the videos tree.

    ``clips.json`` is only written by the judge stage, so a run that was
    generated but never judged had no record of what it produced and a later
    ``--stage segment`` reported "no clips on disk" while two mp4 files sat
    there. Falling back to the directory means any stage can pick up a run that
    was left part-finished.
    """
    # The manifest carries the prompts the run was started with, keyed by slug.
    # Both branches below need it: judge_all groups clips by (prompt, seed), so
    # four models carrying four different "prompts" collapse into four
    # single-model groups and the dynamic dimensions — the ones that compare
    # models inside one call — silently become per-model judgements. That
    # happened, and it is why a stale clips.json cannot be trusted for the text.
    manifest_prompts: list[dict] = []
    manifest_path = run_dir / "manifest_prompts.json"
    if manifest_path.exists():
        manifest_prompts = json.loads(manifest_path.read_text(encoding="utf-8"))

    def _true_prompt(slug: str) -> str:
        return next(
            (row["text"] for row in manifest_prompts if row.get("slug") == slug),
            slug,
        )

    clips = run_dir / "clips.json"
    if clips.exists():
        out = []
        for row in json.loads(clips.read_text(encoding="utf-8")):
            if row["model"] not in models:
                continue
            stored = row["prompt"]
            out.append(Generation(
                model=row["model"], prompt=_true_prompt(stored), seed=row["seed"],
                path=run_dir / row["file"], seconds=row["seconds"], width=0, height=0, fps=0,
                modality=row.get("modality", ""), elapsed_s=row.get("elapsed_s", 0.0),
                note="reused",
            ))
        return out

    # The manifest carries the prompts the run was started with. Without it the
    # discovered prompt has to come from the filename, which holds a slug, not
    # the text — and a wrong prompt is not cosmetic: judge_all groups clips by
    # (prompt, seed), so four models with four different "prompts" collapse into
    # four single-model groups and the dynamic dimensions stop comparing them.
    # That happened: temporal_consistency was scored in four separate calls.
    manifest_prompts: list[dict] = []
    manifest_path = run_dir / "manifest_prompts.json"
    if manifest_path.exists():
        manifest_prompts = json.loads(manifest_path.read_text(encoding="utf-8"))

    out = []
    for model in models:
        for path in sorted((run_dir / "videos" / model).glob("*.mp4")):
            # Filenames are `<slug>-<seed>.mp4`; the seed is what varies within
            # one prompt, and it is the only part a resumed run needs.
            try:
                seed = int(path.stem.rsplit("-", 1)[-1])
            except ValueError:
                seed = 0
            slug = path.stem.rsplit("-", 1)[0]
            prompt = next(
                (row["text"] for row in manifest_prompts if row.get("slug") == slug),
                slug,
            )
            out.append(Generation(
                model=model, prompt=prompt, seed=seed,
                path=path, seconds=0.0, width=0, height=0, fps=0,
                modality="", elapsed_s=0.0, note="discovered",
            ))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
