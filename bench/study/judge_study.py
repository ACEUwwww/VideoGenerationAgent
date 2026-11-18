"""Score the study's clips with the local judge, one dimension at a time.

Separate from ``bench.runner``'s judge stage for two reasons that matter to how
the result is read:

* the runner refuses to judge without ``BENCH_ALLOW_PAID`` because its judge can
  be a hosted paid model. This study's judge is local (``qwen2.5-vl-32b`` on
  127.0.0.1), so scoring costs nothing and must not require a paid gate that
  would imply otherwise;
* the study has to be able to score *some* models while others still render. A
  dimension that took only the finished models is recorded with the models it
  actually covered, so a partial pass can never be mistaken for a full one.

Every dimension is written to ``judge/<dimension>.json`` in the same shape the
runner uses, so ``report_study.py`` and the existing report code read it
unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

from bench.config import ALL_DIMENSIONS, JudgeConfig  # noqa: E402
from bench.frames import build_payload  # noqa: E402
from bench.judge import Judge  # noqa: E402


def local_judge(model: str = "qwen2.5-vl-32b-local", base: str = "http://127.0.0.1:8791/v1") -> Judge:
    config = JudgeConfig(model=model, base_url=base, api_key="local", request_timeout_s=1800.0)
    return Judge(config)


def clips_for(run: Path, models: list[str]) -> dict[tuple[str, int], dict[str, Path]]:
    """{prompt_slug: {model: path}} for every clip on disk.

    Grouped by prompt *and* seed, because the same prompt at two seeds is two
    different videos; the dynamic dimensions compare models inside one call and
    must see the same seed group.
    """
    groups: dict[tuple[str, int], dict[str, Path]] = {}
    for model in models:
        for path in sorted((run / "videos" / model).glob("*.mp4")):
            stem = path.stem
            try:
                seed = int(stem.rsplit("-", 1)[-1])
                slug = stem.rsplit("-", 1)[0]
            except ValueError:
                continue
            groups.setdefault((slug, seed), {})[model] = path
    return groups


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score the study clips with the local judge")
    parser.add_argument("--run", type=Path, default=ROOT / "bench/results/study")
    parser.add_argument("--models", nargs="+", default=["cogvideox5b", "wan21_1_3b", "longlive_1_3b"])
    parser.add_argument("--dimensions", nargs="+", default=[
        "imaging_quality", "aesthetic_quality", "temporal_consistency", "motion_effects"])
    parser.add_argument("--limit", type=int, default=0, help="only the first N prompt groups (0 = all)")
    parser.add_argument("--judge-model", default="qwen2.5-vl-32b-local")
    parser.add_argument("--judge-base", default="http://127.0.0.1:8791/v1")
    args = parser.parse_args(argv[1:])

    unknown = [d for d in args.dimensions if d not in ALL_DIMENSIONS]
    if unknown:
        raise SystemExit(f"unknown dimensions {unknown}")

    groups = clips_for(args.run, args.models)
    keys = sorted(groups)
    if args.limit:
        keys = keys[: args.limit]
    if not keys:
        raise SystemExit("no clips to judge")
    usable = [k for k in keys if len(groups[k]) >= 2]
    print(f"{len(keys)} prompt/seed groups, {len(usable)} with >=2 models", flush=True)

    judge = local_judge(args.judge_model, args.judge_base)
    out_dir = args.run / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dimension in args.dimensions:
        cache = out_dir / f"{dimension}.json"
        if cache.exists():
            print(f"  reuse {cache.name}", flush=True)
            continue
        print(f"  judge {dimension} ...", flush=True)
        started = time.monotonic()
        scores: dict[str, list[float]] = {}
        per_group: dict[str, dict] = {}
        errors: list[str] = []
        covered: set[str] = set()

        # Resume from a partial pass. Scoring 240 group-dimensions takes hours
        # at ~40s each, so an interruption must not throw the finished work
        # away. Partial progress goes to a `.partial` file that the cache check
        # never reads, and only the completed pass is promoted to the real name
        # — otherwise a half-scored dimension would look like a finished one to
        # every later run and to the report.
        partial = out_dir / f"{dimension}.partial.json"
        if partial.exists():
            saved = json.loads(partial.read_text(encoding="utf-8"))
            per_group = saved.get("per_video") or {}
            scores = {m: [float(x) for x in v] for m, v in (saved.get("samples") or {}).items()}
            errors = [saved["error"]] if saved.get("error") else []
            covered = set(saved.get("models_covered") or [])
            print(f"  resume {dimension}: {len(per_group)} groups already scored", flush=True)

        def _save_partial() -> None:
            partial.write_text(json.dumps({
                "dimension": dimension,
                "scores": {m: (sum(v) / len(v) if v else None) for m, v in scores.items()},
                "samples": scores, "models_covered": sorted(covered),
                "per_video": per_group, "usage": dict(judge.usage),
                "elapsed_s": round(time.monotonic() - started, 1),
                "error": "; ".join(errors)[:2000], "partial": True,
                "judge": {"model": judge.config.model, "base_url": judge.config.base_url},
            }, indent=2, ensure_ascii=False), encoding="utf-8")

        for slug, seed in keys:
            group_key = f"{slug}|seed={seed}"
            if group_key in per_group:
                continue
            per_model = groups[(slug, seed)]
            payloads = {m: build_payload(p, judge.config) for m, p in per_model.items()}
            # The video-text dimensions need the prompt text; the four scored
            # here do not, so the slug is what travels with the group.
            reply = judge.score(dimension, payloads, slug.replace("_", " "))
            if reply.error:
                errors.append(f"{slug} seed={seed}: {reply.error}")
            per_group[group_key] = {
                "scores": reply.scores, "error": reply.error,
                "raw": {k: v[-300:] for k, v in reply.raw.items()},
            }
            for model, score in reply.scores.items():
                covered.add(model)
                if score is not None:
                    scores.setdefault(model, []).append(float(score))
            print(f"    {slug[:32]:34s} seed={seed} {reply.scores}", flush=True)
            _save_partial()

        payload = {
            "dimension": dimension,
            "scores": {m: (sum(v) / len(v) if v else None) for m, v in scores.items()},
            "samples": scores,
            # Which models this dimension actually covered, so a partial pass
            # cannot be read as a complete one.
            "models_covered": sorted(covered),
            "per_video": per_group,
            "groups_scored": len(per_group),
            "groups_expected": len(keys),
            "usage": dict(judge.usage),
            "elapsed_s": round(time.monotonic() - started, 1),
            "error": "; ".join(errors)[:2000],
            "judge": {"model": judge.config.model, "base_url": judge.config.base_url},
        }
        cache.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        partial.unlink(missing_ok=True)
        print(f"  {dimension}: {payload['scores']}"
              + (f"  [{payload['error'][:80]}]" if errors else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
