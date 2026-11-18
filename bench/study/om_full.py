"""Drive OpenMontage's full workflow as a benchmark baseline, stage by stage.

Why not its own ``auto-produce`` entry point, which looks like the obvious way in:

measured on a probe run (2026-11-17), ``POST /api/auto-produce`` advances through
script and entities and then loses its own job record at the storyboard step —
``GET .../auto-produce`` returns ``job: null`` and ``project.autoProduce`` is
absent from the state file — while a *stale* copy of the state is written back
over the entity work (the log reported ``素材图达标 10/10`` but 9 of 10 entities
had lost ``selectedImageId``). A run driven through it can neither be watched nor
resumed. That is a defect in the vendor's orchestration, not in this harness, so
it is reported rather than patched: the vendored checkout stays untouched.

Driving the documented routes in order reaches the same workflow and is
inspectable at every step:

    create -> script -> confirm -> entity stills -> storyboard
           -> batch-video -> audio -> edit ensure -> export

``confirm`` is the step the previous driver omitted and the reason its storyboard
call returned HTTP 500 "请先定稿本集再拆分镜"; ``generate_storyboard`` refuses to
run until the episode's status is ``confirmed``.

Scope is held comparable to ``ours`` rather than to a series: one episode, and
``--shots`` is recorded next to each result so the report can state what each
side was actually asked for.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
OM = ROOT / "bench/_vendor/OpenMontage"
BASE = "http://127.0.0.1:8787"
TOKEN_FILE = OM / "bench_run" / ".token"
PROJECTS = OM / "bench_run" / "projects"
OUT_ROOT = ROOT / "bench/results/om_full"


@dataclass
class Stage:
    name: str
    ok: bool
    seconds: float
    detail: str = ""


@dataclass
class EpisodeResult:
    brief: str
    seed: int
    project_id: str
    ok: bool
    export_path: str = ""
    stages: list[dict] = field(default_factory=list)
    shots: int = 0
    shot_seconds: int = 0
    entity_stills: int = 0
    total_seconds: float = 0.0
    error: str = ""
    notes: str = ""


def _token() -> str:
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


def call(method: str, path: str, body: dict | None = None, *, timeout: int = 1800) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_token()}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail[:400]}") from exc


def state(project_id: str) -> dict:
    """Read the project off disk, not through the API's bundle.

    The bundle is fine for the fields the studio exposes, but the run needs the
    per-shot video status and entity selection, which only the state file holds.
    """
    path = PROJECTS / project_id / "studio_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def wait_shots(project_id: str, shot_ids: list[str], *, timeout_s: float = 3600,
               poll_s: float = 15.0) -> list[dict]:
    """Poll until every named shot reaches a terminal status."""
    started = time.monotonic()
    done = {"success", "failed", "error", "cancelled"}
    while time.monotonic() - started < timeout_s:
        rows = state(project_id).get("shots") or []
        selected = [s for s in rows if s.get("id") in set(shot_ids)]
        if selected and all(str(s.get("status")) in done for s in selected):
            return selected
        time.sleep(poll_s)
    return [s for s in (state(project_id).get("shots") or []) if s.get("id") in set(shot_ids)]


def _stage(results: list[Stage], name: str, fn):
    """Run one stage, record it, and let a failure stop the episode.

    Unlike the vendor runner this does *not* swallow the error: a stage that
    fails means the episode is not a comparable artefact, and reporting it as
    "done with a missing step" is how the earlier baseline ended up claiming a
    workflow it never ran.
    """
    print(f"    {name} ...", flush=True)
    started = time.monotonic()
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        results.append(Stage(name, False, round(time.monotonic() - started, 1), str(exc)[:400]))
        print(f"    {name} FAILED: {str(exc)[:200]}", flush=True)
        raise
    results.append(Stage(name, True, round(time.monotonic() - started, 1)))
    return value


def recover_stuck_shots(project_id: str, shots: list[dict]) -> list[dict]:
    """Re-fetch any shot whose Ark task finished while the studio kept polling.

    Measured on the first full run: seven shots sat at ``generating`` for
    400-650s while their tasks had already returned ``succeeded`` at Ark. The
    studio's poller had simply stopped following them. ``sync-video`` is the
    studio's own recovery route, so nothing is invented here — but without it
    the episode never completes and the run pays for clips it cannot use.
    """
    stuck = [s for s in shots if str(s.get("status")) in {"generating", "pending"}
             and s.get("providerTaskId")]
    for shot in stuck:
        try:
            call("POST", f"/api/projects/{project_id}/storyboard",
                 {"action": "sync-video", "shotId": shot["id"]}, timeout=300)
        except Exception as exc:  # noqa: BLE001 - recovery is best effort
            print(f"    sync-video failed for {shot['id']}: {str(exc)[:100]}", flush=True)
    return [s for s in state(project_id).get("shots") or [] if s.get("id") in {x["id"] for x in shots}]


def rebuild_timeline(project_id: str, episode_id: str) -> dict:
    """Rewrite the edit timeline from every shot that has a selected take.

    The studio's ``ensure`` only builds a timeline once, from the shots that
    already had video at that moment, and never adds later ones. On the measured
    run that turned 15 rendered shots into a 4-clip export.
    """
    rows = [s for s in (state(project_id).get("shots") or [])
            if s.get("episodeId") == episode_id and not s.get("deletedAt")
            and s.get("selectedVideoId")]
    rows.sort(key=lambda s: int(s.get("shotOrder") or 0))
    clips = []
    for shot in rows:
        take = next((v for v in shot.get("videos") or [] if v["id"] == shot["selectedVideoId"]), None)
        clips.append({
            "id": f"clip_{shot['id'][-10:]}",
            "shotId": shot["id"],
            "enabled": True,
            "start": 0,
            "end": float((take or {}).get("duration") or shot.get("duration") or 5),
        })
    print(f"    timeline: {len(clips)} clips / {sum(c['end'] for c in clips):.1f}s", flush=True)
    return call("POST", f"/api/projects/{project_id}/edits",
                {"action": "save", "episodeId": episode_id, "clips": clips}, timeout=300)


def resume_project(project_id: str, *, resolution: str = "480p") -> EpisodeResult:
    """Finish an episode whose rendering already started, without redoing stages.

    Needed because ``batch-video`` submits shots one at a time and a submit for
    13 shots takes ~9 minutes; the HTTP call timed out mid-way while the studio
    kept accepting shots. Re-running ``run_episode`` would create a *second*
    project and pay for every keyframe and clip again, so this resumes the
    existing one: sync whatever Ark already finished, submit only shots that
    never got a task id, wait, then rebuild the timeline and export.
    """
    started = time.monotonic()
    stages: list[Stage] = []
    # Recover the seed from the project name (benchs2 -> seed 2). A resumed row
    # is the same episode the batch would have written; leaving seed at -1 puts
    # "-1" in the report's repeat column for a perfectly real episode.
    tail = project_id.rsplit("-", 1)[-1]
    result = EpisodeResult(brief="", seed=int(tail) if tail.isdigit() else -1,
                           project_id=project_id, ok=False)
    try:
        snapshot = state(project_id)
        episode_id = (snapshot.get("episodes") or [{}])[0].get("id", "")
        shots = [s for s in (snapshot.get("shots") or []) if not s.get("deletedAt")]
        shot_ids = [s["id"] for s in shots]
        result.shots = len(shots)
        result.shot_seconds = sum(int(s.get("duration") or 0) for s in shots)

        finished = _stage(stages, "sync", lambda: recover_stuck_shots(project_id, shots))
        todo = [s for s in (state(project_id).get("shots") or [])
                if not s.get("deletedAt") and str(s.get("status")) == "pending"
                and not s.get("providerTaskId")]
        if todo:
            print(f"    submitting {len(todo)} unsubmitted shots ...", flush=True)
            for index, shot in enumerate(todo, start=1):
                try:
                    call("POST", f"/api/projects/{project_id}/storyboard",
                         {"action": "video", "shotId": shot["id"], "resolution": resolution,
                          "forceEntityQa": True}, timeout=600)
                except Exception as exc:  # noqa: BLE001 - one shot must not sink the episode
                    print(f"    shot {shot.get('shotOrder')} submit failed: {str(exc)[:100]}", flush=True)
                print(f"      {index}/{len(todo)}", flush=True)

        finished = wait_shots(project_id, shot_ids, timeout_s=5400)
        finished = recover_stuck_shots(project_id, finished)
        good = [s for s in finished if str(s.get("status")) == "success"]
        print(f"    shots success {len(good)}/{len(shot_ids)}", flush=True)
        if not good:
            raise RuntimeError("no shot reached success")

        _stage(stages, "edit_ensure", lambda: call("POST", f"/api/projects/{project_id}/edits",
                                                  {"action": "ensure", "episodeId": episode_id},
                                                  timeout=300))
        _stage(stages, "edit_rebuild", lambda: rebuild_timeline(project_id, episode_id))
        exported = _stage(stages, "export", lambda: call("POST", f"/api/projects/{project_id}/edits",
                                                        {"action": "export", "episodeId": episode_id},
                                                        timeout=1800))
        result.export_path = str(exported.get("path") or exported.get("relPath") or "")
        result.ok = bool(result.export_path)
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)[:500]

    result.stages = [s.__dict__ for s in stages]
    result.total_seconds = round(time.monotonic() - started, 1)
    result.notes = f"resumed resolution={resolution}"
    return result


def run_episode(brief: str, seed: int, *, resolution: str = "480p",
                max_shots: int = 15) -> EpisodeResult:
    started = time.monotonic()
    stages: list[Stage] = []
    result = EpisodeResult(brief=brief, seed=seed, project_id="", ok=False)

    created = _stage(stages, "create", lambda: call("POST", "/api/projects", {
        "title": f"bench_s{seed}",
        "brief": brief,
        "genre": "悬疑",
        "ratio": "9:16",
        "visualStyle": "当代都市夜戏，冷调霓虹，35mm 胶片质感",
    }, timeout=300))
    pid = created["project"]["id"]
    result.project_id = pid
    print(f"    project={pid}", flush=True)

    try:
        _stage(stages, "script", lambda: call("POST", f"/api/projects/{pid}/script",
                                             {"action": "generate", "episodeCount": 1}, timeout=1800))
        bundle = call("GET", f"/api/projects/{pid}")
        episodes = [e for e in (bundle.get("episodes") or []) if not e.get("deletedAt")]
        if not episodes:
            raise RuntimeError("no episode produced by the script stage")
        episode_id = episodes[0]["id"]

        # The step the previous driver omitted. Without it generate_storyboard
        # raises 请先定稿本集再拆分镜 and the whole baseline collapses to zero
        # shots, which is exactly how openmontage_shot ended up comparing a
        # shot tool instead of a workflow.
        _stage(stages, "confirm", lambda: call("PATCH", f"/api/projects/{pid}/episodes",
                                              {"status": "confirmed", "episodeIds": [episode_id]},
                                              timeout=120))

        entities = [e for e in (state(pid).get("entities") or []) if not e.get("deletedAt")]
        print(f"    entities={len(entities)}", flush=True)
        for index, entity in enumerate(entities, start=1):
            name = entity.get("name") or entity.get("id")
            _stage(stages, f"entity[{index}/{len(entities)}]{name}", lambda e=entity: call(
                "POST", f"/api/projects/{pid}/entities",
                {"action": "generate", "entityId": e["id"], "wait": True, "force": True},
                timeout=900,
            ))
        result.entity_stills = sum(
            1 for e in (state(pid).get("entities") or []) if e.get("selectedImageId")
        )

        shots = _stage(stages, "storyboard", lambda: call("POST", f"/api/projects/{pid}/storyboard",
                                                         {"action": "generate", "episodeId": episode_id},
                                                         timeout=1800))["shots"]
        # The studio plans 12-30 shots over 150-360s by default; measured once at
        # 56 shots / 176s for a one-line brief. Ours is 15 shots / 60s, so the
        # untrimmed baseline would be ~4x longer and cost ~4x more, and the
        # comparison would be a duration difference rather than a workflow one.
        # Trim to the requested count, keeping the studio's own shot order.
        if max_shots and len(shots) > max_shots:
            ordered = sorted(shots, key=lambda s: int(s.get("shotOrder") or 0))
            for extra in ordered[max_shots:]:
                try:
                    call("POST", f"/api/projects/{pid}/storyboard",
                         {"action": "delete", "shotId": extra["id"]}, timeout=120)
                except Exception as exc:  # noqa: BLE001 - trimming is best effort
                    print(f"    trim failed for {extra['id']}: {str(exc)[:80]}", flush=True)
            shots = ordered[:max_shots]
            print(f"    trimmed to {len(shots)} shots "
                  f"({sum(int(s.get('duration') or 0) for s in shots)}s)", flush=True)
        shot_ids = [s["id"] for s in shots]
        result.shots = len(shots)
        result.shot_seconds = sum(int(s.get("duration") or 0) for s in shots)
        print(f"    shots={result.shots} planned={result.shot_seconds}s", flush=True)

        # One generation per shot, not per segment. A segment is 10-15s and the
        # studio renders all its beats in a single Seedance call, which would
        # make each clip ~3x ours and the comparison a duration difference.
        started = _stage(stages, "video", lambda: call("POST", f"/api/projects/{pid}/storyboard", {
            "action": "batch-video",
            "episodeId": episode_id,
            "shotIds": shot_ids,
            "resolution": resolution,
            # The studio blocks rendering when its own still-image QA dislikes
            # a character sheet. That check runs a *different* model over the
            # reference art, and it fires on wording rather than on anything a
            # viewer sees; leaving it on would mean this baseline sometimes
            # renders nothing at all. The override is the studio's own
            # documented escape hatch.
            "forceEntityQa": True,
        }, timeout=600))

        # ``batch-video`` catches per-shot exceptions and still answers HTTP
        # 200, putting the failure inside ``started``. Measured on episode 2:
        # two of fifteen shots came back with no task id and no error surfaced
        # anywhere, so the shot count looked complete while two shots had never
        # been submitted. Reading ``started`` is the only way to see it.
        rejected = [row for row in (started.get("started") or []) if row.get("error")]
        if rejected:
            raise RuntimeError(f"studio refused {len(rejected)}/{len(shot_ids)} shots: "
                               f"{rejected[0].get('error')}")

        finished = wait_shots(pid, shot_ids)
        finished = recover_stuck_shots(pid, finished)
        good = [s for s in finished if str(s.get("status")) == "success"]
        print(f"    shots rendered {len(good)}/{len(shot_ids)}", flush=True)
        if not good:
            raise RuntimeError("no shot reached success")

        _stage(stages, "edit_ensure", lambda: call("POST", f"/api/projects/{pid}/edits",
                                                  {"action": "ensure", "episodeId": episode_id},
                                                  timeout=300))
        # Voice is optional in the vendor flow (edge-tts needs egress); it is
        # attempted and recorded so its absence is visible rather than assumed.
        try:
            _stage(stages, "audio", lambda: call("POST", f"/api/projects/{pid}/audio",
                                                {"action": "generate", "episodeId": episode_id},
                                                timeout=1800))
        except Exception as exc:  # noqa: BLE001
            print(f"    audio unavailable: {str(exc)[:120]}", flush=True)

        # The studio builds the edit timeline from whatever shots had video at
        # the moment ensure ran, and never backfills it. Measured: 15 rendered
        # shots exported as a 4-clip film. Rebuilding the timeline from the
        # state's own shot list before exporting is what makes the delivered
        # file contain the work that was paid for.
        _stage(stages, "edit_rebuild", lambda: rebuild_timeline(pid, episode_id))
        exported = _stage(stages, "export", lambda: call("POST", f"/api/projects/{pid}/edits",
                                                       {"action": "export", "episodeId": episode_id},
                                                       timeout=1800))
        result.export_path = str(exported.get("path") or exported.get("relPath") or "")
        result.ok = bool(result.export_path)
    except Exception as exc:  # noqa: BLE001 - one episode must not sink the batch
        result.error = str(exc)[:500]

    result.stages = [s.__dict__ for s in stages]
    result.total_seconds = round(time.monotonic() - started, 1)
    result.notes = f"resolution={resolution}"
    return result


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="OpenMontage full workflow baseline")
    parser.add_argument("--briefs", type=Path, default=ROOT / "bench/study/briefs.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-project", default="",
                        help="finish an already-created project id instead of creating a new one")
    args = parser.parse_args(argv[1:])

    if args.resume_project:
        # Resuming must not create a second project: that would re-pay for every
        # keyframe and clip. It finishes the existing one and appends the result
        # so the batch keeps one row per episode.
        print(f"resume {args.resume_project}")
        row = resume_project(args.resume_project, resolution=args.resolution).__dict__
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = args.out / "episodes.json"
        results = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else []
        results = [r for r in results if r.get("project_id") != args.resume_project] + [row]
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> ok={row['ok']} shots={row['shots']} export={row['export_path'][:60]!r} "
              f"err={row['error'][:100]!r}")
        return 0

    briefs = json.loads(args.briefs.read_text(encoding="utf-8"))
    plan = [(b, s) for b in briefs for s in args.seeds]
    print(f"{len(briefs)} briefs x {len(args.seeds)} seeds = {len(plan)} episodes")
    if args.dry_run:
        for brief, seed in plan:
            print(f"  seed={seed} {brief[:60]}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    out_file = args.out / "episodes.json"
    # The plan is written before any work, so the report's denominator is the
    # number of episodes that were asked for rather than the number that happen
    # to have finished. Reading the row count instead made an in-progress batch
    # look like it had delivered everything it was given.
    (args.out / "plan.json").write_text(json.dumps(
        {"briefs": [b[:60] for b in briefs], "seeds": args.seeds,
         "episodes": [{"brief_index": briefs.index(b), "seed": s} for b, s in plan]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    for index, (brief, seed) in enumerate(plan, start=1):
        print(f"  [{index}/{len(plan)}] seed={seed} {brief[:50]!r}", flush=True)
        row = run_episode(brief, seed, resolution=args.resolution).__dict__
        results.append(row)
        print(f"  -> ok={row['ok']} shots={row['shots']} export={row['export_path'][:60]!r} "
              f"err={row['error'][:100]!r}", flush=True)
        # Written after every episode, so an interrupted batch keeps its evidence.
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
