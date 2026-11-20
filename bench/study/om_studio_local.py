"""Launch the Drama Studio with a local open-weight model registered as a backend.

Same contract as ``om_studio.sh`` — same ports, same judge, same "no
Postgres/Redis/MinIO" posture — with one addition: before the server starts,
``om_local_backend.register()`` puts a ``local`` video backend into the studio's
own registries. Nothing under ``bench/_vendor/`` is edited; the registration
uses the public ``VIDEO_BACKENDS.register`` / ``register_video_job`` calls.

How the injection reaches the server process without touching its code: this
module is imported by ``serve.py``'s process via ``PYTHONSTARTUP``-style entry
below, i.e. the launcher runs this file, which registers the backend and then
executes the vendor's ``serve.py`` in the same interpreter.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
OM = ROOT / "bench/_vendor/OpenMontage"
RUN = OM / "bench_run"


def _env_setup() -> None:
    run_dir = RUN
    run_dir.joinpath("projects").mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("bench/logs").mkdir(parents=True, exist_ok=True)

    os.environ.update(
        PROJECTS_ROOT=str(run_dir / "projects"),
        OPENMONTAGE_ROOT=str(OM),
        STUDIO_DIST=str(OM / "studio" / "dist"),
        HOST="127.0.0.1",
        PORT="8787",
        STUDIO_AUTH_REQUIRED="true",
        STUDIO_DEMO_ENABLED="true",
        STUDIO_DEMO_AUTO_ENTER="false",
        STUDIO_PUBLIC_URL="http://127.0.0.1:8787",
        STUDIO_CORS_ORIGINS="http://127.0.0.1:8787",
        STUDIO_MEDIA_WORKERS="1",
        STUDIO_VIDEO_PARALLEL="1",
        # The local model is the only generator in this configuration.
        STUDIO_VIDEO_BACKEND="local",
        STUDIO_IMAGE_BACKEND="local_diffusion",
        STUDIO_FREE_FIRST="0",
        # One QA model for both workflows, on the port that is actually serving.
        OPENAI_API_KEY="local",
        OPENAI_BASE_URL=os.environ.get("BENCH_JUDGE_BASE_URL", "http://127.0.0.1:8791/v1"),
        OPENAI_MODEL=os.environ.get("BENCH_JUDGE_MODEL", "qwen2.5-vl-32b-local"),
        STUDIO_VIDEO_QA_MAX_ATTEMPTS="2",
        STUDIO_ENTITY_QA_MAX_ATTEMPTS="2",
        STUDIO_REQUIRE_GITHUB_STAR="0",
    )
    os.environ.setdefault("LOCAL_VIDEO_MODEL", "cogvideox5b_i2v")
    # Planning must run on the SAME model our side plans with, or the comparison
    # measures planner capability instead of orchestration. Our pipeline plans
    # with DeepSeek, so this side does too. The QA judge stays on the local VLM
    # above: `_vision_endpoints` tries OPENAI_* first, and both workflows are
    # scored by that same judge.
    _dotenv = {}
    try:
        from dotenv import dotenv_values

        _dotenv = dotenv_values(str(ROOT / ".env"))
    except Exception:
        _dotenv = {}
    if _dotenv.get("DEEPSEEK_API_KEY"):
        os.environ.setdefault("DEEPSEEK_API_KEY", _dotenv["DEEPSEEK_API_KEY"])
        os.environ.setdefault("LLM_PROVIDER", "deepseek")
        os.environ.setdefault("LLM_MODEL", _dotenv.get("LLM_MODEL") or "deepseek-chat")
    else:
        # Without the shared planner key the comparison would silently run a
        # different planner on this side. Say so rather than proceed quietly.
        print("[om_local] WARNING: no DEEPSEEK_API_KEY; OpenMontage will plan with a "
              "different model than our pipeline", flush=True)
    # Store on disk and render in-process: no Postgres, no Redis, no MinIO.
    for key in ("DATABASE_URL", "STUDIO_DATABASE_URL", "REDIS_URL", "STUDIO_REDIS_URL"):
        os.environ.pop(key, None)


def main() -> int:
    _env_setup()
    for path in (str(ROOT / "bench/study"), str(ROOT), str(OM), str(OM / "studio" / "server")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import om_local_backend

    om_local_backend.register()
    print(f"[om_local] registered backend 'local' with model {os.environ['LOCAL_VIDEO_MODEL']}", flush=True)

    os.chdir(OM)
    runpy.run_path(str(RUN / "serve.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
