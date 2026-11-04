from __future__ import annotations

import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas import ProjectCreate
from agent.settings import get_settings
from backend.services import agent_runner, event_store, project_store
from mcp_servers.videobench_eval import DIMENSIONS

app = FastAPI(title="Lightweight Video Generation Agent Demo")

# Same-origin serving needs no CORS at all; the list exists only for the
# separate Vite dev server. CORS_ORIGINS adds hosts when the UI is opened from
# another machine's address rather than through the API port itself.
_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_ORIGINS += [origin for origin in os.environ.get("CORS_ORIGINS", "").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    from mcp_servers.providers import provider_report

    settings = get_settings()
    return {
        "providers": provider_report(),
        "llm_model": settings.llm.model,
        "canvas": {"width": settings.media.width, "height": settings.media.height, "fps": settings.media.fps},
        "video_bench_dimensions": list(DIMENSIONS),
    }


@app.get("/api/projects")
def list_projects(limit: int = 30) -> dict:
    return {"projects": project_store.list_projects(limit)}


@app.post("/api/projects")
def create_project(payload: ProjectCreate) -> dict:
    project = project_store.create_project(
        payload.user_prompt,
        payload.style,
        payload.shot_count,
        script=payload.script,
        genre=payload.genre,
    )
    return project_store.get_project(project["project_id"])


@app.post("/api/projects/{project_id}/run")
async def run_project(project_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    try:
        project = project_store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    if not project_store.claim_run(project_id):
        return {"status": "already_running"}

    background_tasks.add_task(agent_runner.run_project_agent, project_id)
    return {"status": "started"}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    try:
        return project_store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@app.get("/api/projects/{project_id}/events")
def get_events(project_id: str, since: int = 0) -> dict:
    try:
        project_store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    return event_store.list_events(project_id, since)


@app.get("/api/projects/{project_id}/assets/{filename:path}")
def get_asset(project_id: str, filename: str) -> FileResponse:
    root = project_store.project_dir(project_id).resolve()
    path = (root / Path(filename)).resolve()
    if not str(path).startswith(str(root)) or not path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)


# The built frontend is mounted last so /api/* and /health keep priority over
# the catch-all. Serving it here means one exposed port covers the UI and the
# API on the same origin, which also removes CORS from the picture.
_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="ui")
