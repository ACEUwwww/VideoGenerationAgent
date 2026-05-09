from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.schemas import ProjectCreate
from backend.services import agent_runner, event_store, project_store

app = FastAPI(title="Lightweight Video Generation Agent Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/projects")
def create_project(payload: ProjectCreate) -> dict:
    project = project_store.create_project(payload.user_prompt, payload.style, payload.shot_count)
    return project_store.get_project(project["project_id"])


@app.post("/api/projects/{project_id}/run")
async def run_project(project_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    try:
        project = project_store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    if project["state"].get("status") == "running":
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
