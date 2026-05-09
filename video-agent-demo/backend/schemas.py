from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    user_prompt: str = Field(min_length=1)
    style: str = "cinematic"
    shot_count: int = Field(default=3, ge=1, le=8)


class ProjectResponse(BaseModel):
    project_id: str
    status: str
    user_prompt: str
    style: str
    shot_count: int
