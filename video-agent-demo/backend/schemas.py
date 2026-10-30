from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    user_prompt: str = Field(min_length=1)
    script: str = ""
    genre: str = ""
    style: str = "电影感竖屏短剧"
    shot_count: int = Field(default=4, ge=1, le=12)


class ProjectResponse(BaseModel):
    project_id: str
    status: str
    user_prompt: str
    style: str
    shot_count: int
