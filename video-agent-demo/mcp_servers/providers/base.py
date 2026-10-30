"""Provider contracts for media generation.

The agent's tools layer calls these interfaces; swapping a mock for a real
vendor means implementing one class and naming it in .env. Return values are
paths relative to the project directory, which is what the frontend serves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str
    shot_id: str
    project_dir: Path
    width: int
    height: int
    seed: int = 0
    caption: str = ""
    # Keyframe-memory anchor: when set, the model is conditioned on this image
    # so characters keep their face, wardrobe and colour grade across shots.
    reference_image: Path | None = None


@dataclass
class VideoRequest:
    image_path: Path
    motion_prompt: str
    shot_id: str
    project_dir: Path
    duration_s: float
    width: int
    height: int
    fps: int
    seed: int = 0
    caption: str = ""


@dataclass
class SpeechRequest:
    text: str
    speaker: str
    project_dir: Path
    shot_id: str
    duration_s: float
    voice: str = ""


@dataclass
class ImageResult:
    image_path: str
    provider: str
    prompt: str
    seed: int = 0
    reference: str = ""


class ImageProvider(Protocol):
    name: str

    def generate(self, request: ImageRequest) -> dict: ...


class VideoProvider(Protocol):
    name: str

    def generate(self, request: VideoRequest) -> dict: ...


class SpeechProvider(Protocol):
    name: str

    def synthesize(self, request: SpeechRequest) -> dict: ...
