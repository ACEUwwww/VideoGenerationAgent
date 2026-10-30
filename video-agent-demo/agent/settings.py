"""Runtime configuration.

Every provider choice is explicit: there is no silent fallback from a real
API to a mock. Missing credentials raise at the point of use so a half
configured run never produces mock media that looks real.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv is optional at runtime
    pass


DEMO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = DEMO_ROOT / "workspace"
PROJECTS_ROOT = WORKSPACE_ROOT / "projects"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    reasoner_model: str
    temperature: float
    max_tokens: int
    timeout_s: float


@dataclass(frozen=True)
class MediaSettings:
    image_provider: str
    video_provider: str
    tts_provider: str
    image_base_url: str
    image_api_key: str
    image_model: str
    video_base_url: str
    video_api_key: str
    video_model: str
    tts_base_url: str
    tts_api_key: str
    tts_model: str
    tts_voice: str
    ark_base_url: str = ""
    ark_api_key: str = ""
    ark_image_model: str = ""
    ark_video_model: str = ""
    ark_video_resolution: str = "720p"
    image_reference: bool = True
    image_retries: int = 1
    video_retries: int = 1
    width: int = 1080
    height: int = 1920
    fps: int = 30
    @property
    def aspect(self) -> str:
        return f"{self.width}:{self.height}"


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    media: MediaSettings
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    extras: dict[str, str] = field(default_factory=dict)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def load_settings() -> Settings:
    llm = LLMSettings(
        provider=_env("LLM_PROVIDER", "deepseek"),
        base_url=_env("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=_env("DEEPSEEK_API_KEY") or _env("LLM_API_KEY"),
        model=_env("LLM_MODEL", "deepseek-chat"),
        reasoner_model=_env("LLM_REASONER_MODEL", "deepseek-reasoner"),
        temperature=float(_env("LLM_TEMPERATURE", "0.8")),
        max_tokens=int(_env("LLM_MAX_TOKENS", "4096")),
        timeout_s=float(_env("LLM_TIMEOUT_S", "180")),
    )
    media = MediaSettings(
        image_provider=_env("IMAGE_PROVIDER", "mock"),
        video_provider=_env("VIDEO_PROVIDER", "mock"),
        tts_provider=_env("TTS_PROVIDER", "mock"),
        image_base_url=_env("IMAGE_BASE_URL"),
        image_api_key=_env("IMAGE_API_KEY"),
        image_model=_env("IMAGE_MODEL"),
        video_base_url=_env("VIDEO_BASE_URL"),
        video_api_key=_env("VIDEO_API_KEY"),
        video_model=_env("VIDEO_MODEL"),
        tts_base_url=_env("TTS_BASE_URL"),
        tts_api_key=_env("TTS_API_KEY"),
        tts_model=_env("TTS_MODEL", "seed-audio-1.0"),
        tts_voice=_env("TTS_VOICE", "default"),
        ark_base_url=_env("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        ark_api_key=_env("ARK_API_KEY"),
        ark_image_model=_env("ARK_IMAGE_MODEL", "doubao-seedream-4-0-250828"),
        ark_video_model=_env("ARK_VIDEO_MODEL", "doubao-seedance-1-0-pro-fast-251015"),
        ark_video_resolution=_env("ARK_VIDEO_RESOLUTION", "720p"),
        image_reference=_env("IMAGE_REFERENCE", "1") not in ("0", "false", "False"),
        image_retries=int(_env("IMAGE_RETRIES", "1")),
        video_retries=int(_env("VIDEO_RETRIES", "1")),
        width=int(_env("VIDEO_WIDTH", "1080")),
        height=int(_env("VIDEO_HEIGHT", "1920")),
        fps=int(_env("VIDEO_FPS", "30")),
    )
    return Settings(llm=llm, media=media)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def reset_settings() -> None:
    """Tests and the runner call this after changing environment variables."""
    global _SETTINGS
    _SETTINGS = None
