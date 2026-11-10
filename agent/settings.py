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
    ark_vision_model: str = ""
    ark_video_resolution: str = "720p"
    kling_base_url: str = ""
    kling_api_key: str = ""
    kling_model: str = ""
    kling_mode: str = "std"
    image_reference: bool = True
    image_retries: int = 1
    video_retries: int = 1
    width: int = 1080
    height: int = 1920
    fps: int = 30
    transitions: bool = True
    title_card: bool = True
    # Chain each clip's last frame into the next clip as its last_frame role.
    # Requires a model that accepts last_frame (Seedance 2.x or 1.0 pro).
    first_last_frame: bool = True
    bgm_path: str = ""
    bgm_volume: float = 0.18
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
        # pro, not pro-fast: pro-fast rejects last_frame, and first/last-frame chaining
        # is what removes the jump at the cut between shots.
        ark_video_model=_env("ARK_VIDEO_MODEL", "doubao-seedance-1-0-pro-250528"),
        ark_vision_model=_env("ARK_VISION_MODEL", "doubao-seed-2-1-pro-260915"),
        ark_video_resolution=_env("ARK_VIDEO_RESOLUTION", "720p"),
        kling_base_url=_env("KLING_BASE_URL"),
        kling_api_key=_env("KLING_API_KEY"),
        kling_model=_env("KLING_MODEL", "kling-v2-1"),
        kling_mode=_env("KLING_MODE", "std"),
        image_reference=_env("IMAGE_REFERENCE", "1") not in ("0", "false", "False"),
        image_retries=int(_env("IMAGE_RETRIES", "1")),
        video_retries=int(_env("VIDEO_RETRIES", "1")),
        width=int(_env("VIDEO_WIDTH", "1080")),
        height=int(_env("VIDEO_HEIGHT", "1920")),
        fps=int(_env("VIDEO_FPS", "30")),
        transitions=_env("VIDEO_TRANSITIONS", "1") not in ("0", "false", "False"),
        first_last_frame=_env("VIDEO_FIRST_LAST_FRAME", "1") not in ("0", "false", "False"),
        title_card=_env("VIDEO_TITLE_CARD", "1") not in ("0", "false", "False"),
        bgm_path=_env("BGM_PATH", ""),
        bgm_volume=float(_env("BGM_VOLUME", "0.18")),
    )
    return Settings(llm=llm, media=media)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def reset_settings() -> None:
    """Tests and the runner call this after changing environment variables.

    Provider instances are cached too; clearing only the settings leaves a
    stale provider selected, so a reconfigured run would silently keep using
    the previous vendor.
    """
    global _SETTINGS
    _SETTINGS = None
    from mcp_servers.providers import reset_provider_cache

    reset_provider_cache()
