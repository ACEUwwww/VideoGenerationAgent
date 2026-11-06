"""Provider registry.

Selection is by name from ``.env``. An unknown name is a configuration error,
not a reason to quietly fall back to a mock: a run that claims to use a real
vendor must either use it or fail loudly.
"""

from __future__ import annotations

import os
from functools import lru_cache

from agent.settings import get_settings
from mcp_servers.providers.base import (
    ImageProvider,
    ImageRequest,
    SpeechProvider,
    SpeechRequest,
    VideoProvider,
    VideoRequest,
)
from mcp_servers.providers.ark_media import ArkImageProvider, ArkVideoProvider, ArkVisionJudge
from mcp_servers.providers.doubao_speech import DoubaoSpeechProvider
from mcp_servers.providers.mock_media import (
    MockImageProvider,
    MockSpeechProvider,
    MockVideoProvider,
)


class ProviderConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_image_provider() -> ImageProvider:
    name = get_settings().media.image_provider
    if name == "mock":
        return MockImageProvider()
    if name == "ark":
        return ArkImageProvider()
    raise ProviderConfigError(
        f"IMAGE_PROVIDER={name!r} has no implementation yet. "
        "Add a class in mcp_servers/providers/ and register it here."
    )


@lru_cache(maxsize=1)
def get_video_provider() -> VideoProvider:
    name = get_settings().media.video_provider
    if name == "mock":
        return MockVideoProvider()
    if name == "ark":
        return ArkVideoProvider()
    raise ProviderConfigError(
        f"VIDEO_PROVIDER={name!r} has no implementation yet. "
        "Add a class in mcp_servers/providers/ and register it here."
    )


@lru_cache(maxsize=1)
def get_speech_provider() -> SpeechProvider:
    name = get_settings().media.tts_provider
    if name == "mock":
        return MockSpeechProvider()
    if name == "doubao_speech":
        return DoubaoSpeechProvider()
    raise ProviderConfigError(
        f"TTS_PROVIDER={name!r} has no implementation yet. "
        "Add a class in mcp_servers/providers/ and register it here."
    )


def get_vision_judge():
    """Vision QA judge, or None when no vision model is configured.

    Returning None keeps the measured-only path intact: a run without a vision
    model scores exactly as before rather than failing.
    """
    media = get_settings().media
    if not media.ark_vision_model or not media.ark_api_key:
        return None
    if os.environ.get("ARK_ALLOW_PAID") != "1":
        return None
    return ArkVisionJudge()


def provider_report() -> dict[str, str]:
    media = get_settings().media
    return {
        "image": get_image_provider().name,
        "video": get_video_provider().name,
        "tts": get_speech_provider().name,
        "vision": "ark" if get_vision_judge() else "none",
        "canvas": f"{media.width}x{media.height}@{media.fps}",
    }


__all__ = [
    "ImageProvider",
    "ImageRequest",
    "ProviderConfigError",
    "SpeechProvider",
    "SpeechRequest",
    "VideoProvider",
    "VideoRequest",
    "get_image_provider",
    "get_speech_provider",
    "get_video_provider",
    "get_vision_judge",
    "provider_report",
]
