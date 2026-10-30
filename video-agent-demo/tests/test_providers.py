from __future__ import annotations

import os

import pytest

from agent.settings import get_settings
from mcp_servers.providers import ProviderConfigError
from mcp_servers.providers.base import ImageRequest, VideoRequest
from mcp_servers.providers.ark_media import ArkImageProvider
from mcp_servers.providers.mock_media import MockImageProvider, MockVideoProvider


def test_mock_providers_write_real_files(tmp_path):
    image = MockImageProvider().generate(
        ImageRequest(prompt="p", negative_prompt="", shot_id="shot_001", project_dir=tmp_path, width=540, height=960, caption="c")
    )
    assert (tmp_path / image["image_path"]).exists()

    video = MockVideoProvider().generate(
        VideoRequest(
            image_path=tmp_path / image["image_path"],
            motion_prompt="m",
            shot_id="shot_001",
            project_dir=tmp_path,
            duration_s=2.0,
            width=540,
            height=960,
            fps=24,
        )
    )
    assert (tmp_path / video["video_path"]).stat().st_size > 5_000


def test_ark_requires_explicit_paid_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("ARK_ALLOW_PAID", raising=False)
    with pytest.raises(RuntimeError, match="ARK_ALLOW_PAID"):
        ArkImageProvider().generate(
            ImageRequest(prompt="p", negative_prompt="", shot_id="s", project_dir=tmp_path, width=1080, height=1920)
        )


def test_unknown_provider_is_a_config_error(monkeypatch):
    import mcp_servers.providers as providers
    from agent.settings import reset_settings

    monkeypatch.setenv("IMAGE_PROVIDER", "nope")
    reset_settings()
    providers.get_image_provider.cache_clear()
    try:
        with pytest.raises(ProviderConfigError, match="nope"):
            providers.get_image_provider()
    finally:
        monkeypatch.delenv("IMAGE_PROVIDER")
        reset_settings()
        providers.get_image_provider.cache_clear()


def test_default_config_is_vertical_short_drama():
    media = get_settings().media
    assert media.height > media.width
    assert media.aspect == f"{media.width}:{media.height}"


def test_mock_verdict_skips_repair_budget(project_root):
    """A mock render is scored honestly but must not trigger a paid repair call."""
    from mcp_servers.evaluators import evaluate_image, evaluate_video
    from PIL import Image

    image = project_root / "shot_001.png"
    Image.new("RGB", (1080, 1920), (20, 30, 40)).save(image)
    verdict = evaluate_image(image, width=1080, height=1920, prompt="x" * 200, anchor="", provider="mock")
    assert verdict["passed"] is False
    assert verdict["mock"] is True

    # The score must track the actual measurement, not a fixed mock ceiling:
    # a flat frame is genuinely poor, a sharp one genuinely better.
    flat = evaluate_image(image, width=1080, height=1920, prompt="x" * 200, anchor="", provider="mock")
    noise = project_root / "shot_002.png"
    import numpy as np
    from PIL import Image as _Image
    rng = np.random.default_rng(0)
    _Image.fromarray(rng.integers(0, 255, (1920, 1080, 3), dtype=np.uint8)).save(noise)
    busy = evaluate_image(noise, width=1080, height=1920, prompt="x" * 200, anchor="", provider="mock")
    assert busy["score"] != flat["score"], "mock score should reflect the measurement"

    video = project_root / "shot_001.mp4"
    video.write_bytes(b"not really a video")
    verdict = evaluate_video(video, duration_s=3.0, width=1080, height=1920, provider="mock")
    assert verdict["mock"] is True


def test_claim_run_is_atomic(project_root):
    import threading

    from backend.services import project_store

    created = project_store.create_project("并发", "电影感", 1)
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(project_store.claim_run(created["project_id"]))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(results) == 1
