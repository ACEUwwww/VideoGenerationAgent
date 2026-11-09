"""Kling (可灵) video provider.

Kling is an async task API like Ark: create a task, poll it, then download the
result. Authentication is a bearer token; ``KLING_BASE_URL`` selects the region
or a compatible gateway, so the same code works against whichever endpoint the
key belongs to.

Endpoints used (v1):
  POST /v1/videos/image2video          create an image-to-video task
  GET  /v1/videos/image2video/{id}     poll one task
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx

from agent.settings import get_settings
from mcp_servers.providers.base import VideoRequest

POLL_INTERVAL_S = 6.0
VIDEO_TIMEOUT_S = 900.0
CREATE_ATTEMPTS = 3
DOWNLOAD_ATTEMPTS = 4
RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# Kling reports progress through task_status; anything else is terminal.
_PENDING = ("submitted", "processing")
_SUCCESS = ("succeed", "succeeded")
_FAILED = ("failed", "error")


def _credentials() -> tuple[str, str, str]:
    """Base URL, API key and model. Paid like every other real provider."""
    import os

    if os.environ.get("KLING_ALLOW_PAID") != "1":
        raise RuntimeError(
            "Kling provider is paid and disabled. Set KLING_ALLOW_PAID=1 for an explicit paid test run."
        )
    media = get_settings().media
    base_url = (media.kling_base_url or "").rstrip("/")
    api_key = media.kling_api_key
    model = media.kling_model
    if not base_url:
        raise RuntimeError("KLING_BASE_URL is empty")
    if not api_key:
        raise RuntimeError("KLING_API_KEY is empty")
    if not model:
        raise RuntimeError("KLING_MODEL is empty")
    return base_url, api_key, model


def _image_payload(path: Path) -> str:
    """Kling wants raw base64 without the data-URI prefix."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _post_with_retry(url: str, headers: dict, payload: dict, *, timeout: float) -> dict:
    last = ""
    for attempt in range(CREATE_ATTEMPTS):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
            continue
        if response.status_code in RETRYABLE_STATUS:
            last = f"HTTP {response.status_code}: {response.text[:300]}"
            time.sleep(2**attempt)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"Kling HTTP {response.status_code}: {response.text[:500]}")
        body = response.json()
        code = body.get("code")
        if code not in (0, None):
            raise RuntimeError(f"Kling error {code}: {body.get('message')} | {str(body)[:300]}")
        return body
    raise RuntimeError(f"Kling call failed after {CREATE_ATTEMPTS} attempts: {last}")


class KlingVideoProvider:
    """Image-to-video through the Kling task API."""

    name = "kling"

    def generate(self, request: VideoRequest) -> dict:
        base_url, api_key, model = _credentials()
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        payload: dict = {
            "model_name": model,
            "image": _image_payload(request.image_path),
            "prompt": request.motion_prompt,
            "duration": str(max(int(round(request.duration_s)), 1)),
            "mode": get_settings().media.kling_mode,
        }
        # Kling accepts a tail image for first/last-frame generation, which is
        # how one shot's ending is pinned to the next shot's opening.
        if request.last_frame_path is not None and Path(request.last_frame_path).exists():
            payload["image_tail"] = _image_payload(Path(request.last_frame_path))

        created = _post_with_retry(
            f"{base_url}/v1/videos/image2video", headers, payload, timeout=120
        )
        task_id = (created.get("data") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"Kling did not return a task_id: {str(created)[:300]}")

        started = time.monotonic()
        video_url = ""
        last_error = ""
        body: dict = {}
        while time.monotonic() - started < VIDEO_TIMEOUT_S:
            time.sleep(POLL_INTERVAL_S)
            try:
                with httpx.Client(timeout=60) as client:
                    poll = client.get(f"{base_url}/v1/videos/image2video/{task_id}", headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            if poll.status_code != 200:
                last_error = f"HTTP {poll.status_code}: {poll.text[:200]}"
                continue
            body = poll.json()
            data = body.get("data") or {}
            status = str(data.get("task_status", "")).lower()
            if status in _SUCCESS:
                videos = (data.get("task_result") or {}).get("videos") or []
                if videos:
                    video_url = videos[0].get("url", "")
                break
            if status in _FAILED:
                raise RuntimeError(
                    f"Kling task {task_id} failed: {data.get('task_status_msg') or body.get('message')}"
                )
        if not video_url:
            raise RuntimeError(
                f"Kling task {task_id} produced no video within {VIDEO_TIMEOUT_S}s "
                f"(last poll error: {last_error or 'none'})"
            )

        # The result CDN drops TLS mid-transfer often enough that a single GET
        # loses an already-paid render; retry the download only.
        raw = b""
        last_download_error = ""
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                with httpx.Client(timeout=600) as client:
                    raw = client.get(video_url).content
                if raw:
                    break
            except httpx.HTTPError as exc:
                last_download_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
        if not raw:
            raise RuntimeError(
                f"Kling task {task_id} finished but download failed after "
                f"{DOWNLOAD_ATTEMPTS} attempts ({last_download_error or 'empty body'}); url={video_url}"
            )

        target = request.project_dir / "videos" / f"{request.shot_id}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {
            "video_path": str(target.relative_to(request.project_dir)),
            "provider": self.name,
            "model": model,
            "task_id": task_id,
            "motion_prompt": request.motion_prompt,
            "duration_s": request.duration_s,
            "usage": body.get("usage", {}),
        }
