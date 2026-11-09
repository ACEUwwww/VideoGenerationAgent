"""Volcengine Ark providers: Doubao Seedream (image) and Seedance (video).

Both are paid per call. ``ARK_ALLOW_PAID=1`` must be set explicitly before a
provider will spend money, so a stray ``IMAGE_PROVIDER=ark`` in .env cannot
silently bill a full multi-shot run.

Ark has no public URL for local files, so the keyframe is uploaded as a base64
data URI in the ``first_frame`` role.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx

from agent.llm_client import parse_json
from agent.settings import get_settings
from mcp_servers.providers.base import ImageRequest, VideoRequest

POLL_INTERVAL_S = 6.0
VIDEO_TIMEOUT_S = 900.0
# A create call is the money-spending step: retry only what cannot be billed
# twice (connection errors, 429, 5xx). A 4xx business error is raised at once.
CREATE_ATTEMPTS = 3
# The render result CDN drops TLS mid-transfer; retrying the download costs
# nothing extra because the task is already billed and finished.
DOWNLOAD_ATTEMPTS = 4
RETRYABLE_STATUS = (429, 500, 502, 503, 504)


# Seedance 1.0 pro-fast only supports text-to-video and first-frame
# image-to-video; last_frame reached the API with 2.0 and also exists on
# 1.0 pro. Matching on the model id keeps this honest when .env is switched.
_LAST_FRAME_PATTERNS = ("seedance-2", "seedance-1-0-pro-2")


def supports_last_frame(model: str) -> bool:
    """Whether this Seedance model accepts a ``last_frame`` role."""
    name = (model or "").lower()
    if "pro-fast" in name:
        return False
    return any(pattern in name for pattern in _LAST_FRAME_PATTERNS)


def _post_with_retry(url: str, headers: dict, payload: dict, *, timeout: float, attempts: int = CREATE_ATTEMPTS) -> dict:
    """POST with bounded retry on connection errors and transient statuses."""
    last = ""
    for attempt in range(attempts):
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
            raise RuntimeError(f"Ark HTTP {response.status_code}: {response.text[:500]}")
        return response.json()
    raise RuntimeError(f"Ark call failed after {attempts} attempts: {last}")


def _credentials(kind: str) -> tuple[str, str, str]:
    import os

    if os.environ.get("ARK_ALLOW_PAID") != "1":
        raise RuntimeError(
            f"Ark {kind} provider is paid and disabled. Set ARK_ALLOW_PAID=1 for an explicit paid test run."
        )
    media = get_settings().media
    base_url = media.ark_base_url.rstrip("/")
    api_key = media.ark_api_key
    if not api_key:
        raise RuntimeError("ARK_API_KEY is empty")
    model = {
        "image": media.ark_image_model,
        "video": media.ark_video_model,
        "vision": media.ark_vision_model,
    }[kind]
    if not model:
        raise RuntimeError(f"ARK_{kind.upper()}_MODEL is empty")
    return base_url, api_key, model


class ArkImageProvider:
    """POST /images/generations — Doubao Seedream."""

    name = "ark"

    def generate(self, request: ImageRequest) -> dict:
        base_url, api_key, model = _credentials("image")
        payload = {
            "model": model,
            "prompt": request.prompt,
            "size": f"{request.width}x{request.height}",
            "response_format": "b64_json",
            "watermark": False,
            "sequential_image_generation": "disabled",
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.seed:
            payload["seed"] = request.seed
        reference = ""
        if request.reference_image:
            reference = _data_uri(request.reference_image)
            payload["image"] = reference

        body = _post_with_retry(
            f"{base_url}/images/generations",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload,
            timeout=300,
        )
        data = body["data"][0]
        if data.get("b64_json"):
            raw = base64.b64decode(data["b64_json"])
        else:
            raw = httpx.get(data["url"], timeout=300).content

        target = request.project_dir / "images" / f"{request.shot_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {
            "image_path": str(target.relative_to(request.project_dir)),
            "provider": self.name,
            "model": model,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "usage": body.get("usage", {}),
            "width": request.width,
            "height": request.height,
            "reference_image": str(request.reference_image) if request.reference_image else "",
        }


class ArkVideoProvider:
    """POST /contents/generations/tasks — Doubao Seedance, async + polling."""

    name = "ark"

    def generate(self, request: VideoRequest) -> dict:
        base_url, api_key, model = _credentials("video")
        image_uri = _data_uri(request.image_path)
        content = [
            {"type": "text", "text": request.motion_prompt},
            {"type": "image_url", "image_url": {"url": image_uri}, "role": "first_frame"},
        ]
        # First/last-frame chaining pins this clip's ending to the next clip's
        # opening, so the cut does not jump. pro-fast does not accept
        # last_frame at all, and mixing roles across scenarios is rejected, so
        # the flag only appears when the model is known to support it.
        if request.last_frame_path is not None and supports_last_frame(model):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_uri(Path(request.last_frame_path))},
                    "role": "last_frame",
                }
            )
        payload = {
            "model": model,
            "content": content,
            "ratio": "9:16",
            "resolution": get_settings().media.ark_video_resolution,
            "duration": int(round(request.duration_s)),
            "watermark": False,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        created = _post_with_retry(f"{base_url}/contents/generations/tasks", headers, payload, timeout=120)
        task_id = created["id"]

        started = time.monotonic()
        video_url = ""
        last_poll_error = ""
        with httpx.Client(timeout=120) as client:
            while time.monotonic() - started < VIDEO_TIMEOUT_S:
                time.sleep(POLL_INTERVAL_S)
                try:
                    poll = client.get(f"{base_url}/contents/generations/tasks/{task_id}", headers=headers)
                except httpx.HTTPError as exc:
                    last_poll_error = f"{type(exc).__name__}: {exc}"
                    continue
                if poll.status_code != 200:
                    last_poll_error = f"HTTP {poll.status_code}: {poll.text[:200]}"
                    continue
                body = poll.json()
                status = body.get("status")
                if status == "succeeded":
                    video_url = body["content"]["video_url"]
                    break
                if status in ("failed", "cancelled"):
                    raise RuntimeError(f"Ark video task {task_id} {status}: {body.get('error')}")
            if not video_url:
                raise RuntimeError(
                    f"Ark video task {task_id} timed out after {VIDEO_TIMEOUT_S}s (last poll error: {last_poll_error or 'none'})"
                )

            # The CDN drops TLS connections mid-transfer often enough that a
            # single GET loses a render that has already been paid for. Retry
            # the download only; the task itself succeeded and is not re-billed.
            raw = b""
            last_download_error = ""
            for attempt in range(DOWNLOAD_ATTEMPTS):
                try:
                    raw = client.get(video_url, timeout=600).content
                    if raw:
                        break
                except httpx.HTTPError as exc:
                    last_download_error = f"{type(exc).__name__}: {exc}"
                    time.sleep(2**attempt)
            if not raw:
                raise RuntimeError(
                    f"Ark video {task_id} finished but download failed after "
                    f"{DOWNLOAD_ATTEMPTS} attempts ({last_download_error or 'empty body'}); "
                    f"url={video_url}"
                )
        usage = body.get("usage", {})

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
            "resolution": body.get("resolution", ""),
            "usage": usage,
        }


class ArkVisionJudge:
    """Chat-completions call against a vision model for QA judgement.

    The keyframe goes in as a data URI at full resolution: the pixel-based
    evaluator works on 192px thumbnails, which is far too small for a model to
    spot a deformed face. Returns 1-5 scores keyed by dimension plus a reason.
    """

    name = "ark"

    def __call__(self, image_path, prompt: str) -> tuple[dict[str, int], str]:
        """Callable so the instance itself satisfies the mllm_judge hook.

        Accepts either one path (per-clip judging) or a list of paths in
        playback order (sequence judging).
        """
        return self.judge(image_path, prompt)

    def judge(self, image_path, prompt: str) -> tuple[dict[str, int], str]:
        base_url, api_key, model = _credentials("vision")
        paths = [image_path] if isinstance(image_path, (str, Path)) else list(image_path)
        if not paths:
            raise ValueError("vision judge needs at least one image")
        content = [
            {
                "type": "image_url",
                # JPEG re-encode keeps the payload small enough to survive the
                # connection; a 1080x1920 PNG is ~500KB before base64 and the
                # server drops it mid-request.
                "image_url": {"url": _image_uri(Path(path)), "detail": "high"},
            }
            for path in paths
        ]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 600,
            "temperature": 0.2,
        }
        body = _post_with_retry(
            f"{base_url}/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload,
            timeout=120,
        )
        text = body["choices"][0]["message"]["content"]
        data = parse_json(text)
        # Return every numeric 1-5 score the model produced. The per-clip
        # prompt asks for four dimensions and the sequence prompt for three
        # different ones; filtering to a fixed list silently dropped the
        # sequence scores and made the judge look like it failed.
        scores = {
            key: value
            for key, value in data.items()
            if key != "reason" and isinstance(value, (int, float))
        }
        return scores, str(data.get("reason", ""))[:400]


def _image_uri(path: Path, quality: int = 90) -> str:
    """JPEG data URI for the vision judge.

    Base64 inflates the payload by ~33%; a full-resolution PNG of a live-action
    keyframe reaches 500KB and more, which the endpoint drops as
    ``RemoteProtocolError``. JPEG q90 at the same pixel dimensions is ~1/5 the
    size and the model sees the same detail.
    """
    import io

    from PIL import Image

    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "png")
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"
