"""Doubao Speech / Seed-Audio provider.

Runs on the speech product line (``openspeech.bytedance.com``), not Ark. Auth is
a separate speech API key in ``X-Api-Key`` — an Ark key does not work here.

The endpoint answers ``POST /api/v3/tts/create`` with a base64 WAV; the model
generates read-aloud speech from ``text_prompt``. Output is 40kHz stereo
PCM, which ffmpeg normalises to the canvas during composition.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx

from agent.settings import get_settings
from mcp_servers.providers.base import SpeechRequest

DEFAULT_BASE_URL = "https://openspeech.bytedance.com/api/v3/tts/create"
DEFAULT_MODEL = "seed-audio-1.0"
ATTEMPTS = 3
RETRYABLE = (429, 500, 502, 503, 504)


class DoubaoSpeechProvider:
    name = "doubao_speech"

    def synthesize(self, request: SpeechRequest) -> dict:
        media = get_settings().media
        api_key = media.tts_api_key
        if not api_key:
            raise RuntimeError("TTS_API_KEY is empty; set the Doubao speech API key in .env")

        url = media.tts_base_url or DEFAULT_BASE_URL
        model = media.tts_model or DEFAULT_MODEL
        payload = {"model": model, "text_prompt": request.text}
        headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}

        started = time.monotonic()
        last_error = ""
        body: dict | None = None
        for attempt in range(ATTEMPTS):
            try:
                with httpx.Client(timeout=180) as client:
                    response = client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
                continue
            if response.status_code in RETRYABLE:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                time.sleep(2**attempt)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"Doubao speech HTTP {response.status_code}: {response.text[:400]}")
            body = response.json()
            break
        if body is None:
            raise RuntimeError(f"Doubao speech failed after {ATTEMPTS} attempts: {last_error}")

        if not body.get("audio"):
            raise RuntimeError(f"Doubao speech returned no audio: {list(body)}")

        raw = base64.b64decode(body["audio"])
        target = request.project_dir / "audio" / f"{request.shot_id}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {
            "audio_path": str(target.relative_to(request.project_dir)),
            "provider": self.name,
            "model": model,
            "text": request.text,
            "duration_s": float(body.get("duration", 0.0) or 0.0),
            "bytes": len(raw),
            "latency_s": round(time.monotonic() - started, 2),
        }
