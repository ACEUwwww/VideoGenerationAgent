"""LLM access layer.

The agent nodes never talk to a provider SDK directly: they call
``generate_text`` / ``generate_json`` here. Provider quirks (DeepSeek injects
``<ds_safety>`` blocks into message content, reasoning models put the answer
after a long private draft) are normalised in this module.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from agent.settings import LLMSettings, get_settings

_SAFETY_BLOCK = re.compile(r"<ds_safety>.*?</ds_safety>", re.DOTALL)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMReply:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False, model: str | None = None) -> LLMReply: ...


class OpenAICompatibleClient:
    """DeepSeek / vLLM / any ``POST /chat/completions`` endpoint."""

    def __init__(self, settings: LLMSettings, transport: httpx.BaseTransport | None = None) -> None:
        if not settings.api_key:
            raise LLMError(
                "LLM api key is empty. Set DEEPSEEK_API_KEY in video-agent-demo/.env"
            )
        self.settings = settings
        self._transport = transport

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False, model: str | None = None) -> LLMReply:
        payload: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        last_error = ""
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.settings.timeout_s, transport=self._transport) as client:
                    response = client.post(
                        f"{self.settings.base_url.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.settings.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
                continue

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                time.sleep(2**attempt)
                continue
            if response.status_code != 200:
                raise LLMError(f"LLM HTTP {response.status_code}: {response.text[:500]}")

            body = response.json()
            choice = body["choices"][0]["message"]
            usage = body.get("usage", {})
            return LLMReply(
                text=_clean(choice.get("content") or ""),
                model=body.get("model", payload["model"]),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_s=round(time.monotonic() - started, 2),
            )
        raise LLMError(f"LLM call failed after retries: {last_error}")


def _clean(text: str) -> str:
    return _SAFETY_BLOCK.sub("", text).strip()


def _strip_fence(text: str) -> str:
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model output, tolerating fences and chatter."""
    candidate = _strip_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        return json.loads(candidate[start : end + 1])
    raise LLMError(f"model did not return a JSON object: {candidate[:400]}")


class LLMService:
    """Thin facade used by the graph nodes."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = OpenAICompatibleClient(get_settings().llm)
        return self._client

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMReply:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.client.chat(messages, model=model)

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        required_keys: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], LLMReply]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        reply = self.client.chat(messages, json_mode=True, model=model)
        try:
            data = parse_json(reply.text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"invalid JSON from {reply.model}: {reply.text[:400]}") from exc
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise LLMError(f"model JSON missing keys {missing}: {list(data)}")
        return data, reply


_DEFAULT: LLMService | None = None


def get_llm() -> LLMService:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMService()
    return _DEFAULT


def set_llm(service: LLMService | None) -> None:
    """Tests inject a fake service; also used to reset between runs."""
    global _DEFAULT
    _DEFAULT = service
