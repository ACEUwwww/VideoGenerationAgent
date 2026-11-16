"""A local, OpenAI-compatible vision judge.

Both sides of the comparison already speak this protocol: the bench judge builds
an ``OpenAI(...)`` client against a base_url, and OpenMontage's shot QA POSTs
``{base}/chat/completions`` with image_url parts. Serving that shape from a
locally loaded model means one judge scores both workflows' output — which is
what makes the numbers comparable at all.

Why local rather than a hosted endpoint: the same reason the benchmark exists.
A hosted judge sends every generated frame to a third party, costs money per
call, and is a different model from run to run as the provider updates it. A
local Qwen2.5-VL-7B is fixed by its weights.

Only what the two call sites use is implemented: models, and non-streaming
chat/completions with text and image_url content. Anything else is answered with
an explicit error rather than a plausible-looking empty result.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = os.environ.get("VLM_MODEL_DIR", "")
SERVED_NAME = os.environ.get("VLM_NAME", "qwen2.5-vl-7b-local")
HOST = os.environ.get("VLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("VLM_PORT", "8790"))
# The bench judge sends no max_tokens, because the hosted judge's own default was
# enough for it. Qwen2.5-VL reasons at length before writing its final score
# line — motion_effects measured 1354 completion tokens for five videos and was
# cut off mid-sentence at 512, producing no score at all. 2048 leaves room for
# the analysis plus the labelled lines. max_tokens on the request still wins.
DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_DEFAULT_MAX_TOKENS", "2048"))

_model = None
_processor = None
_lock = threading.Lock()


def load() -> None:
    global _model, _processor
    print(f"[vlm] loading {MODEL_ID}", flush=True)
    # flash_attention_2 is required, not a speed preference. With transformers
    # 5.22.2 the default SDPA path degenerates on image input: a JSON request
    # came back as {"score": 100, "reason": "The\n\n addCriterion\n{...repeated",
    # and a two-word OCR answer turned into "addCriterion" repeated forever.
    # Text-only prompts were fine, so the fault is in the vision attention path.
    # Under flash_attention_2 the same prompts answer correctly and terminate.
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("[vlm] ready", flush=True)


def _image_from_url(url: str) -> Image.Image:
    """Decode a data: URI into an RGB image.

    Remote URLs are refused on purpose: this server exists to keep judging local,
    and fetching an arbitrary URL from inside the judge would be a way around
    that. Both call sites already inline their images.
    """
    if not url.startswith("data:"):
        raise ValueError("only data: URIs are accepted")
    _, _, payload = url.partition(",")
    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _build_messages(payload: dict) -> tuple[list[dict], list[Image.Image]]:
    """Split OpenAI content parts into chat messages plus images.

    transformers expects images passed alongside the messages and referenced by
    a ``{"type": "image"}`` part, not OpenAI's ``image_url``. The mapping is
    positional, so order is preserved.
    """
    messages: list[dict] = []
    images: list[Image.Image] = []
    for msg in payload.get("messages") or []:
        role = msg.get("role") or "user"
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts: list[dict] = []
        for part in content or []:
            kind = part.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": part.get("text") or ""})
            elif kind in {"image_url", "image"}:
                url = (part.get("image_url") or {}).get("url") if kind == "image_url" else part.get("image")
                images.append(_image_from_url(url))
                parts.append({"type": "image"})
            else:
                raise ValueError(f"unsupported content part {kind!r}")
        messages.append({"role": role, "content": parts})
    return messages, images


def complete(payload: dict) -> dict:
    messages, images = _build_messages(payload)
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=images or None, return_tensors="pt").to("cuda")
    max_new = int(payload.get("max_tokens") or DEFAULT_MAX_TOKENS)
    with _lock, torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = out[:, inputs["input_ids"].shape[1]:]
    answer = _processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": SERVED_NAME,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated.shape[1]),
            "total_tokens": int(inputs["input_ids"].shape[1] + generated.shape[1]),
        },
    }
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - quieter than the default
        sys.stderr.write("[vlm] " + fmt % args + "\n")

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's API
        if self.path.rstrip("/") in {"", "/v1"}:
            return self._send(200, {"ok": True, "model": SERVED_NAME})
        if self.path.rstrip("/") == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]})
        self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's API
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": f"no route {self.path}"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if payload.get("stream"):
                return self._stream(payload)
            return self._send(200, complete(payload))
        except Exception as exc:  # noqa: BLE001 - the client needs the reason
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _stream(self, payload: dict) -> None:
        """Server-sent events, one chunk per decoded token.

        The bench judge streams because a non-streaming call died at a constant
        ~153s against the egress proxy. Against a local server that constraint
        is gone, but the judge's client has no non-streaming path — so this
        endpoint has to speak SSE or the judge cannot use the local model at all.
        """
        messages, images = _build_messages(payload)
        text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _processor(text=[text], images=images or None, return_tensors="pt").to("cuda")
        max_new = int(payload.get("max_tokens") or DEFAULT_MAX_TOKENS)
        prompt_len = int(inputs["input_ids"].shape[1])

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(obj: dict) -> None:
            data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(_processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs, max_new_tokens=max_new, do_sample=False,
            temperature=None, top_p=None, streamer=streamer,
        )
        thread = threading.Thread(target=_model.generate, kwargs=kwargs, daemon=True)
        thread.start()

        emitted = 0
        for piece in streamer:
            if not piece:
                continue
            emitted += 1
            emit({
                "id": "chatcmpl-local", "object": "chat.completion.chunk",
                "model": SERVED_NAME,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            })
        thread.join()
        emit({
            "id": "chatcmpl-local", "object": "chat.completion.chunk", "model": SERVED_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_len, "completion_tokens": emitted,
                      "total_tokens": prompt_len + emitted},
        })
        done = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n" % len(done) + done + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


if __name__ == "__main__":
    if not MODEL_ID or not Path(MODEL_ID).is_dir():
        raise SystemExit(f"set VLM_MODEL_DIR to a local model directory (got {MODEL_ID!r})")
    load()
    print(f"[vlm] serving {SERVED_NAME} on http://{HOST}:{PORT}/v1", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
