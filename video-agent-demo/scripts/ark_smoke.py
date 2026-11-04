"""Cheapest possible Ark smoke test, run only after activating models.

Usage (spends money on success, nothing on a 404/ModelNotOpen):
    .venv/bin/python scripts/ark_smoke.py --kind image
    .venv/bin/python scripts/ark_smoke.py --kind video
    .venv/bin/python scripts/ark_smoke.py --kind status   # free, lists activation

Video uses 480p / 4s / single clip to keep the charge minimal.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.settings import get_settings  # noqa: E402

PROBE_IMAGE = "Vertical 9:16 portrait, a courier in rain at night, cinematic, 35mm"
PROBE_VIDEO = "a hand reaching for a phone, rain on the screen"


def _client():
    media = get_settings().media
    return media, httpx.Client(timeout=300), {"Authorization": f"Bearer {media.ark_api_key}"}


def status() -> int:
    media, client, headers = _client()
    response = client.get(f"{media.ark_base_url}/models", headers=headers, params={"page_num": 1, "page_size": 200})
    response.raise_for_status()
    models = {m["id"]: m.get("status", "") for m in response.json().get("data", [])}
    wanted = {
        "image": media.ark_image_model,
        "video": media.ark_video_model,
    }
    for kind, model in wanted.items():
        listed = "listed" if model in models else "NOT LISTED"
        print(f"{kind:6s} {model:34s} {listed:12s} state={models.get(model, '-')}")
    print("\n'listed' only means the catalog knows the model. Activation is verified by")
    print("running --kind image / --kind video, which returns ModelNotOpen until the")
    print("model is enabled in the Ark console for this account.")
    return 0


def image() -> int:
    media, client, headers = _client()
    started = time.time()
    payload = {
        "model": media.ark_image_model,
        "prompt": PROBE_IMAGE,
        "size": f"{media.width}x{media.height}",
        "response_format": "b64_json",
        "watermark": False,
        "sequential_image_generation": "disabled",
    }
    response = client.post(f"{media.ark_base_url}/images/generations", headers=headers, json=payload)
    return _report("image", response, started)


def video() -> int:
    media, client, headers = _client()
    started = time.time()
    payload = {
        "model": media.ark_video_model,
        "content": [{"type": "text", "text": PROBE_VIDEO}],
        "ratio": "9:16",
        "duration": 4,
        "resolution": "480p",
        "watermark": False,
    }
    response = client.post(f"{media.ark_base_url}/contents/generations/tasks", headers=headers, json=payload)
    if response.status_code != 200:
        return _report("video", response, started)
    task_id = response.json()["id"]
    print(f"task {task_id} created in {time.time() - started:.1f}s, polling...")
    while time.time() - started < 600:
        time.sleep(8)
        poll = client.get(f"{media.ark_base_url}/contents/generations/tasks/{task_id}", headers=headers)
        if poll.status_code == 200 and poll.json().get("status") in ("succeeded", "failed", "cancelled"):
            body = poll.json()
            print(json.dumps(body, ensure_ascii=False)[:600])
            return 0 if body.get("status") == "succeeded" else 1
    print("timed out")
    return 1


def _report(kind: str, response: httpx.Response, started: float) -> int:
    print(f"{kind}: HTTP {response.status_code} in {time.time() - started:.1f}s")
    body = response.json()
    if response.status_code != 200:
        error = body.get("error", {})
        print(f"  code={error.get('code')} message={error.get('message', '')[:200]}")
        if error.get("code") == "ModelNotOpen":
            print("  -> model is not activated for this account. Activate it in the Ark console.")
        elif error.get("code") == "SetLimitExceeded":
            print("  -> model IS activated but the account hit its usage cap (Safe Experience")
            print("     Mode). Raise or disable that cap on the Model Activation page.")
        elif error.get("code") == "InvalidEndpointOrModel.NotFound":
            print("  -> this model id does not exist for this account; pick one from --kind status.")
        return 1
    data = body["data"][0]
    raw = base64.b64decode(data["b64_json"]) if data.get("b64_json") else httpx.get(data["url"]).content
    out = Path(tempfile.gettempdir()) / "ark_smoke.png"
    out.write_bytes(raw)
    print(f"  ok, {len(raw)} bytes -> {out}, usage={body.get('usage')}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["status", "image", "video"], default="status")
    args = parser.parse_args()
    if args.kind == "status":
        sys.exit(status())
    if not os.environ.get("ARK_ALLOW_PAID") and args.kind in ("image", "video"):
        print("set ARK_ALLOW_PAID=1 to confirm this may spend money")
        sys.exit(2)
    sys.exit(image() if args.kind == "image" else video())
