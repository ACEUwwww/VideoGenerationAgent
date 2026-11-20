"""HTTP render service: the GPU side of the OpenMontage baseline.

Why a separate process rather than an in-process backend: the studio runs in
``bench/_vendor/OpenMontage/.venv-studio``, which contains no CUDA stack, and
installing torch + diffusers into the vendor's virtualenv would modify the
vendored checkout's environment. So the vendor-side backend stays a thin HTTP
client, and this service runs in ``bench/.venv`` where the weights already load.

Two endpoints, both synchronous and single-request:

  POST /render/video  {model, prompt, seconds, seed}      -> video/mp4
  POST /render/image  {prompt, negative, seed, width, height} -> image/png

Synchronous on purpose. The studio's media pool already runs a bounded worker
thread; adding a queue here would hide how long a render actually takes, and
render time is one of the things the report states.

The models are loaded once and reused. A cold load is ~40 s for CogVideoX and
~5 s for SD1.5; reloading per request would corrupt any timing comparison.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

import requests
from flask import Flask, request, send_file  # noqa: E402

app = Flask(__name__)

# Serialises GPU work across threads. The server is threaded so health checks
# answer during a render; the device itself still takes one job at a time.
_GPU_LOCK = threading.Lock()

_VIDEO_PIPES: dict[str, object] = {}
_IMAGE_PIPE = [None]

IMAGE_MODEL_DIR = ROOT / "bench/models/hf/stable-diffusion-v1-5/stable-diffusion-v1-5"


def _video_provider(model: str):
    from mcp_servers.providers.local_media import LocalVideoProvider

    if model not in _VIDEO_PIPES:
        _VIDEO_PIPES[model] = LocalVideoProvider(model)
    return _VIDEO_PIPES[model]


def _image_pipe():
    if _IMAGE_PIPE[0] is None:
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            str(IMAGE_MODEL_DIR), torch_dtype=torch.float16, safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to("cuda")
        _IMAGE_PIPE[0] = pipe
    return _IMAGE_PIPE[0]


@app.get("/health")
def health():
    return {"ok": True, "video_loaded": sorted(_VIDEO_PIPES), "image_loaded": _IMAGE_PIPE[0] is not None}


@app.post("/render/video")
def render_video():
    payload = request.get_json(force=True)
    model = payload["model"]
    prompt = payload.get("prompt") or ""
    seconds = float(payload.get("seconds") or 5.0)
    seed = int(payload.get("seed") or 0)
    image_in = payload.get("image")

    from mcp_servers.providers.base import VideoRequest
    from mcp_servers.providers.local_media import LOCAL_HEIGHT, LOCAL_WIDTH, MODELS, _frames_for

    spec = MODELS[model]
    frames = _frames_for(spec.get("frame_rule", "4k+1"), spec["fps"], seconds)

    # CogVideoX-5B-I2V hard-requires its own latent bucket (60x90 -> 720x480).
    # Any other size trips a learned-positional-embedding assertion in diffusers.
    width, height = (720, 480) if spec.get("i2v") else (LOCAL_WIDTH, LOCAL_HEIGHT)

    started = time.monotonic()
    staging = Path("/tmp/om_render") / f"{int(time.time() * 1000)}"
    staging.mkdir(parents=True, exist_ok=True)

    image_path = None
    if image_in and spec.get("i2v"):
        source = Path(str(image_in))
        if not source.is_file():
            raise RuntimeError(f"keyframe not found: {source}")
        image_path = staging / "keyframe.png"
        image_path.write_bytes(source.read_bytes())

    provider = _video_provider(model)
    with _GPU_LOCK:
        result = provider.generate(VideoRequest(
        image_path=image_path, motion_prompt=prompt, shot_id="shot", project_dir=staging,
        duration_s=frames / spec["fps"], width=width, height=height,
        fps=spec["fps"], seed=seed,
    ))
    produced = staging / result["video_path"]
    data = produced.read_bytes()
    return send_file(io.BytesIO(data), mimetype="video/mp4",
                     download_name="clip.mp4")


@app.post("/render/image")
def render_image():
    payload = request.get_json(force=True)
    prompt = payload.get("prompt") or ""
    negative = payload.get("negative") or ""
    width = int(payload.get("width") or 512)
    height = int(payload.get("height") or 512)
    seed = int(payload.get("seed") or 0)

    import torch

    pipe = _image_pipe()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with _GPU_LOCK:
        out = pipe(
            prompt=prompt, negative_prompt=negative, width=width, height=height,
            num_inference_steps=30, guidance_scale=7.5, generator=generator,
        ).images[0]

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", download_name="still.png")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--warm-image", action="store_true")
    args = parser.parse_args()

    if args.warm_image:
        _image_pipe()
        print("[render] image pipeline warm", flush=True)
    # Threaded so /health still answers while a 35 s render is in flight: the
    # studio polls this endpoint to decide whether the backend is configured,
    # and a single-threaded server made it look dead whenever the GPU was busy.
    # The GPU itself is still serialised by _GPU_LOCK, because two diffusers
    # pipelines cannot share one device safely.
    app.run(host="127.0.0.1", port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
