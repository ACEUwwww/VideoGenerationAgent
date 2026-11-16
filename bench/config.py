"""Configuration for the Video-Bench comparison harness.

Everything the benchmark needs that is not a model weight lives here: where the
vendored Video-Bench checkout is, which judge model scores the videos, and how
heavily each video is downsampled before it is sent.

The judge default is Doubao on Volcengine Ark, called through the
OpenAI-compatible endpoint. Upstream Video-Bench hardcodes ``gpt-4o-2024-08-06``
in five places; this harness never uses those code paths, so the model name is
free to be whatever Ark serves.

Paid calls stay behind ``ARK_ALLOW_PAID=1``, the same gate the pipeline itself
uses. A benchmark run renders nothing without it and spends nothing without the
matching ``BENCH_ALLOW_PAID=1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_ROOT.parent
VENDOR_ROOT = BENCH_ROOT / "_vendor" / "Video-Bench"
# Long-video workflow vendors live beside Video-Bench, one directory per repo.
# Like Video-Bench they are fetched, not committed.
LONGLIVE_ROOT = BENCH_ROOT / "_vendor" / "LongLive" / "LongLive1.0"
OPENMONTAGE_ROOT = BENCH_ROOT / "_vendor" / "OpenMontage"
# Weights land here as plain diffusers repos: <org>/<repo>/model_index.json.
# bench/fetch_models.sh writes exactly this layout, and the backends load from
# it directly, so the two cannot drift apart.
MODEL_ROOT = BENCH_ROOT / "models" / "hf"
RESULT_ROOT = BENCH_ROOT / "results"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _flag(name: str) -> bool:
    return os.environ.get(name, "") == "1"


def _load_dotenv() -> None:
    """Read the project .env without importing python-dotenv.

    The bench runs in its own interpreter, so it cannot rely on the pipeline's
    dotenv import being available. Real environment variables win, which keeps
    a one-off override (``ARK_VISION_MODEL=x python runner.py``) meaningful.
    """
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


load_dotenv = _load_dotenv


# The nine Video-Bench dimensions that actually have both a prompt and a working
# code path once the upstream routing bugs are bypassed. ``overall_consistency``
# is Video-Bench's "video-text consistency" dimension: VideoBench_full.json has
# no entry keyed ``video-text consistency``, and the dispatcher in
# videobench/__init__.py does not route ``overall_consistency`` at all, so it is
# unreachable upstream despite its prompt and its 91 annotated prompts existing.
VIDEO_TEXT_DIMENSIONS = ("overall_consistency", "color", "object_class", "scene", "action")
STATIC_DIMENSIONS = ("imaging_quality", "aesthetic_quality")
DYNAMIC_DIMENSIONS = ("temporal_consistency", "motion_effects")
ALL_DIMENSIONS = STATIC_DIMENSIONS + DYNAMIC_DIMENSIONS + VIDEO_TEXT_DIMENSIONS


@dataclass
class JudgeConfig:
    """Which MLLM scores the videos, and how the frames reach it."""

    model: str = "doubao-seed-2-1-pro-260915"
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    api_key: str = ""
    reasoner_model: str = ""
    temperature: float = 0.0
    # Frames per second sampled from each video before the judge sees it.
    sample_fps: int = 2
    # Frames per grid panel for the dynamic dimensions. Upstream uses 8 in a
    # 1x8 strip; 4 keeps each panel readable after downscaling.
    grid_size: int = 4
    # Longest edge in pixels. 640 measured 4.1 MB of base64 per 32s clip, which
    # Ark accepts; the upstream 1080x1920 path produces 17.8 MB and is prone to
    # request-size rejection.
    max_edge: int = 640
    jpeg_quality: int = 85
    request_timeout_s: float = 300.0
    attempts: int = 4


@dataclass
class BenchConfig:
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    result_root: Path = RESULT_ROOT
    model_root: Path = MODEL_ROOT
    allow_paid: bool = False

    @property
    def vendor(self) -> Path:
        return VENDOR_ROOT


def load_config() -> BenchConfig:
    load_dotenv()
    # The judge endpoint is overridable so both sides of the comparison can be
    # scored by one model. Judging workflow A with a hosted model and workflow B
    # with another measures the judges' disagreement, not the pipelines.
    judge = JudgeConfig(
        model=_env("BENCH_JUDGE_MODEL") or _env("ARK_VISION_MODEL", "doubao-seed-2-1-pro-260915"),
        base_url=_env("BENCH_JUDGE_BASE_URL") or _env("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        api_key=_env("BENCH_JUDGE_API_KEY") or _env("ARK_API_KEY"),
        reasoner_model=_env("BENCH_JUDGE_REASONER_MODEL"),
        sample_fps=int(_env("BENCH_SAMPLE_FPS", "2")),
        grid_size=int(_env("BENCH_GRID_SIZE", "4")),
        max_edge=int(_env("BENCH_MAX_EDGE", "640")),
        jpeg_quality=int(_env("BENCH_JPEG_QUALITY", "85")),
        request_timeout_s=float(_env("BENCH_JUDGE_TIMEOUT", "300")),
        attempts=int(_env("BENCH_JUDGE_ATTEMPTS", "4")),
    )
    return BenchConfig(
        judge=judge,
        allow_paid=_flag("BENCH_ALLOW_PAID") and _flag("ARK_ALLOW_PAID"),
    )


def require_vendor() -> Path:
    """The vendored checkout, with a message that says how to restore it."""
    if not (VENDOR_ROOT / "videobench" / "prompt_dict.py").exists():
        raise SystemExit(
            f"Video-Bench is not vendored at {VENDOR_ROOT}.\n"
            "Run: python bench/setup_vendor.py"
        )
    return VENDOR_ROOT
