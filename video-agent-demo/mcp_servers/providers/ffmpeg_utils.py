"""ffmpeg/ffprobe helpers shared by providers and the composer."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH")
    return path


def ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe not found on PATH")
    return path


def font_path() -> Path:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise RuntimeError("no CJK font found; install fonts-noto-cjk")


def run(args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(args[:3])}...): {proc.stderr[-1200:]}")
    return proc


def probe(path: Path) -> dict:
    proc = run(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(proc.stdout)


def media_facts(path: Path) -> dict:
    """Compact, provider-agnostic description used by the evaluator."""
    path = Path(path)
    if not path.exists():
        return {"exists": False, "bytes": 0}
    facts: dict = {"exists": True, "bytes": path.stat().st_size, "file": path.name}
    try:
        info = probe(path)
    except RuntimeError as exc:
        facts["probe_error"] = str(exc)[:300]
        return facts
    fmt = info.get("format", {})
    facts["duration_s"] = round(float(fmt.get("duration", 0.0) or 0.0), 3)
    facts["format_name"] = fmt.get("format_name", "")
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            facts["width"] = int(stream.get("width", 0))
            facts["height"] = int(stream.get("height", 0))
            facts["codec"] = stream.get("codec_name", "")
            facts["frames"] = int(stream.get("nb_frames", 0) or 0)
            facts["avg_frame_rate"] = stream.get("avg_frame_rate", "")
        elif stream.get("codec_type") == "audio":
            facts["has_audio"] = True
    facts.setdefault("has_audio", False)
    return facts
