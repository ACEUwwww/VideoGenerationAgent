"""ffmpeg/ffprobe helpers shared by providers and the composer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ffmpeg links libass for the subtitles filter and needs a CJK-capable font
# file it can open by path. Windows ships Microsoft YaHei; macOS has PingFang;
# Linux needs fonts-noto-cjk installed. CJK_FONT_PATH overrides all of them.
FONT_CANDIDATES = {
    "win32": (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\Deng.ttf",
    ),
    "darwin": (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ),
    "linux": (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    ),
}

FONT_INSTALL_HINT = {
    "win32": "Windows 10+ 自带微软雅黑；若缺失请设置 CJK_FONT_PATH 指向任意中文字体文件",
    "darwin": "macOS 自带苹方；若缺失请设置 CJK_FONT_PATH 指向任意中文字体文件",
    "linux": "执行 apt-get install -y fonts-noto-cjk，或设置 CJK_FONT_PATH",
}

EXE_SUFFIX = ".exe" if os.name == "nt" else ""


def _which(name: str) -> str | None:
    """Locate an executable, tolerating the .exe suffix on Windows."""
    found = shutil.which(name) or (shutil.which(name + EXE_SUFFIX) if EXE_SUFFIX else None)
    return found


def ffmpeg_bin() -> str:
    path = _which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (Windows: winget install Gyan.FFmpeg) "
            "and make sure the build includes libass for the subtitles filter."
        )
    return path


def ffprobe_bin() -> str:
    path = _which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe not found on PATH; it ships with ffmpeg")
    return path


def platform_key() -> str:
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def font_path() -> Path:
    override = os.environ.get("CJK_FONT_PATH", "").strip()
    if override:
        path = Path(override)
        if not path.exists():
            raise RuntimeError(f"CJK_FONT_PATH points at a missing file: {override}")
        return path

    key = platform_key()
    for candidate in FONT_CANDIDATES[key]:
        path = Path(candidate)
        if path.exists():
            return path
    raise RuntimeError(f"no CJK font found on {key}: {FONT_INSTALL_HINT[key]}")


# ASS styles reference a font family name, which libass resolves through the
# system font provider. The family must match the file that font_path() found.
FONT_FAMILY_BY_FILE = {
    "msyh.ttc": "Microsoft YaHei",
    "msyhbd.ttc": "Microsoft YaHei",
    "simhei.ttf": "SimHei",
    "simsun.ttc": "SimSun",
    "deng.ttf": "DengXian",
    "pingfang.ttc": "PingFang SC",
    "hiragino sans gb.ttc": "Hiragino Sans GB",
    "arial unicode.ttf": "Arial Unicode MS",
    "songti.ttc": "Songti SC",
    "notosanscjk-regular.ttc": "Noto Sans CJK SC",
    "notosanscjk-bold.ttc": "Noto Sans CJK SC",
    "wqy-zenhei.ttc": "WenQuanYi Zen Hei",
    "wqy-microhei.ttc": "WenQuanYi Micro Hei",
}


def font_family() -> str:
    """ASS/VSFilter family name for the font file font_path() resolves to."""
    override = os.environ.get("CJK_FONT_FAMILY", "").strip()
    if override:
        return override
    name = font_path().name.lower()
    return FONT_FAMILY_BY_FILE.get(name, "Noto Sans CJK SC")


def has_libass() -> bool:
    """The subtitles burn-in needs an ffmpeg built with libass (--enable-libass)."""
    try:
        proc = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return " subtitles " in proc.stdout or " ass " in proc.stdout


# A console app spawned from a GUI or service context flashes a black window on
# Windows. We capture output anyway, so the window is never useful.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def run(args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        creationflags=NO_WINDOW,
    )
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
