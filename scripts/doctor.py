"""Environment self-check. Runs on Windows, macOS and Linux.

Checks the things that actually break a run: ffmpeg and its libass build, a
usable CJK font, the three credential sets, and which provider each capability
will use. Never spends money.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.settings import get_settings  # noqa: E402
from mcp_servers.providers.ffmpeg_utils import (  # noqa: E402
    FONT_INSTALL_HINT,
    ffmpeg_bin,
    font_family,
    font_path,
    has_libass,
    platform_key,
)

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def _line(status: str, label: str, detail: str) -> None:
    print(f"  [{status:4s}] {label:22s} {detail}")


def check_binaries() -> list[str]:
    problems = []
    print("依赖")
    try:
        path = ffmpeg_bin()
        _line(OK, "ffmpeg", path)
    except RuntimeError as exc:
        problems.append(str(exc))
        _line(FAIL, "ffmpeg", str(exc)[:80])
        return problems

    if has_libass():
        _line(OK, "libass", "subtitle burn-in available")
    else:
        problems.append("ffmpeg 缺少 libass，字幕无法烧录")
        _line(FAIL, "libass", "缺少，字幕烧录会失败；换一个包含 libass 的构建")

    if shutil.which("ffprobe"):
        _line(OK, "ffprobe", shutil.which("ffprobe"))
    else:
        problems.append("ffprobe 缺失")
        _line(FAIL, "ffprobe", "缺失，应从同一 ffmpeg 包中获得")

    try:
        path = font_path()
        _line(OK, "CJK font", f"{path.name} -> family {font_family()}")
    except RuntimeError:
        problems.append(FONT_INSTALL_HINT[platform_key()])
        _line(FAIL, "CJK font", FONT_INSTALL_HINT[platform_key()])
    return problems


def check_config() -> list[str]:
    problems = []
    print("\n配置")
    settings = get_settings()
    llm, media = settings.llm, settings.media

    _line(OK if llm.api_key else FAIL, "DeepSeek", f"{llm.model}, key {'已设置' if llm.api_key else '缺失'}")
    if not llm.api_key:
        problems.append("DEEPSEEK_API_KEY 未设置")

    for kind, provider, key in (
        ("image", media.image_provider, media.ark_api_key),
        ("video", media.video_provider, media.ark_api_key),
    ):
        if provider == "mock":
            _line(OK, kind, "mock（本地渲染，零成本）")
        elif kind == "video" and provider == "kling":
            if media.kling_api_key and media.kling_base_url:
                _line(OK, kind, f"kling / {media.kling_model} @ {media.kling_base_url}")
            else:
                problems.append("VIDEO_PROVIDER 是 kling 但 KLING_API_KEY / KLING_BASE_URL 为空")
                _line(FAIL, kind, "kling 已选但凭据或端点为空")
        elif key:
            _line(OK, kind, f"ark / {media.ark_image_model if kind == 'image' else media.ark_video_model}")
        else:
            problems.append(f"{kind} provider 是 {provider} 但 ARK_API_KEY 为空")
            _line(FAIL, kind, f"{provider} 已选但 ARK_API_KEY 为空")

    if media.tts_provider == "mock":
        _line(OK, "tts", "mock（静音轨）")
    elif media.tts_api_key:
        _line(OK, "tts", f"doubao_speech / {media.tts_model}")
    else:
        problems.append("TTS provider 不是 mock 但 TTS_API_KEY 为空")
        _line(FAIL, "tts", "已选 doubao_speech 但 TTS_API_KEY 为空")

    _line(OK, "canvas", f"{media.width}x{media.height}@{media.fps}")
    _line(OK, "video res", media.ark_video_resolution)
    paid = media.image_provider == "ark" or media.video_provider == "ark"
    if paid:
        print("\n  注意：已选择付费 provider，真实调用需要 ARK_ALLOW_PAID=1")
    if media.video_provider == "kling":
        print("  注意：Kling 是付费 provider，真实调用需要 KLING_ALLOW_PAID=1")
    return problems


def main() -> int:
    print(f"平台：{platform_key()}  Python {sys.version.split()[0]}\n")
    problems = check_binaries() + check_config()
    print()
    if problems:
        print(f"发现 {len(problems)} 个问题，按上面的提示处理后重跑。")
        return 1
    print("环境检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
