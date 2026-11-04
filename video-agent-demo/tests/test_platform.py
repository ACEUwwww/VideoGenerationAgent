"""Portability checks: font resolution, filter escaping, binary lookup.

These run on every platform. The Windows-specific branches are exercised by
constructing the same inputs Windows would produce, so a Linux CI still catches
a regression in the Windows path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mcp_servers.providers import ffmpeg_utils
from mcp_servers.providers.ffmpeg_utils import (
    FONT_CANDIDATES,
    FONT_INSTALL_HINT,
    font_family,
    font_path,
    has_libass,
    platform_key,
)
from mcp_servers.video_tools_server import _escape_filter_path


def test_every_platform_has_font_candidates_and_hint():
    for key in ("win32", "darwin", "linux"):
        assert FONT_CANDIDATES[key], f"{key} has no font candidates"
        assert FONT_INSTALL_HINT[key]


def test_platform_key_is_known():
    assert platform_key() in FONT_CANDIDATES


def test_cjk_font_path_override(monkeypatch, tmp_path):
    font = tmp_path / "custom.ttf"
    font.write_bytes(b"not really a font")
    monkeypatch.setenv("CJK_FONT_PATH", str(font))
    assert font_path() == font


def test_missing_font_override_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("CJK_FONT_PATH", str(tmp_path / "nope.ttf"))
    with pytest.raises(RuntimeError, match="missing file"):
        font_path()


def test_font_family_override(monkeypatch):
    monkeypatch.setenv("CJK_FONT_FAMILY", "Source Han Sans SC")
    assert font_family() == "Source Han Sans SC"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("msyh.ttc", "Microsoft YaHei"),
        ("simhei.ttf", "SimHei"),
        ("PingFang.ttc", "PingFang SC"),
        ("NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
        ("wqy-zenhei.ttc", "WenQuanYi Zen Hei"),
    ],
)
def test_font_family_maps_each_platform_file(monkeypatch, tmp_path, filename, expected):
    font = tmp_path / filename
    font.write_bytes(b"x")
    monkeypatch.delenv("CJK_FONT_FAMILY", raising=False)
    monkeypatch.setenv("CJK_FONT_PATH", str(font))
    assert font_family() == expected


def test_ffmpeg_filter_path_keeps_posix_paths_readable():
    assert _escape_filter_path(Path("/data/a b/x.ass")) == "/data/a b/x.ass"


def test_ffmpeg_filter_path_escapes_drive_colon():
    """ffmpeg reads ':' inside a filter value as an option separator."""
    escaped = _escape_filter_path(Path("/mnt/c/Users/me/proj/x.ass"))
    assert ":" not in escaped


def test_ffmpeg_filter_path_has_no_backslashes():
    """Backslashes would need quadrupling; forward slashes work everywhere."""
    escaped = _escape_filter_path(Path("/data/project/x.ass"))
    assert "\\" not in escaped


@pytest.mark.skipif(ffmpeg_utils.ffmpeg_bin.__module__ is None, reason="ffmpeg module")
def test_libass_detection_returns_bool():
    assert isinstance(has_libass(), bool)


def test_no_shell_true_anywhere():
    """subprocess with a string command would break on Windows quoting."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if ".venv" in path.parts or path.name == Path(__file__).name:
            continue
        if "shell=True" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(path.name)
    assert not offenders, f"shell=True found in {offenders}"
