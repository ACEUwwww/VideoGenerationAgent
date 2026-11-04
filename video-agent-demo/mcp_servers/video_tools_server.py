"""MCP-like media tools.

Media generation is dispatched to a provider chosen in ``.env``; composition,
subtitles and the cover frame are always done locally with ffmpeg so they stay
deterministic and free. Every function returns paths relative to the project
directory, which is what the frontend serves.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mcp_servers.providers import (
    ImageRequest,
    SpeechRequest,
    VideoRequest,
    get_image_provider,
    get_speech_provider,
    get_video_provider,
)
from mcp_servers.providers.ffmpeg_utils import ffmpeg_bin, font_path, font_family, media_facts, run

from agent.settings import PROJECTS_ROOT, get_settings


def _project_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def ensure_project_dirs(project_id: str) -> Path:
    root = _project_dir(project_id)
    for name in ["shots", "images", "videos", "audio", "subtitles", "final"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_image(
    prompt: str,
    project_id: str,
    shot_id: str,
    *,
    negative_prompt: str = "",
    caption: str = "",
    seed: int = 0,
    reference_path: str = "",
) -> dict:
    media = get_settings().media
    root = ensure_project_dirs(project_id)
    reference = None
    if reference_path:
        candidate = root / reference_path
        if candidate.exists():
            reference = candidate
    request = ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        shot_id=shot_id,
        project_dir=root,
        width=media.width,
        height=media.height,
        seed=seed,
        caption=caption,
        reference_image=reference,
    )
    result = get_image_provider().generate(request)
    _write_meta(request.project_dir / "images" / f"{shot_id}.json", result)
    return result


def generate_video(
    image_path: str,
    motion_prompt: str,
    project_id: str,
    shot_id: str,
    *,
    duration_s: float,
    caption: str = "",
    seed: int = 0,
) -> dict:
    media = get_settings().media
    root = ensure_project_dirs(project_id)
    request = VideoRequest(
        image_path=root / image_path,
        motion_prompt=motion_prompt,
        shot_id=shot_id,
        project_dir=root,
        duration_s=duration_s,
        width=media.width,
        height=media.height,
        fps=media.fps,
        seed=seed,
        caption=caption,
    )
    result = get_video_provider().generate(request)
    _write_meta(root / "videos" / f"{shot_id}.json", result)
    return result


def synthesize_speech(text: str, project_id: str, shot_id: str, *, speaker: str, duration_s: float, voice: str = "") -> dict:
    root = ensure_project_dirs(project_id)
    request = SpeechRequest(
        text=text,
        speaker=speaker,
        project_dir=root,
        shot_id=shot_id,
        duration_s=duration_s,
        voice=voice or get_settings().media.tts_voice,
    )
    result = get_speech_provider().synthesize(request)
    _write_meta(root / "audio" / f"{shot_id}.json", result)
    return result


# --------------------------------------------------------------------------- #
# Audio fitting
# --------------------------------------------------------------------------- #
# Speech length and shot length are decided independently: the LLM writes the
# line, the shot list sets the duration. A line that runs long must not cut the
# picture, and one that runs short must not leave silence driving the mux.
MAX_SPEEDUP = 1.45
MAX_SLOWDOWN = 0.85


def fit_audio(audio_path: str, project_id: str, shot_id: str, *, target_duration: float) -> dict:
    """Return an audio track whose length matches the shot.

    Within the tempo limits the speech is time-stretched, which keeps delivery
    natural. Beyond them the clip is padded or trimmed rather than squeezed
    into something unintelligible.
    """
    root = ensure_project_dirs(project_id)
    source = root / audio_path
    actual = float(media_facts(source).get("duration_s", 0.0) or 0.0)
    target = max(float(target_duration), 0.3)
    if actual <= 0:
        raise ValueError(f"cannot read duration of {audio_path}")

    ratio = actual / target
    if ratio > MAX_SPEEDUP:
        tempo, action = MAX_SPEEDUP, "speedup+pad"
    elif ratio > 1.001:
        tempo, action = ratio, "speedup"
    elif ratio < MAX_SLOWDOWN:
        tempo, action = MAX_SLOWDOWN, "slowdown+pad"
    elif ratio < 0.999:
        tempo, action = ratio, "slowdown"
    else:
        tempo, action = 1.0, "pad"

    chain = [f"atempo={tempo:.4f}"] if abs(tempo - 1.0) > 1e-3 else []
    chain.append(f"apad=whole_dur={target:.3f}")
    chain.append(f"atrim=0:{target:.3f}")

    out = root / "audio" / f"{shot_id}_fit.m4a"
    run(
        [
            ffmpeg_bin(), "-y", "-i", str(source),
            "-af", ",".join(chain),
            "-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "128k",
            str(out),
        ]
    )
    return {
        "audio_path": str(out.relative_to(root)),
        "source_duration_s": round(actual, 3),
        "target_duration_s": round(target, 3),
        "action": action,
        "tempo": round(tempo, 3),
    }


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def compose_video(
    clips: list[dict],
    project_id: str,
    *,
    title: str = "",
    subtitle_lines: list[dict] | None = None,
    bgm_path: str | None = None,
) -> dict:
    """Concat shot clips, burn subtitles, and write the final vertical episode.

    ``clips`` is a list of ``{video_path, audio_path, duration_s}`` in story
    order. Audio is normalised to AAC so clips with and without a speech track
    concatenate without stream mismatch.
    """
    media = get_settings().media
    root = ensure_project_dirs(project_id)
    if not clips:
        raise ValueError("compose_video needs at least one clip")

    normalised: list[Path] = []
    for position, clip in enumerate(clips, start=1):
        source = root / clip["video_path"]
        audio = root / clip["audio_path"] if clip.get("audio_path") else None
        out = root / "videos" / f"_norm_{position:03d}.mp4"
        args = [ffmpeg_bin(), "-y", "-i", str(source)]
        if audio and audio.exists():
            args += ["-i", str(audio)]
            args += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        else:
            args += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
            args += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        args += [
            "-vf",
            f"scale={media.width}:{media.height}:force_original_aspect_ratio=increase,"
            f"crop={media.width}:{media.height},setsar=1,fps={media.fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out),
        ]
        run(args)
        normalised.append(out)

    concat_list = root / "final" / "_concat.txt"
    concat_list.parent.mkdir(parents=True, exist_ok=True)
    concat_list.write_text("".join(f"file '{path.name}'\n".replace(path.name, str(path)) for path in normalised), encoding="utf-8")

    joined = root / "final" / "_joined.mp4"
    run([ffmpeg_bin(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(joined)])

    ass_path = None
    if subtitle_lines:
        ass_path = _write_ass(root, subtitle_lines, media)
    target = root / "final" / "final_video.mp4"

    if ass_path and bgm_path:
        run([
            ffmpeg_bin(), "-y", "-i", str(joined), "-i", str(root / bgm_path),
            "-filter_complex",
            f"[0:v]ass={_escape_filter_path(ass_path)}[v];"
            f"[1:a]volume=0.18,aloop=loop=-1:size=2e9[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(target),
        ])
    elif ass_path:
        run([
            ffmpeg_bin(), "-y", "-i", str(joined),
            "-vf", f"ass={_escape_filter_path(ass_path)}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(target),
        ])
    else:
        run([ffmpeg_bin(), "-y", "-i", str(joined), "-c", "copy", "-movflags", "+faststart", str(target)])

    for path in normalised + [joined, concat_list]:
        path.unlink(missing_ok=True)

    _write_meta(
        root / "final" / "final_video.json",
        {
            "clips": [clip["video_path"] for clip in clips],
            "title": title,
            "subtitle_lines": len(subtitle_lines or []),
            "canvas": f"{media.width}x{media.height}@{media.fps}",
        },
    )
    return {
        "final_video_path": str(target.relative_to(root)),
        "subtitle_path": str(ass_path.relative_to(root)) if ass_path else "",
        "clip_count": len(clips),
    }


def make_cover(project_id: str, *, title: str, subtitle: str = "") -> dict:
    """Episode cover: first shot's keyframe with the title burned in."""
    media = get_settings().media
    root = ensure_project_dirs(project_id)
    source = root / "images" / "shot_001.png"
    if not source.exists():
        raise FileNotFoundError("cover needs images/shot_001.png")
    target = root / "final" / "cover.png"
    safe_title = _escape_drawtext(title)[:24]
    safe_sub = _escape_drawtext(subtitle)[:30]
    font = _escape_filter_path(font_path())
    drawtext = (
        f",drawtext=fontfile='{font}':text='{safe_title}':fontcolor=white:fontsize={int(media.width * 0.085)}"
        f":x=(w-text_w)/2:y=h*0.72:borderw=6:bordercolor=black@0.85"
    )
    if safe_sub:
        drawtext += (
            f",drawtext=fontfile='{font}':text='{safe_sub}':fontcolor=white@0.88:fontsize={int(media.width * 0.042)}"
            f":x=(w-text_w)/2:y=h*0.72+{int(media.width * 0.11)}:borderw=4:bordercolor=black@0.8"
        )
    run([ffmpeg_bin(), "-y", "-i", str(source), "-vf", f"scale={media.width}:{media.height}" + drawtext, "-frames:v", "1", str(target)])
    return {"cover_path": str(target.relative_to(root))}


def _write_ass(root: Path, lines: list[dict], media) -> Path:
    target = root / "subtitles" / "subtitles.ass"
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {media.width}\nPlayResY: {media.height}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_family()},64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,4,2,2,80,80,260,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for line in lines:
        events.append(
            "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}".format(
                start=_ass_time(line["start_s"]),
                end=_ass_time(line["end_s"]),
                text=str(line["text"]).replace("\n", " "),
            )
        )
    target.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return target


def _ass_time(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


def _escape_filter_path(path: Path | str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter argument.

    Forward slashes work on every platform and avoid the backslash doubling
    that Windows paths otherwise need. The drive colon must still be escaped,
    because ffmpeg reads ':' inside a filter value as an option separator.
    """
    candidate = Path(path)
    text = candidate.as_posix() if candidate.is_absolute() else str(path).replace("\\", "/")
    return text.replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace(",", "\\,")
    )


def _write_meta(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
