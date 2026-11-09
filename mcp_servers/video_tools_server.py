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
    last_frame_path: str = "",
) -> dict:
    media = get_settings().media
    root = ensure_project_dirs(project_id)
    last_frame = root / last_frame_path if last_frame_path else None
    if last_frame is not None and not last_frame.exists():
        last_frame = None
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
        last_frame_path=last_frame,
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
# Storyboards author transitions as labels. Each label maps to an ffmpeg xfade
# transition plus the overlap it needs; the overlap is real time, so the
# subtitles and the audio bed are shifted by the same amounts.
TRANSITIONS: dict[str, tuple[str | None, float]] = {
    "叠化": ("fade", 0.40),
    "dissolve": ("fade", 0.40),
    "淡入": ("fade", 0.40),
    "闪白": ("fadewhite", 0.25),
    "推黑": ("fadeblack", 0.30),
    "淡出": ("fadeblack", 0.30),
    "硬切": (None, 0.0),
    "cut": (None, 0.0),
}
TITLE_CARD_SECONDS = 1.2
TITLE_CARD_TRANSITION = ("fade", 0.40)


def _transition_for(name: str | None, enabled: bool) -> tuple[str | None, float]:
    """Map a storyboard transition label onto (xfade name, overlap seconds)."""
    if not enabled or not name:
        return None, 0.0
    for label, spec in TRANSITIONS.items():
        if label in str(name):
            return spec
    return None, 0.0


def _make_title_card(root: Path, media, title: str, subtitle: str) -> Path:
    """A short 9:16 card carrying the episode title over a silent audio bed."""
    target = root / "videos" / "_title_card.mp4"
    font = _escape_filter_path(font_path())
    filters = (
        f"drawtext=fontfile='{font}':text='{_escape_drawtext(title)[:20]}':fontcolor=white"
        f":fontsize={int(media.width * 0.11)}:x=(w-text_w)/2:y=(h-text_h)/2"
        f":borderw=6:bordercolor=black@0.85"
    )
    if subtitle:
        filters += (
            f",drawtext=fontfile='{font}':text='{_escape_drawtext(subtitle)[:26]}'"
            f":fontcolor=white@0.82:fontsize={int(media.width * 0.05)}"
            f":x=(w-text_w)/2:y=(h-text_h)/2+{int(media.width * 0.13)}"
        )
    run([
        ffmpeg_bin(), "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x0b0d18:s={media.width}x{media.height}:d={TITLE_CARD_SECONDS:.3f}:r={media.fps}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-vf", filters,
        "-t", f"{TITLE_CARD_SECONDS:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-pix_fmt", "yuv420p", str(target),
    ])
    return target


def _combine(previous: Path, incoming: Path, out: Path, kind: str | None, overlap: float, elapsed: float) -> None:
    """Append one segment, using a cross-transition or a hard cut."""
    if kind and overlap > 0:
        offset = max(elapsed - overlap, 0.0)
        graph = (
            f"[0:v][1:v]xfade=transition={kind}:duration={overlap:.3f}:offset={offset:.3f}[v];"
            f"[0:a][1:a]acrossfade=d={overlap:.3f}[a]"
        )
    else:
        graph = "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"
    run([
        ffmpeg_bin(), "-y", "-i", str(previous), "-i", str(incoming),
        "-filter_complex", graph,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out),
    ])


def compose_video(
    clips: list[dict],
    project_id: str,
    *,
    title: str = "",
    subtitle_lines: list[dict] | None = None,
    bgm_path: str | None = None,
) -> dict:
    """Concat shot clips with their transitions, burn subtitles, mix BGM.

    ``clips`` is a list of ``{video_path, audio_path, duration_s, transition,
    subtitle_lines}`` in story order. ``transition`` is the boundary *after*
    that clip. Subtitle times on each clip are relative to its own start and are
    re-based onto the encoded timeline, which differs from a plain sum once
    transitions overlap and a title card is prepended.
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

    # Build the segment list: optional title card, then one per shot.
    segments: list[dict] = []
    if media.title_card and (title or ""):
        segments.append(
            {
                "path": _make_title_card(root, media, title, ""),
                "transition": TITLE_CARD_TRANSITION if media.transitions else (None, 0.0),
                "lines": [],
            }
        )
    for position, clip in enumerate(clips):
        segments.append(
            {
                "path": normalised[position],
                "transition": _transition_for(clip.get("transition"), media.transitions),
                "lines": clip.get("subtitle_lines") or [],
            }
        )
    for segment in segments:
        segment["duration"] = float(media_facts(segment["path"]).get("duration_s", 0.0) or 0.0)

    # Walk the timeline once so subtitle placement and the concat offsets agree.
    starts: list[float] = []
    cursor = 0.0
    for index, segment in enumerate(segments):
        starts.append(cursor)
        overlap = segment["transition"][1] if index < len(segments) - 1 else 0.0
        cursor += segment["duration"] - overlap
    total_duration = cursor

    placed: list[dict] = []
    for index, segment in enumerate(segments):
        for line in segment["lines"]:
            start = starts[index] + float(line["start_s"])
            end = starts[index] + float(line["end_s"])
            placed.append(
                {
                    "text": line["text"],
                    "start_s": round(min(start, total_duration), 2),
                    "end_s": round(min(end, total_duration), 2),
                }
            )
    if not any(segment["lines"] for segment in segments):
        placed = list(subtitle_lines or [])

    joined = root / "final" / "_joined.mp4"
    current = segments[0]["path"]
    elapsed = segments[0]["duration"]
    for index in range(1, len(segments)):
        previous_transition = segments[index - 1]["transition"]
        out = root / "final" / f"_join_{index:03d}.mp4"
        _combine(
            current,
            segments[index]["path"],
            out,
            previous_transition[0],
            previous_transition[1],
            elapsed,
        )
        if current.name.startswith("_join_"):
            current.unlink(missing_ok=True)
        current = out
        elapsed += segments[index]["duration"] - previous_transition[1]
    if current != joined:
        run([ffmpeg_bin(), "-y", "-i", str(current), "-c", "copy", "-movflags", "+faststart", str(joined)])
        current.unlink(missing_ok=True)

    ass_path = _write_ass(root, placed, media) if placed else None
    target = root / "final" / "final_video.mp4"

    bgm = _resolve_bgm(root, bgm_path or media.bgm_path)
    video_filter = f"ass={_escape_filter_path(ass_path)}" if ass_path else "null"
    args = [ffmpeg_bin(), "-y", "-i", str(joined)]
    if bgm:
        args += ["-stream_loop", "-1", "-i", str(bgm)]
        graph = (
            f"[0:v]{video_filter}[v];"
            f"[1:a]volume={media.bgm_volume:.3f}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
    else:
        graph = f"[0:v]{video_filter}[v];[0:a]anull[a]"
    # Bound the output explicitly. A looped BGM never ends, so -shortest picks
    # the first stream to finish and truncated the episode to the BGM length in
    # practice (3.80s cut to 3.23s). The known segment total is authoritative.
    args += [
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-t", f"{total_duration:.3f}", str(target),
    ]
    run(args)

    for path in normalised + [joined]:
        path.unlink(missing_ok=True)

    # A clip's transition describes the boundary that follows it, so the last
    # clip contributes none. Segment indices are offset by the title card.
    offset = len(segments) - len(clips)
    transitions = [
        {
            "after": index,
            "label": clip.get("transition", ""),
            "applied": segments[index + offset]["transition"][0],
        }
        for index, clip in enumerate(clips[:-1])
    ]
    _write_meta(
        root / "final" / "final_video.json",
        {
            "clips": [clip["video_path"] for clip in clips],
            "title": title,
            "subtitle_lines": len(placed),
            "canvas": f"{media.width}x{media.height}@{media.fps}",
            "duration_s": round(total_duration, 3),
            "title_card": bool(media.title_card and title),
            # The card itself cross-fades into the first shot, which is an
            # overlap on the timeline just like any clip boundary.
            "title_card_transition": (
                segments[0]["transition"][0]
                if len(segments) > len(clips) and segments[0]["transition"][0]
                else ""
            ),
            "bgm": str(bgm) if bgm else "",
            "transitions": transitions,
        },
    )
    return {
        "final_video_path": str(target.relative_to(root)),
        "subtitle_path": str(ass_path.relative_to(root)) if ass_path else "",
        "clip_count": len(clips),
        "duration_s": round(total_duration, 3),
    }


def _resolve_bgm(root: Path, configured: str | None) -> Path | None:
    """BGM may be absolute or relative to the project; missing means none."""
    if not configured:
        return None
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    local = root / candidate
    return local if local.exists() else None


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
