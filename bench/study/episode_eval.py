"""Episode-level evaluation: what a workflow adds that clip scores cannot show.

Video-Bench scores a five-second clip. Every dimension it has is frame-level
technical quality or text alignment, so a pipeline that writes a script, keeps a
cast consistent across shots, and delivers subtitles plus audio is
indistinguishable from one that renders a single prompt — if the clip that comes
out looks similar. `bench/DESIGN_longvideo.md` already established this; this
module is the measurement side of it.

Three layers, scored separately and never summed:

  layer 1  per-segment quality      existing Video-Bench judge, per segment
  layer 2  cross-shot identity      fingerprint drift, cut-anchored
  layer 3  structure                shot count, duration, cuts, audio, subtitles

Layer 2 has one trap that has already produced a wrong number in this project
and must not be repeated: an episode with cuts cannot be compared against its
own first frame, because a cut changes the picture on purpose. Each shot is
anchored on *its own* first window, cuts are detected rather than assumed from
the storyboard grid, and windows that straddle a cut or a crossfade are dropped
rather than scored. Done the naive way this measured our own episode at
identity_min -0.83 and nearly went into the write-up as a regression.

Cost is a first-class metric here rather than a footnote: the interesting claim
about a workflow is quality *per unit of delivered video*, so every row carries
wall time and finished seconds.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EpisodeReport:
    model: str
    path: Path
    duration_s: float
    shots: int
    cuts: list[float] = field(default_factory=list)
    identity: list[dict] = field(default_factory=list)
    structure: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def probe(path: Path) -> dict:
    """Container facts, read from ffprobe rather than trusted from the run."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {
        "duration_s": round(float((data.get("format") or {}).get("duration") or 0.0), 2),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": _fps(video.get("r_frame_rate") or "0/1"),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def _fps(value: str) -> float:
    num, _, den = value.partition("/")
    try:
        return round(float(num) / float(den or 1), 2)
    except (ValueError, ZeroDivisionError):
        return 0.0


def scene_cuts(path: Path, threshold: float = 0.25) -> list[float]:
    """Cut times from ffmpeg's scene score, de-duplicated to one per cut."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path),
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    )
    times = sorted({
        round(float(m), 2) for m in re.findall(r"pts_time:([0-9.]+)", out.stderr)
    })
    merged: list[float] = []
    for t in times:
        if not merged or t - merged[-1] > 0.08:
            merged.append(t)
    return merged


def title_card_end(path: Path, max_s: float = 2.5) -> float:
    """Where the opening title card stops, or 0.0 when there is none.

    The composition step can burn a black card with the logline on it for the
    first ~1.2 s. It shares no structure with the first shot, so a window that
    includes it measures the card rather than the clip. Left in, it dragged
    shot 0 to 0.37 and the whole episode's identity_min with it — the same
    artifact, from the same cause, that README 11.8 already had to exclude.
    Detected rather than assumed: an empty project has no card and must not
    lose its first window.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(fps / 10))
        last_dark = 0.0
        index = 0
        while index < int(max_s * fps):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            # A title card is near-black with type on it: very low mean level.
            if float(frame.mean()) < 40.0:
                last_dark = index / fps
            index += step
        return round(last_dark, 2)
    finally:
        capture.release()


def verify_bounds(path: Path, bounds: list[float], *, window: float = 0.25,
                  floor: float = 6.0) -> list[float]:
    """Keep only the declared boundaries that a real picture change backs up.

    Needed because the two sides edit differently. A hard cut is found by
    ffmpeg's scene detector; a 0.25-0.40s crossfade is not, because the frames
    either side of it are blended rather than replaced. Measured on our own
    15-shot episode: the scene detector returned **zero** cuts, so the whole
    64s film was scored as one continuous shot and reported identity_min -0.537
    — an artefact of the edit, not a property of the generation.

    So the declared storyboard boundaries are used when the detector is blind,
    but only after each one is checked against the decoded frames. A boundary
    with no visible change is dropped rather than trusted, which keeps this from
    becoming "believe whatever the pipeline claims".
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    import numpy as np

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        kept: list[float] = []
        for bound in bounds:
            frames = []
            for t in (bound - window, bound + window):
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
                ok, frame = capture.read()
                frames.append(frame.astype(np.float32) if ok else None)
            before, after = frames
            if before is None or after is None:
                continue
            # A crossfade is a gradual change, so the difference at a soft cut is
            # smaller than at a hard one (measured 9-35 here, hard cuts 25-35).
            # The floor rejects a boundary where nothing changed at all.
            if float(np.abs(before - after).mean()) >= floor:
                kept.append(round(bound, 2))
        return kept
    finally:
        capture.release()


def end_fade_start(path: Path, min_s: float = 1.0, ramp: float = 6.0) -> float:
    """Where the closing fade-out starts, or the duration when there is none.

    Symmetric to :func:`title_card_end`, and needed for the same reason. Our
    composition fades the picture out over the last few seconds, so that shot's
    later windows are compared against a first window at full exposure while
    themselves getting darker — which is a change of exposure, not of identity.
    Measured on the 15-shot study episode: without this, the final shot read
    0.386 / 0.190 / 0.311 and pulled the episode's identity_min from 0.71 to
    0.19. Detected from the mean-level ramp rather than assumed, so a film that
    ends on a hard cut keeps its last window.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return 0.0
        duration = total / fps
        step = max(1, int(fps / 2))
        samples: list[tuple[float, float]] = []
        index = max(0, int((duration - max(min_s, ramp) - 1.0) * fps))
        while index < total:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            samples.append((index / fps, float(frame.mean())))
            index += step
        if len(samples) < 4:
            return duration
        peak = max(v for _, v in samples)
        # Walk back from the end while the level is still falling. The first
        # sample of that run is where the ramp begins.
        start = duration
        for t, level in reversed(samples):
            if level < peak * 0.85:
                start = t
            else:
                break
        # A ramp shorter than half a second is a hard cut, not a fade.
        return round(start, 2) if duration - start >= 0.5 else duration
    finally:
        capture.release()


def within_shot_identity(path: Path, cuts: list[float], window: float = 1.0,
                         guard: float = 0.7, skip_before: float = 0.0,
                         skip_after: float = 0.0) -> list[dict]:
    """Identity per window, each compared to its own shot's first window.

    Windows overlapping a cut (plus ``guard`` on both sides) are excluded: their
    fingerprint changes because the film cut, not because the model drifted, and
    including them measures the editor instead of the generator.

    ``skip_before`` drops the opening title card. It is the first shot's anchor
    window that suffers most, and since every later window in that shot is
    compared against it, one contaminated anchor corrupts the whole shot.
    ``skip_after`` does the same for a closing fade-out.
    """
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bench.segments import cross_segment, segment_clip

    segments = segment_clip(path, window)
    bounds = [0.0, *cuts]

    def _straddles(seg) -> bool:
        if skip_before and seg.start_s < skip_before:
            return True
        # Any overlap with the fade, not just a window that starts after it: a
        # 1s window beginning at 62.0 still holds frames up to 63.0, and the fade
        # began at 62.27. Checking the start alone left that window in and it read
        # 0.386 against a full-exposure anchor.
        if skip_after and seg.end_s > skip_after:
            return True
        return any(seg.start_s - guard < c < seg.end_s + guard for c in cuts)

    by_shot: dict[int, list] = {i: [] for i in range(len(bounds))}
    for seg in segments:
        if _straddles(seg) or not seg.frame_count:
            continue
        shot = max(i for i, b in enumerate(bounds) if seg.start_s >= b - 1e-6)
        by_shot[shot].append(seg)

    rows: list[dict] = []
    for shot in sorted(by_shot):
        group = by_shot[shot]
        if len(group) < 2:
            # A shot shorter than two clean windows yields no within-shot
            # measurement. Reporting the shot's absence is honest; inventing a
            # value by comparing across the cut is not.
            continue
        ident, _ = cross_segment(group)
        for item in ident:
            rows.append({
                "shot": shot,
                "start_s": item["start_s"],
                "identity": round(min(item["color"], item["structure"]), 3),
                # The two components are kept because they measure different
                # things and can disagree by a whole unit. On shot_009 of a
                # measured episode the picture went from a wide back view to a
                # front close-up of the same character in the same clothing:
                # colour stayed at 0.885 while structure fell to 0.694, and on
                # shot_010 they read 0.901 and -0.071. Structure is a
                # mean-removed luma fingerprint, so it tracks *layout and
                # framing*; it drops when the camera moves even though the
                # character is unchanged. Reporting only their minimum and
                # calling it "identity" would claim a character-consistency
                # result this metric cannot support.
                "color": item["color"],
                "structure": item["structure"],
            })
    return rows


def summarise(rows: list[dict], structure: dict, cuts: list[float], duration: float) -> dict:
    values = [r["identity"] for r in rows]
    # Colour and structure are summarised separately as well as combined. The
    # combined figure is dominated by structure whenever the framing changes,
    # so a reader who wants "did the palette hold" needs the colour column and
    # a reader who wants "did the framing hold" needs the structure column. See
    # within_shot_identity for the measured example where they diverge.
    colors = [r["color"] for r in rows if "color" in r]
    structures = [r["structure"] for r in rows if "structure" in r]
    per_shot: dict[int, list[float]] = {}
    for r in rows:
        per_shot.setdefault(r["shot"], []).append(r["identity"])
    shot_min = {k: min(v) for k, v in per_shot.items()}

    def _median(items: list[float]) -> float | None:
        return round(sorted(items)[len(items) // 2], 3) if items else None

    return {
        "measurements": len(values),
        "shots_measured": len(per_shot),
        "identity_min": round(min(values), 3) if values else None,
        "identity_median": _median(values),
        "identity_mean": round(sum(values) / len(values), 3) if values else None,
        "identity_spread": round(max(values) - min(values), 3) if values else None,
        "color_median": _median(colors),
        "color_min": round(min(colors), 3) if colors else None,
        "structure_median": _median(structures),
        "structure_min": round(min(structures), 3) if structures else None,
        "worst_shot_identity": round(min(shot_min.values()), 3) if shot_min else None,
        "shots_below_075": sum(1 for v in shot_min.values() if v < 0.75),
        "cuts": len(cuts),
        "duration_s": duration,
        "seconds_per_shot": round(duration / structure["shots"], 2) if structure.get("shots") else None,
    }


def evaluate_episode(path: Path, model: str, *, shots: int = 0,
                     elapsed_s: float = 0.0,
                     declared_bounds: list[float] | None = None) -> EpisodeReport:
    """All layers for one episode.

    ``declared_bounds`` are the workflow's own shot boundaries. When they are
    supplied they are the primary source and are checked against decoded frames
    by ``verify_bounds``; the scene detector is only a fallback for calls that
    have no declared timeline.

    That precedence is a correction, not a preference. The first version used
    declared bounds only when the detector returned *zero* cuts. On the second
    episode of the same workflow the detector returned two spurious cuts (5.13 s
    and 35.07 s), so none of the fourteen real boundaries was used and a 64 s,
    15-shot film was scored as **three** shots — identity_median -0.157,
    identity_min -0.993. The first episode happened to return zero cuts, which
    is why the bug was invisible there. A detector missing crossfades is
    unreliable; a detector that also fires on them is worse, because it silently
    overrides the timeline instead of deferring to it.
    """
    facts = probe(path)
    cut_source = "detected"
    cuts: list[float] = []
    if declared_bounds:
        cuts = verify_bounds(path, declared_bounds)
        cut_source = "declared-verified"
    if not cuts:
        cuts = scene_cuts(path)
        cut_source = "detected"
    lead_in = title_card_end(path)
    fade_out = end_fade_start(path)
    rows = within_shot_identity(path, cuts, skip_before=lead_in, skip_after=fade_out)
    structure = {
        "shots": shots or (len(cuts) + 1),
        "title_card_s": lead_in,
        "end_fade_s": fade_out if fade_out < facts["duration_s"] else None,
        "cut_source": cut_source,
        "width": facts["width"],
        "height": facts["height"],
        "fps": facts["fps"],
        "has_audio": facts["has_audio"],
        "elapsed_s": elapsed_s,
        "delivered_seconds": facts["duration_s"],
    }
    report = EpisodeReport(
        model=model, path=path, duration_s=facts["duration_s"],
        shots=structure["shots"], cuts=cuts, identity=rows, structure=structure,
    )
    report.summary = summarise(rows, structure, cuts, facts["duration_s"])
    if elapsed_s and facts["duration_s"]:
        report.summary["realtime_factor"] = round(elapsed_s / facts["duration_s"], 2)
    return report


def to_dict(report: EpisodeReport) -> dict:
    return {
        "model": report.model,
        "path": str(report.path),
        "duration_s": report.duration_s,
        "shots": report.shots,
        "cuts": report.cuts,
        "structure": report.structure,
        "summary": report.summary,
        "identity": report.identity,
    }
