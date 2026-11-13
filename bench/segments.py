"""Segmented evaluation: score long videos the way long videos actually fail.

A single score for a two-minute clip hides the thing worth knowing. The failure
mode that separates long-video workflows is not "the whole thing is mediocre" —
it is "shot 4 is fine, shot 9 has a different face, shot 14 wanders off". That
only shows up when the clip is cut into segments and each segment is measured
against its neighbours.

So this module does three things in sequence:

  layer 1  segment a clip into fixed windows and score each window's quality
  layer 2  measure what changes *between* windows: identity, scene, tone, cuts
  layer 3  summarise the sequence as a whole (narrative shape, pacing drift)

Why fixed windows rather than shot boundaries: a shot boundary is a decision the
workflow itself made, so detecting shots would reward or punish a pipeline for
its own editing choices before it is scored. Constant windows give every model
the same ruler. The cost is that a cut can land mid-window; the transition
metrics in layer 2 are what catch that, so it is detected rather than ignored.

Everything here is local and free — no judge calls. Layer 1's *quality* scores
come from the existing judge; this module handles the geometry, the sampling and
the cross-segment metrics.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Segment boundaries. 5 seconds is long enough that a clip contains real motion
# (shorter windows make `motion_effects` meaningless) and short enough that a
# two-minute clip splits into two dozen segments, which is enough resolution to
# see drift begin.
DEFAULT_SEGMENT_S = 5.0

# Frames sampled inside one segment for the cross-segment metrics. Two is the
# minimum that can express motion; eight gives the mode enough support without
# making payloads large.
FRAMES_PER_SEGMENT = 8


@dataclass
class Segment:
    """One fixed window of a clip, with its own measurements."""

    index: int
    start_s: float
    end_s: float
    frame_count: int
    # Local metrics, computed without any judge call.
    motion: float = 0.0
    flicker: float = 0.0
    brightness: float = 0.0
    colorfulness: float = 0.0
    # Fingerprint of the segment's first frame, for layer 2.
    hist: list[float] = field(default_factory=list)
    thumb: list[float] = field(default_factory=list)


@dataclass
class SegmentReport:
    clip: Path
    model: str
    duration_s: float
    segments: list[Segment]

    # layer 2 results
    identity: list[dict] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    # layer 3 summary
    summary: dict = field(default_factory=dict)


def _cv2():
    """OpenCV lives in the bench environment only."""
    import cv2

    return cv2


def _fingerprint(frame: np.ndarray) -> tuple[list[float], list[float]]:
    """Colour and structure vectors, matching keyframe_memory's split.

    Reusing that decomposition keeps this comparable with the pipeline's own
    drift metric instead of introducing a second, incomparable notion of
    "similar". Imported lazily: the pipeline modules live outside the bench
    environment's package path when only the bench venv is active.
    """
    from mcp_servers.metrics import histogram_vector, thumb_vector

    small = _resize(frame, 192)
    return histogram_vector(small).tolist(), thumb_vector(small)[:256].tolist()


def _resize(frame: np.ndarray, target_width: int) -> np.ndarray:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame.astype(np.float32)
    scale = target_width / width
    return cv2.resize(
        frame, (target_width, max(2, int(height * scale))), interpolation=cv2.INTER_AREA
    ).astype(np.float32)


def _cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity, with a floor for degenerate vectors.

    A flat, single-colour frame produces a near-zero vector: the histogram has
    one bin and the mean-removed thumbnail is flat. Comparing two of those gives
    a ratio of two noise terms — measured at -0.089 for two identical frames —
    which reads as "completely different" when the truth is "there is nothing
    here to compare". Values below the noise floor report 1.0 (indistinguishable)
    rather than an arbitrary sign.
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    # 1e-3 sits above float32 noise on a 256-vector of image statistics and far
    # below any real content vector.
    if na < 1e-3 or nb < 1e-3:
        return 1.0
    return float(np.dot(va, vb) / (na * nb))


def segment_clip(video_path: Path, segment_s: float = DEFAULT_SEGMENT_S) -> list[Segment]:
    """Cut a clip into fixed windows and measure each one locally.

    Frames are decoded once, in order, and bucketed by timestamp — seeking per
    segment would re-open the file dozens of times and, on short clips, land on
    the same keyframe repeatedly.
    """
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        buckets: dict[int, list[np.ndarray]] = {}
        index = 0
        while index < total:
            ok, frame = capture.read()
            if not ok:
                break
            bucket = int((index / fps) // segment_s)
            buckets.setdefault(bucket, []).append(frame)
            index += 1
    finally:
        capture.release()

    if not buckets:
        return []

    # A final window holding a handful of frames is a rounding artefact, not a
    # segment: a 5.06s clip splits 80 frames / 1 frame, and scoring that
    # one-frame tail produces a spurious "drift" reading. Fold anything shorter
    # than half a window into its predecessor.
    minimum = max(2, int(FRAMES_PER_SEGMENT))
    last = max(buckets)
    if last > 0 and len(buckets[last]) < minimum:
        buckets[last - 1].extend(buckets.pop(last))

    segments: list[Segment] = []
    for bucket in sorted(buckets):
        frames = buckets[bucket]
        start = bucket * segment_s
        segments.append(_measure_segment(bucket, start, segment_s, frames))
    return segments


def _measure_segment(index: int, start: float, segment_s: float, frames: list[np.ndarray]) -> Segment:
    """Local measurements for one window.

    These deliberately duplicate nothing from the judge: they are the cheap,
    deterministic signals that can be computed for every segment without a
    paid call, so the curve has full resolution even when judging is sparse.

    The fingerprint averages several frames rather than using the first one.
    A single frame is too unstable a representative: measured on a clip
    repeated three times — which should read as perfectly stable — first-frame
    fingerprints gave identity 0.449, no better than a genuinely mixed clip at
    0.546. Averaging over the window makes the fingerprint describe the
    segment rather than whichever instant happened to be sampled.
    """
    import cv2

    small = [cv2.resize(f, (160, 90)).astype(np.float32) for f in frames]
    diffs = [float(np.abs(small[i] - small[i - 1]).mean()) for i in range(1, len(small))]
    # Flicker is the second difference of the mean level: a smooth pan changes
    # brightness steadily, a flicker reverses direction every frame.
    levels = [float(f.mean()) for f in small]
    second = [abs(levels[i] - 2 * levels[i - 1] + levels[i - 2]) for i in range(2, len(levels))]

    from mcp_servers.metrics import histogram_vector, thumb_vector

    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)]
    resized = np.stack([cv2.resize(f, (192, 342), interpolation=cv2.INTER_AREA).astype(np.float32) for f in picks])
    mean_frame = resized.mean(axis=0)
    hist = histogram_vector(mean_frame).tolist()
    thumb = thumb_vector(mean_frame)[:256].tolist()

    hsv = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2HSV)
    return Segment(
        index=index,
        start_s=start,
        end_s=start + segment_s,
        frame_count=len(frames),
        motion=float(np.mean(diffs)) if diffs else 0.0,
        flicker=float(np.mean(second)) if second else 0.0,
        brightness=levels[0] if levels else 0.0,
        colorfulness=float(hsv[:, :, 1].mean()),
        hist=hist,
        thumb=thumb,
    )


def cross_segment(segments: list[Segment]) -> tuple[list[dict], list[dict]]:
    """Layer 2: what changes between windows.

    Two different questions, kept apart because they fail independently:

    *identity* — is it still the same scene/palette? Compares each segment
    against the first, never against its neighbour. Anchoring on the first
    segment is the same choice the pipeline makes for keyframes: a chain
    comparison accumulates error and makes a slow drift look stable.

    *transitions* — is the boundary between consecutive segments a cut, a
    fade, or a mistake? Large first-frame differences plus a spike in
    brightness change mean the workflow jumped; a black gap means it dropped
    frames.
    """
    usable = [s for s in segments if s.frame_count and s.hist]
    if len(usable) < 2:
        return [], []

    anchor = usable[0]
    identity = [
        {
            "index": s.index,
            "start_s": s.start_s,
            "color": round(_cosine(anchor.hist, s.hist), 3),
            "structure": round(_cosine(anchor.thumb, s.thumb), 3),
            "brightness_delta": round(s.brightness - anchor.brightness, 2),
        }
        for s in usable[1:]
    ]

    transitions = []
    for prev, cur in zip(usable, usable[1:]):
        gap = cur.start_s - prev.end_s
        transitions.append(
            {
                "from": prev.index,
                "to": cur.index,
                "brightness_jump": round(abs(cur.brightness - prev.brightness), 2),
                "motion_delta": round(abs(cur.motion - prev.motion), 2),
                "color_delta": round(1.0 - _cosine(prev.hist, cur.hist), 3),
                "structure_delta": round(1.0 - _cosine(prev.thumb, cur.thumb), 3),
                "empty_segment": cur.frame_count == 0 or prev.frame_count == 0,
                "gap_s": round(gap, 3),
            }
        )
    return identity, transitions


def summarise(segments: list[Segment], identity: list[dict], transitions: list[dict]) -> dict:
    """Layer 3: the sequence as a whole.

    Reported as shape, not as one number. "Worst segment", "where it starts to
    drift" and "how much the score moves" answer different questions, and
    averaging them together would hide the one that matters.
    """
    usable = [s for s in segments if s.frame_count]
    if not usable:
        return {"segments": 0}

    identity_scores = [min(item["color"], item["structure"]) for item in identity] or [1.0]
    motions = [s.motion for s in usable]
    colors = [s.colorfulness for s in usable]
    levels = [s.brightness for s in usable]

    # Drift onset: the first segment that has fallen away from the anchor on
    # either axis. Checking structure alone misses a palette change — measured
    # on a clip whose colour was replaced but whose layout was kept: structure
    # stayed at 0.997 while colour fell to 0.416, and the old check reported no
    # drift at all. Either kind of change is drift to a viewer.
    onset = None
    for item in identity:
        if min(item["structure"], item["color"]) < 0.75:
            onset = item["index"]
            break

    return {
        "segments": len(usable),
        "duration_s": round(sum(s.end_s - s.start_s for s in usable), 2),
        "identity_min": round(min(identity_scores), 3),
        "identity_median": round(statistics.median(identity_scores), 3),
        "identity_spread": round(max(identity_scores) - min(identity_scores), 3),
        "drift_onset_segment": onset,
        "motion_median": round(statistics.median(motions), 2),
        "motion_spread": round(max(motions) - min(motions), 2),
        "colorfulness_median": round(statistics.median(colors), 2),
        "brightness_spread": round(max(levels) - min(levels), 2),
        "empty_segments": sum(1 for s in segments if s.frame_count == 0),
        "worst_transition": max(transitions, key=lambda t: t["structure_delta"], default=None),
    }


def evaluate_clip(video_path: Path, model: str, segment_s: float = DEFAULT_SEGMENT_S) -> SegmentReport:
    """All three layers for one clip, with no paid calls."""
    segments = segment_clip(video_path, segment_s)
    identity, transitions = cross_segment(segments)
    report = SegmentReport(
        clip=video_path,
        model=model,
        duration_s=sum(s.end_s - s.start_s for s in segments if s.frame_count),
        segments=segments,
        identity=identity,
        transitions=transitions,
        summary=summarise(segments, identity, transitions),
    )
    return report


def to_dict(report: SegmentReport) -> dict:
    """Serialisable form, without the per-segment fingerprints.

    The vectors are 256 floats each; keeping them in the JSON makes the file
    unreadable for a person and useless for a report. The derived numbers are
    what anyone reads.
    """
    return {
        "clip": str(report.clip),
        "model": report.model,
        "duration_s": round(report.duration_s, 2),
        "summary": report.summary,
        "segments": [
            {
                "index": s.index,
                "start_s": round(s.start_s, 2),
                "end_s": round(s.end_s, 2),
                "frame_count": s.frame_count,
                "motion": round(s.motion, 2),
                "flicker": round(s.flicker, 3),
                "brightness": round(s.brightness, 2),
                "colorfulness": round(s.colorfulness, 2),
            }
            for s in report.segments
        ],
        "identity": report.identity,
        "transitions": report.transitions,
    }
