"""Turn a video file into the two frame payloads Video-Bench judges consume.

Upstream's ``Video_Dataset`` decodes at the source resolution and base64-encodes
whatever comes out: a 32-second 1080x1920 clip measured 17.8 MB per video per
judge call. That is large enough to be rejected by request-size limits and large
enough to make a 9-dimension run expensive. Both payload shapes are already
downscaled by the judge before it looks at them, so encoding full resolution
only wastes bandwidth.

Two payloads per video, matching upstream:

* ``frames``     — stills sampled at ``sample_fps``, shown one per image part.
* ``grid_frames`` — 1xN strips of consecutive sampled frames, used by the two
  dynamic dimensions so the judge can compare neighbouring frames.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import JudgeConfig


def _cv2():
    """OpenCV is only in the bench environment, not the pipeline's.

    Importing it at module scope would make the whole harness unimportable
    from the main test suite, where the parser and message-shape tests among
    others do not need it.
    """
    import cv2

    return cv2


@dataclass
class FramePayload:
    frames: list[str]
    grid_frames: list[str]
    source: str


def _resize_to_max_edge(frame: np.ndarray, max_edge: int) -> np.ndarray:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return frame
    scale = max_edge / longest
    return cv2.resize(frame, (max(2, int(width * scale)), max(2, int(height * scale))), interpolation=cv2.INTER_AREA)


def _encode(frame: np.ndarray, quality: int) -> str:
    cv2 = _cv2()
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buffer).decode("ascii")


def _sample(video_path: Path, judge: JudgeConfig) -> list[np.ndarray]:
    """Frames sampled at ``judge.sample_fps``, downscaled once here.

    Seeking by frame index rather than decoding every frame keeps a 30fps,
    32-second clip at 64 decoded frames.
    """
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        step = max(1, int(round(fps / max(1, judge.sample_fps))))
        out: list[np.ndarray] = []
        index = 0
        while index < total:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            out.append(_resize_to_max_edge(frame, judge.max_edge))
            index += step
        return out
    finally:
        capture.release()


def _grids(frames: list[np.ndarray], grid_size: int, quality: int) -> list[str]:
    """1xN strips, N consecutive sampled frames each.

    Upstream builds these per 8 frames. Frames beyond the last full strip are
    dropped rather than padded, which is what upstream does too.
    """
    strips: list[str] = []
    for start in range(0, len(frames) - grid_size + 1, grid_size):
        group = frames[start:start + grid_size]
        strip = np.concatenate(group, axis=1)
        strips.append(_encode(strip, quality))
    return strips


def build_payload(video_path: Path, judge: JudgeConfig) -> FramePayload:
    frames = _sample(Path(video_path), judge)
    return FramePayload(
        frames=[_encode(frame, judge.jpeg_quality) for frame in frames],
        grid_frames=_grids(frames, judge.grid_size, judge.jpeg_quality),
        source=str(video_path),
    )


def payload_bytes(payload: FramePayload) -> int:
    """Encoded size of one video's payload, for budgeting before a paid run."""
    return sum(len(x) for x in payload.frames) + sum(len(x) for x in payload.grid_frames)
