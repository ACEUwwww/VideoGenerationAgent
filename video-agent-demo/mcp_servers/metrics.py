"""Frame-level measurements behind the Video-Bench style evaluator.

Every number here is computed from decoded pixels. Nothing is guessed from a
prompt string, so a broken render cannot score well on momentum alone.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from mcp_servers.providers.ffmpeg_utils import ffmpeg_bin

THUMB = (96, 170)


def extract_frames(video_path: Path, count: int = 8, width: int = 192) -> list[np.ndarray]:
    """Decode ``count`` evenly spaced frames as RGB uint8 arrays."""
    video_path = Path(video_path)
    if not video_path.exists():
        return []
    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        out = Path(tmp) / "f%03d.png"
        proc = subprocess.run(
            [
                ffmpeg_bin(), "-y", "-i", str(video_path),
                "-vf", f"fps=4,scale={width}:-2",
                "-frames:v", str(max(count * 4, count)),
                str(out),
            ],
            capture_output=True, text=True, timeout=180,
        )
        files = sorted(Path(tmp).glob("f*.png"))
        if proc.returncode != 0 or not files:
            return []
        if len(files) > count:
            step = len(files) / count
            files = [files[min(int(i * step), len(files) - 1)] for i in range(count)]
        return [np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) for f in files]


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _laplacian_var(gray: np.ndarray) -> float:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    total = 0.0
    for y in range(1, gray.shape[0] - 1):
        row = (
            gray[y - 1, 1:-1] + gray[y + 1, 1:-1] + gray[y, :-2] + gray[y, 2:]
            - 4 * gray[y, 1:-1]
        )
        total += float(np.mean(row * row))
    return total / max(gray.shape[0] - 2, 1)


def _colorfulness(frame: np.ndarray) -> float:
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))


def thumb_vector(frame: np.ndarray) -> np.ndarray:
    """Public structure descriptor used by keyframe memory."""
    return _thumb(frame)


def histogram_vector(frame: np.ndarray) -> np.ndarray:
    """Public colour descriptor used by keyframe memory."""
    return _histogram(frame)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Public similarity used by keyframe memory."""
    return _cosine(a, b)


def _thumb(frame: np.ndarray) -> np.ndarray:
    """Structure descriptor: mean-removed luma, so a global brightness shift
    does not read as a structural match while layout differences do."""
    gray = _gray(frame)
    image = Image.fromarray(gray.astype(np.uint8)).resize(THUMB, Image.BILINEAR)
    vector = np.asarray(image, dtype=np.float32).reshape(-1)
    vector = vector - vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _histogram(frame: np.ndarray, bins: int = 16) -> np.ndarray:
    hist = []
    for channel in range(3):
        h, _ = np.histogram(frame[..., channel], bins=bins, range=(0, 255))
        hist.append(h.astype(np.float32))
    vector = np.concatenate(hist)
    total = vector.sum()
    return vector / total if total else vector


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0 else float(np.dot(a, b) / denominator)


def _band(value: float, low: float, ideal_low: float, ideal_high: float, high: float) -> float:
    """Score a measurement against an acceptable band.

    Full marks inside [ideal_low, ideal_high], falling to 0 at ``low`` and
    ``high``. Real generated footage sits well above what a static card does,
    so a single-target ramp penalises genuine motion as if it were flicker.
    """
    if value < ideal_low:
        return _squash(value, low, ideal_low)
    if value > ideal_high:
        return 1.0 - _squash(value, ideal_high, high)
    return 1.0


def _squash(value: float, low: float, high: float) -> float:
    """Map a raw measurement to 0..1 with a linear ramp between low and high."""
    if high == low:
        return 0.0
    return float(min(max((value - low) / (high - low), 0.0), 1.0))


def measure(frames: list[np.ndarray]) -> dict[str, float]:
    """Raw per-clip measurements, before any normalisation to scores."""
    if not frames:
        return {}
    grays = [_gray(f) for f in frames]
    diffs = [float(np.mean(np.abs(grays[i + 1] - grays[i]))) for i in range(len(grays) - 1)]
    luminance = [float(np.mean(g)) for g in grays]
    return {
        "frames": len(frames),
        "sharpness": float(np.mean([_laplacian_var(g) for g in grays])),
        "contrast": float(np.mean([np.std(g) for g in grays])),
        "colorfulness": float(np.mean([_colorfulness(f) for f in frames])),
        "brightness": float(np.mean(luminance)),
        "brightness_jitter": float(np.std(luminance)),
        "mean_frame_delta": float(np.mean(diffs)) if diffs else 0.0,
        "delta_std": float(np.std(diffs)) if diffs else 0.0,
        "flicker": float(np.max([abs(diffs[i + 1] - diffs[i]) for i in range(len(diffs) - 1)])) if len(diffs) > 1 else 0.0,
    }


def center_crop(frame: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """Crop the central region, where the subject of a portrait shot sits.

    Comparing whole frames across different shot sizes punishes any framing
    change. The subject lives in the middle of a 9:16 frame, so the centre
    crop is the part that should stay consistent when the camera moves.
    """
    height, width = frame.shape[:2]
    half_h, half_w = int(height * ratio / 2), int(width * ratio / 2)
    top, left = max(height // 2 - half_h, 0), max(width // 2 - half_w, 0)
    return frame[top : height // 2 + half_h, left : width // 2 + half_w]


def compare_to_reference(
    frames: list[np.ndarray],
    reference: np.ndarray,
    *,
    framing_comparable: bool = True,
) -> dict[str, float]:
    """Drift against a reference keyframe (the character/scene memory anchor)."""
    if not frames:
        return {}
    ref_hist = _histogram(reference)
    ref_thumb = _thumb(reference)
    ref_center_thumb = _thumb(center_crop(reference))
    ref_center_hist = _histogram(center_crop(reference))
    hist_scores = [_cosine(_histogram(f), ref_hist) for f in frames]
    thumb_scores = [_cosine(_thumb(f), ref_thumb) for f in frames]
    center_scores = [_cosine(_thumb(center_crop(f)), ref_center_thumb) for f in frames]
    center_hist_scores = [_cosine(_histogram(center_crop(f)), ref_center_hist) for f in frames]
    return {
        "color_consistency": float(np.mean(hist_scores)),
        "structure_consistency": float(np.mean(thumb_scores)),
        "structure_drift_max": float(max(1.0 - s for s in thumb_scores)),
        # Subject-local drift: stable under camera moves, sensitive to the
        # character changing face, wardrobe or colour grade.
        "subject_consistency": float(np.mean(center_scores)),
        "subject_color_consistency": float(np.mean(center_hist_scores)),
        "framing_comparable": framing_comparable,
    }


def score_dimensions(raw: dict[str, float], reference: dict[str, float] | None = None) -> dict[str, dict]:
    """Convert raw measurements into Video-Bench style dimension scores.

    Each entry is ``{score, detail}`` with score in 0..1 and higher is better.
    Thresholds come from typical 1080x1920 short-drama footage: a clean clip
    sits around sharpness 60-200, frame delta 2-12, jitter under 3.
    """
    if not raw:
        return {}
    reference = reference or {}

    sharp = _squash(raw.get("sharpness", 0.0), 8.0, 90.0)
    contrast = _squash(raw.get("contrast", 0.0), 18.0, 62.0)
    colorful = _squash(raw.get("colorfulness", 0.0), 12.0, 62.0)
    exposure = 1.0 - min(abs(raw.get("brightness", 0.0) - 118.0) / 118.0, 1.0)

    delta = raw.get("mean_frame_delta", 0.0)
    # 2-6 is a near-static card; 8-45 is a normally moving live-action shot;
    # above ~70 the clip is degenerating into noise.
    motion = _band(delta, 0.5, 8.0, 45.0, 70.0)
    # Flicker is the second difference of frame deltas. Real footage with fast
    # motion legitimately reaches ~10; a strobing render keeps climbing.
    smooth = 1.0 - _squash(raw.get("flicker", 0.0), 12.0, 40.0)
    # Brightness jitter of a whole frame: generated clips swing more than
    # locked-off footage, so the tolerance starts at 6.
    stable = 1.0 - _squash(raw.get("brightness_jitter", 0.0), 6.0, 30.0)
    temporal = float(np.mean([smooth, stable]))

    color_consistency = reference.get("color_consistency", 0.0)
    structure_consistency = reference.get("structure_consistency", 0.0)

    return {
        "imaging_quality": {
            "score": round(float(np.mean([sharp, contrast, exposure])), 3),
            "detail": {"sharpness": round(raw.get("sharpness", 0), 2), "contrast": round(raw.get("contrast", 0), 2)},
        },
        "aesthetic_quality": {
            "score": round(float(np.mean([colorful, exposure, contrast])), 3),
            "detail": {"colorfulness": round(raw.get("colorfulness", 0), 2)},
        },
        "temporal_consistency": {
            "score": round(temporal, 3),
            "detail": {"flicker": round(raw.get("flicker", 0), 2), "brightness_jitter": round(raw.get("brightness_jitter", 0), 2)},
        },
        "motion_effects": {
            "score": round(float(np.mean([motion, 1.0 - _squash(raw.get("delta_std", 0.0), 8.0, 30.0)])), 3),
            "detail": {"mean_frame_delta": round(delta, 2), "delta_std": round(raw.get("delta_std", 0), 2)},
        },
        "color_consistency": {
            "score": round(color_consistency, 3),
            "detail": {"reference_hist_cosine": round(color_consistency, 3)},
        },
        "scene_consistency": {
            "score": round(structure_consistency, 3),
            "detail": {"reference_thumb_cosine": round(structure_consistency, 3)},
        },
        "object_consistency": {
            # Subject structure is only comparable at similar framing; the
            # subject's colour grade survives a camera change, so it carries
            # the judgement alone when framing differs.
            "score": round(
                float(
                    np.mean(
                        [
                            reference.get("subject_consistency", structure_consistency),
                            reference.get("subject_color_consistency", color_consistency),
                        ]
                    )
                    if reference.get("framing_comparable", True)
                    else reference.get("subject_color_consistency", color_consistency)
                ),
                3,
            ),
            "detail": {
                "subject_structure": round(reference.get("subject_consistency", 0.0), 3),
                "subject_color": round(reference.get("subject_color_consistency", 0.0), 3),
                "drift_max": round(reference.get("structure_drift_max", 0.0), 3),
            },
        },
    }
