"""Keyframe memory: the cross-shot consistency anchor.

Every accepted clip contributes a keyframe. The project keeps the first
accepted keyframe as its global visual memory; later shots are scored against
it so character/scene drift shows up as a number instead of a vibe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from mcp_servers.metrics import cosine, histogram_vector, thumb_vector
from mcp_servers.providers.ffmpeg_utils import ffmpeg_bin, run


def extract_keyframe(video_path: Path, target: Path, at_s: float = 0.15) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    run([ffmpeg_bin(), "-y", "-ss", f"{at_s:.3f}", "-i", str(video_path), "-frames:v", "1", str(target)])
    return target


def extract_last_frame(video_path: Path, target: Path, back_s: float = 0.1) -> Path:
    """Last usable frame of a clip, for first/last-frame chaining.

    Seeking from the end rather than the start: the final frame can be a
    partially decoded frame on some encoders, so backing off a fraction of a
    second gives a clean one.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    duration = 0.0
    try:
        from mcp_servers.providers.ffmpeg_utils import media_facts

        duration = float(media_facts(video_path).get("duration_s", 0.0) or 0.0)
    except Exception:
        duration = 0.0
    seek = max(duration - back_s, 0.0)
    run([
        ffmpeg_bin(), "-y", "-ss", f"{seek:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-update", "1", str(target),
    ])
    return target


def load_reference(path: Path) -> np.ndarray:
    data = np.asarray(Image.open(path).convert("RGB").resize((192, 342), Image.BILINEAR), dtype=np.float32)
    return data


def fingerprint(path: Path) -> dict:
    frame = load_reference(path)
    return {"hist": histogram_vector(frame).tolist(), "thumb": thumb_vector(frame)[:256].tolist()}


def similarity(a: dict, b: dict) -> dict:
    return {
        "color": round(cosine(np.array(a["hist"], dtype=np.float32), np.array(b["hist"], dtype=np.float32)), 3),
        "structure": round(cosine(np.array(a["thumb"], dtype=np.float32), np.array(b["thumb"], dtype=np.float32)), 3),
    }


def memory_path(project_root: Path) -> Path:
    return project_root / "keyframes" / "memory.json"


def read_memory(project_root: Path) -> dict:
    path = memory_path(project_root)
    if not path.exists():
        return {"anchor": "", "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def record(
    project_root: Path,
    shot_id: str,
    keyframe_path: Path,
    score: float,
    shot_size: str = "",
) -> dict:
    """Add a keyframe to memory; the first accepted shot becomes the anchor."""
    memory = read_memory(project_root)
    fingerprint_data = fingerprint(keyframe_path)
    entry = {
        "shot_id": shot_id,
        "path": str(keyframe_path.relative_to(project_root)),
        "score": score,
        "shot_size": shot_size,
        "fingerprint": fingerprint_data,
        "drift_vs_anchor": None,
    }
    anchor = next((e for e in memory["entries"] if e["path"] == memory["anchor"]), None)
    if anchor is None:
        memory["anchor"] = entry["path"]
    else:
        entry["drift_vs_anchor"] = similarity(anchor["fingerprint"], fingerprint_data)

    memory["entries"] = [e for e in memory["entries"] if e["shot_id"] != shot_id] + [entry]
    memory_path(project_root).parent.mkdir(parents=True, exist_ok=True)
    memory_path(project_root).write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def compact(memory: dict) -> dict:
    """Memory without the per-keyframe fingerprints, for the project state.

    The fingerprints are 256-float vectors per shot — at 12 shots they were 55%
    of state.json, which every node rewrites in full. They stay in memory.json
    where the drift maths needs them; state carries only what the UI shows.
    """
    entries = [
        {
            "shot_id": entry.get("shot_id", ""),
            "path": entry.get("path", ""),
            "score": entry.get("score", 0.0),
            "shot_size": entry.get("shot_size", ""),
            "drift_vs_anchor": entry.get("drift_vs_anchor"),
        }
        for entry in memory.get("entries", [])
    ]
    return {"anchor": memory.get("anchor", ""), "entries": entries}


def anchor_entry(project_root: Path) -> dict | None:
    memory = read_memory(project_root)
    return next((e for e in memory["entries"] if e["path"] == memory["anchor"]), None)


def anchor_frame(project_root: Path) -> np.ndarray | None:
    entry = anchor_entry(project_root)
    return load_reference(project_root / entry["path"]) if entry else None
