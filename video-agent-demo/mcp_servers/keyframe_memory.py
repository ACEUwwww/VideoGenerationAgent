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


def anchor_entry(project_root: Path) -> dict | None:
    memory = read_memory(project_root)
    return next((e for e in memory["entries"] if e["path"] == memory["anchor"]), None)


def anchor_frame(project_root: Path) -> np.ndarray | None:
    entry = anchor_entry(project_root)
    return load_reference(project_root / entry["path"]) if entry else None
