"""Which prompts the comparison runs on.

The prompt suite is Video-Bench's own (``videobench/VideoBench_full.json``, 419
annotated prompts tagged with the dimension each one tests). Using it has two
concrete benefits over inventing prompts:

* The numbers are directly comparable to the published leaderboard.
* Every prompt already carries the dimension it was written for, so a
  ``color`` prompt is scored by the ``color`` judge rather than hoping a random
  sentence exercises one.

``custom_*`` mode in upstream expects prompts to be derived from filenames, and
``get_prompt_from_filename`` strips a trailing ``-<int>``. Prompts carrying
characters that a filesystem rejects are therefore unsafe as filenames; this
module gives each one a stable slug and keeps the mapping alongside the data so
the original text is never lost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import require_vendor


@dataclass(frozen=True)
class Prompt:
    text: str
    dimensions: tuple[str, ...]
    slug: str

    def to_dict(self) -> dict:
        return {"text": self.text, "dimensions": list(self.dimensions), "slug": self.slug}


def _slug(text: str) -> str:
    """A filename-safe identifier that stays stable across runs.

    The hash guards against two prompts whose readable part coincides after
    non-ASCII and punctuation are stripped — Video-Bench's suite is entirely
    English, but the pipeline's own prompts are not.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    readable = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in text)[:48].strip()
    readable = readable.replace(" ", "_") or "prompt"
    return f"{readable}_{digest}"


def load_suite(vendor: Path | None = None) -> list[dict]:
    root = vendor or require_vendor()
    path = root / "videobench" / "VideoBench_full.json"
    return json.loads(path.read_text(encoding="utf-8"))


def select_prompts(
    count: int,
    dimension: str = "",
    vendor: Path | None = None,
) -> list[Prompt]:
    """The first ``count`` prompts, optionally filtered to one dimension.

    Order comes from the suite file, so the same ``count`` always selects the
    same prompts — a benchmark whose prompt set drifted between runs could not
    be compared to itself. ``count=0`` means the whole suite.
    """
    entries = load_suite(vendor)
    chosen = []
    for entry in entries:
        dims = entry["dimension"] if isinstance(entry["dimension"], list) else [entry["dimension"]]
        if dimension and dimension not in dims:
            continue
        chosen.append(Prompt(text=entry["prompt"], dimensions=tuple(dims), slug=_slug(entry["prompt"])))
        if count and len(chosen) >= count:
            break
    return chosen


def load_prompt_file(path: Path) -> list[Prompt]:
    """Custom prompts from a plain JSON list or an ``{index: text}`` mapping."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        texts = [str(v) for _, v in sorted(raw.items(), key=lambda kv: str(kv[0]))]
    elif isinstance(raw, list):
        texts = [str(item["text"] if isinstance(item, dict) else item) for item in raw]
    else:
        raise SystemExit(f"{path} must hold a JSON list or an index->text mapping")
    return [Prompt(text=text, dimensions=(), slug=_slug(text)) for text in texts]


def write_manifest(prompts: list[Prompt], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.to_dict() for p in prompts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
