"""Character consistency across shots, scored by the local VLM.

Why this exists, and why it is a separate script: the fingerprint metric the
bench reports as "identity" does not measure who is in the frame. It is a
mean-removed luma descriptor, so it collapses when the camera moves even though
the character is unchanged. Measured on one clip that cut from a wide back view
to a front close-up of the same character in the same clothing, colour stayed at
0.885 while structure fell to 0.694; another clip read 0.901 and -0.071. A paper
that reported those numbers as character consistency would be describing the
wrong thing.

So this asks a vision-language model the question the project actually claims:
given several shots from the same episode, is the person recognisably the same
person — same face, hair, build, clothing — and did anything in their appearance
change. It is scored per episode, not per frame, because the question is
relational.

Cost: nothing. The judge is the local Qwen2.5-VL on 127.0.0.1, already running
for the clip bench. Requests are serialised by the server's own lock, so this
only slows the other dimension passes rather than fighting them.

Design choices worth stating:

* One frame per shot, taken at 60% into the clip. The first frames of a Seedance
  render are the keyframe the workflow itself supplied; scoring those would
  measure the keyframe model and not the video. The midpoint is where the
  render's own drift has had time to show.
* Shots are presented in their own order and each labelled with its index, so a
  dropped or reordered frame is visible in the raw reply rather than silently
  changing what was compared.
* The model must answer with a number on a fixed scale and name the specific
  trait that changed. A bare number cannot be audited; the named trait can be
  checked against the frames by hand.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

OM = ROOT / "bench/_vendor/OpenMontage"
OM_PROJECTS = OM / "bench_run/projects"
OUT_DIR = ROOT / "bench/results/compare_workflows"

SYSTEM = (
    "You are comparing frames taken from different shots of one short vertical drama "
    "episode. Each frame is labelled with its shot index. The question is CHARACTER "
    "CONSISTENCY: is the main person in these frames the same person throughout?\n\n"
    "Judge only these traits: face and facial features, hairstyle, body build and height, "
    "and clothing (garment type, colour, pattern).\n\n"
    "Ignore everything else. Camera angle, framing, background, lighting, colour grading, "
    "image sharpness and composition are NOT part of this question and must not affect "
    "the score. A shot from behind and a close-up of the same person is a perfect match.\n\n"
    "Answer in exactly this format:\n"
    "SCORE: <a number from 1 to 5, where 5 = clearly the same person in every frame, "
    "1 = a different person in most frames>\n"
    "CHANGED: <the single trait that changed most, or NONE if nothing changed>\n"
    "WHY: <one sentence, at most 30 words, naming what you actually saw>"
)


def grab(path: Path, at_fraction: float = 0.6) -> bytes | None:
    """One JPEG frame from a fixed fraction into the clip, via ffmpeg."""
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip() or 0)
    if not duration:
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        out = Path(handle.name)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{duration * at_fraction:.3f}", "-i", str(path),
             "-frames:v", "1", "-vf", "scale='min(640,iw)':-2", "-y", str(out)],
            capture_output=True, timeout=120)
        return out.read_bytes() if out.exists() and out.stat().st_size else None
    finally:
        out.unlink(missing_ok=True)


def ask(frames: list[tuple[int, bytes]], *, base: str, model: str,
        timeout: float = 900.0) -> str:
    content = [{"type": "text", "text": SYSTEM}]
    for index, jpeg in frames:
        content.append({"type": "text", "text": f"Shot {index}:"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    content.append({"type": "text", "text": "Now give SCORE / CHANGED / WHY."})
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 200,
    }).encode()
    import urllib.request
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    return payload["choices"][0]["message"]["content"]


def parse(reply: str) -> dict:
    """Read the three labelled lines, tolerating the markdown the model adds.

    The first pass over a live reply came back as ``**SCORE:** 5`` — the model
    leans on bold headings even when asked for plain labels — and a strict
    ``SCORE:`` pattern returned None for every episode while ``CHANGED`` still
    parsed, which read as "no score anywhere" instead of "the parser was too
    literal". Asterisks and underscores are stripped first.
    """
    text = reply.replace("*", "").replace("`", "")
    score = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text)
    changed = re.search(r"CHANGED:\s*(.+)", text)
    why = re.search(r"WHY:\s*(.+)", text)
    return {
        "score": float(score.group(1)) if score else None,
        "changed": (changed.group(1).strip()[:120] if changed else ""),
        "why": (why.group(1).strip()[:240] if why else ""),
        "raw": reply[-600:],
    }


def our_shots(project_id: str) -> list[Path]:
    directory = ROOT / "workspace/projects" / project_id / "videos"
    return sorted(directory.glob("shot_*.mp4")) if directory.is_dir() else []


def om_shots(project_id: str) -> list[Path]:
    state_path = OM_PROJECTS / project_id / "studio_state.json"
    if not state_path.exists():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = sorted((s for s in (state.get("shots") or []) if not s.get("deletedAt")),
                  key=lambda s: int(s.get("shotOrder") or 0))
    paths = []
    for shot in rows:
        take = next((v for v in shot.get("videos") or []
                     if v.get("id") == shot.get("selectedVideoId")), None)
        if take:
            path = OM_PROJECTS / project_id / take["path"]
            if path.exists():
                paths.append(path)
    return paths


def score_episode(label: str, shots: list[Path], *, base: str, model: str,
                  max_shots: int = 12) -> dict:
    """Evenly sample up to ``max_shots`` shots; more frames than that get crowded."""
    if len(shots) > max_shots:
        step = len(shots) / max_shots
        chosen = [shots[int(i * step)] for i in range(max_shots)]
    else:
        chosen = shots
    frames = []
    for index, path in enumerate(chosen):
        jpeg = grab(path)
        if jpeg:
            frames.append((index, jpeg))
    if len(frames) < 2:
        return {"label": label, "score": None, "error": "not enough frames", "shots": 0}
    started = time.monotonic()
    reply = ask(frames, base=base, model=model)
    result = parse(reply)
    result.update(label=label, shots=len(frames), elapsed_s=round(time.monotonic() - started, 1))
    return result


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Character consistency across shots")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "character_consistency.json")
    parser.add_argument("--base", default="http://127.0.0.1:8791/v1")
    parser.add_argument("--model", default="qwen2.5-vl-32b-local")
    parser.add_argument("--max-shots", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=1,
                        help="ask the same episode this many times; the model is deterministic "
                             "at temperature 0, so >1 checks stability of the answer itself")
    args = parser.parse_args(argv[1:])


    tasks: list[tuple[str, list[Path]]] = []
    # Score exactly the episodes the workflow comparison reports on, and nothing
    # else. Two earlier attempts got this wrong in the same direction: the first
    # scored every project in the workspace (seven of them from before the
    # study), and the second scored every completed episode of ours — which by
    # then included briefs 0 and 2, none of which OpenMontage ran. Either way the
    # character score would have described a different episode set than the rest
    # of the report. compare_workflows.py owns the pairing decision, so its
    # output is the only correct source.
    comparison = ROOT / "bench/results/compare_workflows/episodes.json"
    if not comparison.exists():
        raise SystemExit("no compare_workflows/episodes.json; run compare_workflows.py first")
    paired_rows = json.loads(comparison.read_text(encoding="utf-8")).get("episodes") or []
    for row in paired_rows:
        project_id = row.get("project_id") or ""
        if not project_id:
            continue
        if row.get("side") == "ours":
            shots = our_shots(project_id)
            label = f"ours:{project_id}"
        else:
            shots = om_shots(project_id)
            label = f"openmontage:{project_id}"
        if shots:
            tasks.append((label, shots))

    print(f"{len(tasks)} episodes to score")
    results = []
    for label, shots in tasks:
        for repeat in range(args.repeats):
            try:
                row = score_episode(label, shots, base=args.base, model=args.model,
                                    max_shots=args.max_shots)
            except Exception as exc:  # noqa: BLE001 - one episode must not sink the batch
                row = {"label": label, "score": None, "error": str(exc)[:300]}
            row["repeat"] = repeat
            results.append(row)
            print(f"  {label:28s} shots={row.get('shots', 0):2d} "
                  f"score={row.get('score')} changed={row.get('changed', '')[:40]!r}", flush=True)
            # Written after every episode: a long run must keep its evidence.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
