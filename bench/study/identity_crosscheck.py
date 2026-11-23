"""Pixel-side identity cross-check for the repeated-measure study.

Why this exists: the tier3a result ("carried text state raises cross-shot
character consistency from 2.28 to 4.61") is scored by one local vision-language
model. A single judge, with no second opinion on the same clips, is the weakest
link in that claim. This script asks a different question of the same 180
renders with a measurement that shares no code path with the VLM: do the faces
in shot 1 and shot 5 carry the same *identity embedding*?

The two metrics are deliberately not the same object:

* the VLM answers "is this recognisably the same person" from full frames;
* this answers "how close are the face embeddings" from a detected face crop.

Agreement between them is evidence the claim is about the character and not an
artefact of one judge's prompt. Disagreement is equally informative and is
reported, not smoothed over.

Design choices worth stating, because each one can move the number:

* **Several frames per shot, not one.** A single frame at a fixed fraction
  catches the character turned away often enough that it silently removes shots
  from the sample. Each shot is sampled at ``FRAME_FRACTIONS`` and the largest
  detectable face across those frames is used. The cost is that this no longer
  sees exactly the frame the VLM judge saw, so agreement between the two is
  structural rather than frame-exact; the earlier single-frame behaviour is
  still reachable with ``--frames 0.6``.
* **The no-face case is defined, not left implicit.** DreamFactory's CSFD (the
  metric this is a stand-in for) averages over all ``n(n-1)/2`` pairs and never
  says what a pair contributes when one side has no detectable face. A frame
  without a face is a normal outcome here: wide shots, over-the-shoulder
  framings and empty establishing shots all produce them. This script drops
  such pairs, and reports ``skipped_pair_ratio`` next to every score, because a
  dropped pair biases the mean toward whichever shots happen to be close-ups.
  An episode whose shot-level detection rate is below ``MIN_DETECTION_RATE`` is
  marked inapplicable instead of being scored on the two shots that happened to
  have faces.
* **Largest face, one per shot.** The study's briefs are single-protagonist,
  so the largest box is the character under test. Smaller boxes are counted and
  ignored.
* **Faces smaller than ``MIN_FACE_PX`` are not embedded.** An identity vector
  from a 30-pixel face is noise that still looks like a number. Those frames
  count as no-face and inflate ``skipped_pair_ratio`` where that actually
  happens.
* **Frames are not upscaled.** They are decoded at native clip resolution; the
  detector runs on whatever the model actually rendered. Upscaling was measured
  as a control and recovered only a handful of frames.

Backbone: InceptionResnetV1 pretrained on VGGFace2 (``facenet-pytorch``), face
detection by the bundled MTCNN. This is not ArcFace; it is the same class of
measurement (a face-identity embedding, cosine-scored) from a checkpoint that
downloads without the InsightFace package. Swapping the backbone changes the
absolute cosines and should not change the paired comparison, which is what the
report uses.

Cost: nothing. No paid calls; one frame per clip, CPU is sufficient.

Usage:
    python bench/study/identity_crosscheck.py            # score and write JSON
    python bench/study/identity_crosscheck.py --limit 3  # smoke test
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "bench/results/tier3a_repeats"
CLIPS = REP / "clips"
OUT = REP / "identity_crosscheck.json"

MODELS = ("cogvideox5b", "wan21_1_3b")
ARMS = ("independent", "planned")

# Frames sampled per shot. 60% alone matched the VLM judge's frame but missed
# the character whenever the shot framed them away from camera.
FRAME_FRACTIONS = (0.15, 0.3, 0.45, 0.6, 0.8)
# Below this the crop is too small to carry identity; see the module docstring.
MIN_FACE_PX = 48
# An episode with fewer detectable faces than this is not scored at all.
MIN_DETECTION_RATE = 0.6

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# ---------------------------------------------------------------- data layout


def episode_dirs() -> list[dict]:
    """Every arm/model/brief/repeat cell, named as the repeat study names them."""
    episodes = []
    if not CLIPS.is_dir():
        return episodes
    # Model ids contain underscores (wan21_1_3b), so the cell name is parsed by
    # regex rather than split: independent_wan21_1_3b_b0_r0 is six fields, not four.
    pattern = re.compile(r"^(independent|planned)_(.+)_b(\d+)_r(\d+)$")
    for path in sorted(CLIPS.iterdir()):
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        arm, model = match.group(1), match.group(2)
        if model not in MODELS:
            continue
        brief = int(match.group(3))
        repeat = int(match.group(4))
        episodes.append(
            {
                "label": path.name,
                "dir": path,
                "arm": arm,
                "model": model,
                "brief_index": brief,
                "repeat": repeat,
            }
        )
    return episodes


def shot_videos(path: Path) -> list[Path]:
    return sorted((path / "videos").glob("shot_*.mp4"))


# --------------------------------------------------------------- frame access


def duration_s(path: Path) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def grab_frame(path: Path, fraction: float = 0.6) -> np.ndarray | None:
    """One RGB frame from a fixed fraction into the clip."""
    total = duration_s(path)
    if total <= 0:
        return None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        out = Path(handle.name)
    try:
        subprocess.run(
            [FFMPEG, "-v", "error", "-ss", f"{total * fraction:.3f}", "-i", str(path),
             "-frames:v", "1", "-y", str(out)],
            capture_output=True, timeout=120,
        )
        if not out.exists() or out.stat().st_size == 0:
            return None
        return np.asarray(Image.open(out).convert("RGB"))
    finally:
        out.unlink(missing_ok=True)


# ------------------------------------------------------------------- identity


class FaceEncoder:
    """Largest-face detection plus a face-identity embedding, on CPU."""

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN

        self._torch = torch
        self.device = torch.device(device if device != "cpu" and torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(keep_all=True, device=self.device, post_process=False)
        self.encoder = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(self.device)

    def encode(self, frame: np.ndarray) -> dict:
        """The largest usable face in ``frame``, or the reason there is none."""
        image = Image.fromarray(frame)
        boxes, probs = self.detector.detect(image)
        if boxes is None or len(boxes) == 0:
            return {"vec": None, "reason": "no_face", "faces": 0, "face_px": 0}

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best = int(np.argmax(areas))
        box = boxes[best]
        face_px = int(min(box[2] - box[0], box[3] - box[1]))
        if face_px < MIN_FACE_PX:
            return {"vec": None, "reason": "face_too_small", "faces": len(boxes), "face_px": face_px}

        with self._torch.no_grad():
            aligned = self.detector.extract(image, [box], save_path=None)
            if aligned is None or len(aligned) == 0:
                return {"vec": None, "reason": "align_failed", "faces": len(boxes), "face_px": face_px}
            vec = self.encoder(aligned.to(self.device).float())
        vec = vec.detach().cpu().numpy().reshape(-1)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return {
            "vec": vec,
            "reason": "ok",
            "faces": int(len(boxes)),
            "face_px": face_px,
            "det_prob": float(probs[best]) if probs is not None else None,
        }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ------------------------------------------------------------- one episode


def score_episode(encoder: FaceEncoder, episode: dict, fractions: tuple[float, ...]) -> dict:
    videos = shot_videos(episode["dir"])
    per_shot = []
    for index, video in enumerate(videos):
        chosen: dict | None = None
        for fraction in fractions:
            frame = grab_frame(video, fraction)
            if frame is None:
                continue
            result = encoder.encode(frame)
            result["shot"] = index
            result["fraction"] = fraction
            if chosen is None:
                chosen = result
            elif result["vec"] is not None and (
                chosen["vec"] is None or result["face_px"] > chosen["face_px"]
            ):
                # A usable face always beats an unusable crop; among usable
                # crops the largest face wins.
                chosen = result
        per_shot.append(chosen or {"shot": index, "vec": None, "reason": "no_frame", "faces": 0, "face_px": 0})

    detected = [s for s in per_shot if s["vec"] is not None]
    detection_rate = len(detected) / len(per_shot) if per_shot else 0.0

    sims = []
    skipped = 0
    for a, b in itertools.combinations(range(len(per_shot)), 2):
        va, vb = per_shot[a]["vec"], per_shot[b]["vec"]
        if va is None or vb is None:
            skipped += 1
            continue
        sims.append(cosine(va, vb))

    pairs_total = len(sims) + skipped
    return {
        "label": episode["label"],
        "arm": episode["arm"],
        "model": episode["model"],
        "brief_index": episode["brief_index"],
        "repeat": episode["repeat"],
        "shots": len(per_shot),
        "detected_shots": len(detected),
        "detection_rate": round(detection_rate, 3),
        "pairs_total": pairs_total,
        "pairs_scored": len(sims),
        "skipped_pair_ratio": round(skipped / pairs_total, 3) if pairs_total else 1.0,
        "mean_cosine": round(statistics.mean(sims), 4) if sims else None,
        "min_cosine": round(min(sims), 4) if sims else None,
        "median_cosine": round(statistics.median(sims), 4) if sims else None,
        "applicable": bool(detection_rate >= MIN_DETECTION_RATE and len(sims) > 0),
        "per_shot": [
            {
                "shot": s["shot"],
                "reason": s["reason"],
                "faces": s["faces"],
                "face_px": s["face_px"],
            }
            for s in per_shot
        ],
    }


# --------------------------------------------------------------- statistics


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "median": round(statistics.median(values), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def sign_test(diffs: list[float]) -> dict:
    """Exact one-sided sign test, in the direction the hypothesis predicts."""
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return {"wins": 0, "ties": len(diffs), "losses": 0, "n": 0, "p_greater": 1.0}
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return {
        "wins": wins,
        "ties": len(diffs) - n,
        "losses": losses,
        "n": n,
        "p_greater": round(p, 8),
    }


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation test on the mean difference."""
    if not diffs:
        return 1.0
    observed = abs(statistics.mean(diffs))
    hits = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        if abs(statistics.mean([s * d for s, d in zip(signs, diffs)])) >= observed - 1e-12:
            hits += 1
    return round(hits / total, 8)


def bootstrap_ci(diffs: list[float], trials: int = 5000, seed: int = 0) -> list[float]:
    if not diffs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(trials))
    return [round(means[int(0.025 * trials)], 4), round(means[int(0.975 * trials)], 4)]


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, no scipy dependency."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den else None


def pair_arms(scored: list[dict], arm_better: str = "planned", arm_worse: str = "independent") -> dict | None:
    """Paired difference of the two arms within one model, keyed by cell."""
    by_cell: dict[tuple[int, int], dict[str, float]] = {}
    for row in scored:
        if not row["applicable"] or row["mean_cosine"] is None:
            continue
        by_cell.setdefault((row["brief_index"], row["repeat"]), {})[row["arm"]] = row["mean_cosine"]

    diffs = [
        cell[arm_better] - cell[arm_worse]
        for cell in by_cell.values()
        if arm_better in cell and arm_worse in cell
    ]
    if not diffs:
        return None
    return {
        "n": len(diffs),
        "mean_diff": round(statistics.mean(diffs), 4),
        "sd_diff": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else 0.0,
        "ci95": bootstrap_ci(diffs),
        "sign_test": sign_test(diffs),
        "sign_flip_p": sign_flip_p(diffs),
        "diffs": [round(d, 4) for d in diffs],
    }


def load_vlm_scores() -> dict[tuple[str, str, int, int], float]:
    """Episode-level VLM scores from the repeat study, for the agreement check.

    The judge's records label a cell ``arm:model:b{brief}:r{repeat}``, which is
    not the clip directory name, so the key is rebuilt field by field.
    """
    scores: dict[tuple[str, str, int, int], float] = {}
    for model in MODELS:
        path = REP / f"consistency_{model}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("score") is None:
                continue
            label = row.get("label", "")
            parts = label.split(":")
            if len(parts) != 4:
                continue
            try:
                key = (parts[0], parts[1], int(parts[2].lstrip("b")), int(parts[3].lstrip("r")))
            except ValueError:
                continue
            scores[key] = float(row["score"])
    return scores


# --------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N episodes")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--frames", default=",".join(str(f) for f in FRAME_FRACTIONS),
        help="comma-separated fractions into each clip to sample (0.6 alone matches the VLM judge)",
    )
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    fractions = tuple(float(f) for f in args.frames.split(",") if f.strip())

    episodes = episode_dirs()
    if not episodes:
        print(f"no episodes under {CLIPS}", file=sys.stderr)
        return 1
    if args.limit:
        episodes = episodes[: args.limit]

    encoder = FaceEncoder(args.device)
    scored = []
    for position, episode in enumerate(episodes, start=1):
        row = score_episode(encoder, episode, fractions)
        scored.append(row)
        print(
            f"[{position}/{len(episodes)}] {row['label']:<34} "
            f"det={row['detection_rate']:.2f} cos={row['mean_cosine']} "
            f"skip={row['skipped_pair_ratio']:.2f} ok={row['applicable']}",
            flush=True,
        )

    vlm = load_vlm_scores()

    arms: dict[str, dict] = {}
    for model in MODELS:
        arms[model] = {}
        for arm in ARMS:
            values = [
                row["mean_cosine"]
                for row in scored
                if row["model"] == model and row["arm"] == arm
                and row["applicable"] and row["mean_cosine"] is not None
            ]
            subset = [row for row in scored if row["model"] == model and row["arm"] == arm]
            arms[model][arm] = {
                **describe(values),
                "episodes": len(subset),
                "applicable": len(values),
                "mean_skipped_pair_ratio": round(
                    statistics.mean([r["skipped_pair_ratio"] for r in subset]), 3
                ) if subset else None,
                "episode_labels": [r["label"] for r in subset if r["applicable"]],
            }

    pooled_arms: dict[str, dict] = {}
    for arm in ARMS:
        values = [
            row["mean_cosine"]
            for row in scored
            if row["arm"] == arm and row["applicable"] and row["mean_cosine"] is not None
        ]
        pooled_arms[arm] = describe(values)

    pairs = {model: pair_arms([r for r in scored if r["model"] == model]) for model in MODELS}
    pairs["pooled"] = pair_arms(scored)

    agreement = {}
    for model in MODELS + ("pooled",):
        xs, ys = [], []
        for row in scored:
            if not row["applicable"] or row["mean_cosine"] is None:
                continue
            if model != "pooled" and row["model"] != model:
                continue
            key = (row["arm"], row["model"], row["brief_index"], row["repeat"])
            if key in vlm:
                xs.append(row["mean_cosine"])
                ys.append(vlm[key])
        agreement[model] = {"n": len(xs), "spearman": spearman(xs, ys), "xs": xs, "ys": ys}

    # When one arm has too few detectable faces to score, the pooled figure has
    # no spread to correlate with. Report the arm that does have coverage, and
    # label it as such rather than presenting it as the whole study.
    by_arm_agreement = {}
    for arm in ARMS:
        xs, ys = [], []
        for row in scored:
            if row["arm"] != arm or not row["applicable"] or row["mean_cosine"] is None:
                continue
            key = (row["arm"], row["model"], row["brief_index"], row["repeat"])
            if key in vlm:
                xs.append(row["mean_cosine"])
                ys.append(vlm[key])
        by_arm_agreement[arm] = {"n": len(xs), "spearman": spearman(xs, ys), "xs": xs, "ys": ys}

    payload = {
        "meta": {
            "root": str(ROOT),
            "clips": str(CLIPS),
            "frames_per_clip": len(fractions),
            "frame_fractions": list(fractions),
            "min_face_px": MIN_FACE_PX,
            "min_detection_rate": MIN_DETECTION_RATE,
            "backbone": "InceptionResnetV1/vggface2 (facenet-pytorch), MTCNN detector",
            "metric": "cosine between L2-normalised face embeddings, averaged over scorable shot pairs",
            "scored_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        },
        "episodes": scored,
        "arms": arms,
        "pooled_arms": pooled_arms,
        "pairs": pairs,
        "agreement_with_vlm": agreement,
        "agreement_with_vlm_by_arm": by_arm_agreement,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    print("\nshot-level face coverage (usable faces, i.e. >= %d px)" % MIN_FACE_PX)
    for arm in ARMS:
        subset = [row for row in scored if row["arm"] == arm]
        shots = sum(row["shots"] for row in subset)
        detected = sum(row["detected_shots"] for row in subset)
        usable = sum(1 for row in subset if row["applicable"])
        rate = detected / shots if shots else 0.0
        print(
            f"  {arm:<12} shots_with_face={detected}/{shots} ({rate:.1%}) "
            f"episodes_applicable={usable}/{len(subset)}"
        )

    print("\narm means (face-embedding cosine, applicable episodes only)")
    for model in MODELS:
        row = arms[model]
        print(
            f"  {model:<14} independent={row['independent'].get('mean')} "
            f"(n={row['independent'].get('applicable')}) "
            f"planned={row['planned'].get('mean')} "
            f"(n={row['planned'].get('applicable')})"
        )
        if pairs[model]:
            p = pairs[model]
            print(
                f"    paired diff={p['mean_diff']:+.4f} CI95={p['ci95']} "
                f"w/t/l={p['sign_test']['wins']}/{p['sign_test']['ties']}/{p['sign_test']['losses']} "
                f"p={p['sign_test']['p_greater']}"
            )
        print(f"    agreement with VLM: rho={agreement[model]['spearman']} n={agreement[model]['n']}")

    if pairs["pooled"]:
        p = pairs["pooled"]
        print(
            f"  {'pooled':<14} paired diff={p['mean_diff']:+.4f} CI95={p['ci95']} "
            f"w/t/l={p['sign_test']['wins']}/{p['sign_test']['ties']}/{p['sign_test']['losses']} "
            f"p={p['sign_test']['p_greater']}"
        )
    print(f"  {'pooled':<14} agreement with VLM: rho={agreement['pooled']['spearman']} n={agreement['pooled']['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
