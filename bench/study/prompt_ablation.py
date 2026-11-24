"""Controlled prompt ablation for the cross-shot consistency claim.

Why this exists: the repeated-measure study compares two arms, ``planned`` and
``independent``, that differ in three things at once — prompt length, the
presence of plan-derived scene/action content, and the verbatim character
appearance anchor. A difference between them therefore cannot be attributed to
any one of those. This script rebuilds the comparison as a factorial design in
which every arm is derived from a *single* planner output, so model, seed,
brief, repeat, shot index, resolution and frame count are identical across arms
and the only thing that changes is the prompt string.

## Design

For one (brief, repeat) the planner produces, per shot, one prompt. That prompt
is decomposed clause by clause into:

* **S** — scene / camera / lighting / action clauses
* **A** — character-appearance clauses (face, hair, build, wardrobe, marks)

Five arms are then assembled from S, A and the brief:

| arm | prompt | isolates |
| --- | --- | --- |
| `brief` | the brief verbatim | control, as in the existing ``independent`` arm |
| `brief_anchor` | brief + A | the anchor alone, with no plan structure |
| `scene` | S only | plan content with the anchor removed |
| `planned` | S + A (the planner's own text) | the existing ``planned`` arm |
| `planned_trunc` | the planner's text cut to the brief's length | prompt length alone |

`brief` vs `scene` isolates plan-derived content. `scene` vs `planned` isolates
the anchor. `brief` vs `brief_anchor` isolates the anchor again, with plan
structure held absent rather than present — the two anchor contrasts together
are what make it a factorial rather than a chain. `planned` vs `planned_trunc`
asks whether the effect is length.

## What this still does not isolate

* A is not one factor: it bundles face, hair, build and wardrobe. Splitting it
  further is possible from the same decomposition but multiplies the render
  count; the keyword groups are printed so a later run can do it.
* The planner writes S and A interleaved in one sentence; the clause split is a
  heuristic. Every arm's text is written to disk so the classification can be
  audited without rerunning anything, and the per-arm character counts are
  printed next to the score they produce.
* Arms share seeds with each other, so model noise is common across arms within
  a cell but the clips are not identical — that is what pairing is for.

Usage:
    python bench/study/prompt_ablation.py --inspect            # show the split, render nothing
    python bench/study/prompt_ablation.py --models cogvideox5b --briefs 0 --repeats 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/study"))

OUT = ROOT / "bench/results/prompt_ablation"
BRIEFS = ROOT / "bench/study/briefs.json"
SHOTS = 5

ARMS = ("brief_en", "brief_en_anchor", "scene", "planned", "planned_trunc")

# The length-control arm keeps this share of the planned prompt. 0.45 puts it in
# the same band as the other short arms without matching any one of them.
TRUNC_FRACTION = 0.45

# Appearance-only vocabulary. A clause carrying any of these is describing the
# character, not the scene — that is the whole basis of the split, so the list
# is deliberately narrow and auditable rather than clever.
ANCHOR_TERMS = (
    "face", "facial", "eyelid", "monolid", "cheekbone", "jawline",
    "hair", "hairstyle", "buzz cut", "ponytail", "beard", "stubble",
    "build", "slender", "lean", "slim", "muscular", "stocky",
    "wearing", "wears", "wore", "dressed", "clad",
    "uniform", "shirt", "vest", "jacket", "coat", "hoodie", "sweater",
    "trousers", "pants", "jeans", "shorts", "skirt", "dress",
    "sneakers", "shoes", "boots", "sandals", "barefoot",
    "badge", "name tag", "nametag", "gloves", "glove", "backpack", "satchel",
    "scar", "tattoo", "mole", "freckle", "wrist", "complexion", "pale skin",
    "age", "22-26", "24-28", "years old",
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r"(?<=[,;])\s+")


# A clause that describes what the camera does is scene direction even when it
# happens to mention a body part ("tilts up to his face"). Without this the
# anchor arm silently collected camera moves.
CAMERA_TERMS = (
    "camera", "shot", "tilt", "pan ", "pans", "zoom", "dolly", "push in", "pushes in",
    "pull back", "tracking", "close-up", "wide shot", "medium shot", "frame",
)


def _is_anchor(text: str) -> bool:
    low = text.lower()
    if any(term in low for term in CAMERA_TERMS):
        return False
    return any(re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", low) for term in ANCHOR_TERMS)


def split_prompt(prompt: str) -> tuple[list[str], list[str]]:
    """Split one shot prompt into (scene sentences, appearance sentences).

    Two levels, because one is not enough. Sentence level first: a sentence
    that describes only the character ("He has a round face, short buzz cut,
    ...") is pure anchor and moves whole. A sentence that is clearly scene moves
    whole. Only a *mixed* sentence — the camera clause that ends in a garment
    description — is split at its commas, and its parts are classified
    individually.

    The earlier comma-only split was tried and rejected: without a sentence
    boundary, a scene sentence silently absorbed the following wardrobe clause,
    which inflated the anchor arm with setting text.
    """
    scene, anchor = [], []
    for sentence in _SENTENCE.split(prompt.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        clauses = [c.strip() for c in _CLAUSE.split(sentence) if c.strip()]
        anchor_hits = sum(1 for c in clauses if _is_anchor(c))
        if anchor_hits == 0:
            scene.append(sentence)
        elif anchor_hits == len(clauses):
            anchor.append(sentence)
        else:
            scene.extend(c for c in clauses if not _is_anchor(c))
            anchor.extend(c for c in clauses if _is_anchor(c))
    return scene, anchor


def _tidy(parts: list[str]) -> list[str]:
    """Drop dangling connectives left behind by removing a neighbouring clause."""
    out = []
    for part in parts:
        text = part.strip()
        text = re.sub(r"^(and|but|with|while|then|also|plus)\s+", "", text, flags=re.I)
        text = text.strip(" ,;")
        if len(text) > 2:
            out.append(text)
    return out


def join(clauses: list[str]) -> str:
    """Reassemble parts into one readable prompt string."""
    parts = _tidy(clauses)
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        separator = " " if out.endswith((".", "!", "?")) else ", "
        out += separator + part
    return out.rstrip(",; ").strip()


def truncate_to(prompt: str, fraction: float) -> str:
    """Cut on a sentence boundary at or under ``fraction`` of the prompt's length.

    Measured in characters of the *same* prompt, never against the brief: the
    briefs are Chinese and the planner writes English, so a brief-length limit
    truncates the English prompt to a fragment. That mistake was made once and
    produced an 18-character arm.
    """
    limit = int(len(prompt) * fraction)
    sentences = [s.strip() for s in _SENTENCE.split(prompt.strip()) if s.strip()]
    out = ""
    for sentence in sentences:
        candidate = (out + " " + sentence).strip() if out else sentence
        if len(candidate) > limit:
            break
        out = candidate
    return out or prompt[:limit]


def build_arms(brief_en: str, prompt: str) -> dict[str, dict]:
    """Every arm for one shot, each with the text and how it was made.

    ``brief_en`` is the English rendering of the brief. Sending the raw Chinese
    brief to these models is not a neutral control: CogVideoX and Wan both use
    English text encoders, and an unmatched-language prompt fails for reasons
    that have nothing to do with carried state. The earlier comparison used the
    Chinese brief and therefore had language as a live confound.
    """
    scene, anchor = split_prompt(prompt)
    planned = join(scene + anchor)
    arms = {
        "brief_en": {"text": brief_en},
        "brief_en_anchor": {"text": join([brief_en] + anchor)},
        "scene": {"text": join(scene)},
        "planned": {"text": planned},
        "planned_trunc": {"text": truncate_to(planned, TRUNC_FRACTION)},
    }
    for entry in arms.values():
        entry["chars"] = len(entry["text"])
    return {
        "arms": arms,
        "split": {"scene_clauses": scene, "anchor_clauses": anchor,
                  "scene_chars": len(join(scene)), "anchor_chars": len(join(anchor))},
    }


def brief_en_path(brief_index: int) -> Path:
    return OUT / "briefs_en" / f"b{brief_index}.json"


def get_brief_en(brief: str, brief_index: int) -> str:
    """English rendering of one brief, cached. The control arm's text.

    Translated once and reused by every arm and repeat so the control is a
    fixed string, not a fresh translation each time.
    """
    path = brief_en_path(brief_index)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]

    from agent.llm_client import get_llm

    reply = get_llm().generate_text(
        "You translate Chinese short-drama briefs into English for a text-to-video "
        "model. Output the translation only: no preamble, no commentary, no quotes.",
        "Translate this brief into one English paragraph of 3-5 sentences. Keep every "
        "concrete detail: place, time, weather, who is present, and what happens. "
        "Do not add plot that is not in the source.\n\n" + brief,
        model=None,
    )
    text = (reply.text if hasattr(reply, "text") else str(reply)).strip().strip('"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"brief_index": brief_index, "source": brief, "text": text},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return text


def plan_path(brief_index: int, repeat: int) -> Path:
    return OUT / "plans" / f"b{brief_index}_r{repeat}.json"


def get_plan(brief: str, brief_index: int, repeat: int) -> dict:
    """One planner run, cached. This is the single source every arm derives from."""
    path = plan_path(brief_index, repeat)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    from tier3a_story import story_prompts

    started = time.monotonic()
    prompts = [p for p in story_prompts(brief, "cogvideox5b", repeat, shots=SHOTS) if p][:SHOTS]
    row = {
        "brief_index": brief_index,
        "repeat": repeat,
        "brief": brief,
        "prompts": prompts,
        "plan_seconds": round(time.monotonic() - started, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def seed_for(repeat: int, shot: int) -> int:
    """Same band the repeated-measure study used, shared by every arm."""
    return repeat * 100 + shot


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["cogvideox5b"])
    parser.add_argument("--briefs", nargs="+", type=int, default=[0])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--judge-base", default="http://127.0.0.1:8791/v1")
    parser.add_argument("--judge-model", default="qwen2.5-vl-32b-local")
    parser.add_argument("--inspect", action="store_true",
                        help="print the clause split and stop; renders nothing")
    args = parser.parse_args(argv[1:])

    briefs = json.loads(BRIEFS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    total = len(args.models) * len(args.briefs) * args.repeats * len(args.arms) * SHOTS
    print(f"{len(args.models)} models x {len(args.briefs)} briefs x {args.repeats} repeats "
          f"x {len(args.arms)} arms x {SHOTS} shots = {total} clips")

    if args.inspect:
        for brief_index in args.briefs:
            brief = briefs[brief_index]
            brief_en = get_brief_en(brief, brief_index)
            plan = get_plan(brief, brief_index, 0)
            print(f"\n########## brief {brief_index} ##########")
            print(f"source ({len(brief)} chars): {brief}")
            print(f"english ({len(brief_en)} chars): {brief_en}")
            for shot, prompt in enumerate(plan["prompts"][:2]):
                built = build_arms(brief_en, prompt)
                print(f"\n=== brief {brief_index} shot {shot} ({len(prompt)} chars) ===")
                print(f"scene {len(built['split']['scene_clauses'])} clauses / "
                      f"{built['split']['scene_chars']} chars | "
                      f"anchor {len(built['split']['anchor_clauses'])} clauses / "
                      f"{built['split']['anchor_chars']} chars")
                for clause in built["split"]["anchor_clauses"]:
                    print(f"  [A] {clause[:150]}")
                for clause in built["split"]["scene_clauses"]:
                    print(f"  [S] {clause[:150]}")
                for arm, entry in built["arms"].items():
                    print(f"  {arm:<14} {entry['chars']:>4} chars | {entry['text'][:110]}")
        return 0

    from tier3a_story import render
    from character_consistency import ask, grab

    log = OUT / "clips.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["arm"], row["model"], row["brief_index"], row["repeat"]))

    for model in args.models:
        for brief_index in args.briefs:
            brief_en = get_brief_en(briefs[brief_index], brief_index)
            for repeat in range(args.repeats):
                if all((a, model, brief_index, repeat) in done for a in args.arms):
                    print(f"skip {model} b{brief_index} r{repeat} (complete)")
                    continue
                plan = get_plan(briefs[brief_index], brief_index, repeat)
                tag_root = OUT / "clips"
                for arm in args.arms:
                    tag = f"{arm}_{model}_b{brief_index}_r{repeat}"
                    clips = []
                    chars = []
                    started = time.monotonic()
                    for shot, prompt in enumerate(plan["prompts"]):
                        text = build_arms(brief_en, prompt)["arms"][arm]["text"]
                        chars.append(len(text))
                        path = render(text, seed_for(repeat, shot), model, tag, tag_root, shot)
                        if path is None:
                            continue
                        frame = grab(path)
                        if frame:
                            clips.append(frame)
                    if len(clips) < 2:
                        print(f"  {tag}: only {len(clips)} clips, skipped")
                        continue
                    score = ask(list(enumerate(clips)), base=args.judge_base, model=args.judge_model)
                    row = {
                        "arm": arm, "model": model, "brief_index": brief_index,
                        "repeat": repeat, "label": f"{arm}:{model}:b{brief_index}:r{repeat}",
                        "shots": len(clips), "reply": score,
                        "elapsed_s": round(time.monotonic() - started, 1),
                        "prompt_chars": chars[: len(clips)],
                    }
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"  {row['label']} -> {score.splitlines()[0][:70]}")
    print(f"\nwrote {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
