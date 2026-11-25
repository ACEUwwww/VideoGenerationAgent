# Controlled prompt ablation: what in the carried state actually does the work

> Companion to the matched comparison ([`om_matched_v2.md`](om_matched_v2.md)) and
> the repeated-measure study ([`technical_report.md`](technical_report.md)).
> Scripts: [`prompt_ablation.py`](../study/prompt_ablation.py),
> [`ablation_stats.py`](../study/ablation_stats.py),
> [`noise_floor.py`](../study/noise_floor.py).
> Raw: [`prompt_ablation/clips.jsonl`](prompt_ablation/clips.jsonl),
> [`prompt_ablation/summary.json`](prompt_ablation/summary.json).

## Why this exists

The repeated-measure study compares `planned` against `independent` and reports a
rise in cross-shot character consistency. Those two arms differ in **three**
things at once:

1. prompt length (49 → ~700–880 characters),
2. plan-derived scene and action content,
3. the character's appearance anchor, restated verbatim in every shot.

A difference between them therefore cannot be attributed to any one of the
three. This study rebuilds the comparison so that each factor moves alone.

## Design

Every arm is derived from **one planner output per (brief, repeat)**, so the
model, brief, repeat, shot index, seed, resolution, frame count and judge are
identical across arms. Only the prompt string changes.

The planner's prompt for a shot is decomposed into **S** (scene / camera /
lighting / action) and **A** (character appearance), then reassembled:

| Arm | Prompt | Isolates |
| --- | --- | --- |
| `brief_en` | the English brief verbatim | control |
| `brief_en_anchor` | brief + A | anchor, no plan structure |
| `scene` | S only | plan content, no anchor |
| `planned` | S + A (the planner's own text) | the original `planned` arm |
| `planned_trunc` | first 45 % of `planned` | *intended* as length control |

Two corrections were made before running, both documented in the commit log:

- **Language.** The briefs are Chinese and the planner writes English, while
  both checkpoints use English text encoders. The control arm was therefore the
  brief translated into English and cached — not the raw Chinese string, which
  would have differed from every other arm in language as well as content.
- **Splitting.** Sentence level first, comma level only for mixed sentences;
  camera vocabulary forces a clause into S so that "tilts up to his face" is not
  counted as an appearance clause. `--inspect` prints the split without
  rendering, so the classification can be audited before any GPU time.

### Scale

2 checkpoints × 3 briefs × 3 repeats × 5 arms × 5 shots = **450 clips**,
18 cells, every cell with all five arms, one judge pass per episode
(the same `qwen2.5-vl-32b-local` prompt the reports already use).

## Results

Arm means (1–5, higher is better; n = 18 cells each):

| Arm | Mean | SD | ≈ chars |
| --- | ---: | ---: | ---: |
| `brief_en` | 3.17 | 1.20 | 167 |
| `scene` | 3.50 | 0.71 | 570 |
| `planned_trunc` | 3.00 | 0.91 | 330 |
| `planned` | 4.28 | 0.67 | 863 |
| `brief_en_anchor` | 4.56 | 0.86 | 461 |

Paired contrasts (within model/brief/repeat, n = 18 pairs; CI is a 5 000-sample
bootstrap, p is the exact one-sided sign test):

| Contrast | Mean diff | 95 % CI | w/t/l | p | Verdict |
| --- | ---: | --- | ---: | ---: | --- |
| `brief_en − scene` | −0.33 | [−0.89, +0.22] | 5/0/9 | 0.21 | CI includes zero |
| `scene − planned` | −0.78 | [−1.11, −0.44] | 1/0/13 | 0.0009 | **CI excludes zero** |
| `brief_en − brief_en_anchor` | −1.39 | [−2.00, −0.78] | 1/0/13 | 0.0009 | **CI excludes zero** |
| `planned − planned_trunc` | +1.28 | [+0.72, +1.83] | 12/0/1 | 0.0017 | **CI excludes zero** |

Negative signs on the anchor rows mean the **anchor arm scores higher** — the
contrast is written `arm_a − arm_b`, so a negative value is the anchor winning.

## What this establishes

**The appearance anchor is the active ingredient, and it replicates three
times with different counterfactuals:**

| Anchor contrast | Context of the comparison | Gain |
| --- | --- | ---: |
| `scene → planned` | plan structure present | **+0.78** |
| `brief_en → brief_en_anchor` | no plan structure | **+1.39** |
| `planned → planned_trunc` | see the caveat below | **+1.28** |

All three move in the same direction with 12–13 of 14–18 pairs concordant and
intervals excluding zero. The two cleanest (`scene` vs `planned`,
`brief_en` vs `brief_en_anchor`) hold different things constant — one keeps plan
content and varies only the anchor, the other holds plan content absent — so this
is a factorial result, not a chain of comparisons.

**Plan-derived scene/action content on its own shows no measurable effect.**
`brief_en` (3.17) → `scene` (3.50) is +0.33 with a CI spanning zero. Writing a
detailed scene description without the character anchor does not make the person
more consistent, which is what the earlier comparison could not separate.

## A caveat that invalidates the length arm

`planned_trunc` does **not** control for length. Measured after the fact:

```
anchor clauses surviving truncation: 0 / 296  (0 %)
```

Truncating `planned` to its first 45 % removes **every** appearance clause,
because the planner writes scene first and appearance second. So the
`planned − planned_trunc` row is a third **anchor** contrast, not a length
contrast, and its agreement with the other two is expected rather than
independent.

Consequently: **this study says nothing about prompt length.** The earlier
observation that trimming a planned prompt to 400 characters recovers only ~0.075
of the deficit remains the only length evidence, and it was not a controlled arm.
A real length control would need to cut the scene sentences while keeping the
anchor, or pad a short arm with scene-neutral filler to the anchor arm's length.

## What it does not establish

- **Not a causal story about pixels.** Both checkpoints are text-to-video; no
  keyframe, no reference image, no visual conditioning is involved anywhere in
  this study. Every gain measured here travels as text.
- **Not general.** Two checkpoints of 1.3B and 5B parameters, three briefs, one
  judge. Nothing here speaks to larger models or to human raters.
- **Not attributable to wording.** The anchor's effect could be the specific
  traits it names, or simply that a repeated noun phrase keeps the model's
  attention on one subject. Splitting the anchor into face / hair / build /
  wardrobe arms would separate those; this run did not.

## Comparison with the earlier claim

The repeated-measure study reported consistency rising under carried state. That
result stands. What this ablation adds is that the rise is produced by the
**appearance anchor**, and not by the longer or richer scene description that
travels with it — which is the factor the earlier two-arm design could not
isolate.
