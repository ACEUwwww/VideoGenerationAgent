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
| `planned_trunc` | first 45 % of `planned` | *intended* as length control — it is not one |
| `planned_padded` | `planned` + neutral production vocabulary to ~2× | length, anchor and scene held constant |

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
| `planned_padded` | 4.22 | 1.00 | 1695 |
| `brief_en_anchor` | 4.56 | 0.86 | 461 |

Paired contrasts (within model/brief/repeat, n = 18 pairs; CI is a 5 000-sample
bootstrap, p is the exact one-sided sign test):

| Contrast | Mean diff | 95 % CI | w/t/l | p | Verdict |
| --- | ---: | --- | ---: | ---: | --- |
| `brief_en − scene` | −0.33 | [−0.89, +0.22] | 5/0/9 | 0.21 | CI includes zero |
| `scene − planned` | −0.78 | [−1.11, −0.44] | 1/0/13 | 0.0009 | **CI excludes zero** |
| `brief_en − brief_en_anchor` | −1.39 | [−2.00, −0.78] | 1/0/13 | 0.0009 | **CI excludes zero** |
| `planned − planned_trunc` | +1.28 | [+0.72, +1.83] | 12/0/1 | 0.0017 | **CI excludes zero** |
| `planned − planned_padded` | +0.06 | [−0.28, +0.39] | 5/0/4 | 0.50 | CI includes zero |

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

**Prompt length shows no measurable effect.** `planned` → `planned_padded` is
+0.06 with a CI spanning zero, at a 1.95× length ratio with the anchor and scene
held byte-identical. The three results together say the carried state works
because of *what it names*, not because of *how much it says*.

## A caveat that invalidates the first length arm

`planned_trunc` does **not** control for length. Measured after the fact:

```
anchor clauses surviving truncation: 0 / 296  (0 %)
```

Truncating `planned` to its first 45 % removes **every** appearance clause,
because the planner writes scene first and appearance second. So the
`planned − planned_trunc` row is a third **anchor** contrast, not a length
contrast, and its agreement with the other two is expected rather than
independent.

`planned_padded` was built to replace it. It appends only production vocabulary —
lens, depth of field, grain, light wrapping, colour grade — until the prompt is
about twice as long, so the anchor and the scene text are byte-identical to
`planned` and only length moves. Measured ratio: **1.94–1.95×**. A unit test
asserts the appended text contains no appearance vocabulary and that `planned`
appears as a prefix.

**Length has no effect on this metric:**

```
planned (863 chars)   4.28   ─┐
                              ├─ +0.06, CI [−0.28, +0.39], 5/0/4 wins, p = 0.50
planned_padded (1695) 4.22   ─┘
```

That closes the question the earlier comparison left open: the gain from carried
state is **not** a gain from writing more words. Doubling the prompt with
detail-free production vocabulary moves nothing, while adding ~450 characters of
appearance anchor moves the score by 0.8–1.4. Note also that `planned_padded`
(1695 chars) scores slightly **below** `planned` (863), so if anything there is a
weak hint of diminishing returns past a point — not evidence, but worth not
reading as "more is better".

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
