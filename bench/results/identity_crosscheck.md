# Pixel-side identity cross-check

> Companion measurement to [the repeated-measure study](../results/orchestration_report.md).
> Same 180 clips, no new rendering, no paid calls.
> Script: [`bench/study/identity_crosscheck.py`](../../study/identity_crosscheck.py).
> Raw output: [`identity_crosscheck.json`](identity_crosscheck.json).

## What this was for

The headline result — cross-shot character consistency **2.28 → 4.61** — is
scored by a single local vision-language model. One judge, with no second
opinion on the same renders, is the weakest link in that claim. The obvious
cross-check is a face-identity measurement, which is also what
[DreamFactory](https://arxiv.org/abs/2408.11788) proposes with **CSFD**
(cross-scene face distance): detect a face per keyframe, embed it, average the
cosine over every frame pair.

This document reports what happened when that measurement was run on our own
clip set.

## Method

| | |
| --- | --- |
| Clips | 180 (36 episodes × 5 shots), the exact set behind the repeated-measure study |
| Sampling | 5 frames per shot at 15/30/45/60/80 % of the clip; the largest detectable face across them is used |
| Detector | MTCNN (facenet-pytorch) |
| Embedding | InceptionResnetV1 / VGGFace2, L2-normalised |
| Metric | cosine between face embeddings, averaged over shot pairs where **both** sides have a usable face |
| Usable face | at least 48 px on its short side |
| Episode scored | at least 60 % of its shots have a usable face |

This is not ArcFace. It is the same class of measurement — a face-identity
embedding, cosine-scored — from a checkpoint that installs without the
InsightFace package. Swapping the backbone moves the absolute cosines; it would
not make the coverage problem below go away.

## Result 1 — the metric cannot be computed on the control arm

| Arm | Shots with a usable face | Episodes scored |
| --- | --- | --- |
| independent | **4 / 90 (4.4 %)** | **0 / 18** |
| planned | 39 / 90 (43.3 %) | 9 / 18 |

The independent arm produces almost no detectable faces. This is not a detector
setting: lowering MTCNN's thresholds to 0.3 changed nothing, 2× upsampling
recovered a handful of frames, and a second detector (OpenCV SSD res10) found
nothing on the frames where MTCNN also found nothing. The frames themselves are
wide and medium shots of a character who is frequently turned away, rendered at
832×480 by a 1.3B–5B model.

Consequence: **the paired comparison this study needs is not computable from
this metric.** With zero scored episodes in one arm there is no pair to
difference, so no sign test, no confidence interval, no p-value.

That is the direct answer to "can CSFD be lifted into this benchmark": not on
this clip set, and the reason is structural rather than a bug. CSFD's implicit
assumption — that a face is present and detectable in every frame being
compared — does not hold for multi-shot drama rendered at this model tier. The
paper never defines what a frame pair contributes when one side has no
detectable face, and on our data that case is the majority.

## Result 2 — where faces are detectable, the two metrics agree

Restricting to the 9 planned-arm episodes that clear the coverage threshold:

| Model | Mean cosine | SD | n |
| --- | ---: | ---: | ---: |
| CogVideoX-5B | 0.7312 | 0.0488 | 5 |
| Wan2.1-1.3B | 0.8142 | 0.0524 | 4 |
| pooled | 0.7681 | — | 9 |

Episode-level cosines run 0.647–0.851; the lowest single shot pair is 0.534.
Mean fraction of shot pairs skipped for lack of a face: **81 %**.

Agreement between the face metric and the VLM judge on the same episodes:

| Model | Spearman ρ | n |
| --- | ---: | ---: |
| CogVideoX-5B | 0.866 | 5 |
| Wan2.1-1.3B | 0.775 | 4 |
| pooled | 0.822 | 9 |

Two things to hold against that ρ:

- **The judge's scores have almost no spread.** All nine episodes scored 4 or 5.
  A rank correlation over two distinct values is arithmetic, not evidence of
  convergent validity.
- **n = 9, and the sample is the arm that already looked good.** It says the two
  methods do not contradict each other where both can speak. It does not say
  either one is calibrated.

## Result 3 — detectability itself separates the arms

4.4 % vs 43.3 % of shots carrying a usable face is a large difference, and it
tracks the direction of the study's claim. It is **not** an identity
measurement, and it should not be reported as one: a blurred, distorted or
half-turned face fails detection for reasons that have as much to do with
render quality and framing as with who the person is. Treat it as a
quality-correlated signal, not as consistency evidence.

## What this establishes and what it does not

Establishes:

- A CSFD-style face metric is **not applicable** to this clip set, and the
  reason is the source material, not the implementation.
- Any future report built on this data must either report face coverage
  alongside the score or state that the pixel metric was unavailable.

Does not establish:

- That the VLM's 2.28 → 4.61 is correct. This was meant to be an independent
  check of it and, on this data, it cannot be one.
- Anything about faces at higher model tiers. A production render at 1080p from
  Seedance would very likely clear the coverage threshold; the measurement is
  untested there.

## What would make the check possible

In rough order of cost:

1. **Render one close-up per shot.** A 3-second insert per character per shot
   costs one extra clip and gives every shot a detectable face. This is the
   cheapest fix and does not change the creative brief.
2. **Use the keyframe stills instead of video frames.** The image stage renders
   1080×1920; those stills have faces even where the video is a wide shot. It
   measures the image model rather than the video model, which is a different
   claim, but it is a valid one.
3. **Score on 1080p production renders** from the paid provider rather than on
   local 1.3B–5B output.
