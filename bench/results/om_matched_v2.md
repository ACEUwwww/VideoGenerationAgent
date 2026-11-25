# Matched comparison v2: OpenMontage vs ours on an identical configuration

> Supersedes the OpenMontage numbers in
> [`orchestration_report.md`](orchestration_report.md) §5.2, which compared a
> 12-shot OpenMontage episode against a 5-shot run of ours and disclosed only
> the shot-count difference.
>
> Raw: [`om_matched_v2/consistency.json`](om_matched_v2/consistency.json).
> Drivers: [`om_full.py`](../study/om_full.py),
> [`om_local_backend.py`](../study/om_local_backend.py),
> [`ours_matched_arm.py`](../study/ours_matched_arm.py),
> [`score_matched_v2.py`](../study/score_matched_v2.py).

## Why this run exists

The earlier comparison held the brief and the checkpoint class fixed but let
four other things float. Each one can move the score on its own, so the earlier
difference between 2.0 and 4.0 could not be attributed to orchestration.

| Knob | Earlier run | This run (both sides) |
| --- | --- | --- |
| Checkpoint | `cogvideox5b` (ours) vs i2v path (OM) | `cogvideox5b` text-to-video |
| Seed | `seed=index` (ours) vs hardcoded `seed=0` (OM) | shot position `0..N-1` |
| Clip length | 5 s (ours) vs storyboard's 3–5 s (OM) | 5 s |
| Shots | 5 (ours) vs 12 (OM) | 12 |
| Prompt language | English (ours) vs **Chinese** (OM) | English on both sides |
| Brief | same | same |
| Judge | same | same |

Two of these were hard failures, not merely differences:

**The image-to-video path returned HTTP 500 for every shot.** OpenMontage
attaches entity references to shots whenever a character bible exists, which
selects `cogvideox5b_i2v`. That checkpoint needs a usable first frame, and on
local weights the entity stills never clear the vendor's plate gate. The first
attempt at this rerun failed **56 of 56** shots this way. The adapter now forces
text-to-video (`LOCAL_VIDEO_FORCE_T2V`), which is what the reference arm was
always doing.

**The Studio writes Chinese shot prompts; CogVideoX's text encoder is
English-only.** The repository already documented this for the *stills* path —
a Chinese prompt tokenised into fragments with no representation produced "an
aerial photograph of a hillside shantytown" — and fixed it inside the stills
backend. The *video* path had no equivalent translation, so every OpenMontage
shot would have been rendered from a prompt the encoder could not read. That is
a property of the substituted model, not of the workflow under test, so the
translation was added on the adapter side and OpenMontage's own prompt text was
left untouched.

## Configuration actually executed

| | OpenMontage | Ours |
| --- | --- | --- |
| Project / run | `benchs0-4` | `results/om_matched_v2/ours` |
| Brief | `深夜的便利店，一个外卖骑手发现收银员不是人。` | same |
| Planner | DeepSeek | DeepSeek |
| Checkpoint | `cogvideox5b` (t2v) | `cogvideox5b` (t2v) |
| Resolution / fps | 832×480 @ 8 fps | 832×480 @ 8 fps |
| Frames per clip | 41 (4k+1 rule, 5 s) | 41 |
| Seed | shot position | shot position |
| Shots scored | 12 | 12 |
| Judge | qwen2.5-vl-32b-local | same |

## What still differs, and what that bounds

- **OpenMontage's internal planning granularity.** Its storyboard stage has no
  shot-count parameter: `SKILL.md` sets min 10 / max 14 / target 12, and
  `skills_root()` is fixed inside the vendor tree. For this brief its natural
  plan was **56 shots**; the 12 scored here are its own first 12, and it deleted
  the rest through its own API. Both sides agree on the *scored* shot count, but
  OpenMontage planned against a coarser episode than the one it delivered.
- **The translation layer is ours, not theirs.** OpenMontage's prompts are
  Chinese by design. On a bilingual backend they are used as written; here they
  pass through one translation step before reaching the encoder. This is
  unavoidable on an English-only checkpoint, and it is recorded rather than
  hidden.
- **n = 1 per side.** One brief, one episode each, one judge pass. No
  significance is claimed and none is available.

## Result

| Arm | Shots | Cross-shot consistency (1–5) | Largest changed trait |
| --- | ---: | ---: | --- |
| OpenMontage | 12 | **2.0** | Face and Facial Features |
| Ours | 12 | **3.0** | Clothing |

Judge's reason, verbatim:

> **OpenMontage (2.0):** *"the face and facial features are inconsistent across
> the frames, with some showing a person wearing glasses and others showing
> indistinct or obscured faces."*

> **Ours (3.0):** *"the main character's face, hairstyle, and body build remain
> consistent across most frames, but there is a noticeable change in clothing in
> Shot 4."*

The same pattern as before — ours holds the face and loses wardrobe on one
shot; OpenMontage loses the face itself — but the size of the gap is not the
same.

## What this changes about the earlier claim

| Comparison | Gap |
| --- | ---: |
| Earlier run (OM 12 shots vs ours 5 shots) | 2.0 vs 4.0 → **2.0 points** |
| This run (both 12 shots, matched config) | 2.0 vs 3.0 → **1.0 point** |

**Half the earlier gap was the configuration, not the orchestration.** Ours
scored 4.0 on a 5-shot episode and 3.0 on a 12-shot one from the same brief and
planner — so a five-shot episode is systematically easier to look consistent on,
and the earlier comparison handed our side that advantage.

Against the measured noise floor of ~1.0 point on an unchanged prompt
([`noise_floor.json`](noise_floor.json)), the remaining 1-point difference is
**at the resolution limit of this metric**. With n = 1 per side there is no
statistical claim to make: what this run supports is that the direction is
unchanged and the magnitude was previously overstated.

## Honest reading

- The direction (ours above OpenMontage on cross-shot character consistency)
  survives full configuration control.
- The magnitude does not. The defensible statement is a **1-point** difference
  on one episode, not a 2-point one.
- Neither number is distinguishable from render/judge noise at n = 1.
- OpenMontage's remaining handicap is not configuration: its own prompts retain
  an `@entity` reference syntax (`@阿明-外卖骑手装`) and per-shot Chinese staging
  notes that survive translation as literal text. That is what its workflow
  actually emits, so it is left alone — but it is another reason not to read
  this as a clean estimate of orchestration quality.

