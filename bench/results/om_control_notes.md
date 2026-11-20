# OpenMontage full-workflow control: findings while wiring it up

Working notes for the control arm that replaces the constructed `independent`
baseline. Not part of the report yet — this is what happened while building it,
including the parts that cost most of the time.

Date: 2026-11-19. Hardware: 1x B300 (275 GB), VLM judge resident on the same GPU.

## Environment facts (measured, not assumed)

- The studio runs in `bench/_vendor/OpenMontage/.venv-studio`, which has **no
  torch, diffusers, transformers or safetensors** — only PIL, requests, httpx.
  A local renderer therefore cannot run inside the server process without
  installing a CUDA stack into the vendor's virtualenv.
  → Rendering is a separate Flask service in `bench/.venv`
  (`bench/study/local_render_service.py`, port 8788); the studio-side backend is
  a thin HTTP client.
- Postgres, Redis and MinIO are all **optional**. With `DATABASE_URL` /
  `REDIS_URL` unset the studio falls back to JSON files and a bounded
  ThreadPoolExecutor. The earlier report claim that the studio "requires
  Postgres/Redis/MinIO" was wrong.
- `VIDEO_BACKENDS.register` and `register_video_job` are public, and
  `IMAGE_BACKENDS.register` likewise. A `local` video backend and a
  `local_diffusion` image backend were added from outside the vendored tree.
  No file under `bench/_vendor/` was edited.

## CogVideoX-5B-I2V: the earlier "unavailable" verdict was wrong

The Tier 3a report said I2V could not be used because diffusers raises
`It is currently not possible to generate videos at a different resolution`.
The guard is real but its meaning was misread. In `diffusers/models/embeddings.py`
it fires when a *learned* positional embedding does not match the checkpoint's
own latents:

| checkpoint | sample_height | sample_width | pixels (VAE x8) |
| --- | --- | --- | --- |
| CogVideoX-5B | 60 | 90 | 480 x 720 |
| CogVideoX-5B-I2V | 60 | 90 | 480 x 720 |

So the native bucket is **720x480 landscape (3:2)**, neither 832x480 nor
portrait. Probed: 41 frames at 720x480 render in 34.5 s (load 14.4 s).
The model was always fine; it was being asked for the wrong shape.

This matters for the comparison: OpenMontage's consistency mechanism is
*reference image → keyframe → i2v*. Declaring the local video backend
text-to-video would delete the only cross-shot mechanism the vendor has, and
any weakness measured afterwards would be an artefact of the wiring.

## Defects found and fixed

1. **`_ratio_pixels` parsed SIZES as a string.** `providers.nvidia_flux.SIZES`
   maps ratio → `(width, height)` tuples. `str(...).split("x")` gave
   `int("(1344, 768)")` → ValueError, so every entity still failed after the
   backend was otherwise working. Fixed by accepting both tuple and string.

2. **Chinese prompts to an English-only text encoder.** This was the expensive
   one. The studio writes entity bibles and shot prompts in Chinese; its own
   stills backends (Seedream, the Comfy Qwen path) are bilingual. Stable
   Diffusion 1.5's CLIP encoder is not.

   Measured directly: the prompt
   `二十多岁东亚男性，短寸头，浓眉…穿深蓝色外卖工装外套` produced **an aerial
   photograph of a hillside shantytown** — the model had no representation for
   those tokens and latched onto noise. The English rendering of the same
   description produced the intended standing figure.

   Consequence before the fix: every entity scored 0.0–0.3 against a 0.7 bar,
   the entity stage retried forever, and the episode could not reach the video
   stage at all.

   Fixed inside the backend (`LocalDiffusionImageBackend._translate`), which is
   the right place: the gap belongs to the *substituted* model, not to
   OpenMontage's orchestration. The vendor's prompts are left untouched so both
   sides keep their own logic. Translations are cached because the studio
   re-sends the same description on every QA retry.

3. **Prompts longer than the text encoder can read.** Second half of the same
   root cause. SD1.5's CLIP encoder truncates at 77 tokens and says so in the
   log:

   ```
   The following part of your input was truncated because CLIP can only handle
   sequences up to 77 tokens: [...]
   ```

   OpenMontage's stills prompts are a long English boilerplate preamble plus
   the Chinese subject: "FULL BODY only. Head-to-toe standing figure…", then on
   a retry "Fix previous QA failure: The image fails to meet several PRIMARY
   criteria…", then "qa retry 2/3 product photo of the single named object…".
   All of that is English and all of it lands *before* the subject. So the part
   the model actually read was the scaffolding, and the character description —
   the only part that identifies who is in the frame — was cut off entirely.

   The translation prompt was rewritten to compress to ≤60 words, discard the
   boilerplate, and lead with the subject in the first five words. Compressing
   is part of adapting the prompt to the substitute model; it does not change
   what OpenMontage asks for, only whether its request survives the encoder.

## QA asymmetry between the two sides (must be disclosed, not equalised)

| | our pipeline | OpenMontage |
| --- | --- | --- |
| Image gate | geometric + metadata: resolution, file size, anchor present in prompt, prompt length. `IMAGE_PASS = 0.78` | VLM semantic fit against the entity bible. `STUDIO_QA_MIN_SCORE = 0.7` |
| Fail behaviour | rewrite prompt, retry | retry up to N; if no character passes, the whole episode fails |

After the translation fix, entity scores rose (scenes 0.2 → 0.5, one scene 0.9)
but characters stayed at 0.2–0.3, i.e. **1 of 11 entities cleared OpenMontage's
own bar**.

Decision recorded here because it is a judgement call: the episode was resumed
with `requireEntityQa=false` and `requireVideoQa=false`. Falling back to the
constructed control.

Rationale: our side has no equivalent semantic gate — it is text-to-video, no
keyframe stage, no entity bible. Letting OpenMontage die at a gate our side does
not have would measure *SD1.5's weakness at full-body character stills*, and
then report "it produced no film" as our advantage. That is not a finding about
orchestration.

What is kept as a fact, separately: with a bilingual stills model (Seedream) the
vendor's gate is presumably passable; with SD1.5 it is not. That is a statement
about the substitute, and the report says so.

## Still open

- The 44-shot episode is in the video stage; no I2V clip has been produced yet.
- Both sides have not yet been scored with the same cross-shot consistency VLM.
- LongLive still has no OpenMontage variant, so it remains unpaired.

## Update: the video stage was reachable after all

Three further defects stood between the entity stage and a rendered clip. All
three were in my wiring, not in the vendor's code.

4. **`run_local_shot` was imported, not registered.** The studio has two video
   paths: a legacy per-shot path that goes through the job table, and a
   segment-QA path that does `from providers.video_jobs import run_local_shot`
   directly. Only the job table was populated, so every segment failed with
   `cannot import name 'run_local_shot'` and the stage looped.

5. **The model followed the config, not the shot.** The default was
   `cogvideox5b_i2v`, which cannot render without a first frame. On the
   text-only path the studio passes no reference, so the backend now picks
   `cogvideox5b` when there is no keyframe and `cogvideox5b_i2v` when there is.

6. **Several episodes ran concurrently on one GPU.** Restarting the server
   does not stop an in-flight auto-produce thread, and it rewrites the state
   file it was given, so deleted projects reappeared and competed for the same
   device. Every run now starts from a killed server and a purged project
   directory.

With those fixed, a full episode reached the video stage and rendered clips.

Still open: in the live server, rendered clips reached `/tmp/om_render`
(45 staged files) while the project's `media/videos` stayed empty and its shots
stayed at `generating`. Isolated, the same call writes back correctly — run in
a separate process against the `omlast` project, `run_local_shot` returned
`status: success` with `videos: 1` — so the function is not at fault, but *why*
the live path differs was **not established**. A lost update between the media
worker and the auto-produce thread (both rewriting one `studio_state.json`) is
the leading hypothesis and is unverified; no traceback reached the log, and the
media worker was observed holding a single job at a time rather than erroring.

Recorded as an open defect in this integration. It is the reason the text-only
control arm cannot yet contribute a scored episode, and it is not attributed to
the vendor's code.

Separately, and verified: the studio's video stage hands the backend vendor
model ids (`doubao-seedance-2-0-260128`) unless the option default is
overridden. Passing one straight through to the local worker raised
`KeyError` and a 500 per shot until the backend mapped foreign ids onto the
checkpoints it can actually load.

## What this means for the comparison

The control arm is a real run, not a constructed baseline, and what it shows is
narrower than "they lose":

1. **The reference-image path cannot run on local open-weight models at all.**
   The vendor's entity stage requires a pure-white seamless cutout plate per
   character and gates it with a VLM at 0.7. Stable Diffusion 1.5 does not
   produce that plate: measured 0 of 5 characters passing across two complete
   runs (13 and 15 entities). The video stage then hard-refuses to start,
   because the whole consistency mechanism is downstream of those stills.

   This is the substance of the architectural difference. OpenMontage's
   coherence is *load-bearing on a capable image model*: break that link and
   the pipeline stops rather than degrading. Our layer, on the same weak
   models, still delivers shots — because its cross-shot state travels as text
   in the prompt, not as a keyframe through an image model.

2. **The text-only path runs but is not yet scored.** With
   `textOnlyVideo=true, generateEntities=false` the vendor's own configuration
   for exactly this situation, the pipeline reaches the video stage and renders.
   Until the write-back defect above is resolved, no finished episode exists to
   score, so no comparative number is claimed.

3. Nothing here shows our films look better than OpenMontage's. What it shows
   is a difference in failure mode: one architecture keeps producing output
   when the image model is weak, the other stops.


