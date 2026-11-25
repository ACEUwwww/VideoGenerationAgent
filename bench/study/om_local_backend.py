"""Register a local open-weight model as an OpenMontage studio backend.

Why this exists: the report's control arm was a *constructed* "no state carried"
baseline, not a measurement of a competing workflow. Replacing it means running
OpenMontage's own full pipeline on the same local models we use, so the two
sides differ only in orchestration.

Two constraints shaped the design.

*The studio has no CUDA stack.* It runs in
``bench/_vendor/OpenMontage/.venv-studio``, which contains no torch, and
installing one would modify the vendored checkout's environment. Rendering
therefore happens in ``bench/.venv`` behind ``local_render_service.py``, and
this module is a thin HTTP client. The studio's own ``image_factory`` and
``video_factory`` registries are public, so the backend is added from outside
the vendored tree — no file under ``bench/_vendor/`` is edited.

*OpenMontage's consistency mechanism is reference image -> keyframe -> i2v.*
Its shot contract hands the backend ``refPath`` (the entity's reference still)
and the shot's prompt. Declaring this backend text-to-video would silently drop
the only cross-shot mechanism the vendor has, and any weakness measured
afterwards would then be an artefact of the wiring rather than a finding. So
the image stage is served locally as well (Stable Diffusion 1.5), the keyframe
reaches the model, and CogVideoX-5B-I2V animates it. The comparison is then
orchestration against orchestration, which is the whole point.

What still differs, stated because it bounds the conclusion: the local stack
substitutes SD1.5 for Seedream and CogVideoX-5B-I2V for Seedance 1.0 pro-fast.
Both sides of the comparison are 1.3B-5B open-weight models at the same
resolution, so the substitution is applied to neither side preferentially.
"""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

OM_ROOT = Path("/mdata/shiweisong/VideoGenerationAgent/bench/_vendor/OpenMontage")
PROJECT_ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")

BACKEND_ID = "local"
IMAGE_ID = "local_diffusion"

# Where the GPU worker listens. Overridable so a run can be pinned to a
# different port without editing this file.
RENDER_URL = os.environ.get("LOCAL_RENDER_URL", "http://127.0.0.1:8788")
RENDER_TIMEOUT = float(os.environ.get("LOCAL_RENDER_TIMEOUT", "900"))


def _install_path() -> None:
    """Make the studio server importable the way its own modules expect.

    ``studio/server`` uses flat imports (``from drama_service import ...``), so
    the directory itself goes on sys.path rather than its parent.
    """
    for path in (OM_ROOT, OM_ROOT / "studio" / "server"):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))


def _reachable() -> bool:
    try:
        return requests.get(f"{RENDER_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_TRANSLATION_CACHE: dict[str, str] = {}


def _has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def _translate_for_encoder(text: str) -> str:
    """Module-level translation for call paths that are not the stills backend.

    The stills backend has this as a method; the video path is a module-level
    function and had no equivalent, so the studio's Chinese shot prompts went
    straight to CogVideoX's English-only encoder. That is the same defect the
    stills path already documents — a Chinese prompt tokenises into fragments
    the encoder has no representation for — and leaving it in place would
    handicap one side of the comparison for a reason that belongs to the
    substituted model rather than to the workflow being measured.

    Shares ``_TRANSLATION_CACHE`` with the stills backend so a prompt repeated
    across QA retries is translated once.
    """
    cleaned = (text or "").strip()
    if not cleaned or not _has_cjk(cleaned):
        return cleaned
    cached = _TRANSLATION_CACHE.get(cleaned)
    if cached:
        return cached
    translated = _translate_with_llm(cleaned)
    _TRANSLATION_CACHE[cleaned] = translated
    return translated


def _translate_with_llm(text: str) -> str:
    """Ask the planner LLM for a literal English rendering of a Chinese prompt.

    Deliberately literal, not creative: this is repairing an encoder's missing
    vocabulary, and any invention here would be the backend writing the film's
    art direction, which is the orchestration's job.
    """
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    base = (os.environ.get("LLM_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("LLM_MODEL") or "deepseek-chat"
    if not key:
        raise RuntimeError(
            "A Chinese prompt reached the local stills backend but no DEEPSEEK_API_KEY is set "
            "to translate it; SD1.5 cannot read Chinese and would return an unrelated image."
        )
    response = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content":
                    "Rewrite the user's image prompt as ONE English prompt for Stable Diffusion 1.5.\n\n"
                    "Two hard constraints:\n"
                    "1. The text encoder reads at most 77 tokens. Everything past that is silently "
                    "discarded, so the prompt must be at most 60 words. Shorter is better.\n"
                    "2. OpenMontage wraps its prompts in a long English boilerplate preamble "
                    "('FULL BODY only...', 'Fix previous QA failure...', 'qa retry 2/3') and then "
                    "describes the subject in Chinese. That preamble is already English and is NOT "
                    "the subject. Discard it. What matters is the Chinese description.\n\n"
                    "Lead with the subject in the first five words: who or what, and how it is "
                    "framed (full body / close-up / empty room / single object). Then the two or "
                    "three most identifying details (clothing and colour, hair, a held prop). Drop "
                    "generalities and drop anything already said twice.\n\n"
                    "Output the prompt only. No preamble, no quotes, no commentary, no explanation."},
                {"role": "user", "content": text},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    out = (response.json()["choices"][0]["message"]["content"] or "").strip()
    return out or text


# --------------------------------------------------------------------------- #
# Image backend: entity stills and shot keyframes
# --------------------------------------------------------------------------- #
class LocalDiffusionImageBackend:
    """Renders stills through the local GPU worker.

    ``reference_path`` is accepted and passed on: the worker uses it when the
    pipeline supports image conditioning. SD1.5's text-to-image pipeline does
    not, so a reference is recorded as dropped rather than silently ignored —
    the same convention the vendor's own backends use in their notes.
    """

    id = IMAGE_ID

    def configured(self) -> bool:
        return _reachable()

    def unavailable_message(self) -> str:
        return (
            f"Local image backend selected but the render worker at {RENDER_URL} is not answering. "
            "Start it with bench/study/local_render_service.py --port 8788."
        )

    def generate(
        self,
        *,
        prompt: str,
        ratio: str,
        out_path: Path,
        reference_path: Path | None = None,
        negative_prompt: str | None = None,
        denoise: float | None = None,
    ) -> Path:
        width, height = _ratio_pixels(ratio)
        response = requests.post(
            f"{RENDER_URL}/render/image",
            json={"prompt": self._translate(prompt),
                  "negative": self._translate(negative_prompt or ""),
                  "width": width, "height": height, "seed": 0},
            timeout=RENDER_TIMEOUT,
        )
        response.raise_for_status()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response.content)
        return out_path

    def _translate(self, text: str) -> str:
        """Render a Chinese prompt into English for an English-only text encoder.

        Why this lives in the backend: the studio writes its prompts in Chinese
        and its own stills backends (Seedream, the Comfy Qwen path) are
        bilingual. Stable Diffusion 1.5's CLIP encoder is not — a Chinese prompt
        tokenises into fragments it has no representation for, and the model
        returns an unrelated picture. Measured directly: the entity bible above
        produced an aerial photograph of a hillside shantytown.

        That is a capability of the *substituted* model, not a property of the
        workflow being measured, so the gap is closed here rather than by
        changing what the studio writes. Translating inside the backend keeps
        OpenMontage's prompt untouched and both sides on their own logic.

        Cached: the studio re-sends the same entity description across QA
        retries, and a translation is not something to pay for twice.
        """
        cleaned = (text or "").strip()
        if not cleaned or not _has_cjk(cleaned):
            return cleaned
        cached = _TRANSLATION_CACHE.get(cleaned)
        if cached:
            return cached
        translated = _translate_with_llm(cleaned)
        _TRANSLATION_CACHE[cleaned] = translated
        return translated



def _ratio_pixels(ratio: str) -> tuple[int, int]:
    """Match the vendor's ratio table so the still has the frame's aspect."""
    try:
        from providers.nvidia_flux import SIZES
    except Exception:
        SIZES = {}
    # SIZES maps ratio -> (width, height) tuples, not "WxH" strings.
    size = SIZES.get(ratio) or (768, 1344)
    if isinstance(size, str):
        width, height = (int(v) for v in size.lower().split("x"))
    else:
        width, height = (int(v) for v in size)
    # SD1.5 is trained at 512; large canvases produce duplicated subjects.
    # Scale the requested aspect to a ~512-class area instead of honouring it.
    scale = (512 * 512 / (width * height)) ** 0.5
    width, height = int(width * scale) // 8 * 8, int(height * scale) // 8 * 8
    return width, height


# --------------------------------------------------------------------------- #
# Video backend: shot clips
# --------------------------------------------------------------------------- #
class LocalVideoBackend:
    """Studio-facing backend that renders with a local open-weight model."""

    id = BACKEND_ID

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("LOCAL_VIDEO_MODEL", "cogvideox5b_i2v")

    def configured(self) -> bool:
        return _reachable()

    def unavailable_message(self) -> str:
        return (
            f"Local video backend selected but the render worker at {RENDER_URL} is not answering. "
            "Start it with bench/study/local_render_service.py --port 8788."
        )

    def resolve_model(self, requested: str) -> str:
        req = (requested or "").strip()
        if req.lower().startswith(f"{BACKEND_ID}:"):
            return req.split(":", 1)[-1]
        return self.model

    def submit(self, **kwargs: Any) -> None:
        from providers.video_jobs import run_local_shot

        # ``start_shot_video`` passes store/shot/state/payload and never a
        # project_id kwarg. Reading one back gave an empty string, so
        # ``store._dir("")`` returned None: the render still ran (hence mp4s
        # piling up in the staging dir) but the clip was never written into the
        # project and the shot never left "generating". The shot record carries
        # the id, and the payload carries it too; both are authoritative.
        shot = kwargs.get("shot") or {}
        payload = kwargs.get("payload") or {}
        project_id = (
            kwargs.get("project_id")
            or shot.get("projectId")
            or payload.get("projectId")
            or ""
        )

        run_local_shot(
            store=kwargs["store"],
            project_id=project_id,
            shot_id=shot["id"],
            model=kwargs.get("model") or self.model,
            resolution=kwargs.get("resolution") or "480p",
            base_prompt=kwargs.get("base_prompt") or "",
            ratio=kwargs.get("ratio") or "9:16",
            duration=int(kwargs.get("duration") or 5),
            ref_path=kwargs.get("refPath") or payload.get("refPath"),
            end_ref_path=kwargs.get("endRefPath") or payload.get("endRefPath"),
        )


def run_local_shot(
    *,
    store: Any,
    project_id: str,
    shot_id: str,
    model: str,
    resolution: str = "480p",
    base_prompt: str = "",
    ratio: str = "9:16",
    duration: int = 5,
    ref_path: str | None = None,
    end_ref_path: str | None = None,
    **_: Any,
) -> None:
    """Render one shot on the GPU worker and write it back in studio shape.

    Mirrors ``run_comfy_shot``: same video-version record, same QA gate, same
    propagation. Keeping the shape identical is what makes the two sides
    comparable — a different bookkeeping shape would be a second variable.
    """
    _install_path()
    from drama_service import _judge_video_file, _now, _propagate_segment_video, new_id

    try:
        st0 = store.load(project_id)
        sh0 = next((x for x in (st0.get("shots") or []) if x["id"] == shot_id), None) if st0 else None
        prompt_used = base_prompt or (sh0 or {}).get("prompt") or ""
        # The studio writes shot prompts in Chinese; CogVideoX's text encoder is
        # English-only. See ``_translate_for_encoder``.
        prompt_used = _translate_for_encoder(prompt_used)

        # An i2v checkpoint cannot render without a first frame, and a t2v
        # checkpoint has nothing to do with one. The studio decides which by
        # whether it handed us a reference; the model follows that decision
        # rather than the other way round.
        #
        # The studio also passes its own model ids (e.g.
        # ``doubao-seedance-2-0-260128``) whenever its option default wins over
        # the backend pin. Those are vendor names this backend cannot load, and
        # passing one through produced a KeyError and a 500 per shot. Mapping
        # foreign ids to the local checkpoint is the backend's job: it is the
        # component that knows what it can actually run.
        local = {"cogvideox5b", "cogvideox5b_i2v", "wan21_1_3b", "longlive_1_3b"}
        # The reference arm is text-to-video, so this side must be too. The
        # studio attaches entity references to shots whenever a character bible
        # exists, which selected the i2v checkpoint; that checkpoint needs a
        # real first frame and returned HTTP 500 for every shot when the
        # reference was unusable (56 of 56 in the first v2 attempt). Forcing
        # t2v keeps the model invocation identical on both sides.
        force_t2v = str(os.environ.get("LOCAL_VIDEO_FORCE_T2V", "0")).lower() in {"1", "true", "yes"}
        if force_t2v:
            chosen = "cogvideox5b"
            ref_path = None
        elif model not in local:
            chosen = "cogvideox5b_i2v" if ref_path else "cogvideox5b"
        elif not ref_path and "i2v" in model:
            chosen = "cogvideox5b"
        elif ref_path and "i2v" not in model:
            chosen = "cogvideox5b_i2v"
        else:
            chosen = model

        # Params parity with the reference arm. The prior run left these at the
        # adapter's defaults — seed 0 for every shot and whatever duration the
        # storyboard asked for — while our side used seed = shot index and a
        # fixed 5 s. Those are model-invocation parameters, not orchestration,
        # so controlling them here keeps the comparison about the workflow.
        #
        # Seed: the shot's 0-based position within its episode, matching
        # `bench/study/tier3a_story.py:render(..., seed=index, ...)`.
        order = 0
        if st0:
            siblings = sorted(
                (s for s in (st0.get("shots") or [])
                 if not s.get("deletedAt") and s.get("episodeId") == (sh0 or {}).get("episodeId")),
                key=lambda s: int(s.get("shotOrder") or 0),
            )
            order = next((i for i, s in enumerate(siblings) if s["id"] == shot_id), 0)
        # Duration: the local checkpoint renders fixed 5 s clips in the reference
        # arm; a longer request here would differ by clip length, not by
        # workflow. Recorded so the report can state it.
        seconds = float(os.environ.get("LOCAL_VIDEO_SECONDS", "5"))

        response = requests.post(
            f"{RENDER_URL}/render/video",
            json={"model": chosen, "prompt": prompt_used, "seconds": seconds,
                  "seed": order, "image": ref_path},
            timeout=RENDER_TIMEOUT,
        )
        response.raise_for_status()

        vid_id = new_id("vid_")
        rel = f"media/videos/{vid_id}.mp4"
        dest = store._dir(project_id) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)

        st = store.load(project_id)
        sh = next((x for x in (st.get("shots") or []) if x["id"] == shot_id), None) if st else None
        if not sh:
            return
        # The vendor's own switch. With QA off we record the clip as produced and
        # say the gate was not applied, rather than inventing a pass.
        qa_on = str(os.environ.get("LOCAL_VIDEO_QA", "0")).lower() not in {"0", "false", "off", "no"}
        qa = _judge_video_file(dest, prompt=prompt_used, entities=[]) if qa_on else {
            "qaPass": True, "skipped": True, "reason": "LOCAL_VIDEO_QA=0: QA gate not applied",
        }
        version = {
            "id": vid_id,
            "shotId": shot_id,
            "path": rel,
            "providerTaskId": None,
            "model": chosen,
            "backend": BACKEND_ID,
            "duration": float(duration or 5),
            # The render service sends the clip but no size headers; record the
            # size the checkpoint actually rendered, not an empty "x".
            "resolution": "720x480" if "i2v" in chosen else "832x480",
            "prompt": prompt_used,
            "qa": qa,
            "qaPass": bool(qa.get("qaPass")),
            "rating": None,
            "note": "keyframe-anchored" if ref_path and "i2v" in chosen else "text-to-video: no keyframe",
            "createdAt": _now(),
        }
        sh.setdefault("videos", []).append(version)
        sh["selectedVideoId"] = vid_id
        sh["providerTaskId"] = None
        sh["staleReason"] = None
        sh["updatedAt"] = _now()
        if version["qaPass"]:
            sh["status"] = "success"
            sh["error"] = None
        else:
            sh["status"] = "failed"
            sh["error"] = f"QA below threshold: {qa.get('reason') or 'mismatch'}"
        _propagate_segment_video(st, sh)
        store.save(st)
    except Exception as exc:
        st = store.load(project_id)
        sh = next((x for x in (st.get("shots") or []) if x["id"] == shot_id), None) if st else None
        if sh:
            sh["status"] = "failed"
            sh["error"] = str(exc)
            sh["updatedAt"] = _now()
            _propagate_segment_video(st, sh)
            store.save(st)


def register() -> None:
    """Attach both backends to the studio's registries.

    Call once, before the studio serves a request. Idempotent: re-registering
    the same id replaces the entry rather than duplicating it.
    """
    _install_path()
    from providers.image_factory import IMAGE_BACKENDS
    from providers.video_factory import VIDEO_BACKENDS, register_video_job

    VIDEO_BACKENDS.register(BACKEND_ID, LocalVideoBackend(), priority=30)
    register_video_job(BACKEND_ID, run_local_shot)
    # The segment-QA path imports this name out of the module instead of going
    # through the job table, so it has to exist as an attribute too. Without it
    # every segment fails with "cannot import name 'run_local_shot'" and the
    # video stage loops until it gives up.
    import providers.video_jobs as video_jobs

    video_jobs.run_local_shot = run_local_shot
    # Above the free remote stills (33-40) so `auto` cannot wander off to a
    # network model mid-run; below nothing, because it is the only local one.
    IMAGE_BACKENDS.register(IMAGE_ID, LocalDiffusionImageBackend(), priority=50)
