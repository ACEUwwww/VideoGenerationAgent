"""Judge videos with an OpenAI-compatible multimodal model.

The scoring prompts, the 1-5 / 1-3 scales and the output formats come verbatim
from the vendored Video-Bench checkout. Only the execution loop is rewritten
here, because upstream's has three defects that make a run either impossible or
misleading:

1. ``videobench/VideoTextAlignment.py:327`` reads ``modelname`` before the loop
   that binds it, so every video-text dimension raises ``UnboundLocalError``
   on the first video. Confirmed against the pinned checkout.
2. The judge model name is hardcoded (``gpt-4o-2024-08-06`` four times,
   ``gpt-4o-mini`` once), so pointing ``base_url`` at another provider 404s.
3. ``videobench/__init__.py`` does not route ``overall_consistency``, and
   ``prompt_dict`` has no ``video-text consistency`` key although
   ``VideoBench_full.json`` and the README both use that name.

Each dimension keeps upstream's contract: what the judge is asked, how many
calls it takes, and how the answer is parsed. Where upstream's parser is a bare
``re.search(r':\\s*(\\d+)')`` that silently grabs the wrong number when the judge
reasons first, this module anchors on the model label instead.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ALL_DIMENSIONS, DYNAMIC_DIMENSIONS, STATIC_DIMENSIONS, VIDEO_TEXT_DIMENSIONS, JudgeConfig, require_vendor
from .frames import FramePayload


def _retryable(exc: Exception) -> bool:
    """Whether the same call could plausibly succeed on a second attempt."""
    status = getattr(exc, "status_code", None)
    if status is None:
        # No HTTP status means the request never completed: connection reset,
        # timeout, DNS. Those are worth another try.
        return True
    return status in (408, 409, 425, 429) or 500 <= status < 600


def _prompts() -> dict:
    """Dimension prompts, imported from the vendored checkout."""
    vendor = str(require_vendor())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from videobench.prompt_dict import prompt

    return prompt


@dataclass
class JudgeReply:
    dimension: str
    scores: dict[str, int | float | None]
    raw: dict[str, str] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str = ""


class Judge:
    """Scores prepared frame payloads; never touches a video file directly."""

    def __init__(self, config: JudgeConfig):
        if not config.api_key:
            raise SystemExit("ARK_API_KEY is empty; the judge cannot run without it.")
        self.config = config
        self.prompts = _prompts()
        self._client = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    # -- transport ---------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.request_timeout_s,
            )
        return self._client

    def complete(self, messages: list[dict], *, model: str = "") -> str:
        """One judge call, streamed, retried only on failures that can succeed again.

        Streaming is not an optimisation here, it is what makes the call
        survive. A non-streaming call sends nothing while the judge reasons —
        first token measured at 10-58s — and the egress proxy drops the
        connection at a near-constant 153s. Measured on the same four static
        calls: non-streaming returned 2 of 4, streaming 4 of 4.

        A malformed request is not transient: retrying a 400 four times with
        exponential backoff turns a one-second mistake into a 25-second one and
        still fails. Only transport errors, 429 and 5xx are retried.

        ``stream_options.include_usage`` keeps the token counts, which a
        streamed response otherwise omits; Ark supports it.
        """
        last = ""
        for attempt in range(self.config.attempts):
            try:
                stream = self.client.chat.completions.create(
                    model=model or self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                parts: list[str] = []
                usage = None
                for chunk in stream:
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last = f"{type(exc).__name__}: {exc}"
                if not _retryable(exc):
                    raise RuntimeError(last) from exc
                time.sleep(2 ** attempt)
                continue
            if usage is not None:
                self.usage["prompt_tokens"] += usage.prompt_tokens or 0
                self.usage["completion_tokens"] += usage.completion_tokens or 0
            self.usage["calls"] += 1
            return "".join(parts)
        raise RuntimeError(f"judge call failed after {self.config.attempts} attempts: {last}")

    # -- message builders --------------------------------------------------

    @staticmethod
    def _image(b64: str) -> dict:
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}

    @staticmethod
    def _text(text: str) -> dict:
        """Every content part is an object.

        Ark rejects a content array containing bare strings ("expected a string
        or array of objects"), which is exactly the shape upstream builds. The
        OpenAI SDK does not normalise it, so it has to be done here.
        """
        return {"type": "text", "text": text}

    def _static_messages(self, prompt: str, model: str, payload: FramePayload, examples: dict[str, FramePayload]) -> list[dict]:
        """Upstream staticquality.py: one call per model, other models as examples.

        Only the first frame of each example is shown, which is what upstream
        does — the static dimensions judge a still, not a sequence.
        """
        content: list = [
            self._text("According to **Important Notes** in system message, there are examples from other models.\n")
        ]
        for position, (name, example) in enumerate(examples.items(), start=1):
            if not example.frames:
                continue
            content.append(self._text(f"This example is from model {position} \n"))
            content.append(self._image(example.frames[0]))
        content.append(self._text("These are the frames from the video you are evaluating. \n"))
        content.extend(self._image(frame) for frame in payload.frames)
        content.append(self._text(
            "Assuming there are a video ' scoring 'x',provide your analysis and explanation in the output format as follows:\n"
            "- video: x ,because ..."
        ))
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]

    def _dynamic_messages(self, prompt: str, dimension: str, payloads: dict[str, FramePayload]) -> list[dict]:
        """Upstream dynamicquality.py: every model in one call, grid strips only."""
        label = {"temporal_consistency": "temporal consistency", "motion_effects": "motion effects"}[dimension]
        content: list = [
            self._text(
                "The following images are concatenated by the key frames of the video."
                f"And one of the following images arranges {self.config.grid_size} key frames per second from a video in a 1*{self.config.grid_size} grid view.\n"
            ),
            self._text("Please associate the images in time order to help you watch the whole video.\n"),
        ]
        labels = list(payloads)
        for name in labels:
            # The grid strips carry no identifying text of their own, and
            # upstream's gridview module concatenates them without saying which
            # model each belongs to — so the judge cannot attribute a score to
            # a model. Naming each block is what makes a multi-model call
            # parseable at all.
            content.append(self._text(f"\nFrames from the video generated by {name}:\n"))
            content.extend(self._image(strip) for strip in payloads[name].grid_frames)
        content.append(self._text(f"Find the issues of the video in {label} and then evaluate the {label} of the video.\n"))
        content.append(self._text(self._format_instruction(labels)))
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _format_instruction(labels: list[str]) -> str:
        """Ask for one labelled line per model, with the score filled in.

        Upstream's instruction ("assuming a video scoring 'x'") shows the score
        slot as the literal letter ``x`` and never names the model, so a real
        reply came back as ``- video: x ,because ...`` with the number in prose
        at the end. Stating the label and an example with a real number gets a
        parser-friendly answer.
        """
        lines = ["Provide your analysis, then end with one line per video in exactly this format:",
                 "Final Scores:"]
        for label in labels:
            lines.append(f"- {label}: <integer 1-5> ,because ...")
        return "\n".join(lines) + "\n"

    # -- parsers -----------------------------------------------------------

    @classmethod
    def _extract_scores(cls, text: str, labels: list[str], maximum: int = 5) -> dict[str, int | None]:
        """Scores per model, from the most reliable signal available.

        Three shapes appear in practice, so they are tried in order:

        1. ``<label>: <n>`` — what the instruction asks for. Labels are matched
           longest-first so ``model_10`` cannot be read as ``model_1``.
        2. A single model with an unlabelled score line — ``Score: 4`` or
           ``- video: 4 ,because``. Safe, because there is only one video.
        3. Nothing usable. Returns ``None`` for every model rather than reusing
           one number across models, which would invent agreement.
        """
        by_label: dict[str, int | None] = {}
        for label in sorted(labels, key=len, reverse=True):
            # The label must stand alone: without the boundary, ``model_1``
            # matches inside ``model_10: 2`` and reads the 0.
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])\s*[:\-]?\s*(\d{{1,2}})"
            match = re.search(pattern, text, re.IGNORECASE)
            if match is None:
                by_label[label] = None
                continue
            value = int(match.group(1))
            by_label[label] = value if 1 <= value <= maximum else None

        if all(by_label.get(label) is not None for label in labels):
            return {label: by_label[label] for label in labels}

        if len(labels) == 1:
            label = labels[0]
            if by_label[label] is not None:
                return {label: by_label[label]}
            return {label: cls._loose_score(text, maximum)}

        return {label: by_label.get(label) for label in labels}

    @staticmethod
    def _loose_score(text: str, maximum: int) -> int | None:
        """The score in a reply that did not follow the requested format.

        Anchored phrases only. A bare ``\\d`` scan would happily return a frame
        count or a step number; these patterns all attach the number to the act
        of scoring.

        The first pattern is the loosest and is what catches real replies. A
        measured example, from ``imaging_quality`` on a Wan clip:

            - video: x ,score 2, because the video has prominent motion blur ...

        The judge keeps the template's literal ``x`` and puts the real number in
        its own clause. Requiring a numeric tail (``/5``, ``out of 5``, end of
        line, a full stop) missed every comma-terminated occurrence, so the
        reply scored ``None`` even though it plainly said 2.
        """
        patterns = (
            # A sentence-ending full stop is fine; a decimal point is not, so the
            # lookahead only rejects a point followed by another digit.
            r"(?<![A-Za-z])scores?(?![A-Za-z])\s*(?:is|:|=)?\s*[\"']?(\d{1,2})[\"']?(?!\d)(?!\.\d)",
            r"scoring\s+['\"]?(\d{1,2})['\"]?",
            r"- video:\s*(\d{1,2})\s*,?\s*because",
            r"because\s*\(\s*(\d{1,2})\s*/\s*5\s*\)",
        )
        for pattern in patterns:
            found = [
                int(match.group(1))
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            ]
            valid = [value for value in found if 1 <= value <= maximum]
            # The last mention is the verdict. A reply can refer to a score and
            # then revise it ("that would score 5, but this clip scores 2"), and
            # the conclusion comes last, so the first hit would be the wrong one.
            if valid:
                return valid[-1]
        return None

    # -- dimensions --------------------------------------------------------

    def score_static(self, dimension: str, payloads: dict[str, FramePayload]) -> JudgeReply:
        prompt = self.prompts[dimension]
        started = time.monotonic()
        scores: dict[str, int | None] = {}
        raw: dict[str, str] = {}
        errors: list[str] = []
        for name, payload in payloads.items():
            examples = {k: v for k, v in payloads.items() if k != name}
            try:
                text = self.complete(self._static_messages(prompt, name, payload, examples))
            except Exception as exc:  # noqa: BLE001 - one model must not sink the run
                errors.append(f"{name}: {exc}")
                scores[name] = None
                continue
            raw[name] = text
            scores[name] = self._extract_scores(text, [name])[name]
        return JudgeReply(
            dimension=dimension,
            scores=scores,
            raw=raw,
            usage=dict(self.usage),
            elapsed_s=time.monotonic() - started,
            error="; ".join(errors),
        )

    def score_dynamic(self, dimension: str, payloads: dict[str, FramePayload]) -> JudgeReply:
        prompt = self.prompts[dimension]
        started = time.monotonic()
        labels = list(payloads)
        try:
            text = self.complete(self._dynamic_messages(prompt, dimension, payloads))
        except Exception as exc:  # noqa: BLE001
            return JudgeReply(dimension, {k: None for k in labels}, {}, dict(self.usage),
                              time.monotonic() - started, str(exc))
        return JudgeReply(
            dimension=dimension,
            scores=self._extract_scores(text, labels),
            raw={"all": text},
            usage=dict(self.usage),
            elapsed_s=time.monotonic() - started,
        )

    def score_video_text(
        self,
        dimension: str,
        prompt_spec: dict,
        payloads: dict[str, FramePayload],
        video_prompt: str,
    ) -> JudgeReply:
        """Upstream VideoTextAlignment.py's two-assistant chain, repaired.

        Per model: one call to describe, two calls to ask questions, one to
        answer them, one to summarise into a score. Five calls, which is what
        upstream does — the multi-round design is the paper's "chain of query".

        Two upstream defects are fixed here. ``modelname`` is bound before it is
        read (upstream raises ``UnboundLocalError`` on the first video), and the
        video's text prompt is actually passed: this dimension asks how well the
        clip matches what was asked for, so a judge that never receives the
        prompt is not measuring consistency at all.

        ``prompt_spec`` is the dimension's system-prompt bundle (describe / two
        assistants / answer / summarise), not the video prompt.
        """
        if not video_prompt:
            raise ValueError("video-text dimensions need the clip's text prompt")
        started = time.monotonic()
        scores: dict[str, int | float | None] = {}
        raw: dict[str, str] = {}
        errors: list[str] = []
        for name, payload in payloads.items():
            try:
                scores[name], raw[name] = self._video_text_one(name, prompt_spec, payload, video_prompt)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
                scores[name] = None
        return JudgeReply(
            dimension=dimension,
            scores=scores,
            raw=raw,
            usage=dict(self.usage),
            elapsed_s=time.monotonic() - started,
            error="; ".join(errors),
        )

    def _video_text_one(
        self, name: str, prompt: dict, payload: FramePayload, video_prompt: str
    ) -> tuple[float | None, str]:
        frames = [self._image(frame) for frame in payload.frames]
        model_line = f"{len(payload.frames)} frames from {name}."
        video_prompt_block = f"This is the text prompt:\n{video_prompt}"

        # 1. Describe the video.
        describe = self.complete([
            {"role": "system", "content": prompt["gpt4o-system"]},
            {"role": "user", "content": [
                {"type": "text", "text": "You must think following the 'Evaluation Steps' one by one.\n"},
                {"type": "text", "text": f"\n\nThese are the frames from the video generated by {name}\n"},
                {"type": "text", "text": model_line},
                *frames,
            ]},
        ])
        description = f"This is the video's initial description:\n<description>\n{describe}\n</description>"

        # 2. Two assistants each raise one question, seeing the other's.
        questions: list[str] = []
        running = description
        for role in ("Assistant-one", "Assistant-two"):
            reply = self.complete([
                {"role": "system", "content": prompt[role]},
                {"role": "user", "content": (
                    "It's the first round." + prompt[role] + video_prompt_block
                    + "\n\n" + running
                    + "\n\nYour questions must be deisplayed in the format: \n<question>\n[your questions] or [I have no question]\n</question>."
                    + "\n\nDisplay the results in the specified Output Format"
                )},
            ])
            match = re.search(r"\[Your questions?\]\s*:?\s*(.*)", reply, re.DOTALL)
            questions.append(match.group(1).strip() if match else "No question")
            running = running + f"\nThis is a question of another assistant:\n<question-one>\n{questions[-1]}\n</question-one>"

        # 3. Answer the collected questions with the frames in view. Upstream
        # puts the text prompt first in this turn; keeping that order matters
        # because the judge is asked to check the clip against this prompt.
        qa = "\n\nThere are the questions of two assistants:\n\n<qa_history>\n" + "\n".join(questions) + "\n</qa_history>\n"
        answer = self.complete([
            {"role": "system", "content": prompt["gpt4o-answer"]},
            {"role": "user", "content": [
                {"type": "text", "text": video_prompt_block},
                {"type": "text", "text": qa},
                {"type": "text", "text": f"These are the frames from the video generated by {name}\n"},
                {"type": "text", "text": f"{model_line}\n"},
                *frames,
            ]},
        ])

        # 4. Summarise into the 1-3 consistency score. Upstream appends the text
        # prompt to the summary message too, so the final judgement is made
        # against the requirement rather than only against the description.
        history = (
            "There are the two informations:\n<history>\n"
            + description + "\n" + answer
            + "\n</history>"
        )
        final = self.complete([
            {"role": "system", "content": prompt["summer-system"]},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "You must think following the 'evaluation steps' one by one.\n\n"
                    + history + "\n\n" + video_prompt_block + "\n\n"
                )},
                {"type": "text", "text": f"{model_line}\n"},
                *frames,
                {"type": "text", "text": f"The name of the AI medel is {name}\n"},
            ]},
        ])
        return self._parse_consistency(final), final

    @staticmethod
    def _parse_consistency(text: str) -> float | None:
        """Read the score that follows "Evaluation Result", as upstream does.

        Upstream additionally requires the word "because" before the number,
        which yields a raw string "None" the moment the judge phrases it
        differently. The number after the labelled result is what matters.
        """
        match = re.search(r"Evaluation Result[^0-9]{0,80}?(\d+)", text, re.DOTALL | re.IGNORECASE)
        if match is None:
            return None
        value = int(match.group(1))
        return value if 1 <= value <= 3 else None

    def score(self, dimension: str, payloads: dict[str, FramePayload], video_prompt: str = "") -> JudgeReply:
        """Score one dimension for every model.

        ``video_prompt`` is required by the video-text dimensions, which ask how
        well each clip matches the text it was generated from.
        """
        if dimension in STATIC_DIMENSIONS:
            return self.score_static(dimension, payloads)
        if dimension in DYNAMIC_DIMENSIONS:
            return self.score_dynamic(dimension, payloads)
        if dimension in VIDEO_TEXT_DIMENSIONS:
            return self.score_video_text(dimension, self.prompts[dimension], payloads, video_prompt)
        raise ValueError(f"unknown dimension: {dimension}; expected one of {ALL_DIMENSIONS}")
