"""Tests for the benchmark harness that need no GPU money and no model weights.

The judge's message building, scale parsing and payload construction are
deterministic, so they are pinned here. Anything that calls Ark or loads a
diffusion pipeline is exercised by ``bench/runner.py --stage generate`` instead;
a unit test that spends money is not a unit test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.config import ALL_DIMENSIONS, DYNAMIC_DIMENSIONS, STATIC_DIMENSIONS, VIDEO_TEXT_DIMENSIONS, JudgeConfig
from bench.frames import FramePayload, _grids, _resize_to_max_edge
from bench.judge import Judge
from bench.prompts import select_prompts
from bench.report import GROUPS, LABELS


def test_dimension_groups_cover_every_dimension():
    """Every dimension belongs to exactly one report group.

    A dimension outside GROUPS is scored but never printed, which reads as "not
    measured" in the report rather than as an oversight.
    """
    grouped = [name for _, dimensions, _ in GROUPS for name in dimensions]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(ALL_DIMENSIONS)


def test_video_text_dimensions_stay_in_the_one_to_three_scale():
    """The 1-3 group must not leak into the 1-5 group.

    Averaging the two groups, or printing them in one column, would compare a
    1-3 score against a 1-5 score as though they shared a scale.
    """
    assert set(VIDEO_TEXT_DIMENSIONS).isdisjoint(set(STATIC_DIMENSIONS) | set(DYNAMIC_DIMENSIONS))
    ranges = {name: maximum for _, dims, maximum in GROUPS for name in dims}
    assert all(ranges[d] == 3 for d in VIDEO_TEXT_DIMENSIONS)
    assert all(ranges[d] == 5 for d in STATIC_DIMENSIONS + DYNAMIC_DIMENSIONS)


def test_every_dimension_has_a_label():
    missing = [d for d in ALL_DIMENSIONS if d not in LABELS]
    assert not missing, f"dimensions without a report label: {missing}"


@pytest.mark.parametrize(
    "text,labels,expected",
    [
        # The format the instruction asks for.
        ("Final Scores:\n- ours: 5 ,because sharp\n- cogvideox5b: 3 ,because blurry",
         ["ours", "cogvideox5b"], {"ours": 5, "cogvideox5b": 3}),
        # What a real judge reply looked like before the instruction was fixed.
        ("- video: x ,because the frames are sharp.\nScore: 4.", ["demo"], {"demo": 4}),
        # An unlabelled score is only safe when there is one model.
        ("Score: 4", ["demo"], {"demo": 4}),
        # A missing model stays missing rather than borrowing a neighbour's score.
        ("- ours: 4 ,because ok", ["ours", "wan21_1_3b"], {"ours": 4, "wan21_1_3b": None}),
        # Nothing parseable.
        ("I cannot evaluate this.", ["demo"], {"demo": None}),
    ],
)
def test_score_extraction(text, labels, expected):
    assert Judge._extract_scores(text, labels) == expected


def test_label_matching_is_longest_first():
    """A prefix label must not swallow a longer one's score."""
    result = Judge._extract_scores("model_10: 2 ,because x\nmodel_1: 5 ,because y", ["model_1", "model_10"])
    assert result == {"model_1": 5, "model_10": 2}


def test_out_of_range_scores_are_rejected():
    """A 1-5 dimension must not record a 7."""
    assert Judge._extract_scores("- demo: 7 ,because", ["demo"], maximum=5) == {"demo": None}


# A verbatim reply from `imaging_quality` on a Wan clip. The judge kept the
# template's literal `x` and wrote the number in its own clause; the earlier
# patterns all required a numeric tail and returned None for this.
MEASURED_COMMA_REPLY = (
    "- video: x ,score 2, because the video has prominent persistent motion blur across "
    "almost all frames: the first two frames only show completely indistinct blurry paved "
    "ground with no clear subject, and after the walking person's lower legs and feet "
    "appear, obvious motion distortion and soft blurriness remains, with no sharp fine "
    "details, though the content (a person wearing jeans and sneakers walking on pavement) "
    "is vaguely recognizable. Brightness is generally reasonable without extreme "
    "overexposure, and there is no overwhelming severe noise, but the heavy blurriness and "
    "motion distortion lead to an unnatural, poor viewing experience that matches the poor "
    "quality standard."
)


@pytest.mark.parametrize(
    "text,maximum,expected",
    [
        # The measured reply: comma-terminated, template `x` still present.
        (MEASURED_COMMA_REPLY, 5, 2),
        ("Final Score is 3, because ...", 5, 3),
        # The last mention is the verdict, not the first.
        ("a reference model would score 5, but this clip scores 2", 5, 2),
        # Decimals are not integer scores.
        ("score 2.5 out of 5", 5, None),
        # A frame count is not a score.
        ("the video has 81 frames and scores of blur", 5, None),
        # Out of range for the dimension is not a score.
        ("score: 9", 5, None),
        ("- video: x ,because the video looks bad", 5, None),
    ],
)
def test_loose_score_covers_measured_reply_shapes(text, maximum, expected):
    assert Judge._loose_score(text, maximum) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("...\nEvaluation Result: 2 because colors match", 2),
        ("Evaluation Result\n<result> 3 </result>", 3),
        ("no result here", None),
        ("Evaluation Result: 9 because", None),
    ],
)
def test_consistency_parse(text, expected):
    assert Judge._parse_consistency(text) == expected


def test_static_message_parts_are_all_objects():
    """Ark rejects a content array containing bare strings.

    Upstream builds exactly that shape, so this guards the fix rather than
    restating it.
    """
    judge = object.__new__(Judge)
    judge.config = JudgeConfig()
    judge.prompts = {"imaging_quality": "system prompt"}
    payload = FramePayload(frames=["abc"], grid_frames=[], source="x")
    messages = judge._static_messages("system prompt", "ours", payload, {"other": payload})
    for part in messages[1]["content"]:
        assert isinstance(part, dict), f"bare content part: {part!r}"
        assert part["type"] in ("text", "image_url")


def test_dynamic_message_names_each_model():
    """Grid strips carry no text, so the model must be named around them.

    Without the label the judge cannot attribute a score in a multi-model call,
    which is what made upstream's gridview scores unparseable.
    """
    judge = object.__new__(Judge)
    judge.config = JudgeConfig()
    payload = FramePayload(frames=["a"], grid_frames=["g1"], source="x")
    messages = judge._dynamic_messages("system", "motion_effects", {"ours": payload, "other": payload})
    texts = " ".join(p["text"] for p in messages[1]["content"] if p["type"] == "text")
    assert "ours" in texts and "other" in texts


def test_resize_caps_the_longest_edge():
    pytest.importorskip("cv2", reason="frame decoding needs the bench environment")
    import numpy as np

    frame = np.zeros((1920, 1080, 3), dtype="uint8")
    resized = _resize_to_max_edge(frame, 640)
    assert max(resized.shape[:2]) == 640


def test_grid_strips_drop_the_remainder():
    pytest.importorskip("cv2", reason="jpeg encoding needs the bench environment")
    import numpy as np

    frames = [np.zeros((8, 8, 3), dtype="uint8") for _ in range(10)]
    strips = _grids(frames, 4, 85)
    assert len(strips) == 2


def test_prompt_selection_is_stable_and_ordered():
    """The same count must select the same prompts between runs."""
    first = select_prompts(5, "color")
    second = select_prompts(5, "color")
    assert [p.slug for p in first] == [p.slug for p in second]
    assert len(first) == 5
    assert all("color" in p.dimensions for p in first)
    assert len({p.slug for p in first}) == len(first), "slugs must be unique"


def test_prompt_slugs_are_filename_safe():
    """Slugs become filenames, so they may not carry path separators."""
    prompt = select_prompts(1)[0]
    assert "/" not in prompt.slug and "\\" not in prompt.slug
    assert prompt.slug.isascii()


def test_video_text_refuses_to_score_without_the_prompt():
    """The video-text dimensions ask "does this clip match the text prompt".

    A judge that never receives the prompt cannot answer that, so the call must
    fail loudly rather than return a 1-3 score with no meaning.
    """
    judge = object.__new__(Judge)
    judge.config = JudgeConfig()
    judge.prompts = {"color": {"gpt4o-system": "s"}}
    payload = FramePayload(frames=["a"], grid_frames=[], source="x")
    with pytest.raises(ValueError, match="text prompt"):
        judge.score_video_text("color", judge.prompts["color"], {"ours": payload}, "")


def test_clips_outside_the_run_dir_are_recorded_not_crashed(tmp_path):
    """A reused clip may live outside the run directory.

    ``--run-dir`` reuses videos from an earlier run, and a backend may write
    where it likes, so recording ``path.relative_to(run_dir)`` unconditionally
    raised ValueError and lost the whole result file. The absolute path is the
    correct fallback.
    """
    from bench.backends import Generation
    from bench.config import load_config
    from bench.judge import JudgeReply
    from bench.runner import write_run

    config = load_config()
    outside = tmp_path / "elsewhere" / "clip.mp4"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"x" * 2048)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generation = Generation(
        model="ours", prompt="a prompt", seed=0, path=outside,
        seconds=4.0, width=1080, height=1920, fps=30, modality="i2v", elapsed_s=1.0,
    )
    replies = {"imaging_quality": JudgeReply("imaging_quality", {"ours": 4.0})}
    write_run(config, run_dir, [], ["ours"], [0], 4.0, ["imaging_quality"], [generation], replies)

    recorded = json.loads((run_dir / "clips.json").read_text(encoding="utf-8"))
    assert recorded[0]["file"] == str(outside)


def test_estimate_counts_the_real_call_pattern():
    """Five calls per model per video for video-text, one per video for dynamic."""
    from bench.config import load_config
    from bench.runner import estimate

    config = load_config()
    budget = estimate(config, prompts=1, models=2, seeds=1,
                      dimensions=["imaging_quality", "motion_effects", "color"])
    # static: 2 calls; dynamic: 1 call; video-text: 5*2 calls. One video, so
    # the totals are the same number.
    assert budget["per_video_calls"] == 2 + 1 + 10
    assert budget["videos"] == 2
    assert budget["judge_calls"] == 2 * (2 + 1 + 10)


# --- LongLive frame arithmetic ------------------------------------------------
#
# `num_output_frames` is a LATENT count, and Wan's VAE upsamples time 4x, so a
# target duration has to be converted before it is passed. The first version
# divided by 4 and rounded down, which turned a 5-second request into a 19.3s
# clip: latent 78 is what "5 seconds" produced when the argument was taken as a
# pixel-frame count. These pin the conversion, not the model.

def _longlive_frames(seconds: float, fps: int = 16, block: int = 3) -> tuple[int, int]:
    """Mirror of LongLiveBackend.generate's arithmetic."""
    target_frames = int(round(seconds * fps))
    latent = max(block, int(round((target_frames + 3) / 4 / block)) * block)
    return latent, 4 * (latent - 1) + 1


@pytest.mark.parametrize("seconds", [3, 5, 8, 10, 20, 60])
def test_longlive_duration_is_close_to_the_request(seconds):
    """The clip must land within 10% of the requested length."""
    _, pixel = _longlive_frames(seconds)
    achieved = pixel / 16
    assert abs(achieved - seconds) / seconds < 0.10, f"{seconds}s -> {achieved:.2f}s"


@pytest.mark.parametrize("seconds", [3, 5, 8, 10, 20, 60])
def test_longlive_latent_count_is_a_valid_block_multiple(seconds):
    """The pipeline asserts num_output_frames % num_frame_per_block == 0."""
    latent, _ = _longlive_frames(seconds)
    assert latent % 3 == 0
    assert latent >= 3


def test_longlive_five_seconds_needs_latent_21_not_the_pixel_count():
    """The exact regression: 5s at 16fps is 80 pixel frames, but latent must be 21.

    Passing 80 (rounded to a block multiple) produced 309 frames = 19.3s.
    """
    latent, pixel = _longlive_frames(5)
    assert latent == 21
    assert pixel == 81
    assert _longlive_frames(5)[0] != 80


# --- segmented evaluation ----------------------------------------------------
#
# The cross-segment metrics must separate "stable" from "drifted". These tests
# use tiny synthetic frames rather than real clips: the question is whether the
# arithmetic distinguishes the two, not whether a codec decoded correctly.

# OpenCV exists only in the bench environment; these tests need it for resizing
# and fingerprinting, so they skip rather than fail in the pipeline's venv.
# Module-level importorskip would skip the whole file, including the score
# parser tests that have nothing to do with video.
def _has_cv2() -> bool:
    import importlib.util

    return importlib.util.find_spec("cv2") is not None


needs_cv2 = pytest.mark.skipif(not _has_cv2(), reason="cv2 is only in the bench venv")


def _frame(color: tuple[int, int, int], noise: float = 0.0, seed: int = 0):
    """A synthetic frame with real spatial structure.

    Flat colour is not a useful fixture: a single-colour frame gives a histogram
    with one bin and a flat mean-removed thumbnail, so its fingerprint is a zero
    vector and cosine comparison is undefined. These fixtures vary spatially so
    the fingerprint has something to describe, which is what a real frame always
    has.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    height, width = 90, 160
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    # A smooth gradient plus a moving blob: structure that survives resizing.
    base = np.stack([
        (xx / width) * color[0] + 20,
        (yy / height) * color[1] + 20,
        ((xx + yy) / (width + height)) * color[2] + 20,
    ], axis=-1).astype(np.float32)
    cx = 40 + 10 * seed
    base += 40 * np.exp(-(((xx - cx) ** 2 + (yy - 45) ** 2) / 800.0))[..., None]
    if noise:
        base += rng.normal(0, noise, base.shape).astype(np.float32)
    return np.clip(base, 0, 255)


@needs_cv2
def test_segment_measures_motion_and_flicker():
    from bench.segments import _measure_segment

    frames = [_frame((100, 120, 140), noise=1.0, seed=i) for i in range(20)]
    seg = _measure_segment(0, 0.0, 5.0, frames)
    assert seg.frame_count == 20
    assert seg.motion > 0
    # A tiny amount of noise is not flicker; the second difference stays small.
    assert seg.flicker < 5


@needs_cv2
def test_stable_segments_show_high_identity():
    from bench.segments import _measure_segment, cross_segment, summarise

    segments = [_measure_segment(i, i * 5.0, 5.0, [_frame((100, 120, 140), 1.0, i) for _ in range(10)])
                for i in range(4)]
    identity, transitions = cross_segment(segments)
    summary = summarise(segments, identity, transitions)
    assert summary["identity_min"] > 0.9
    assert summary["drift_onset_segment"] is None


@needs_cv2
def test_a_changed_palette_triggers_drift():
    from bench.segments import _measure_segment, cross_segment, summarise

    colours = [(100, 120, 140), (100, 120, 140), (30, 40, 60), (30, 40, 60)]
    segments = [_measure_segment(i, i * 5.0, 5.0, [_frame(c, 1.0, i) for _ in range(10)])
                for i, c in enumerate(colours)]
    identity, transitions = cross_segment(segments)
    summary = summarise(segments, identity, transitions)
    assert summary["identity_min"] < 0.9
    assert summary["drift_onset_segment"] == 2
    assert summary["drift_onset_segment"] != 1


@needs_cv2
def test_identity_anchors_on_the_first_segment_not_the_previous_one():
    """A gradual drift must not read as stable.

    Neighbour-to-neighbour comparison hides slow drift: every step is small
    while the total is large. Anchoring on segment 0 makes the accumulated
    difference visible.
    """
    from bench.segments import _measure_segment, cross_segment, summarise

    colours = [(100, 120, 140), (85, 105, 125), (70, 90, 110), (55, 75, 95)]
    segments = [_measure_segment(i, i * 5.0, 5.0, [_frame(c, 1.0, i) for _ in range(10)])
                for i, c in enumerate(colours)]
    identity, transitions = cross_segment(segments)
    summary = summarise(segments, identity, transitions)
    # The last segment differs most from the anchor, more than from its neighbour.
    assert identity[-1]["structure"] < identity[0]["structure"]
    assert summary["identity_spread"] > 0.05


def test_tiny_tail_segment_is_folded_into_its_neighbour(tmp_path):
    """A 5.06s clip must not report a one-frame second segment.

    It did, and that one-frame tail produced a spurious drift onset.
    """
    import numpy as np

    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (160, 90))
    for i in range(81):
        writer.write(np.full((90, 160, 3), (i * 2) % 255, dtype=np.uint8))
    writer.release()

    from bench.segments import segment_clip

    segments = segment_clip(path, segment_s=5.0)
    assert len(segments) == 1
    assert all(s.frame_count > 8 for s in segments)
