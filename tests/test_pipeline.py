from __future__ import annotations

import json

import pytest

from agent.graph import build_graph, run_agent
from agent.settings import get_settings
from backend.services import project_store
from mcp_servers.metrics import compare_to_reference, extract_frames, measure, score_dimensions
from mcp_servers.providers.ffmpeg_utils import media_facts

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None, reason="ffmpeg is required for media tests"
)


@pytest.fixture
def project(project_root, fake_llm):
    created = project_store.create_project("电梯里的倒影慢半拍", "电影感竖屏短剧", 2, genre="悬疑")
    state = project_store.load_state(created["project_id"])
    state["status"] = "running"
    return created["project_id"], state


def test_graph_has_expected_nodes():
    nodes = set(build_graph().get_graph().nodes)
    for name in ("plan_story", "plan_shots", "evaluate_clip", "synthesize_dialogue", "compose_video"):
        assert name in nodes


def test_full_pipeline_produces_real_media(project):
    project_id, state = project
    final = run_agent(state)

    assert final["status"] == "completed"
    assert len(final["shots"]) == 2
    assert final["characters"][0]["appearance"]

    root = project_store.project_dir(project_id)
    video = root / final["final_video_path"]
    assert video.exists() and video.stat().st_size > 100_000

    facts = media_facts(video)
    settings = get_settings().media
    assert (facts["width"], facts["height"]) == (settings.width, settings.height)
    assert facts["codec"] == "h264"
    assert facts["duration_s"] > 5.0

    for shot in final["shots"]:
        assert (root / shot["image_path"]).exists()
        assert (root / shot["video_path"]).stat().st_size > 10_000
        assert shot["multidim_score"] > 0
        assert len(shot["dims"]) == 7
        assert shot["keyframe_path"]


def test_subtitles_are_burned_and_timed(project):
    project_id, state = project
    final = run_agent(state)
    root = project_store.project_dir(project_id)

    ass = root / final["subtitle_path"]
    assert ass.exists()
    body = ass.read_text(encoding="utf-8")
    assert "Dialogue:" in body
    assert "谁？" in body

    lines = [shot for shot in final["shots"] if shot.get("subtitle_lines")]
    assert lines, "dialogue shot should produce subtitle lines"
    for line in lines[0]["subtitle_lines"]:
        assert 0 <= line["start_s"] < line["end_s"] <= lines[0]["duration_s"] + 0.01


def test_keyframe_memory_anchors_first_shot(project):
    project_id, state = project
    final = run_agent(state)
    memory = final["keyframe_memory"]

    assert memory["anchor"] == "keyframes/shot_001.png"
    assert len(memory["entries"]) == 2
    assert memory["entries"][0]["drift_vs_anchor"] is None
    assert memory["entries"][1]["drift_vs_anchor"] is not None


def test_episode_report_covers_all_dimensions(project):
    _project_id, state = project
    final = run_agent(state)
    report = final["episode_report"]

    assert report["score"] > 0
    assert set(report["dimensions"]) == set(final["shots"][0]["dims"])
    for entry in report["dimensions"].values():
        assert 0.0 <= entry["worst"] <= entry["score"] <= 1.0


def test_llm_prompts_carry_character_anchor(project):
    _project_id, state = project
    final = run_agent(state)
    anchor_words = final["characters"][0]["appearance"].split(",")[0]
    for shot in final["shots"]:
        assert anchor_words in shot["image_prompt"]


def test_mock_run_skips_repair_calls(project):
    """Regression: mock renders used to guarantee a failing verdict, burning
    two LLM repair calls per shot for output that could never change."""
    project_id, state = project
    final = run_agent(state)
    stages = final["usage"]["stages"]
    assert not [name for name in stages if "repair" in name]
    for shot in final["shots"]:
        assert shot.get("image_mock") is True
        assert shot.get("video_mock") is True
        assert shot["image_retry_count"] == 0
        assert shot["video_retry_count"] == 0


def test_usage_is_tracked(project):
    _project_id, state = project
    final = run_agent(state)
    assert final["usage"]["calls"] >= 6
    assert final["usage"]["prompt_tokens"] > 0


@pytest.mark.parametrize("count", [1, 3])
def test_shot_count_is_respected(project_root, fake_llm, count):
    created = project_store.create_project("测试", "电影感", count)
    state = project_store.load_state(created["project_id"])
    final = run_agent(state)
    assert len(final["shots"]) == count


def test_transitions_overlap_shortens_the_timeline(project_root, fake_llm, monkeypatch):
    """A storyboard transition is real time, not a label: an overlapping
    transition must shorten the encoded episode by its overlap."""
    import agent.settings as settings_mod

    settings_mod.reset_settings()
    state = project_store.load_state(project_store.create_project("转场", "电影感", 2)["project_id"])
    state["status"] = "running"
    final = run_agent(state)

    shots = final["shots"]
    meta = json.loads((project_root / state["project_id"] / "final" / "final_video.json").read_text(encoding="utf-8"))
    planned = sum(float(shot["duration_s"]) for shot in shots)
    # Each non-hard-cut boundary subtracts its overlap; there is no title card.
    overlaps = {"fade": 0.40, "fadewhite": 0.25, "fadeblack": 0.30}
    applied = [t["applied"] for t in meta["transitions"]]
    total_overlap = sum(overlaps.get(kind, 0.0) for kind in applied)
    expected = planned - total_overlap
    assert abs(meta["duration_s"] - expected) < 0.35
    facts = media_facts(project_root / state["project_id"] / final["final_video_path"])
    assert abs(facts["duration_s"] - meta["duration_s"]) < 0.35


def test_subtitles_are_rebased_onto_the_encoded_timeline(project_root, fake_llm):
    """Subtitle times are relative to each shot and must be shifted onto the
    encoded timeline, where transitions have already eaten their overlap."""
    state = project_store.load_state(project_store.create_project("字幕位移", "电影感", 2)["project_id"])
    state["status"] = "running"
    final = run_agent(state)
    body = (project_root / state["project_id"] / final["subtitle_path"]).read_text(encoding="utf-8")
    dialogue = [line for line in body.splitlines() if line.startswith("Dialogue:")]
    assert dialogue
    for line in dialogue:
        assert line.split(",")[1] != "0:00:00.00"


def test_bgm_is_mixed_and_preserves_length(project_root, fake_llm, monkeypatch):
    """A looping BGM bed must not truncate the episode: -shortest used to cut
    the video to the BGM's own length."""
    import agent.settings as settings_mod
    import mcp_servers.video_tools_server as tools

    state = project_store.load_state(project_store.create_project("配乐", "电影感", 1)["project_id"])
    state["status"] = "running"
    final = run_agent(state)

    root = project_root / state["project_id"]
    bgm = root / "audio" / "bgm.m4a"
    tools.run([tools.ffmpeg_bin(), "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(bgm)])

    clips = [
        {
            "video_path": final["shots"][0]["video_path"],
            "audio_path": final["shots"][0].get("audio_path", ""),
            "duration_s": final["shots"][0]["duration_s"],
            "transition": "硬切",
            "subtitle_lines": final["shots"][0].get("subtitle_lines") or [],
        }
    ]
    with_bgm = tools.compose_video(clips, state["project_id"], title="片名", bgm_path="audio/bgm.m4a")
    without = tools.compose_video(clips, state["project_id"], title="片名")

    mixed = media_facts(root / with_bgm["final_video_path"])
    plain = media_facts(root / without["final_video_path"])
    assert mixed["has_audio"] and plain["has_audio"]
    assert abs(mixed["duration_s"] - plain["duration_s"]) < 0.35


def test_vision_judge_covers_all_dimensions_in_one_call(tmp_path):
    """One keyframe upload must score every MLLM dimension: calling per
    dimension re-uploads the same image and triples the token cost."""
    from mcp_servers.videobench_eval import MLLM_DIMENSIONS, evaluate_clip

    clip = tmp_path / "clip.mp4"
    keyframe = tmp_path / "keyframe.png"
    subprocess_ffmpeg = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x334455:s=540x960:d=2:r=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clip),
    ]
    import subprocess as sp

    sp.run(subprocess_ffmpeg, capture_output=True, check=True)
    sp.run(["ffmpeg", "-y", "-i", str(clip), "-frames:v", "1", str(keyframe)], capture_output=True, check=True)

    calls: list[str] = []

    def judge(image_path, prompt):
        calls.append(str(image_path))
        return {"imaging_quality": 5, "aesthetic_quality": 4, "motion_effects": 3, "object_consistency": 2}, "脸部轻微变形"

    report = evaluate_clip(clip, shot={"shot_size": "特写"}, keyframe_path=keyframe, mllm_judge=judge)

    assert len(calls) == 1, f"expected one vision call, got {len(calls)}"
    assert calls[0] == str(keyframe), "judge must receive the full-resolution keyframe"
    for name in MLLM_DIMENSIONS:
        entry = report["dimensions"][name]
        assert entry.get("mllm_score_1to5") is not None, f"{name} missing judge score"
        assert entry["mllm_reason"] == "脸部轻微变形"
    # 2/5 maps to 0.25 and is max-merged with the measured estimate, so the
    # recorded score never falls below the judge's own floor.
    subject = report["dimensions"]["object_consistency"]
    assert subject["score"] >= 0.25 - 1e-9


def test_vision_judge_is_absent_without_paid_gate(monkeypatch):
    """No key or no ARK_ALLOW_PAID means no judge, and the measured path stays."""
    from mcp_servers.providers import get_vision_judge

    monkeypatch.delenv("ARK_ALLOW_PAID", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    import agent.settings as settings_mod

    settings_mod.reset_settings()
    assert get_vision_judge() is None


def test_last_frame_capability_is_model_gated():
    """Seeding last_frame into a pro-fast task is rejected by the API, so the
    capability check must be explicit rather than assumed."""
    from mcp_servers.providers.ark_media import supports_last_frame

    assert supports_last_frame("doubao-seedance-1-0-pro-fast-251015") is False
    assert supports_last_frame("doubao-seedance-1-0-pro-250528") is True
    assert supports_last_frame("doubao-seedance-2-0-260128") is True
    assert supports_last_frame("doubao-seedance-2-5-260628") is True


def test_extract_last_frame_reads_a_real_clip(tmp_path):
    """Chaining needs the tail frame of the previous clip, not its opening."""
    import subprocess as sp

    from mcp_servers.keyframe_memory import extract_last_frame

    clip = tmp_path / "clip.mp4"
    sp.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x334455:s=320x568:d=2:r=30",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clip)],
        capture_output=True, check=True,
    )
    out = extract_last_frame(clip, tmp_path / "last.png")
    assert out.exists() and out.stat().st_size > 1000


def test_sequence_judge_scores_the_ordered_keyframes(tmp_path):
    """Per-clip scoring cannot see cross-shot contradictions; the sequence
    judge must receive every keyframe in playback order in one call."""
    from mcp_servers.videobench_eval import SEQUENCE_DIMENSIONS, evaluate_sequence

    frames = []
    for index in range(3):
        frame = tmp_path / f"k{index}.png"
        frame.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(64))
        frames.append(frame)

    shots = [
        {"shot_size": "特写", "description": "他推开门", "dialogue": "有人吗？"},
        {"shot_size": "中景", "description": "街上没人", "dialogue": "怎么都一样"},
        {"shot_size": "全景", "description": "集市出现", "dialogue": "有人！"},
    ]

    seen: list[list[str]] = []

    def judge(paths, prompt):
        seen.append([str(p) for p in paths])
        assert "story_coherence" in prompt and "continuity" in prompt and "pacing" in prompt
        return {"story_coherence": 4, "continuity": 3, "pacing": 5, "reason": "闭环成立"}, "闭环成立"

    result = evaluate_sequence(frames, shots, mllm_judge=judge)

    assert len(seen) == 1, f"expected one sequence call, got {len(seen)}"
    assert seen[0] == [str(p) for p in frames], "keyframes must arrive in playback order"
    assert set(result["dimensions"]) == set(SEQUENCE_DIMENSIONS)
    assert result["dimensions"]["pacing"]["score_1to5"] == 5
    assert result["score"] is not None


def test_sequence_judge_absent_without_a_judge():
    """No vision judge must report unconfigured, not fabricate a score."""
    from mcp_servers.videobench_eval import evaluate_sequence

    result = evaluate_sequence([], [], mllm_judge=None)
    assert result["configured"] is False
    assert result["score"] is None


def test_tail_to_head_join_reuses_the_previous_clip(project):
    """The cut between two clips must land on an identical picture.

    Shot N+1 starts from the real last frame of clip N, so the boundary is not
    a jump. This is the branch pro-fast needs: it rejects the last_frame role
    outright, so the join has to be built on the head side.
    """
    from agent.nodes.video_nodes import _previous_tail_frame

    project_id, state = project
    final = run_agent(state)
    shots = final["shots"]
    assert len(shots) >= 2

    root = project_store.project_dir(project_id)
    joined = dict(shots[1])
    assert joined.get("video_path"), "mock clip 1 should exist"

    tail = _previous_tail_frame(final, joined)
    assert tail, "clip 2 must be joinable to clip 1"
    assert tail.endswith(f"{shots[0]['id']}_last.png")
    assert (root / tail).exists() and (root / tail).stat().st_size > 1000

    # The first clip has nothing before it to be joined to.
    assert _previous_tail_frame(final, shots[0]) == ""


def test_authored_join_targets_the_next_keyframe(project):
    """The other branch: an authored last frame drawn from the next keyframe.

    Both ends are then authored pictures, so drift cannot accumulate across the
    sequence. Available only on models that accept the last_frame role.
    """
    from agent.nodes.video_nodes import _next_keyframe

    project_id, state = project
    final = run_agent(state)
    shots = final["shots"]

    assert _next_keyframe(final, shots[0]) == shots[1]["image_path"]
    # The last shot has no successor to land on.
    assert _next_keyframe(final, shots[-1]) == ""


def test_image_phase_finishes_before_any_clip_is_rendered(project):
    """Every keyframe must exist before shot 1 renders.

    Interleaving the phases would mean keyframe[N+1] does not exist yet when
    shot N asks for it as a final frame, and the chain silently degrades to
    first-frame-only generation.
    """
    import json as _json

    project_id, state = project
    final = run_agent(state)
    root = project_store.project_dir(project_id)

    events = [
        _json.loads(line)
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Tool calls land as (type="Action", message="MCP.<tool>").
    kinds = [event.get("message") for event in events]
    first_video = kinds.index("MCP.generate_video")
    image_actions = [i for i, kind in enumerate(kinds) if kind == "MCP.generate_image"]
    assert image_actions, "image phase should have run"
    assert max(image_actions) < first_video, "all keyframes must precede the first clip"


def test_a_moderated_clip_is_skipped_instead_of_killing_the_episode(project, monkeypatch):
    """One rejected shot must not discard the shots that already rendered.

    Observed for real: a 1135 s run died whole when a single clip came back
    ``OutputVideoSensitiveContentDetected``. Every other clip in that episode
    had been rendered and paid for, and the pipeline had no branch for the
    error, so the exception unwound the entire run.
    """
    from agent.nodes import video_nodes
    from mcp_servers.providers.ark_media import ContentModerationError

    real = video_nodes.tools.generate_video

    def reject_one(*args, **kwargs):
        shot_id = args[3]
        if shot_id.endswith("002"):
            raise ContentModerationError("Ark video task cgt-x failed: OutputVideoSensitiveContentDetected")
        return real(*args, **kwargs)

    project_id, state = project
    monkeypatch.setattr(video_nodes.tools, "generate_video", reject_one)
    final = run_agent(state)

    # The episode still completes, one shot short.
    assert final.get("final_video_path")
    skipped = [s for s in final["shots"] if s.get("status") == "video_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["id"].endswith("002")
    assert skipped[0].get("video_path") == ""
    assert "SensitiveContentDetected" in skipped[0]["video_skip_reason"]

    # Every other clip is untouched: the rejection is isolated, not contagious.
    rendered = [s for s in final["shots"] if s.get("video_path")]
    assert len(rendered) == len(final["shots"]) - 1


def test_the_skip_route_bypasses_qa_and_keyframe_extraction(project):
    """A skipped shot has no pixels, so it must not reach evaluate_clip.

    The state handed to the video node is the one the planner produced; a fresh
    project has no shots yet, which is why the fixture runs the agent first.
    """
    from agent.nodes.video_nodes import clip_was_generated

    project_id, state = project
    final = run_agent(state)
    assert final["shots"]

    final["current_shot_index"] = 0
    assert clip_was_generated(final) == "generated"

    final["shots"] = [dict(final["shots"][0], video_path="", status="video_skipped")]
    assert clip_was_generated(final) == "skipped"


@pytest.mark.parametrize("model", ["cogvideox5b", "wan21_1_3b", "longlive_1_3b"])
def test_text_to_video_models_declare_no_frame_roles(model):
    """A T2V model consumes no image; declaring otherwise sends the run through
    the keyframe stage to render pictures the model silently ignores."""
    from mcp_servers.providers.local_media import LocalVideoProvider

    assert LocalVideoProvider(model).frame_roles == frozenset()


def test_the_i2v_model_declares_a_first_frame_role():
    from mcp_servers.providers.local_media import LocalVideoProvider

    assert LocalVideoProvider("cogvideox5b_i2v").frame_roles == frozenset({"first_frame"})


@pytest.mark.parametrize("provider,expected", [
    ("mock", True),
    ("ark", True),
])
def test_every_video_provider_declares_its_frame_roles(provider, expected):
    """The orchestration reads this attribute instead of matching a model id,
    so a provider without it would fail at the first generated clip."""
    from mcp_servers.providers import get_video_provider

    assert hasattr(get_video_provider(), "frame_roles") is expected


@pytest.mark.parametrize("model,fps,frames,duration", [
    ("cogvideox5b", 8, 41, 5.125),
    ("wan21_1_3b", 16, 81, 5.0625),
])
def test_local_model_frame_contract(model, fps, frames, duration):
    """Each family gets its own frame rule: CogVideoX needs 4k+1 at 8 fps, Wan
    4k+1 up to 81 at 16 fps. Feeding one the other's numbers yields a clip of
    the wrong length, not an error."""
    from mcp_servers.providers.local_media import _frames_for, MODELS

    spec = MODELS[model]
    assert spec["fps"] == fps
    assert _frames_for(spec["frame_rule"], fps, 5.0) == frames
    assert frames / fps == pytest.approx(duration)


def _use_local(monkeypatch, model: str, roles) -> None:
    """Point settings at a local video model without loading any weights.

    ``get_settings`` memoises its result, so patching the environment is not
    enough: the cache has to be cleared or the run keeps the provider it read
    at import time. Patching the provider name here rather than the env var
    keeps the test independent of whatever .env the developer has.
    """
    import agent.settings as settings_mod
    import mcp_servers.providers.local_media as local_media
    from mcp_servers.providers import reset_provider_cache

    base = settings_mod.get_settings()

    from mcp_servers.providers.mock_media import MockVideoProvider

    class Stub(local_media.LocalVideoProvider):
        """Declares real capabilities, renders nothing.

        This test is about routing, not inference: loading a 5B model to assert
        that the image stage was skipped would make the assertion cost two
        minutes and depend on weights. Rendering is delegated to the mock, which
        already handles a missing image_path.
        """

        def __init__(self, _name=""):
            self.model = model
            self.spec = dict(local_media.MODELS[model], roles=frozenset(roles))
            self.model_root = local_media._bench_root() / "models" / "hf"
            self._pipe = None
            self._mock = MockVideoProvider()

        def generate(self, request):
            return self._mock.generate(request)

    import dataclasses

    stub_settings = dataclasses.replace(base, media=dataclasses.replace(
        base.media, video_provider="local", video_model=model))
    monkeypatch.setattr(settings_mod, "_SETTINGS", stub_settings)
    # The factory resolves the class from its own module globals, not from
    # local_media, so patching local_media alone leaves the real provider in
    # place and the test silently loads a 5B model or fails on torch.
    import mcp_servers.providers as providers_mod

    monkeypatch.setattr(providers_mod, "LocalVideoProvider", Stub)
    reset_provider_cache()
    return local_media


def test_t2v_run_skips_the_whole_image_phase(project_root, fake_llm, monkeypatch):
    """With a text-to-video model the keyframe stage must not run at all.

    Rendering placeholders would both waste a stage and leave every
    anchor-derived metric scoring against a picture the video never consumed.
    """
    from mcp_servers.providers import reset_provider_cache

    created = project_store.create_project("电梯里的倒影慢半拍", "电影感竖屏短剧", 2, genre="悬疑")
    state = project_store.load_state(created["project_id"])
    state["status"] = "running"

    local_media = _use_local(monkeypatch, "cogvideox5b", set())
    try:
        final = run_agent(state)
    finally:
        reset_provider_cache()

    # No keyframe was ever requested.
    assert all(not shot.get("image_path") for shot in final["shots"])
    assert all(not shot.get("image_prompt") for shot in final["shots"])
    # And the join recorded is the honest one, not "authored".
    events = [json.loads(line) for line in
              (project_store.project_dir(created["project_id"]) / "events.jsonl").read_text().splitlines()]
    joins = [e for e in events if e.get("data", {}).get("join")]
    assert joins, "video stage should have run"
    assert all(e["data"]["join"] == "none" for e in joins)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_t2v_handles_single_and_multi_shot_lists(project_root, fake_llm, monkeypatch, count):
    """Boundary: one shot, a pair, and a longer list all route the same way."""
    from mcp_servers.providers import reset_provider_cache

    created = project_store.create_project("电梯里的倒影慢半拍", "电影感竖屏短剧", count, genre="悬疑")
    state = project_store.load_state(created["project_id"])
    state["status"] = "running"
    assert state["shot_count"] == count

    _use_local(monkeypatch, "wan21_1_3b", set())
    try:
        final = run_agent(state)
    finally:
        reset_provider_cache()

    assert len(final["shots"]) == count


def test_transitions_overlap_shortens_the_timeline(project_root, fake_llm, monkeypatch):
    """A storyboard transition is real time, not a label: an overlapping
    transition must shorten the encoded episode by its overlap."""
    import agent.settings as settings_mod

    settings_mod.reset_settings()
    state = project_store.load_state(project_store.create_project("转场", "电影感", 2)["project_id"])
    state["status"] = "running"
    final = run_agent(state)

    shots = final["shots"]
    meta = json.loads((project_root / state["project_id"] / "final" / "final_video.json").read_text(encoding="utf-8"))
    planned = sum(float(shot["duration_s"]) for shot in shots)
    # Each non-hard-cut boundary subtracts its overlap; there is no title card.
    overlaps = {"fade": 0.40, "fadewhite": 0.25, "fadeblack": 0.30}
    applied = [t["applied"] for t in meta["transitions"]]
    total_overlap = sum(overlaps.get(kind, 0.0) for kind in applied)
    expected = planned - total_overlap
    assert abs(meta["duration_s"] - expected) < 0.35
    facts = media_facts(project_root / state["project_id"] / final["final_video_path"])
    assert abs(facts["duration_s"] - meta["duration_s"]) < 0.35


def test_subtitles_are_rebased_onto_the_encoded_timeline(project_root, fake_llm):
    """Subtitle times are relative to each shot and must be shifted onto the
    encoded timeline, where transitions have already eaten their overlap."""
    state = project_store.load_state(project_store.create_project("字幕位移", "电影感", 2)["project_id"])
    state["status"] = "running"
    final = run_agent(state)
    body = (project_root / state["project_id"] / final["subtitle_path"]).read_text(encoding="utf-8")
    dialogue = [line for line in body.splitlines() if line.startswith("Dialogue:")]
    assert dialogue
    for line in dialogue:
        assert line.split(",")[1] != "0:00:00.00"


def test_bgm_is_mixed_and_preserves_length(project_root, fake_llm, monkeypatch):
    """A looping BGM bed must not truncate the episode: -shortest used to cut
    the video to the BGM's own length."""
    import agent.settings as settings_mod
    import mcp_servers.video_tools_server as tools

    state = project_store.load_state(project_store.create_project("配乐", "电影感", 1)["project_id"])
    state["status"] = "running"
    final = run_agent(state)

    root = project_root / state["project_id"]
    bgm = root / "audio" / "bgm.m4a"
    tools.run([tools.ffmpeg_bin(), "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(bgm)])

    clips = [
        {
            "video_path": final["shots"][0]["video_path"],
            "audio_path": final["shots"][0].get("audio_path", ""),
            "duration_s": final["shots"][0]["duration_s"],
            "transition": "硬切",
            "subtitle_lines": final["shots"][0].get("subtitle_lines") or [],
        }
    ]
    with_bgm = tools.compose_video(clips, state["project_id"], title="片名", bgm_path="audio/bgm.m4a")
    without = tools.compose_video(clips, state["project_id"], title="片名")

    mixed = media_facts(root / with_bgm["final_video_path"])
    plain = media_facts(root / without["final_video_path"])
    assert mixed["has_audio"] and plain["has_audio"]
    assert abs(mixed["duration_s"] - plain["duration_s"]) < 0.35


def test_vision_judge_covers_all_dimensions_in_one_call(tmp_path):
    """One keyframe upload must score every MLLM dimension: calling per
    dimension re-uploads the same image and triples the token cost."""
    from mcp_servers.videobench_eval import MLLM_DIMENSIONS, evaluate_clip

    clip = tmp_path / "clip.mp4"
    keyframe = tmp_path / "keyframe.png"
    subprocess_ffmpeg = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x334455:s=540x960:d=2:r=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clip),
    ]
    import subprocess as sp

    sp.run(subprocess_ffmpeg, capture_output=True, check=True)
    sp.run(["ffmpeg", "-y", "-i", str(clip), "-frames:v", "1", str(keyframe)], capture_output=True, check=True)

    calls: list[str] = []

    def judge(image_path, prompt):
        calls.append(str(image_path))
        return {"imaging_quality": 5, "aesthetic_quality": 4, "motion_effects": 3, "object_consistency": 2}, "脸部轻微变形"

    report = evaluate_clip(clip, shot={"shot_size": "特写"}, keyframe_path=keyframe, mllm_judge=judge)

    assert len(calls) == 1, f"expected one vision call, got {len(calls)}"
    assert calls[0] == str(keyframe), "judge must receive the full-resolution keyframe"
    for name in MLLM_DIMENSIONS:
        entry = report["dimensions"][name]
        assert entry.get("mllm_score_1to5") is not None, f"{name} missing judge score"
        assert entry["mllm_reason"] == "脸部轻微变形"
    # 2/5 maps to 0.25 and is max-merged with the measured estimate, so the
    # recorded score never falls below the judge's own floor.
    subject = report["dimensions"]["object_consistency"]
    assert subject["score"] >= 0.25 - 1e-9


def test_vision_judge_is_absent_without_paid_gate(monkeypatch):
    """No key or no ARK_ALLOW_PAID means no judge, and the measured path stays."""
    from mcp_servers.providers import get_vision_judge

    monkeypatch.delenv("ARK_ALLOW_PAID", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    import agent.settings as settings_mod

    settings_mod.reset_settings()
    assert get_vision_judge() is None


def test_last_frame_capability_is_model_gated():
    """Seeding last_frame into a pro-fast task is rejected by the API, so the
    capability check must be explicit rather than assumed."""
    from mcp_servers.providers.ark_media import supports_last_frame

    assert supports_last_frame("doubao-seedance-1-0-pro-fast-251015") is False
    assert supports_last_frame("doubao-seedance-1-0-pro-250528") is True
    assert supports_last_frame("doubao-seedance-2-0-260128") is True
    assert supports_last_frame("doubao-seedance-2-5-260628") is True


def test_extract_last_frame_reads_a_real_clip(tmp_path):
    """Chaining needs the tail frame of the previous clip, not its opening."""
    import subprocess as sp

    from mcp_servers.keyframe_memory import extract_last_frame

    clip = tmp_path / "clip.mp4"
    sp.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x334455:s=320x568:d=2:r=30",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clip)],
        capture_output=True, check=True,
    )
    out = extract_last_frame(clip, tmp_path / "last.png")
    assert out.exists() and out.stat().st_size > 1000


def test_sequence_judge_scores_the_ordered_keyframes(tmp_path):
    """Per-clip scoring cannot see cross-shot contradictions; the sequence
    judge must receive every keyframe in playback order in one call."""
    from mcp_servers.videobench_eval import SEQUENCE_DIMENSIONS, evaluate_sequence

    frames = []
    for index in range(3):
        frame = tmp_path / f"k{index}.png"
        frame.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(64))
        frames.append(frame)

    shots = [
        {"shot_size": "特写", "description": "他推开门", "dialogue": "有人吗？"},
        {"shot_size": "中景", "description": "街上没人", "dialogue": "怎么都一样"},
        {"shot_size": "全景", "description": "集市出现", "dialogue": "有人！"},
    ]

    seen: list[list[str]] = []

    def judge(paths, prompt):
        seen.append([str(p) for p in paths])
        assert "story_coherence" in prompt and "continuity" in prompt and "pacing" in prompt
        return {"story_coherence": 4, "continuity": 3, "pacing": 5, "reason": "闭环成立"}, "闭环成立"

    result = evaluate_sequence(frames, shots, mllm_judge=judge)

    assert len(seen) == 1, f"expected one sequence call, got {len(seen)}"
    assert seen[0] == [str(p) for p in frames], "keyframes must arrive in playback order"
    assert set(result["dimensions"]) == set(SEQUENCE_DIMENSIONS)
    assert result["dimensions"]["pacing"]["score_1to5"] == 5
    assert result["score"] is not None


def test_sequence_judge_absent_without_a_judge():
    """No vision judge must report unconfigured, not fabricate a score."""
    from mcp_servers.videobench_eval import evaluate_sequence

    result = evaluate_sequence([], [], mllm_judge=None)
    assert result["configured"] is False
    assert result["score"] is None


def test_tail_to_head_join_reuses_the_previous_clip(project):
    """The cut between two clips must land on an identical picture.

    Shot N+1 starts from the real last frame of clip N, so the boundary is not
    a jump. This is the branch pro-fast needs: it rejects the last_frame role
    outright, so the join has to be built on the head side.
    """
    from agent.nodes.video_nodes import _previous_tail_frame

    project_id, state = project
    final = run_agent(state)
    shots = final["shots"]
    assert len(shots) >= 2

    root = project_store.project_dir(project_id)
    joined = dict(shots[1])
    assert joined.get("video_path"), "mock clip 1 should exist"

    tail = _previous_tail_frame(final, joined)
    assert tail, "clip 2 must be joinable to clip 1"
    assert tail.endswith(f"{shots[0]['id']}_last.png")
    assert (root / tail).exists() and (root / tail).stat().st_size > 1000

    # The first clip has nothing before it to be joined to.
    assert _previous_tail_frame(final, shots[0]) == ""


def test_authored_join_targets_the_next_keyframe(project):
    """The other branch: an authored last frame drawn from the next keyframe.

    Both ends are then authored pictures, so drift cannot accumulate across the
    sequence. Available only on models that accept the last_frame role.
    """
    from agent.nodes.video_nodes import _next_keyframe

    project_id, state = project
    final = run_agent(state)
    shots = final["shots"]

    assert _next_keyframe(final, shots[0]) == shots[1]["image_path"]
    # The last shot has no successor to land on.
    assert _next_keyframe(final, shots[-1]) == ""


def test_image_phase_finishes_before_any_clip_is_rendered(project):
    """Every keyframe must exist before shot 1 renders.

    Interleaving the phases would mean keyframe[N+1] does not exist yet when
    shot N asks for it as a final frame, and the chain silently degrades to
    first-frame-only generation.
    """
    import json as _json

    project_id, state = project
    final = run_agent(state)
    root = project_store.project_dir(project_id)

    events = [
        _json.loads(line)
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Tool calls land as (type="Action", message="MCP.<tool>").
    kinds = [event.get("message") for event in events]
    first_video = kinds.index("MCP.generate_video")
    image_actions = [i for i, kind in enumerate(kinds) if kind == "MCP.generate_image"]
    assert image_actions, "image phase should have run"
    assert max(image_actions) < first_video, "all keyframes must precede the first clip"


def test_a_moderated_clip_is_skipped_instead_of_killing_the_episode(project, monkeypatch):
    """One rejected shot must not discard the shots that already rendered.

    Observed for real: a 1135 s run died whole when a single clip came back
    ``OutputVideoSensitiveContentDetected``. Every other clip in that episode
    had been rendered and paid for, and the pipeline had no branch for the
    error, so the exception unwound the entire run.
    """
    from agent.nodes import video_nodes
    from mcp_servers.providers.ark_media import ContentModerationError

    real = video_nodes.tools.generate_video

    def reject_one(*args, **kwargs):
        shot_id = args[3]
        if shot_id.endswith("002"):
            raise ContentModerationError("Ark video task cgt-x failed: OutputVideoSensitiveContentDetected")
        return real(*args, **kwargs)

    project_id, state = project
    monkeypatch.setattr(video_nodes.tools, "generate_video", reject_one)
    final = run_agent(state)

    # The episode still completes, one shot short.
    assert final.get("final_video_path")
    skipped = [s for s in final["shots"] if s.get("status") == "video_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["id"].endswith("002")
    assert skipped[0].get("video_path") == ""
    assert "SensitiveContentDetected" in skipped[0]["video_skip_reason"]

    # Every other clip is untouched: the rejection is isolated, not contagious.
    rendered = [s for s in final["shots"] if s.get("video_path")]
    assert len(rendered) == len(final["shots"]) - 1


def test_the_skip_route_bypasses_qa_and_keyframe_extraction(project):
    """A skipped shot has no pixels, so it must not reach evaluate_clip.

    The state handed to the video node is the one the planner produced; a fresh
    project has no shots yet, which is why the fixture runs the agent first.
    """
    from agent.nodes.video_nodes import clip_was_generated

    project_id, state = project
    final = run_agent(state)
    assert final["shots"]

    final["current_shot_index"] = 0
    assert clip_was_generated(final) == "generated"

    final["shots"] = [dict(final["shots"][0], video_path="", status="video_skipped")]
    assert clip_was_generated(final) == "skipped"


@pytest.mark.parametrize("model", ["cogvideox5b", "wan21_1_3b", "longlive_1_3b"])
def test_text_to_video_models_declare_no_frame_roles(model):
    """A T2V model consumes no image; declaring otherwise sends the run through
    the keyframe stage to render pictures the model silently ignores."""
    from mcp_servers.providers.local_media import LocalVideoProvider

    assert LocalVideoProvider(model).frame_roles == frozenset()


def test_the_i2v_model_declares_a_first_frame_role():
    from mcp_servers.providers.local_media import LocalVideoProvider

    assert LocalVideoProvider("cogvideox5b_i2v").frame_roles == frozenset({"first_frame"})


@pytest.mark.parametrize("provider,expected", [
    ("mock", True),
    ("ark", True),
])
def test_every_video_provider_declares_its_frame_roles(provider, expected):
    """The orchestration reads this attribute instead of matching a model id,
    so a provider without it would fail at the first generated clip."""
    from mcp_servers.providers import get_video_provider

    assert hasattr(get_video_provider(), "frame_roles") is expected


@pytest.mark.parametrize("model,fps,frames,duration", [
    ("cogvideox5b", 8, 41, 5.125),
    ("wan21_1_3b", 16, 81, 5.0625),
])
def test_local_model_frame_contract(model, fps, frames, duration):
    """Each family gets its own frame rule: CogVideoX needs 4k+1 at 8 fps, Wan
    4k+1 up to 81 at 16 fps. Feeding one the other's numbers yields a clip of
    the wrong length, not an error."""
    from mcp_servers.providers.local_media import _frames_for, MODELS

    spec = MODELS[model]
    assert spec["fps"] == fps
    assert _frames_for(spec["frame_rule"], fps, 5.0) == frames
    assert frames / fps == pytest.approx(duration)
