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
    # Title card adds, each non-hard-cut boundary subtracts its overlap.
    overlaps = {"fade": 0.40, "fadewhite": 0.25, "fadeblack": 0.30}
    applied = [t["applied"] for t in meta["transitions"]]
    # The title card cross-fades into the first shot, which is one more overlap.
    if meta.get("title_card_transition"):
        applied.append(meta["title_card_transition"])
    total_overlap = sum(overlaps.get(kind, 0.0) for kind in applied)
    card = 1.2 if meta["title_card"] else 0.0
    expected = planned + card - total_overlap
    assert abs(meta["duration_s"] - expected) < 0.35
    facts = media_facts(project_root / state["project_id"] / final["final_video_path"])
    assert abs(facts["duration_s"] - meta["duration_s"]) < 0.35


def test_subtitles_shift_with_the_title_card(project_root, fake_llm):
    """Subtitle times are re-based onto the encoded timeline; a prepended card
    must not leave dialogue sitting before its shot."""
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
