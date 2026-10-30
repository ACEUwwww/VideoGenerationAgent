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
