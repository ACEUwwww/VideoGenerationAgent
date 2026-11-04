from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.services import project_store


@pytest.fixture
def project(project_root):
    return project_store.create_project("恢复测试", "电影感", 1)["project_id"]


def test_live_run_blocks_a_second_claim(project):
    assert project_store.claim_run(project) is True
    assert project_store.claim_run(project) is False


def test_stale_run_can_be_reclaimed(project):
    """A worker killed mid-run must not pin the project forever."""
    project_store.claim_run(project)
    state = project_store.load_state(project)
    state["run_heartbeat_at"] = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    project_store.save_state(project, state)

    assert project_store.is_run_stale(state) is True
    assert project_store.claim_run(project) is True


def test_recent_heartbeat_is_not_stale(project):
    project_store.claim_run(project)
    state = project_store.load_state(project)
    assert project_store.is_run_stale(state) is False


def test_legacy_running_state_without_heartbeat_is_reclaimable(project):
    """States written before heartbeats existed have no marker at all."""
    state = project_store.load_state(project)
    state["status"] = "running"
    project_store.save_state(project, state)

    assert project_store.is_run_stale(state) is True
    assert project_store.claim_run(project) is True


def test_heartbeat_advances_the_marker(project):
    project_store.claim_run(project)
    before = project_store.load_state(project)["run_heartbeat_at"]
    project_store.heartbeat(project)
    after = project_store.load_state(project)["run_heartbeat_at"]
    assert after >= before


def test_heartbeat_ignores_finished_projects(project):
    """A late beat must not resurrect a completed project."""
    project_store.claim_run(project)
    state = project_store.load_state(project)
    state["status"] = "completed"
    project_store.save_state(project, state)

    project_store.heartbeat(project)
    assert project_store.load_state(project)["status"] == "completed"


def test_reclaim_clears_the_previous_error(project):
    state = project_store.load_state(project)
    state.update({"status": "failed", "error": "boom"})
    project_store.save_state(project, state)

    assert project_store.claim_run(project) is True
    assert "error" not in project_store.load_state(project)
