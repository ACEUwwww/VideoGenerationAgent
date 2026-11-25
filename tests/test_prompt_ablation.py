"""Unit tests for the prompt-ablation arm construction.

The ablation's conclusions rest on one claim: every arm is assembled from the
same planner output and differs only in the prompt string. That claim is only
true if the split and reassembly do what the docstring says, so it is tested
here without a GPU — no model is loaded and no clip is rendered.

The last test pins a known defect rather than a desired property. ``planned_trunc``
was built as a length control and is not one, because truncation removes the
appearance anchor. That arm is still reported as a third anchor replication, and
the test exists so nobody later "fixes" the truncation and silently changes what
the published numbers mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "study"))

import prompt_ablation as pa  # noqa: E402

PROMPT = (
    "A medium close-up inside a convenience store at night. "
    "A young man stands behind the counter. "
    "He has a thin face, short black hair and a slim build, "
    "wearing a dark blue uniform vest with a red name badge. "
    "The camera tilts up to his face. Rain streaks the glass behind him."
)

BRIEF_EN = "Late at night in a convenience store, the clerk hears his name in the rain."


def test_split_separates_appearance_from_scene():
    scene, anchor = pa.split_prompt(PROMPT)
    anchor_text = " ".join(anchor).lower()
    scene_text = " ".join(scene).lower()

    assert "black hair" in anchor_text
    assert "uniform vest" in anchor_text
    assert "convenience store" not in anchor_text

    assert "convenience store" in scene_text
    assert "rain streaks" in scene_text


def test_camera_clause_is_scene_even_when_it_names_a_body_part():
    """'tilts up to his face' is direction, not appearance."""
    scene, anchor = pa.split_prompt(PROMPT)
    joined_anchor = " ".join(anchor).lower()
    joined_scene = " ".join(scene).lower()
    assert "tilts up to his face" not in joined_anchor
    assert "tilts up to his face" in joined_scene


def test_planned_rebuild_keeps_every_clause():
    scene, anchor = pa.split_prompt(PROMPT)
    planned = pa.join(scene + anchor)
    for clause in scene + anchor:
        head = pa._tidy([clause])[0][:24].lower()
        assert head in planned.lower(), clause


def test_scene_arm_carries_no_appearance_vocabulary():
    scene, _ = pa.split_prompt(PROMPT)
    scene_text = pa.join(scene).lower()
    for term in ("hair", "uniform vest", "name badge", "slim build"):
        assert term not in scene_text, term


def test_padding_appends_only_production_vocabulary():
    scene, anchor = pa.split_prompt(PROMPT)
    planned = pa.join(scene + anchor)
    padded = pa.pad_to_ratio(planned, 2.0)

    # The padded arm must contain the planned prompt unchanged as a prefix.
    assert padded.startswith(planned)
    assert len(padded) > len(planned) * 1.5

    appended = padded[len(planned):].lower()
    for term in ("hair", "vest", "badge", "face", "build", "wearing"):
        assert term not in appended, term


def test_arms_are_built_from_one_prompt_and_differ_only_in_text():
    built = pa.build_arms(BRIEF_EN, PROMPT)["arms"]
    assert set(built) == set(pa.ARMS)
    for name, entry in built.items():
        assert entry["chars"] == len(entry["text"]), name

    # Sanity on length ordering: anchor arms longer than the bare brief, and the
    # padded arm roughly twice the planned one.
    assert built["brief_en_anchor"]["chars"] > built["brief_en"]["chars"]
    assert built["planned_padded"]["chars"] > built["planned"]["chars"]
    assert built["planned_trunc"]["chars"] < built["planned"]["chars"]


def test_truncation_is_known_to_remove_the_anchor():
    """Pins the published defect: planned_trunc is not a length control.

    If a future change makes truncation keep the anchor, this test fails and
    forces a re-read of prompt_ablation.md, where `planned − planned_trunc` is
    documented as an anchor contrast rather than a length contrast.
    """
    scene, anchor = pa.split_prompt(PROMPT)
    planned = pa.join(scene + anchor)
    truncated = pa.truncate_to(planned, pa.TRUNC_FRACTION).lower()

    survivors = [a for a in anchor if pa._tidy([a])[0][:24].lower() in truncated]
    assert survivors == []
