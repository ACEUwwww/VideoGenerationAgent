"""Planning nodes: goal analysis, story bible, and the shot list.

Character consistency for a short drama lives here: the bible produces one
appearance anchor per character, and every downstream image prompt reuses that
anchor verbatim instead of re-inventing the character per shot.
"""

from __future__ import annotations

import json

from agent.llm_client import get_llm
from agent.prompts import BIBLE_SYSTEM, SHOTS_SYSTEM
from agent.state import AgentState, add_event, persist, track_usage
from backend.services import project_store

# Seedance 1.0 pro / 2.x render up to 12s in one call. The old 6s cap made a
# long shot impossible no matter what the prompt asked for: 8 shots over a
# 32s target can only average 4s, so the shot list came back uniformly 4.0s.
MAX_SHOT_SECONDS = 12.0
MIN_SHOT_SECONDS = 3.0


def analyze_user_goal(state: AgentState) -> AgentState:
    state["status"] = "running"
    add_event(state, "Thought", "解析用户创意，确定题材、风格与镜头数量。")
    add_event(
        state,
        "Decision",
        f"按竖屏短剧生产：{state['shot_count']} 个镜头，风格为 {state['style']}。",
        {"shot_count": state["shot_count"], "style": state["style"]},
    )
    return persist(state)


def plan_story(state: AgentState) -> AgentState:
    llm = get_llm()
    add_event(state, "Action", "LLM.plan_bible", {"model": "deepseek"})
    data, reply = llm.generate_json(
        BIBLE_SYSTEM,
        _bible_request(state),
        required_keys=("logline", "genre", "characters", "outline"),
    )
    track_usage(state, "plan_bible", reply)

    characters = data.get("characters") or []
    beats = data.get("beats") or []
    # Beats and shots are deliberately not 1:1: one beat can span several shots
    # and one long take can carry two beats. Only the ordering is binding.
    if len(beats) > state["shot_count"]:
        add_event(
            state,
            "Observation",
            f"设定集给了 {len(beats)} 个节拍，多于 {state['shot_count']} 镜，分镜时会合并低信息量的节拍。",
        )
    for position, character in enumerate(characters, start=1):
        character.setdefault("id", f"c{position}")
    state["logline"] = data.get("logline", "")
    state["genre"] = data.get("genre", "")
    state["tone"] = data.get("tone", "")
    state["setting"] = data.get("setting", "")
    state["outline"] = data.get("outline", "")
    state["hook"] = data.get("hook", "")
    state["ending"] = data.get("ending", "")
    state["beats"] = data.get("beats") or []
    state["characters"] = characters
    project_store.save_bible(state["project_id"], {"characters": characters, **data})

    add_event(
        state,
        "Observation",
        f"设定集完成：{len(characters)} 个角色，题材 {state['genre']}。",
        {"logline": state["logline"], "characters": [c.get("name") for c in characters]},
    )
    return persist(state)


def plan_shots(state: AgentState) -> AgentState:
    llm = get_llm()
    add_event(state, "Action", "LLM.plan_shots", {"shot_count": state["shot_count"]})
    data, reply = llm.generate_json(
        SHOTS_SYSTEM,
        _shots_request(state),
        required_keys=("shots",),
    )
    track_usage(state, "plan_shots", reply)

    raw_shots = data["shots"]
    if len(raw_shots) != state["shot_count"]:
        add_event(
            state,
            "Observation",
            f"模型给了 {len(raw_shots)} 镜，要求 {state['shot_count']} 镜，补一次分镜。",
        )
        data, reply = llm.generate_json(
            SHOTS_SYSTEM,
            _shots_request(state) + f"\n上一次只给了 {len(raw_shots)} 个镜头，这次必须恰好 {state['shot_count']} 个。",
            required_keys=("shots",),
        )
        track_usage(state, "plan_shots_retry", reply)
        raw_shots = data["shots"]
        if len(raw_shots) != state["shot_count"]:
            raise ValueError(
                f"shot list mismatch: model returned {len(raw_shots)} shots, "
                f"requested {state['shot_count']}"
            )

    shots: list[dict] = []
    for position, raw in enumerate(raw_shots, start=1):
        shot_id = f"shot_{position:03d}"
        duration = float(raw.get("duration_s", 4) or 4)
        shot = {
            "id": shot_id,
            "index": position - 1,
            "scene": raw.get("scene", ""),
            "shot_size": raw.get("shot_size", ""),
            "camera": raw.get("camera", ""),
            "description": raw.get("action", ""),
            "dialogue": raw.get("dialogue", "") or "",
            "speaker": raw.get("speaker", "") or "",
            "emotion": raw.get("emotion", ""),
            "sfx": raw.get("sfx", ""),
            "transition": raw.get("transition", "硬切"),
            "duration_s": min(max(duration, MIN_SHOT_SECONDS), MAX_SHOT_SECONDS),
            "characters": raw.get("characters", []),
            "status": "planned",
            "image_prompt": "",
            "negative_prompt": "",
            "motion_prompt": "",
            "image_path": "",
            "video_path": "",
            "audio_path": "",
            "image_score": 0.0,
            "video_score": 0.0,
            "feedback": "",
            "image_retry_count": 0,
            "video_retry_count": 0,
        }
        shots.append(shot)
        project_store.save_shot(state["project_id"], shot)

    state["shots"] = shots
    state["current_shot_index"] = 0
    add_event(
        state,
        "Observation",
        f"分镜表生成：{len(shots)} 镜，总时长 {sum(s['duration_s'] for s in shots):.1f}s。",
        {"shots": shots},
    )
    return persist(state)


def _bible_request(state: AgentState) -> str:
    parts = [f"用户创意：{state['user_prompt']}", f"风格要求：{state['style']}"]
    if state.get("genre"):
        parts.append(f"指定题材：{state['genre']}")
    if state.get("script"):
        parts.append(f"用户提供的剧本/素材：\n{state['script']}")
    parts.append(f"镜头数量：{state['shot_count']}（请让梗概适配这个长度）")
    target = state.get("target_seconds") or 0
    if target:
        parts.append(
            f"目标总时长：约 {target} 秒，共 {state['shot_count']} 个镜头，"
            f"平均每镜约 {target / max(state['shot_count'], 1):.1f} 秒。"
            f"分镜时会写成 3-5 个节拍，其中一个节拍会被拉成一个 8 秒以上的长镜头，"
            f"所以 beats 应当按情绪转折来分，不要为了凑镜头数切碎。"
        )
    return "\n".join(parts)


def _shots_request(state: AgentState) -> str:
    bible = {
        "logline": state.get("logline", ""),
        "genre": state.get("genre", ""),
        "tone": state.get("tone", ""),
        "setting": state.get("setting", ""),
        "outline": state.get("outline", ""),
        "hook": state.get("hook", ""),
        "ending": state.get("ending", ""),
        "characters": state.get("characters", []),
    }
    parts = [f"镜头数量：{state['shot_count']}", f"风格：{state['style']}"]
    target = state.get("target_seconds") or 0
    if target:
        parts.append(
            f"目标总时长：约 {target} 秒（所有镜头的 duration_s 相加必须落在 "
            f"{max(target - 2, 1)}–{target + 2} 秒之间）"
        )
    beats = state.get("beats") or []
    if beats:
        parts.append(
            "故事节拍（必须全部覆盖，顺序不得调换；一个节拍可以展开成多个镜头，"
            "也可以用一个长镜头完整呈现）：\n"
            + "\n".join(f"  {i}. {b}" for i, b in enumerate(beats, start=1))
        )
    parts.append(f"设定集：\n{json.dumps(bible, ensure_ascii=False, indent=2)}")
    return "\n".join(parts)
