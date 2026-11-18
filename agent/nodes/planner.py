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

# Duration window that every Seedance generation accepts: 1.0 pro / pro-fast
# render 2-12s, 2.0 and 2.5 start at 4s. 4-12 is the intersection, and asking
# for 3s on 2.0 is a hard API error, not a clamp. The old 6s cap made a long
# shot impossible no matter what the prompt asked for: 8 shots over a 32s
# target can only average 4s, so the shot list came back uniformly 4.0s.
MAX_SHOT_SECONDS = 12.0
MIN_SHOT_SECONDS = 4.0


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


LONG_SHOT_SECONDS = 8.0
TOTAL_TOLERANCE_S = 2.0


def fit_durations(durations: list[float], target_seconds: int) -> list[float]:
    """Re-time the shot list into the target window without losing its shape.

    A corrective rewrite is a request, not a guarantee: a real run came back at
    38s against a 30s target and the second pass came back at 38s again. Taking
    that as final ships a film 8 seconds over, so the total is enforced here.

    The constraints do not always coexist. "No three consecutive equal shots"
    is unsatisfiable once three shots are pinned at the floor (6 shots over 30s
    with three already at 4s has nowhere else to put them: 4,4,4 is forced).
    So the order below is deliberate — the total is hard, the long take is hard,
    and avoiding a run of three is best effort, reported rather than faked.

    Everything is whole seconds because that is what the video API accepts.
    """
    if not target_seconds or not durations:
        return durations
    count = len(durations)
    total = sum(durations)
    if abs(total - target_seconds) <= TOTAL_TOLERANCE_S:
        return durations

    low, high = int(MIN_SHOT_SECONDS), int(MAX_SHOT_SECONDS)
    ideal = [value * target_seconds / total for value in durations]
    scaled = [min(max(round(value), low), high) for value in ideal]

    # Reserve the long take before distributing the remainder: it is the one
    # property the storyboard was explicitly asked to guarantee, and plain
    # proportional scaling erodes it (a 10s shot becomes 6.7s in a 45s cut).
    keep = max(range(count), key=lambda i: durations[i])
    if durations[keep] >= LONG_SHOT_SECONDS:
        scaled[keep] = int(LONG_SHOT_SECONDS)
    else:
        keep = -1

    # Move one second at a time towards the target, always on the shot that sits
    # furthest from its proportional share. Temporary runs of three are allowed
    # here; they are repaired below, and refusing them here deadlocks the cases
    # where the only way to reach the total is through one.
    for _ in range(count * high * 2):
        current = sum(scaled)
        if current == target_seconds:
            break
        step = -1 if current > target_seconds else 1
        order = sorted(
            range(count),
            key=lambda i: (scaled[i] - ideal[i]) * step,
            reverse=True,
        )
        for index in order:
            candidate = scaled[index] + step
            if not low <= candidate <= high:
                continue
            if index == keep and candidate < int(LONG_SHOT_SECONDS):
                continue
            scaled[index] = candidate
            break
        else:
            break

    # Break runs of three by trading a second between neighbours. This keeps the
    # total fixed, and is skipped when the long take would be the casualty.
    for _ in range(count * 3):
        index = _triple_index(scaled)
        if index is None:
            break
        run = set(range(index, index + 3))
        moved = False

        def can_drop(i: int) -> bool:
            if scaled[i] - 1 < low:
                return False
            return not (i == keep and scaled[i] - 1 < int(LONG_SHOT_SECONDS))

        def can_raise(i: int) -> bool:
            return scaled[i] + 1 <= high

        # Trade inside the run first: move one second from its head to its tail
        # (or the other way round), which splits 4,4,4 into 5,4,3 / 3,4,5.
        for drop, raise_ in ((index, index + 2), (index + 1, index + 2), (index, index + 1), (index + 2, index)):
            if can_drop(drop) and can_raise(raise_):
                scaled[drop] -= 1
                scaled[raise_] += 1
                moved = True
                break

        # When all three are already at the floor the run has to be broken from
        # outside: raise a member and take the second from a shot that has slack.
        if not moved:
            for raise_ in sorted(run):
                if not can_raise(raise_):
                    continue
                donors = [
                    i for i in range(count)
                    if i not in run and can_drop(i)
                ]
                if not donors:
                    continue
                scaled[donors[0]] -= 1
                scaled[raise_] += 1
                moved = True
                break

        if not moved:
            break

    return [float(value) for value in scaled]


def _triple_index(values: list[int]) -> int | None:
    for index in range(len(values) - 2):
        if values[index] == values[index + 1] == values[index + 2]:
            return index
    return None


def rhythm_problems(durations: list[float], target_seconds: int) -> list[str]:
    """The rhythm rules the prompt states, as checks the model cannot argue with.

    Writing "至少一个 8 秒长镜、不得三连同长" into the system prompt was not
    enough: a real run came back 5/9/6/4/4/4, three identical tail shots. A
    stated requirement that nothing verifies is just a suggestion.
    """
    problems: list[str] = []
    if durations and max(durations) < LONG_SHOT_SECONDS:
        problems.append(
            f"没有 8 秒以上的长镜头（最长 {max(durations):.0f}s），"
            f"请把承载情绪转折的那一镜拉长到 8-12 秒"
        )
    streak = 1
    for index in range(1, len(durations)):
        streak = streak + 1 if durations[index] == durations[index - 1] else 1
        if streak >= 3:
            problems.append(
                f"第 {index - 1}-{index + 1} 镜连续三镜都是 {durations[index]:.0f} 秒，"
                f"请打散节奏"
            )
            break
    if target_seconds:
        total = sum(durations)
        if abs(total - target_seconds) > TOTAL_TOLERANCE_S:
            problems.append(
                f"总时长 {total:.1f}s 超出 {target_seconds}±{TOTAL_TOLERANCE_S:.0f}s"
            )
    return problems


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

    # One corrective pass: hand the model its own numbers back with the rule it
    # broke, which works far better than repeating the rule in isolation.
    problems = rhythm_problems(_durations(raw_shots), state.get("target_seconds") or 0)
    if problems:
        add_event(state, "Observation", "分镜节奏不合格，按缺失的指标重排一次。", {"problems": problems})
        data, reply = llm.generate_json(
            SHOTS_SYSTEM,
            _shots_request(state)
            + "\n\n上一版分镜节奏不合格，必须修正：\n"
            + "\n".join(f"  - {item}" for item in problems),
            required_keys=("shots",),
        )
        track_usage(state, "plan_shots_rhythm", reply)
        if len(data["shots"]) == state["shot_count"]:
            raw_shots = data["shots"]
            remaining = rhythm_problems(_durations(raw_shots), state.get("target_seconds") or 0)
            add_event(
                state,
                "Observation",
                "节奏复检通过。" if not remaining else "节奏复检仍有缺口，按时长硬压缩。",
                {"remaining": remaining},
            )

    # Whatever the rewrite returned, the total must land in the window. Ask
    # first, then enforce: a stated constraint that nothing enforces is how the
    # previous run shipped 38.5s against a 30s request.
    target = state.get("target_seconds") or 0
    before = _durations(raw_shots)
    after = fit_durations(before, target)
    if after != before:
        add_event(
            state,
            "Decision",
            f"总时长 {sum(before):.0f}s 超出目标 {target}±{TOTAL_TOLERANCE_S:.0f}s，"
            f"按比例压缩到 {sum(after):.0f}s（节奏比例保留）。",
            {"before": before, "after": after},
        )
        for raw, duration in zip(raw_shots, after):
            raw["duration_s"] = duration

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


def _durations(raw_shots: list[dict]) -> list[float]:
    return [min(max(float(shot.get("duration_s", 4) or 4), MIN_SHOT_SECONDS), MAX_SHOT_SECONDS) for shot in raw_shots]


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
