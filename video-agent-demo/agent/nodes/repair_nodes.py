from __future__ import annotations

from agent.state import AgentState, add_event, current_shot, update_current_shot


def rewrite_image_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    retry_count = shot.get("image_retry_count", 0) + 1
    prompt = (
        f"{shot['image_prompt']} 修正反馈：{shot.get('feedback', '')}。"
        f"第 {retry_count} 次重试：加强主体、构图、光影和风格一致性。"
    )
    add_event(state, "Thought", f"根据图像评估反馈修正 {shot['id']} 的图片 prompt。")
    add_event(state, "Decision", "重新调用图片生成工具。", {"retry_count": retry_count})
    return update_current_shot(state, {"image_prompt": prompt, "image_retry_count": retry_count})


def rewrite_video_prompt(state: AgentState) -> AgentState:
    shot = current_shot(state)
    retry_count = shot.get("video_retry_count", 0) + 1
    prompt = (
        f"{shot['motion_prompt']} 修正反馈：{shot.get('feedback', '')}。"
        f"第 {retry_count} 次重试：强化镜头运动、节奏、转场和情绪变化。"
    )
    add_event(state, "Thought", f"根据视频评估反馈修正 {shot['id']} 的 motion prompt。")
    add_event(state, "Decision", "重新调用视频生成工具。", {"retry_count": retry_count})
    return update_current_shot(state, {"motion_prompt": prompt, "video_retry_count": retry_count})
