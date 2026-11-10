from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.nodes.composer import compose_video, finish
from agent.nodes.image_nodes import (
    call_evaluate_image,
    call_generate_image,
    generate_image_prompt,
    should_retry_image,
)
from agent.nodes.planner import analyze_user_goal, plan_shots, plan_story
from agent.nodes.repair_nodes import rewrite_image_prompt, rewrite_video_prompt
from agent.nodes.video_nodes import (
    advance_keyframe,
    advance_shot,
    call_evaluate_video,
    call_generate_video,
    evaluate_clip_multidim,
    generate_video_prompt,
    has_more_keyframes,
    has_more_shots,
    should_retry_video,
    start_video_phase,
    synthesize_dialogue,
)
from agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_user_goal", analyze_user_goal)
    graph.add_node("plan_story", plan_story)
    graph.add_node("plan_shots", plan_shots)
    graph.add_node("generate_image_prompt", generate_image_prompt)
    graph.add_node("generate_image", call_generate_image)
    graph.add_node("evaluate_image", call_evaluate_image)
    graph.add_node("rewrite_image_prompt", rewrite_image_prompt)
    graph.add_node("advance_keyframe", advance_keyframe)
    graph.add_node("start_video_phase", start_video_phase)
    graph.add_node("generate_video_prompt", generate_video_prompt)
    graph.add_node("generate_video", call_generate_video)
    graph.add_node("evaluate_video", call_evaluate_video)
    graph.add_node("rewrite_video_prompt", rewrite_video_prompt)
    graph.add_node("evaluate_clip", evaluate_clip_multidim)
    graph.add_node("synthesize_dialogue", synthesize_dialogue)
    graph.add_node("advance_shot", advance_shot)
    graph.add_node("compose_video", compose_video)
    graph.add_node("finish", finish)

    graph.set_entry_point("analyze_user_goal")
    graph.add_edge("analyze_user_goal", "plan_story")
    graph.add_edge("plan_story", "plan_shots")
    graph.add_edge("plan_shots", "generate_image_prompt")
    graph.add_edge("generate_image_prompt", "generate_image")
    graph.add_edge("generate_image", "evaluate_image")
    graph.add_conditional_edges(
        "evaluate_image",
        should_retry_image,
        {"retry": "rewrite_image_prompt", "next": "advance_keyframe"},
    )
    graph.add_edge("rewrite_image_prompt", "generate_image")
    # Image phase first, video phase second: shot N needs keyframe[N+1] as its
    # final frame, which only exists after every keyframe has been rendered.
    graph.add_conditional_edges(
        "advance_keyframe",
        has_more_keyframes,
        {"more": "generate_image_prompt", "all_done": "start_video_phase"},
    )
    graph.add_edge("start_video_phase", "generate_video_prompt")
    graph.add_edge("generate_video_prompt", "generate_video")
    graph.add_edge("generate_video", "evaluate_video")
    graph.add_conditional_edges(
        "evaluate_video",
        should_retry_video,
        {"retry": "rewrite_video_prompt", "next": "evaluate_clip"},
    )
    graph.add_edge("rewrite_video_prompt", "generate_video")
    graph.add_edge("evaluate_clip", "synthesize_dialogue")
    graph.add_edge("synthesize_dialogue", "advance_shot")
    # The video phase loops back into the video prompt, not the image prompt:
    # every keyframe already exists by the time shot 1 is rendered.
    graph.add_conditional_edges(
        "advance_shot",
        has_more_shots,
        {"more": "generate_video_prompt", "compose": "compose_video"},
    )
    graph.add_edge("compose_video", "finish")
    graph.add_edge("finish", END)

    return graph.compile()


def run_agent(initial_state: AgentState) -> AgentState:
    app = build_graph()
    return app.invoke(initial_state, config={"recursion_limit": 200})
