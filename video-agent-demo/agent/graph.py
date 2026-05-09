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
    advance_shot,
    call_evaluate_video,
    call_generate_video,
    generate_video_prompt,
    has_more_shots,
    should_retry_video,
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
    graph.add_node("generate_video_prompt", generate_video_prompt)
    graph.add_node("generate_video", call_generate_video)
    graph.add_node("evaluate_video", call_evaluate_video)
    graph.add_node("rewrite_video_prompt", rewrite_video_prompt)
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
        {"retry": "rewrite_image_prompt", "video": "generate_video_prompt"},
    )
    graph.add_edge("rewrite_image_prompt", "generate_image")
    graph.add_edge("generate_video_prompt", "generate_video")
    graph.add_edge("generate_video", "evaluate_video")
    graph.add_conditional_edges(
        "evaluate_video",
        should_retry_video,
        {"retry": "rewrite_video_prompt", "next": "advance_shot"},
    )
    graph.add_edge("rewrite_video_prompt", "generate_video")
    graph.add_conditional_edges(
        "advance_shot",
        has_more_shots,
        {"more": "generate_image_prompt", "compose": "compose_video"},
    )
    graph.add_edge("compose_video", "finish")
    graph.add_edge("finish", END)

    return graph.compile()


def run_agent(initial_state: AgentState) -> AgentState:
    app = build_graph()
    return app.invoke(initial_state, config={"recursion_limit": 200})
