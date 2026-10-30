from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm_client import LLMReply, LLMService, set_llm  # noqa: E402

BIBLE = {
    "logline": "电梯里的倒影慢半拍",
    "genre": "悬疑",
    "tone": "冷调悬疑",
    "setting": "深夜写字楼电梯",
    "characters": [
        {
            "id": "c1",
            "name": "林夏",
            "role": "主角",
            "age_range": "22-26",
            "persona": "冷静但敏感",
            "appearance": "short black bob, pale skin, thin scar on left eyebrow, grey wool coat",
            "wardrobe": "grey wool coat, black turtleneck",
            "voice": "低声、克制",
        }
    ],
    "outline": "起-冲突-反转",
    "hook": "镜中的自己慢了半拍",
    "ending": "镜中人笑了",
}


class FakeLLMClient:
    """Deterministic stand-in for the DeepSeek endpoint."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, messages, *, json_mode=False, model=None) -> LLMReply:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        self.calls.append(system[:20])

        if "短剧策划" in system:
            payload = BIBLE
        elif "分镜师" in system:
            count = 2
            for token in user.split():
                if token.startswith("镜头数量"):
                    pass
            marker = user.split("镜头数量：")[1].split("\n")[0]
            count = int(marker)
            payload = {
                "shots": [
                    {
                        "index": i,
                        "scene": "电梯内",
                        "shot_size": "特写" if i % 2 else "近景",
                        "camera": "固定机位",
                        "action": f"第{i}镜的动作",
                        "dialogue": "谁？" if i == 2 else "",
                        "speaker": "c1",
                        "emotion": "紧张",
                        "sfx": "电梯嗡鸣",
                        "duration_s": 3,
                        "transition": "硬切",
                        "characters": ["c1"],
                    }
                    for i in range(1, count + 1)
                ]
            }
        elif "运动提示词修复专家" in system:
            payload = {"motion_prompt": "Stable face, no morphing, gentle push in.", "change_log": "修复形变"}
        elif "提示词修复专家" in system:
            payload = {"image_prompt": "Vertical 9:16, fixed composition, " + BIBLE["characters"][0]["appearance"] + ", " + "detail " * 20, "change_log": "加强构图"}
        elif "关键帧提示词工程师" in system:
            payload = {
                "image_prompt": "Vertical 9:16 portrait, a woman in a dim elevator, " + BIBLE["characters"][0]["appearance"] + ", cinematic lighting, 35mm, film grain, " + "detail " * 20,
                "negative_prompt": "text, watermark",
                "rationale": "沿用外观锚点",
            }
        elif "运动提示词工程师" in system:
            payload = {"motion_prompt": "Slow push in, subtle facial motion, no morphing.", "rationale": "克制推进"}
        else:
            payload = {"ok": True}

        return LLMReply(text=json.dumps(payload, ensure_ascii=False), model="fake", prompt_tokens=10, completion_tokens=20, latency_s=0.0)


@pytest.fixture
def fake_llm():
    fake = FakeLLMClient()
    set_llm(LLMService(client=fake))
    yield fake
    set_llm(None)


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    import agent.settings
    import backend.services.project_store as ps
    import mcp_servers.video_tools_server as tools

    root = tmp_path / "workspace" / "projects"
    root.mkdir(parents=True)
    monkeypatch.setattr(agent.settings, "PROJECTS_ROOT", root)
    monkeypatch.setattr(ps, "PROJECTS_ROOT", root)
    monkeypatch.setattr(tools, "PROJECTS_ROOT", root)
    return root
