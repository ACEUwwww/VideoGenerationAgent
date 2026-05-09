# Lightweight Video Generation Agent Demo

一个用于课程展示的轻量级 Video Generation Agent Demo。它用 LangGraph 管理 state-driven pipeline，用 MCP-like 工具抽象模拟图片生成、视频生成、评估和合成，并通过 React 前端展示 Thought / Action / Observation / Decision。

## 技术栈

- 前端：Vite + React + TypeScript
- 后端：FastAPI
- Agent：LangGraph `StateGraph`
- 工具协议：MCP-like Python 工具抽象（后续可替换为 MCP Python SDK 服务）
- 存储：本地 JSON + 文件夹
- 生成能力：mock 图片 / mock 视频 / mock 评估 / mock 合成

## 启动

后端：

```bash
cd video-agent-demo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

前端：

```bash
cd video-agent-demo/frontend
npm install
npm run dev
```

打开 Vite 输出的本地地址，输入故事需求、风格和镜头数量，点击 `Start Agent`。

## Pipeline

1. `analyze_user_goal`：分析用户目标
2. `plan_story`：生成故事大纲
3. `plan_shots`：生成分镜列表
4. 对每个 shot 循环：
   - `generate_image_prompt`
   - `generate_image`
   - `evaluate_image`
   - 不通过则 `rewrite_image_prompt` 并重试
   - `generate_video`
   - `evaluate_video`
   - 不通过则 `rewrite_video_prompt` 并重试
5. `compose_video`
6. `finish`

每个节点都会向 `workspace/projects/<project_id>/events.jsonl` 追加 Agent 事件，前端轮询 `/api/projects/{project_id}/events` 展示日志。

## 本地存储

```text
workspace/
  projects/
    project_xxx/
      project.json
      state.json
      events.jsonl
      shots/
        shot_001.json
      images/
        shot_001.png
      videos/
        shot_001.mp4
      final/
        final_video.mp4
```

当前版本生成的是 mock 文件，方便完整跑通流程。后续替换真实模型 API 时，只需要改 `mcp_servers/video_tools_server.py`，Agent 节点不直接依赖具体模型供应商。
