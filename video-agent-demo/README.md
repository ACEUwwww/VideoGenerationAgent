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

## 当前完成状态

这个项目目前已经完成了一个 **最小可运行的端到端 demo**：

- 前端可以创建项目、启动 Agent、展示 Storyboard、媒体路径、最终结果和 Agent 日志。
- 后端提供基础 FastAPI 接口，不使用数据库，项目状态保存到本地 JSON 和文件夹。
- Agent 使用 LangGraph `StateGraph` 串起节点、边和条件边。
- Pipeline 已经支持多镜头循环、图片评估重试、视频评估重试和最终合成。
- MCP-like 工具层已经定义好图片生成、图片评估、视频生成、视频评估、视频合成这几个标准接口。

但是，当前版本的重点是跑通流程，很多 AI 能力仍然是 mock。

## 当前仍是 Mock 的部分

### 1. Story / Storyboard 规划

对应文件：

```text
agent/nodes/planner.py
```

当前状态：

- `plan_story()` 使用字符串模板生成故事大纲。
- `plan_shots()` 使用简单循环生成分镜描述。
- 没有调用大模型。

后续目标：

- 使用 LLM 分析用户需求。
- 生成结构化 story outline。
- 生成结构化 shot list。
- 每个 shot 包含画面描述、角色、场景、动作、镜头语言、时长、转场等字段。

### 2. Image Prompt 生成

对应文件：

```text
agent/nodes/image_nodes.py
```

当前状态：

- `generate_image_prompt()` 使用模板拼接当前 shot 描述和 style。
- 没有调用大模型。

后续目标：

- 使用 LLM 为每个 shot 生成专业 image prompt。
- 支持主体、构图、光线、风格、景别、角色一致性、负面 prompt 等字段。

### 3. Video / Motion Prompt 生成

对应文件：

```text
agent/nodes/video_nodes.py
```

当前状态：

- `generate_video_prompt()` 使用固定模板生成 motion prompt。
- 没有调用大模型。

后续目标：

- 使用 LLM 生成 motion prompt。
- 包含 camera movement、subject motion、duration、transition、mood 等信息。
- 根据不同视频模型适配不同 prompt 格式。

### 4. Prompt Rewrite

对应文件：

```text
agent/nodes/repair_nodes.py
```

当前状态：

- `rewrite_image_prompt()` 和 `rewrite_video_prompt()` 只是把评估 feedback 拼接回 prompt。
- 没有真正理解失败原因。

后续目标：

- 使用 LLM 根据评估反馈重写 prompt。
- 保留原始创作意图，同时修复画面主体、风格、构图、运动、节奏等问题。
- 在 Agent 日志里解释为什么这样修改。

### 5. 图片生成

对应文件：

```text
mcp_servers/video_tools_server.py
```

当前函数：

```text
generate_image()
```

当前状态：

- 只写入一个 1x1 的占位 PNG。
- 不是真实 AI 生图。

后续目标：

- 接入 Stable Diffusion、Flux、DALL-E、Replicate、ComfyUI 或本地 SD WebUI API。
- 保持函数返回格式不变：

```json
{
  "image_path": "images/shot_001.png"
}
```

### 6. 图片评估

对应文件：

```text
mcp_servers/video_tools_server.py
```

当前函数：

```text
evaluate_image()
```

当前状态：

- 使用描述长度和 style 简单计算 mock 分数。
- 不是真实视觉评估。

后续目标：

- 接入 GPT-4o Vision、Gemini Vision、CLIP score 或自定义视觉评估。
- 检查图片是否符合 shot description、style、主体、构图和清晰度。
- 保持返回格式不变：

```json
{
  "score": 0.82,
  "passed": true,
  "feedback": "主体清晰，风格一致，但背景略空。"
}
```

### 7. 视频生成

对应文件：

```text
mcp_servers/video_tools_server.py
```

当前函数：

```text
generate_video()
```

当前状态：

- 只是写入一个文本文件，文件后缀是 `.mp4`。
- 不是真实可播放视频。

后续目标：

- 接入 Runway、Pika、Luma、Kling、Stable Video Diffusion、AnimateDiff 或 ComfyUI video workflow。
- 处理上传图片、提交生成任务、轮询任务状态、下载视频、保存本地文件。
- 保持返回格式不变：

```json
{
  "video_path": "videos/shot_001.mp4"
}
```

### 8. 视频评估

对应文件：

```text
mcp_servers/video_tools_server.py
```

当前函数：

```text
evaluate_video()
```

当前状态：

- 使用描述长度简单计算 mock 分数。
- 不检查真实视频内容。

后续目标：

- 抽帧后调用视觉模型评估。
- 检查内容一致性、主体稳定性、运动是否符合 motion prompt、是否闪烁、是否变形。

### 9. 视频合成

对应文件：

```text
mcp_servers/video_tools_server.py
```

当前函数：

```text
compose_video()
```

当前状态：

- 只是写入一个 mock `final_video.mp4` 文本文件。
- 没有真正拼接视频。

后续目标：

- 使用 FFmpeg 合成所有 shot 视频。
- 统一分辨率、帧率、编码格式。
- 后续可加入转场、背景音乐、字幕和标题。

## 后续开发路径

建议按下面顺序逐步替换 mock 模块。

### 阶段一：接入 LLM，让 Agent 具备真实规划能力

新增建议文件：

```text
agent/llm_client.py
```

职责：

- 统一管理 LLM API key、base URL、model name。
- 封装 `generate_text()`、`generate_json()` 等调用。
- 后续可以切换 OpenAI、Azure OpenAI、Gemini、Claude、本地 Ollama 等模型。

优先改造文件：

```text
agent/nodes/planner.py
agent/nodes/image_nodes.py
agent/nodes/video_nodes.py
agent/nodes/repair_nodes.py
```

目标：

- `plan_story()` 调 LLM 生成故事大纲。
- `plan_shots()` 调 LLM 生成 JSON 分镜。
- `generate_image_prompt()` 调 LLM 生成图片 prompt。
- `generate_video_prompt()` 调 LLM 生成 motion prompt。
- `rewrite_image_prompt()` / `rewrite_video_prompt()` 调 LLM 根据 feedback 修正 prompt。

### 阶段二：接入真实图片生成

优先改造文件：

```text
mcp_servers/video_tools_server.py
```

优先替换函数：

```text
generate_image()
```

开发目标：

- 根据 image prompt 调用真实图片模型。
- 将生成图片保存到 `workspace/projects/<project_id>/images/`。
- 前端 Storyboard 和 Preview 能显示真实图片。

### 阶段三：接入真实图片评估

优先替换函数：

```text
evaluate_image()
```

开发目标：

- 使用视觉模型或 CLIP 评估图片与 shot description / style 的一致性。
- 返回稳定的 `score`、`passed`、`feedback`。
- 让 LangGraph 的图片重试逻辑真正生效。

### 阶段四：接入真实视频生成

优先替换函数：

```text
generate_video()
```

开发目标：

- 使用 image path + motion prompt 调用真实视频模型。
- 保存真实 `.mp4` 到 `workspace/projects/<project_id>/videos/`。
- 前端可以展示或下载真实视频。

注意：

- 视频生成通常是异步任务，可能需要提交任务、轮询状态、下载结果。
- 当前 demo 使用 FastAPI background task，可以先继续沿用。
- 如果后续任务耗时很长或并发变多，再考虑任务队列。

### 阶段五：接入真实视频评估

优先替换函数：

```text
evaluate_video()
```

开发目标：

- 抽取视频关键帧。
- 调用视觉模型评估内容一致性和运动质量。
- 根据失败反馈触发 `rewrite_video_prompt()`。

### 阶段六：接入 FFmpeg 合成最终视频

优先替换函数：

```text
compose_video()
```

开发目标：

- 用 FFmpeg 拼接所有 shot 视频。
- 输出 `workspace/projects/<project_id>/final/final_video.mp4`。
- 前端 Final Video 区展示真实最终视频。

## 建议保持的模块边界

后续扩展时建议保持下面的职责边界：

```text
frontend/
  只负责输入、展示 storyboard、展示媒体、展示 Agent 日志。

backend/
  只负责 HTTP API、本地项目存储、启动 Agent、提供资源访问。

agent/
  负责 LangGraph 流程、状态演进、规划、决策、重试。

mcp_servers/
  负责外部工具和模型 API 调用，比如图片生成、视频生成、视觉评估、FFmpeg 合成。

workspace/
  保存每次运行产生的 project.json、state.json、events.jsonl、图片、视频和最终结果。
```

核心原则：

- Agent 不直接依赖具体模型供应商。
- Agent 通过 MCP-like tool interface 调用外部能力。
- 替换模型时优先改 `mcp_servers/video_tools_server.py`。
- 替换大模型规划能力时优先改 `agent/llm_client.py` 和 `agent/nodes/*.py`。
- 前端只消费后端 API，不参与生成逻辑。
