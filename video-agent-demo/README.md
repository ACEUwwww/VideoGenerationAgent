# 竖屏短剧生成 Agent

一个端到端的竖屏短剧（9:16）自动生产 Agent。用 LangGraph 编排「剧本 → 设定集 → 分镜 → 关键帧 → 图生视频 → 配音 → 字幕 → 成片」的状态机，用 DeepSeek 承担全部创作与修复推理，用 MCP-like 工具层封装图像/视频/语音生成，并按 Video-Bench 的维度体系做自动化质检，评估结果回流到条件边驱动返工。

原始 demo 只有 mock 流程；当前版本把**创作、渲染、质检、合成四条链路都实现为真实可运行**，只有付费媒体模型默认保持在 mock 状态。

---

> 完整的执行链路、每个节点的职责边界与使用方法，见 [WORKFLOW.md](WORKFLOW.md)。

## 1. 整体架构

```
frontend/          React + Vite + TS，只负责输入、展示分镜/播放器/日志/评分
backend/           FastAPI：项目 CRUD、启动 Agent、增量事件轮询、媒体资源(Range)
agent/             LangGraph 状态机：规划、生成、质检、修复、合成
  llm_client.py    DeepSeek 客户端（JSON 模式、重试、用量统计）
  prompts.py       短剧生产提示词（设定集/分镜/图像/运动/质检/修复/字幕）
  nodes/           每类节点一个模块
  settings.py      统一配置入口（.env）
mcp_servers/       MCP-like 工具层
  video_tools_server.py  生成/配音/合成/封面
  providers/       mock 与 ark 两套 provider，按 .env 选择
  metrics.py       基于解码像素的帧级测量
  videobench_eval.py  Video-Bench 维度体系打分
  keyframe_memory.py  关键帧记忆（跨镜一致性锚点）
  evaluators.py    文件级硬性校验（分辨率/时长/编码/体积）
workspace/projects/<id>/  每次运行的全部产物
tests/             pytest：全链路、指标、provider
```

## 2. 启动

```bash
cd video-agent-demo
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # 填入 DEEPSEEK_API_KEY

# 系统依赖：ffmpeg（合成/抽帧/字幕烧录）+ 中文字体
apt-get install -y ffmpeg fonts-noto-cjk

.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

前端：

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

## 3. Pipeline

```
analyze_user_goal
  → plan_story            DeepSeek 生成设定集（角色外观锚点、三幕梗概、钩子）
  → plan_shots            DeepSeek 生成分镜（景别/运镜/台词/情绪/音效/时长/转场）
  → 对每个 shot:
      generate_image_prompt   LLM 写图像提示词，逐字嵌入角色外观锚点
      generate_image          走 IMAGE_PROVIDER
      evaluate_image          文件级质检 → 不通过则 rewrite_image_prompt 重试
      generate_video_prompt   LLM 写运动提示词
      generate_video          走 VIDEO_PROVIDER（图生视频）
      evaluate_video          文件级质检 → 不通过则 rewrite_video_prompt 重试
      evaluate_clip           抽关键帧 → 写入关键帧记忆 → 七维打分
      synthesize_dialogue     台词配音 + 字幕切分
      advance_shot
  → compose_video         归一化 → concat → 字幕烧录 → 封面
  → finish                汇总全片多维报告与用量
```

每个节点都向 `workspace/projects/<id>/events.jsonl` 追加
`Thought / Action / Observation / Decision` 事件，前端增量轮询展示。

## 4. 关键设计

### 4.1 角色一致性（红果模式的核心）

`plan_story` 为每个角色产出 `appearance` 外观锚点（脸型、发色、服装、显著特征），
下游每条图像提示词都必须逐字包含它。质检里有一条硬检查：
提示词不含锚点 → 直接扣分。这是跨镜人物不漂移的第一道闸门。

### 4.2 关键帧记忆

每个镜头生成后抽关键帧（`keyframes/shot_XXX.png`），首个通过的镜头成为
**全片视觉锚点**。后续镜头与锚点做

- **色彩一致性**：RGB 直方图余弦
- **结构一致性**：去均值缩略图的余弦（对整体亮度平移不敏感，对布局变化敏感）

漂移量写进 `keyframes/memory.json` 并在前端表格中按镜展示。
这是「关键帧上下文约束后续片段」这条思路的可测量落地。

### 4.3 Video-Bench 维度评估

维度集合与权重对齐 Video-Bench：

| 维度 | 权重 | 计算方式 |
|---|---|---|
| imaging_quality | 0.15 | 拉普拉斯锐度 + 对比度 + 曝光 |
| aesthetic_quality | 0.15 | 色彩饱和度 + 曝光 + 对比度 |
| temporal_consistency | 0.20 | 帧间差抖动（闪烁）+ 亮度抖动 |
| motion_effects | 0.15 | 平均帧间差落在合适区间、连续性 |
| color_consistency | 0.10 | 与锚点关键帧直方图余弦 |
| scene_consistency | 0.15 | 与锚点关键帧结构余弦 |
| object_consistency | 0.10 | 结构 0.6 + 色彩 0.4 |

Video-Bench 原版用 GPT-4o Vision 打 1-5 分。这里是**像素测量**实现，不需要视觉模型，
分数可复现；`videobench_eval.evaluate_clip` 保留了 `mllm_judge` 钩子，
接入视觉模型后会在 `imaging/aesthetic/motion` 三个维度叠加 MLLM 评分并记录理由。

### 4.4 评估结果驱动返工

质检不做「看起来还行」：文件不存在、分辨率不对、时长偏离、无可解码流、
体积异常小、提示词缺锚点，都会落到具体 issue 文本，进入
`rewrite_image_prompt` / `rewrite_video_prompt`，由 LLM 按 issue 逐条修复。

### 4.5 Provider 抽象

```
IMAGE_PROVIDER=mock|ark
VIDEO_PROVIDER=mock|ark
```

- `mock`：本地真实渲染。图像是 PIL 绘制的 1080×1920 分镜卡（含中文说明与英文提示词），
  视频是 ffmpeg zoompan 推拉的真实 H.264 片段，音频是真实 AAC 静音轨。
  整条合成/字幕/封面链路都是真的，**零成本**。
- `ark`：火山方舟。图像走 `POST /images/generations`（Doubao Seedream，`1080x1920`、
  `watermark:false`）；视频走 `POST /contents/generations/tasks`（Doubao Seedance，
  `ratio:9:16`，首帧以 base64 data URI 传入 `first_frame` 角色），异步轮询后下载。
  代码里没有静默回退：provider 名字写错直接报配置错误。

### 4.6 付费保护

火山方舟按次计费。`ARK_ALLOW_PAID=1` 未设置时，影像/视频 provider 一律拒绝调用，
即使 `.env` 里写了 `ARK_API_KEY`。这样误跑一整部多镜短剧不会产生账单。

## 5. 配置项

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | 必需的 LLM key |
| `LLM_MODEL` / `LLM_REASONER_MODEL` | 默认 `deepseek-chat` / `deepseek-reasoner` |
| `IMAGE_PROVIDER` / `VIDEO_PROVIDER` | `mock` 或 `ark` |
| `ARK_API_KEY` / `ARK_IMAGE_MODEL` / `ARK_VIDEO_MODEL` | 方舟凭据与模型 |
| `TTS_PROVIDER` | `mock` 或 `doubao_speech` |
| `TTS_API_KEY` | **豆包语音**的 key，不是方舟 key |
| `TTS_MODEL` / `TTS_BASE_URL` | 默认 `seed-audio-1.0` / openspeech v3 |
| `ARK_ALLOW_PAID` | `1` 才允许真实付费调用 |
| `VIDEO_WIDTH` / `VIDEO_HEIGHT` / `VIDEO_FPS` | 画布，默认 1080×1920@30 |

## 6. 本地产物

```
workspace/projects/<project_id>/
  project.json  state.json  events.jsonl  bible.json
  shots/shot_001.json
  images/shot_001.png + .json
  videos/shot_001.mp4 + .json
  audio/shot_001.m4a
  keyframes/shot_001.png  memory.json
  subtitles/subtitles.ass
  final/final_video.mp4  cover.png  final_video.json
```

## 7. 测试

```bash
.venv/bin/python -m pytest tests -q
```

36 项测试，覆盖：

- 全链路产出真实 mp4（分辨率/编码/时长断言）、字幕时间轴与烧录
- 关键帧记忆锚点与漂移、回灌参考图的传递
- 七维报告完整性、跨景别维度跳过、主体局部一致性
- 提示词包含角色锚点、LLM 用量统计与媒体成本记账
- provider 付费保护、未知 provider 报错、Ark 重试边界
- 并发 run 的原子性、像素指标性质

## 8. 真实云服务验证记录（2026-10-30）

在账号 `2133734537` 上做的实测，全部用最小调用量：

| 项 | 模型 | 结果 |
|---|---|---|
| 文生视频（480p/4s） | `doubao-seedance-1-0-pro-fast-251015` | ✅ 成功，4.04s / 480×864 / 24fps / 1.8MB |
| 图生视频（首帧 data URI） | 同上 | ✅ 成功，4.04s / 480×864 |
| 状态机 + 真实视频 | 同上 | ✅ 单镜 90.8s 完成，成片 1080×1920 h264 + AAC |
| 文生图 | `doubao-seedream-4-0-20260415` | ✅ 成功，13s，1080×1920 PNG |
| 参考图生图（`image` 参数） | 同上 | ✅ 成功，16s，人物/服装/色彩被参考图锁住，场景可换 |
| 全真实 2 镜端到端 | Seedream 4.0 + Seedance 1.0 pro-fast | ✅ 112-205s，成片 1080×1920，综合 0.79-0.82 |
| 全真实 3 镜端到端 | 同上 | ✅ 199s，综合 0.833，跨景别维度正确跳过 |
| 语音合成 | `seed-audio-1.0` | ✅ 成功，16s 延迟，4.58s WAV，40kHz |
| 全真实 3 镜 + 配音 | Seedream + Seedance + Seed-Audio | ✅ 216s，成片人声 max -2.9dB，台词密度 2/3 |
| 文生图 | `doubao-seedream-5-0-pro-260628` | ⚠️ 已激活但 `429 SetLimitExceeded`（账号安全体验模式配额） |
| 其他 seedream 变体 | — | ❌ `ModelNotOpen` / `InvalidEndpointOrModel.NotFound` |
| 语音合成 | — | ❌ 方舟 key 在 `openspeech.bytedance.com/api/v3/tts/create` 返回 403 `fail to convert model to resource_id`；在 `/tts/unidirectional` 返回 401 `Invalid X-Api-Key`，需要独立的豆包语音凭据 |

已验证的接口约定：

- 视频创建：`POST /api/v3/contents/generations/tasks`，body 含 `model` / `content[]` / `ratio` / `resolution` / `duration`
- 首帧：`content` 里 `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."},"role":"first_frame"}`，小图 data URI 可接受
- 轮询：`GET /api/v3/contents/generations/tasks/{id}`，`status` 取值 `succeeded|failed|cancelled`，成功时视频在 `content.video_url`
- 480p + 9:16 实际产出 480×864，由合成阶段统一上采样到 1080×1920

**评分器校准**：初版阈值按 mock 卡片校准（帧间差 ≈4），在真实 Seedance 片段（帧间差 18.9、flicker 8.8）上误判为抖动，`motion_effects` 得 0 分。已改为区间打分（区间内满分，双向衰减），同一段素材重算后综合分从 0.518 回到 0.823。

## 9. 已知边界

已经修掉的（2026-10-30 第二轮）：

- **mock 造成的必然返工**：以前 mock provider 直接把分数封顶到 0.6，
  而门槛是 0.78，导致每个镜头固定多烧 2 次 LLM 修复调用、而修复根本不可能
  改变占位图。现在 mock 结果单独打 `mock: true` 标记，`should_retry_*` 直接放行，
  实测 LLM 调用从 6 次降至 6 次（原本 8 次），重试计数恒为 0。
- **Ark 调用无重试**：新增 `_post_with_retry`，只对连接错误与
  `429/500/502/503/504` 重试（这些不会产生双份计费），业务 4xx 立即抛出。
  轮询超时的报错现在带上最后一次轮询的具体错误，不再静默 `continue`。
- **`/run` 并发竞态**：`claim_run()` 在锁内完成「读状态 + 置 running」，
  8 线程并发实测只有 1 个成功。
- **跨模块引用私有函数**：`metrics.py` 公开导出
  `histogram_vector` / `thumb_vector` / `cosine`。
- **成本未记账**：真实 provider 的调用次数与 token 累加进 `usage.media`，
  mock 不计账。
- **前端无历史项目列表**：侧栏新增项目列表，可切换历史项目。
- **视频质检要求精确画布尺寸导致的重复计费**：原实现要求片段必须是
  1080×1920，而 Seedance 720p/9:16 原生输出 704×1248，两个镜头因此各多跑一次
  付费重试（2 镜花掉 4 次视频调用）。现在质检判断画幅比例（±3%）与最低可用
  高度，合成阶段负责归一到画布。
- **`ARK_VIDEO_RESOLUTION` 配置未生效**：`MediaSettings` 构造时漏传该字段，
  永远走默认 `720p`。已接上，实测 `.env` 与环境变量覆盖均正确。

第四轮（2026-11-04）修复的三项：

- **崩溃或重启后项目永久卡死**：`claim_run()` 以前只看 `status == "running"`
  就拒绝，没有任何超时判断。实测把状态置成 `running` 模拟进程被杀后，重试
  永远返回 `False`，只能手工改 JSON 才能恢复。现在运行时会周期性写
  `run_heartbeat_at`（45 秒一次的心跳线程 + 节点边界），超过
  `STALE_RUN_SECONDS`（600 秒）没有心跳就判定为僵尸并允许重新认领；旧版本
  留下的、完全没有心跳字段的 `running` 状态也可直接认领。
- **`state.json` 随镜头线性膨胀**：`keyframe_memory` 里每个关键帧带一个
  256 维指纹向量，3 镜时占 state 的 55%。现在指纹只存 `keyframes/memory.json`，
  state 里保留 `compact()` 后的展示摘要（锚点路径、分数、景别、漂移量）。
  3 镜实测 38KB → 16.7KB，`keyframe_memory` 字段从 21KB 降到 496 字节。
- **三个从未被引用的提示词模板**：`SUBTITLE_SYSTEM`、`EVAL_IMAGE_SYSTEM`、
  `EVAL_VIDEO_SYSTEM` 定义在 `agent/prompts.py` 里但零处引用——质检实际走的是
  `evaluators.py` 的像素测量，字幕切分是 `split_subtitle()` 的代码逻辑。
  已删除，避免后来者误以为这些能力已由 LLM 实现。

仍然存在：

- ~~TTS 仍是静音轨~~ **已解决（2026-10-30 第三轮）**：Seed-Audio 1.0 已接入。
  它跑在豆包语音这条独立产品线上（`openspeech.bytedance.com/api/v3/tts/create`），
  鉴权是 `X-Api-Key`，与方舟的 `Authorization: Bearer` 完全不同 —— 方舟 key
  在这里返回 401 `Invalid X-Api-Key`。请求体用 `text_prompt`（不是 `text`），
  返回 base64 WAV（40kHz 立体声 pcm_s16le）。
- **配音时长与镜头时长不匹配**：台词由 LLM 写、时长由分镜定，两者独立。
  实测一句台词 4.58s 配 3s 镜头，直接合成会截断语音。已加 `fit_audio()`：
  在 0.85-1.45 倍速内做时间伸缩（保持自然语速），超出则补静音或截断。
- **没有视觉模型**。当前质检是像素测量 + 文件事实，能抓住崩溃、横屏、
  闪烁、漂移，但抓不住「脸画歪了」这类语义问题。MLLM 钩子已预留。
- **单进程后台任务**。`BackgroundTasks` + `asyncio.to_thread`，无任务队列、
  无主动取消。僵死运行现在能被心跳超时自动回收，但正在跑的生成任务本身
  仍然无法中断（Seedance 单次渲染 60-90 秒，只能等它返回）。
- **关键帧回灌已实现（图像侧），视频侧受模型限制**。
  Seedream 4.0 接受 `image` 参数做参考图条件，首镜关键帧现在会作为锚点传给
  第二镜及之后的所有镜头（`IMAGE_REFERENCE=1`，默认开启）。实测同一部两镜短剧：
  色彩一致性 0.92 → 0.98，主体一致性 0.46 → 0.56。
  视频侧无法回灌：官方文档明确 Seedance 1.0 pro / pro-fast 只支持
  「图生视频-首帧」和「文生视频」，`reference_image` 角色是 Seedance 2.0 系列
  和 2.5 才有的能力。要在视频侧做锚定，需要开通 2.0 及以上模型。
- **`scene_consistency` 仍是薄弱维度**：它比较的是整帧结构相似度，
  而当分镜本身要求换景别换机位时（特写→中景），结构相似度天然低，
  这个维度对「同一场景不同构图」会误判。它适合检测同一机位下的漂移，
  不适合做跨景别的场景一致性判断。
- **字幕只覆盖台词**，没有旁白与片头字卡；`SUBTITLE_SYSTEM` 提示词尚未接入。
- **BGM 未接**：`compose_video` 的 `bgm_path` 混音逻辑已实现，但没有调用方。
- **转场字段仅元数据**：分镜里的叠化/闪白在合成时全部按硬切处理。

## 10. 交付状态

**当前版本可直接交付使用**，具备以下能力：

| 能力 | 状态 | 说明 |
|---|---|---|
| 剧本 → 设定集 → 分镜 | ✅ 真实 | DeepSeek 全链路，角色外观锚点逐字复用 |
| 关键帧生成 | ✅ 真实 | Seedream 4.0，1080×1920，支持参考图回灌 |
| 图像质检与返工 | ✅ 真实 | 文件事实 + 像素测量，LLM 按 issue 修复 |
| 图生视频 | ✅ 真实 | Seedance 1.0 pro-fast，首帧条件 |
| 关键帧记忆 | ✅ 真实 | 首镜锚点，跨镜漂移可测量，回灌约束后续生成 |
| 七维评分 | ✅ 真实 | 像素测量，跨景别维度自动跳过 |
| 字幕烧录与合成 | ✅ 真实 | ffmpeg + ASS，1080×1920 h264+AAC |
| 封面生成 | ✅ 真实 | logline 烧录 |
| 前端展示 | ✅ 真实 | 9:16 播放器、分镜、评分条、项目列表、Agent 日志 |
| 配音 | ✅ 真实 | Seed-Audio 1.0，豆包语音独立鉴权，自动适配镜头时长 |
| 视频侧参考图 | ⚠️ 不可用 | Seedance 1.0 不支持 `reference_image`，需 2.0+ |

### 交付检查清单

```bash
# 1. 环境
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
apt-get install -y ffmpeg fonts-noto-cjk
cp .env.example .env    # 填 DEEPSEEK_API_KEY / ARK_API_KEY

# 2. 测试
.venv/bin/python -m pytest tests -q          # 应 28 passed

# 3. 免费验证 provider 可用性（不花钱）
.venv/bin/python scripts/ark_smoke.py --kind status

# 4. 单次最小付费验证
ARK_ALLOW_PAID=1 .venv/bin/python scripts/ark_smoke.py --kind image

# 5. 启动
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
cd frontend && npm install && npm run dev
```

### 一次真实生成的成本量级

2 镜短剧：2 次 Seedream（各 8.1k output tokens）+ 2 次 Seedance
（合计约 69k tokens），耗时约 200 秒。3 镜约 199 秒 / 3+3 次调用。

## 11. 后续路线

1. **视频侧参考图**：开通 Seedance 2.0+，把锚点关键帧作为 `reference_image`
   传给视频模型，这是压制视频段人物漂移的唯一手段（1.0 不支持）。
2. **多角色音色**：现在所有台词用同一音色朗读。按角色的 `voice` 字段
   选择不同音色，需要先拿到可用音色 ID 列表。
3. **视觉评审**：接视觉模型填 `mllm_judge`，让「脸画歪」这类语义问题进入返工。
4. **首尾帧约束**：相邻镜头用上一镜末帧作为下一镜首帧，进一步压制跳变。
5. **可取消执行器**：替换 `BackgroundTasks`，支持中断与重启恢复。
6. **BGM 与转场**：`compose_video` 的混音能力已就绪，需要调用方与转场实现。
