# 竖屏短剧生成 Agent

输入一个创意，输出一条带字幕、配音与配乐的 1080×1920 竖屏短剧成片。

用 LangGraph 编排「剧本 → 设定集 → 分镜 → 关键帧 → 图生视频 → 配音 → 字幕 → 合成」的状态机，DeepSeek 承担全部创作与修复推理，MCP-like 工具层封装图像/视频/语音生成，Video-Bench 维度体系做自动化质检，评估结果回流到条件边驱动返工。

> 每个节点的职责边界与执行细节见 [WORKFLOW.md](WORKFLOW.md)。

---

## 1. 架构

```
frontend/          React + Vite + TS，输入、分镜/播放器/评分/日志展示
backend/           FastAPI：项目 CRUD、启动 Agent、增量事件轮询、媒体 Range
agent/             LangGraph 状态机
  graph.py         16 节点、3 条条件边
  state.py         AgentState 定义与持久化
  llm_client.py    DeepSeek 客户端（JSON 模式、重试、用量统计）
  prompts.py       短剧生产提示词
  nodes/           每个节点一个模块
  settings.py      统一配置入口（.env）
mcp_servers/       MCP-like 工具层
  video_tools_server.py  生成 / 配音 / 时长适配 / 合成 / 封面
  providers/             mock 与 ark 两套 provider，按 .env 选择
  metrics.py             基于解码像素的帧级测量
  videobench_eval.py     Video-Bench 维度打分
  keyframe_memory.py     关键帧记忆（跨镜一致性锚点）
  evaluators.py          文件级硬性校验
bench/             与开源模型对比（Video-Bench 裁判），见第 11 节
workspace/projects/<id>/  每次运行的全部产物
tests/             pytest：86 项
```

**三层职责分离**，替换任何一层都不影响其他层：

| 层 | 位置 | 职责 |
|---|---|---|
| 编排 | `agent/` | 状态流转、决策、重试 |
| 工具 | `mcp_servers/` | 外部能力调用，模型无关 |
| 呈现 | `backend/` + `frontend/` | HTTP API、本地存储、界面 |

Agent 节点不直接依赖任何模型供应商，只调用 `mcp_servers` 的标准接口。

---

## 2. LangGraph State

### 2.1 状态定义

`AgentState` 是一个 `TypedDict`，承载整部短剧的全部事实，是运行期的唯一真相源。

```python
class AgentState(TypedDict, total=False):
    # 输入
    project_id: str
    user_prompt: str
    style: str
    genre: str
    shot_count: int

    # 设定集（plan_story 产出）
    logline: str          # 烧到封面上
    tone: str             # 调性，贯穿全部提示词
    setting: str
    outline: str
    hook: str             # 首镜 3 秒内的钩子
    ending: str
    characters: list[dict]  # appearance 是跨镜一致性的命脉

    # 分镜与推进
    shots: list[dict]
    current_shot_index: int

    # 质检与返工
    evaluation_score: float
    retry_count: int

    # 记忆与产出
    keyframe_memory: dict      # compact 后的展示摘要
    episode_report: dict
    final_video_path: str
    subtitle_path: str
    usage: dict[str, Any]
```

`shots[]` 每一项记录单镜的全部状态：分镜字段（景别/运镜/台词/时长/转场）、三类提示词、媒体路径、质检分数、重试计数、字幕行。

### 2.2 状态写入的两条纪律

**事件只写增量。** 事件流写 `events.jsonl`，state 里只留 `event_count` 计数器。早期版本把整份事件列表放进 state，导致每次节点写入都要重写 MB 级 JSON。

**指纹留在 memory.json。** `keyframe_memory` 每个关键帧带 256 维指纹向量，3 镜时占 state 的 55%。现在指纹只存 `keyframes/memory.json`，state 里是 `compact()` 后的摘要（锚点路径、分数、景别、漂移量）。3 镜实测 38KB → 16.7KB。

### 2.3 图的拓扑

```
analyze_user_goal → plan_story → plan_shots
        ↓
┌──────────────────── 每个镜头循环 ─────────────────────┐
│ generate_image_prompt → generate_image → evaluate_image │
│        ↑                      │                         │
│   rewrite_image_prompt ←──────┘  (条件边 retry/video)   │
│        ↓                                                │
│ generate_video_prompt → generate_video → evaluate_video │
│        ↑                      │                         │
│   rewrite_video_prompt ←──────┘  (条件边 retry/next)    │
│        ↓                                                │
│ evaluate_clip → synthesize_dialogue → advance_shot      │
└─────────────────────────────────────────────────────────┘
        ↓ (条件边 more/compose)
compose_video → finish → END
```

三条条件边都由纯函数决定，不含副作用：

| 条件边 | 判据 |
|---|---|
| `should_retry_image` | mock 直接放行；通过放行；未达上限重写提示词 |
| `should_retry_video` | 同上，上限取 `VIDEO_RETRIES` |
| `has_more_shots` | `current_shot_index < len(shots)` |

**mock 结果直接放行，不返工。** 早期 mock provider 把分数封顶在门槛之下，导致每镜固定多烧 2 次 LLM 修复调用，而修复根本不可能改变占位图。现在 mock 结果打 `mock: true` 标记，条件边立即放行。

### 2.4 崩溃恢复

每次节点边界与 45 秒一次的心跳线程都写 `run_heartbeat_at`。超过 `STALE_RUN_SECONDS`（600 秒）没有心跳即判定僵尸，允许重新认领，无需手工改 JSON。`claim_run()` 在锁内完成「读状态 + 置 running」，8 线程并发实测只有 1 个成功。

---

## 3. Video-Bench 打分

### 3.1 七个维度

对齐 Video-Bench 的维度体系，权重与判据如下：

| 维度 | 权重 | 判据 |
|---|---:|---|
| `imaging_quality` | 0.15 | 拉普拉斯清晰度、对比度、曝光 |
| `aesthetic_quality` | 0.15 | 色彩丰富度、曝光、对比度 |
| `temporal_consistency` | 0.20 | 闪烁（帧差的二阶差分）、亮度抖动 |
| `motion_effects` | 0.15 | 帧间差落在运动区间、帧差离散度 |
| `color_consistency` | 0.10 | 与锚点的 RGB 直方图余弦 |
| `scene_consistency` | 0.15 | 与锚点的结构相似度（均值去除后的亮度缩略图） |
| `object_consistency` | 0.10 | 中心裁剪区域的结构 + 色彩一致度 |

综合分 = 各维度分数按权重加权平均，`PASS_SCORE = 0.62`。

### 3.2 区间打分而非单点阈值

早期阈值按静态卡片校准（帧间差 ≈4），在真实 Seedance 片段上（帧间差 18.9、flicker 8.8）被误判为抖动，`motion_effects` 得 0 分。现在改成**区间打分**：区间内满分，双向衰减。

```python
motion = _band(delta, 0.5, 8.0, 45.0, 70.0)
# 2-6 是接近静止的卡片；8-45 是正常运动；>70 是退化成噪声
```

同一段素材重算后综合分从 0.518 回到 0.823。

### 3.3 跨景别自动跳过

`scene_consistency` 比较整帧结构相似度。当分镜本身要求换景别（特写→中景）时，结构相似度天然低，这是构图变化而非场景漂移——实测这类误判落在 0.05–0.28。

现在按景别分组判断可比性：

```python
FRAMING_GROUPS = {"特写": "close", "近景": "close", "中景": "mid",
                  "全景": "wide", "远景": "wide", "过肩": "mid"}
```

跨组时该维度标 `applicable=False`、`score=None`，不参与加权。`object_consistency` 在有参考帧时改用中心裁剪区域，对机位变化免疫。

### 3.4 视觉模型语义打分

像素测量抓不住「脸画歪了」。`evaluate_clip` 接受一个可选的 MLLM 钩子：

```python
mllm_judge(image_path, prompt) -> (scores_dict, reason)
```

接入 `doubao-seed-2-1-pro-260915`（方舟 Chat API），三个设计要点：

**喂全分辨率关键帧，不喂测量帧。** `extract_frames()` 抽的是 192px 宽缩略图，high 模式要求约 226 万像素，喂进去会被放大 34 倍面积、按 1280 token 计费，却看不出人脸形变。judge 收到的是 `keyframe_memory.extract_keyframe()` 存下的 1080×1920 原图。

**一次调用判四个维度。** 按维度循环调用会让同一张图重复上传，一镜 3 次 × 1280 token。`dimension_prompt()` 一次输出 `imaging / aesthetic / motion / object_consistency`，单价降到 1 次。

**与测量值取 max，不覆盖。** judge 的 1-5 分映射到 0-1 后与像素测量取最大值，原始分记 `mllm_score_1to5`、理由记 `mllm_reason`。任一环节失败只记 `judge_error`，测量路径不受影响。

未配置 key 或未开 `ARK_ALLOW_PAID` 时 `get_vision_judge()` 返回 `None`，质检退回纯测量路径，行为与不存在视觉模型时完全一致。

---

## 4. 关键帧提取与一致性

### 4.1 关键帧从哪来

每镜视频渲染完成后，从**成片片段**里抽帧，而不是复用生成时的输入图：

```python
def extract_keyframe(video_path: Path, target: Path, at_s: float = 0.15) -> Path:
    run([ffmpeg_bin(), "-y", "-ss", f"{at_s:.3f}", "-i", str(video_path),
         "-frames:v", "1", str(target)])
```

取 0.15 秒而非第 0 帧：视频模型的第一帧有时仍是未收敛的起始态，推进几帧后画面才稳定。

抽出的帧有两个用途——存进记忆库做漂移度量，以及作为后续镜头的图像生成参考。

### 4.2 三层一致性机制

**第一层：文本锚点（全程）。** 设定集里每个角色的 `appearance` 是一段具体到可被画师复现的外形描述。`shot_anchor()` 把它拼进每一镜的图像提示词，并做兜底校验：

```python
if anchor and anchor.split(":")[0] not in prompt:
    prompt = f"{prompt} Character appearance: {anchor}."
```

**第二层：关键帧记忆（跨镜度量）。** 首镜通过质检的关键帧成为全片锚点，后续每镜与它比较：

```python
memory["anchor"] = 首镜关键帧路径          # 只设一次
entry["drift_vs_anchor"] = similarity(anchor["fingerprint"], fingerprint_data)
```

指纹是两段向量：`hist`（RGB 直方图）与 `thumb`（均值去除后的亮度缩略图）。前者对色彩漂移敏感，后者对结构漂移敏感——均值去除使得整体亮度变化不被读成结构匹配。

**第三层：参考图回灌（图像侧）。** `IMAGE_REFERENCE=1`（默认开）时，锚点关键帧作为 `image` 参数传给 Seedream，后续镜头的构图条件被首镜锁住。实测同一部两镜短剧：色彩一致性 0.92 → 0.98，主体一致性 0.46 → 0.56。

首镜没有锚点，它自己建立锚点；`_anchor_reference()` 返回空串时正常生成。

### 4.3 为什么长视频不会漂

一致性不是靠单点手段，而是三层的叠加与互相兜底：

| 漂移类型 | 压制手段 |
|---|---|
| 人物长相改变 | 文本锚点逐字复用 + 参考图回灌 |
| 服装/配色漂移 | 参考图回灌（`image` 参数）+ 色彩直方图度量 |
| 构图/机位跳变 | 中心裁剪的主体比对，机位变化时不误判 |
| 逐镜误差累积 | 所有镜头都对齐**同一个**首镜锚点，不链式传递 |

最后一条是关键：如果每镜都以上一镜为参考，误差会沿链条累积放大。这里所有镜头都锚定首镜，误差不累积。

---

## 5. 合成

`compose_video` 用 ffmpeg 完成归一化、转场、字幕、配乐、封面。

### 5.1 转场

分镜的 `transition` 映射到真实的 `xfade`/`acrossfade`：

| 分镜写法 | 滤镜 | 重叠 |
|---|---|---:|
| 叠化 / dissolve | `fade` | 0.40s |
| 闪白 | `fadewhite` | 0.25s |
| 推黑 / 淡出 | `fadeblack` | 0.30s |
| 硬切 / cut | 无（concat） | 0s |

重叠是**真实时间**：总时长按 `Σduration − Σoverlap` 缩短。`VIDEO_TRANSITIONS=0` 可回退纯硬切。

### 5.2 字幕

字幕时间在 `compose_video` 内按**编码后的时间线**重排。这不能在外层做——转场重叠会缩短总长，任何一处变化都会让所有台词错位。

字号、边距、描边按 1080×1920 画布固定，字体由 `font_path()` 跨平台探测（Windows 雅黑 / macOS 苹方 / Linux Noto CJK），失败时可用 `CJK_FONT_PATH` 指定。

### 5.3 配乐


BGM 用 `aloop` 循环铺底、与台词音轨 `amix`，音量 `BGM_VOLUME`（默认 0.18）。

**这里踩过一个坑**：用 `-shortest` 收尾会把整集裁到 BGM 自身长度（实测 3.80s → 3.23s，视频轨只剩 3.2s）。现在按已知片段总长显式 `-t` 定长。

### 5.4 时长适配

台词长度由 LLM 决定，镜头时长由分镜决定，两者独立。实测一句 4.58 秒的台词配 3 秒镜头，直接合成会把语音砍断。

`fit_audio()` 的处理：

| 原始/目标比例 | 动作 |
|---|---|
| 0.85 – 1.45 | `atempo` 时间伸缩，保持自然语速 |
| > 1.45 | 压到 1.45 倍后补静音 |
| < 0.85 | 放慢到 0.85 倍后补齐 |

---

## 6. 启动

三个平台的差别只在依赖安装，Python 代码完全一致。

### Linux / macOS

```bash
cd VideoGenerationAgent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                    # 填 DEEPSEEK_API_KEY / ARK_API_KEY / TTS_API_KEY
apt-get install -y ffmpeg fonts-noto-cjk

./run_demo.sh check                     # 环境自检，零成本
./run_demo.sh serve                     # http://localhost:8000
```

### Windows

```bat
cd VideoGenerationAgent
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env

winget install Gyan.FFmpeg               :: 必须含 libass，否则字幕无法烧录
run_demo.bat check
run_demo.bat serve
```

Windows 有三处要注意：ffmpeg 必须带 libass（`full` 构建默认带）；字体自动探测微软雅黑，缺失时设 `CJK_FONT_PATH`；所有子进程带 `CREATE_NO_WINDOW`，不会闪黑框。

### 前端

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

| 用途 | Linux / macOS | Windows |
|---|---|---|
| 环境自检 | `./run_demo.sh check` | `run_demo.bat check` |
| 跑测试 | `./run_demo.sh test` | `run_demo.bat test` |
| 零成本试跑 | `./run_demo.sh simulate` | `run_demo.bat simulate` |
| 启动后端 | `./run_demo.sh serve` | `run_demo.bat serve` |

### 连通性自检

```bash
curl -s http://127.0.0.1:8000/api/health/connectivity | python -m json.tool
```

零计费的三层报告：`server`（从 `/proc/net/tcp` 读实际监听地址 + 真实 socket connect）、`outbound`（并发对 DeepSeek / 方舟 / 豆包语音各发一次不鉴权 GET）、`inbound`（明确说明外部可达性无法从容器内验证）。

`/health` 只说明进程还在，这个接口才回答「真的能接受连接、上游也通」。

判读 `outbound`：

- `200` + `credential accepted` —— 网络与 key 都正常
- `401/403` —— 网络通，key 没配或无效
- `405` —— 豆包语音 TTS 端点只接受 POST，405 本身就是主机可达的证据
- `reachable: false` —— DNS、TLS 或出网被阻断

---

## 7. 配置

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | 必需的 LLM key |
| `LLM_MODEL` / `LLM_REASONER_MODEL` | 默认 `deepseek-chat` / `deepseek-reasoner` |
| `IMAGE_PROVIDER` / `VIDEO_PROVIDER` | `mock` 或 `ark` |
| `ARK_API_KEY` | 方舟凭据（图像 + 视频 + 视觉共用） |
| `ARK_IMAGE_MODEL` / `ARK_VIDEO_MODEL` | Seedream / Seedance 模型 ID |
| `ARK_VISION_MODEL` | 视觉质检模型，留空则关闭 MLLM 打分 |
| `ARK_VIDEO_RESOLUTION` | 默认 `720p` |
| `TTS_PROVIDER` | `mock` 或 `doubao_speech` |
| `TTS_API_KEY` / `TTS_MODEL` / `TTS_BASE_URL` | 豆包语音凭据，默认模型 `seed-audio-1.0`，鉴权见下节 |
| `ARK_ALLOW_PAID` | `1` 才允许真实付费调用 |
| `IMAGE_RETRIES` / `VIDEO_RETRIES` | 返工预算，默认各 1 次 |
| `IMAGE_REFERENCE` | `1` 时回灌锚点关键帧，默认开 |
| `VIDEO_WIDTH` / `VIDEO_HEIGHT` / `VIDEO_FPS` | 画布，默认 1080×1920@30 |
| `VIDEO_TRANSITIONS` | 转场开关，默认开 |
| `BGM_PATH` / `BGM_VOLUME` | 配乐文件与混音音量 |
| `CJK_FONT_PATH` / `CJK_FONT_FAMILY` | 字体自动探测失败时手动指定 |

### 三套独立鉴权

| 产品 | 端点 | 鉴权 |
|---|---|---|
| DeepSeek | `api.deepseek.com` | `Authorization: Bearer` |
| 方舟 Ark | `ark.cn-beijing.volces.com/api/v3` | `Authorization: Bearer` |
| 豆包语音 | `openspeech.bytedance.com/api/v3` | `X-Api-Key` |

豆包语音走独立产品线，**用方舟 key 会返回 401 `Invalid X-Api-Key`**。两个易踩的坑：模型名校验发生在鉴权**之前**，写错返回 403 `fail to convert model to resource_id`，看着像 key 问题其实是模型名不对；请求体字段是 `text_prompt` 而非 `text`，返回 base64 WAV。

### 付费保护

任何真实生成调用前都检查 `ARK_ALLOW_PAID=1`，未设置直接抛错。没有从真实模型静默回退到 mock 的路径——半配置的运行不会产出看起来真实的假媒体。

---

## 8. 产物

```
workspace/projects/<project_id>/
  project.json  state.json  events.jsonl  bible.json
  shots/shot_001.json
  images/shot_001.png + .json
  videos/shot_001.mp4 + .json
  audio/shot_001.wav            原始配音
  audio/shot_001_fit.m4a        适配后的配音
  keyframes/shot_001.png        memory.json（含指纹）
  subtitles/subtitles.ass
  final/final_video.mp4  cover.png  final_video.json
```

`final_video.json` 记录片段列表、标题、字幕行数、画布、实际时长、BGM 路径与每条转场的应用结果。

---

## 9. 测试

```bash
.venv/bin/python -m pytest tests -q     # 86 passed
```

| 文件 | 项数 | 覆盖 |
|---|---:|---|
| `test_pipeline.py` | 15 | 全链路、字幕时间轴、关键帧记忆、七维报告、转场重叠、BGM 定长、视觉 judge |
| `test_platform.py` | 15 | 跨平台路径、字体探测、ffmpeg 封装 |
| `test_framing.py` | 7 | 景别分组与跨景别跳过 |
| `test_recovery.py` | 7 | 心跳、僵尸回收、并发认领 |
| `test_metrics.py` | 6 | 像素指标性质 |
| `test_providers.py` | 6 | 付费保护、未知 provider、重试边界 |
| `test_bench.py` | 21 | 对比工具：分数解析、维度分组、载荷构造、调用量估算 |

测试断言绑定实测数值而非存在性：转场重叠缩短总时长、字幕重排到编码后的时间线、BGM 不改变总长、judge 只发一次调用且收到全分辨率关键帧。

---

## 10. 成本与耗时

| 项 | 量级 |
|---|---|
| Seedream 出图 | 约 13s，每镜 1 次 |
| Seedance 出视频 | 约 60-90s，每镜 1 次 |
| Seed-Audio 配音 | 约 16s，每句台词 1 次 |
| 视觉质检 | 约 12s，每镜 1 次（1280 图像 token） |

2 镜短剧约 112-205 秒；3 镜约 199 秒，加配音约 216 秒。

---

## 11. Benchmark：与开源模型对比

`bench/` 用 [Video-Bench](https://github.com/Video-Bench/Video-Bench) 的维度体系，把本项目的产物和开源文生视频模型放在同一把尺子上打分。

### 11.1 接的是什么

Video-Bench 本身是一个**裁判器**：把视频抽帧交给多模态模型，按 9 个维度打 1-5 / 1-3 分。原版写死 GPT-4o；这里改用 Ark 豆包（`doubao-seed-2-1-pro`），走同一个 OpenAI-compatible 接口。

裁判和生成分开：`bench/judge.py` 只吃已经准备好的帧载荷，`bench/backends.py` 只负责出片。所以重新打分不必重渲染，改裁判不必重出片。

### 11.2 上游的四个坑

vendored 的 Video-Bench 有缺陷，这里**不修改它的源码**（`bench/_vendor/` 不入库，按 commit 拉取），而是在 bench 里重写了执行循环：

| 问题 | 位置 | 后果 |
|---|---|---|
| `modelname` 在赋值前被读取 | `VideoTextAlignment.py:327` | 实测 `UnboundLocalError`，5 个 video-text 维度全部跑不起来 |
| 裁判模型名硬编码 5 处 | static/dynamic/VideoTextAlignment | 换 base_url 即 404 |
| `overall_consistency` 无路由 | `videobench/__init__.py` | 有 prompt、有 91 条标注题，但调度器抛 `ValueError` |
| `video-text consistency` 键不存在 | `prompt_dict.py` | 按 README 的维度名调用抛 `KeyError` |

上游标准模式实测只有 4 个维度可跑（2 static + 2 dynamic）。bench 覆盖全部 9 个。

另外两处按需修正：抽帧在客户端降到最长边 640px（1080×1920 原样编码实测 17.8 MB/视频，易被请求体上限拒绝；640px 下 4 秒片段 1.0 MB），以及给每个模型的图片块加上标签（上游的 grid view 拼接块不带模型名，多模型同调时裁判无法归属分数）。

### 11.3 跑法

```bash
python bench/setup_vendor.py                      # 拉 vendored Video-Bench（不入库）

# 权重：并行分片下载，然后用官方 sha256 校验（见 11.4）
bench/fetch_models.sh zai-org/CogVideoX-5b 12
bench/fetch_models.sh Wan-AI/Wan2.1-T2V-1.3B-Diffusers 12
bench/verify_models.sh zai-org/CogVideoX-5b
bench/verify_models.sh Wan-AI/Wan2.1-T2V-1.3B-Diffusers

bench/.venv/bin/python -m bench.runner --estimate  # 只算预算，不花钱
bench/.venv/bin/python -m bench.runner \
    --models ours cogvideox5b wan21_1_3b \
    --prompts 1 --seeds 0 1 --seconds 5 \
    --dimensions imaging_quality aesthetic_quality motion_effects overall_consistency
```

裁判要花钱，需要 `ARK_ALLOW_PAID=1` 与 `BENCH_ALLOW_PAID=1` 同时为 1。生成和判分可分开跑（`--stage generate` / `--stage judge`），产物按维度缓存。

`ours` 后端要求 `VIDEO_PROVIDER=ark`（mock 出的是占位片，不算竞品）。

### 11.4 权重完整性与一个实测踩到的坑

下载器拿 `curl -r` 分片取，写盘用 `dd of=... bs=1M seek=$((s/1048576)) conv=notrunc`。这行**看起来能用、文件长度也对，但内容全是错的**：`bs=1M` 下 `seek` 的单位是 1 MiB 块，`s/1048576` 向下取整，于是除第一块外每个分片都写到了错误偏移，互相覆盖。

后果不是报错，而是模型正常加载、正常出片、画面**全黑且像素是 NaN**。`stat` 出来的文件大小与远端完全一致，所以只看长度发现不了。

修法是 `bs=1M` 配 `oflag=seek_bytes`（`seek` 按字节解释，实测 1000 字节写入偏移 999 确实落在 999）：
```bash
dd of="$out" bs=1M seek="$s" oflag=seek_bytes conv=notrunc status=none
```

因此下载后**必须校验 sha256**，不能只比大小。HF 对 LFS 文件在 `HEAD` 响应里给出 `x-linked-etag`，脚本据此比对：

```bash
bench/verify_models.sh zai-org/CogVideoX-5b             # 只检查
bench/verify_models.sh zai-org/CogVideoX-5b --fix       # 重下损坏文件
```

### 11.5 对比口径

两类模型做的**不是同一个任务**，报告里分开记录、不做归一：

- `ours`：关键帧 → 图生视频（i2v）
- `cogvideox5b` / `wan21_1_3b`：文生视频（t2v）

因此单镜分数比较的是各自产物质量，不是同一条流水线的高低。时长与帧率也各按模型的训练设定取整（CogVideoX 8fps、Wan 16fps），与目标时长有小幅偏差。

### 11.6 对比口径：workflow 与 model 不是同一类对象

本项目交付的是**工作流**（剧本 → 分镜 → 提示词 → 渲染 → 合成），而 CogVideoX / Wan / LongLive 是**生成模型**，一次调用出一条片段。把两者放在同一张分数图里并列，比的是两个不同层面的东西，结论无法归因。

因此本仓库的对比按对象分层，**任何一层内部只比较同类对象**：

| 层 | 比较对象 | 固定什么 |
| --- | --- | --- |
| 编排层对比 | 本项目 workflow ↔ 另一个 workflow（OpenMontage） | 底模、分辨率、brief、seed |
| 模型对比 | 开源模型 ↔ 开源模型 | prompt、时长、分辨率 |

开源模型在本项目里只作为**实验介质**出现：固定它们，才能把编排层的贡献单独测出来。它们不是竞争对手。

此前版本曾把 `ours` 与 `longlive_1_3b` / `wan21_1_3b` / `cogvideox5b` 画进同一张并列柱状图，那几张图（`quality.png`、`identity.png`、`longform.png`、`capability.png`）与对应报告已删除，因为它们把"一次调用的产物"和"一套编排的产物"当成了可比对象。

### 11.7 编排层在本地模型上的实测（Tier 3a）

完整报告见 `bench/results/tier3a_report.md`（图表 `docs/figures/tier3a_*.png`）。核心设计：同底模、同 832×480、逐镜种子固定，**唯一变量是镜与镜之间是否传递角色状态**。

结果分两部分，缺一不可：

| 结论 | 数值 | 样本 | 显著性 |
| --- | --- | --- | --- |
| 跨镜角色一致性 | 2.00 → 4.17（5 分制） | 6 组 × 5 镜 | 符号检验单侧 p=0.031 |
| 单镜画质（四维均值） | 4.175 → 3.675 | 8 格 | brief 胜 6/8 |

也就是说：**编排层不提升单镜画质，它解决的是"五个镜头里是不是同一个人"**。单镜落后不是提示词长度造成的——把提示词从 900 字符截到 400 只补回 0.075。

### 11.8 这一层没有证明什么

- 不能据此说成片更好看：本层全是 5 秒单镜，没有成片。
- 不能说是赢了 OpenMontage：对照组的 `independent` 是构造出来的无状态下界，不是竞品的实测成绩。OpenMontage 完整 studio 需要 Postgres / Redis / MinIO，且其派发层不路由到本地视频工具，本轮未接入。
- 不能推广到强模型：单镜的负面结果很可能只适用于 1.3B–5B。
- 一致性由单一本地 VLM 判定，未与人类评分比对。

### 11.9 像素侧交叉验证：人脸嵌入指标在本层不可用

上面的结论只有单一 VLM 判分，最直接的独立验证是人脸身份嵌入——也正是
DreamFactory（arXiv 2408.11788）用 CSFD 做的事：每帧检出人脸、编码、对全部
帧对取余弦平均。脚本 `bench/study/identity_crosscheck.py`，完整报告
`bench/results/identity_crosscheck.md`，在**同一批 180 条片段**上测，不产生
新的渲染与费用。

结论分两段，都必须读：

| 臂 | 有可用人脸的镜头 | 可计分 episode |
| --- | --- | --- |
| independent | **4 / 90（4.4%）** | **0 / 18** |
| planned | 39 / 90（43.3%） | 9 / 18 |

**配对比较算不出来。** 独立臂没有一个人脸覆盖率过线的 episode，没有配对就没有
符号检验、没有置信区间。这不是检测器设置问题：把 MTCNN 阈值降到 0.3 无变化，
2 倍上采样只救回个位数帧，换 OpenCV SSD 在同样帧上同样无检出——1.3B–5B 在
832×480 上大量是远景/背身/模糊镜头，脸本来就不在里面。

这直接回答了「CSFD 能不能搬进本项目」：**在这套素材上不能**，而且是结构性
原因而非实现缺陷。CSFD 未定义「一侧无脸」时该帧对如何计入，而在我们的数据里
这种情况占多数。

**能检出的人脸里，两个指标不矛盾。** 计划臂 9 个 episode：余弦
CogVideoX 0.7312（n=5）、Wan2.1 0.8142（n=4），与 VLM 的 Spearman ρ=0.82
（n=9）。但这 9 个 episode 的 VLM 分只有 4 和 5 两个取值，秩相关在两点分布上
几乎不构成效度证据；它只能说两者「互不矛盾」，说不出任何一方是校准过的。

另需注意：4.4% 与 43.3% 的检出率差异本身**不是**一致性证据。模糊、畸变、背身
都会让检测失败，它与渲染质量高度相关，不能当作「同一个人」的度量。

---

## 12. 已知边界

这些是**能力边界**，不是待修的缺陷：

- **视频侧无参考图回灌**。`reference_image` 角色是 Seedance 2.0 及以上才有的能力；首尾帧（`first_frame` + `last_frame`）在 1.0 pro 上就有，但 **1.0 pro-fast 不支持**，选 pro-fast 时拼接会退化成只按首帧生成，运行时会写一条告警。
- **单进程后台任务**。`BackgroundTasks` + `asyncio.to_thread`，无任务队列、无主动取消。僵死运行能被心跳回收，但正在跑的生成任务无法中断（Seedance 单次 60-90 秒，只能等返回）。
- **字幕只覆盖台词**，没有旁白，也没有片头字卡。
- **`scene_consistency` 只适合同机位漂移**。跨景别已自动跳过，但它不表达「同一场景不同构图」的语义一致性——那需要视觉模型判断。
- **视觉质检分两级**。逐镜按 Video-Bench 七维打分；成片后再把全部关键帧按播放顺序一次性交给 MLLM，判 `story_coherence` / `continuity` / `pacing`。序列级只看关键帧，不看动态。

---

## 13. 后续路线

1. **视频侧参考图**：开通 Seedance 2.0+，把锚点关键帧作为 `reference_image` 传给视频模型，压制视频段的人物漂移。
2. **多角色音色**：现在所有台词用同一音色。按角色的 `voice` 字段选择不同音色，需要先拿到可用音色 ID 列表。
3. **可取消执行器**：替换 `BackgroundTasks`，支持中断与重启恢复。

已完成：首尾帧拼接（第 N 镜 = keyframe[N] → keyframe[N+1]）、序列级视觉质检。
