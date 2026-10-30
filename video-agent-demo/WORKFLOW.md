# 工作流与使用方法

本文档说明这部竖屏短剧生成 Agent 的完整执行链路、每个环节的职责边界，以及本地如何跑起来。

---

## 一、全局视角

一句话：**输入一个创意，输出一条带字幕和配音的 1080×1920 竖屏成片。**

```
用户创意 + 题材 + 镜头数
        │
        ▼
   ┌─────────────────────────────────────────────┐
   │  LangGraph 状态机（16 个节点 / 3 条条件边）    │
   └─────────────────────────────────────────────┘
        │
        ├─ 创作层：DeepSeek 写设定集、分镜、各类提示词、修复
        ├─ 生成层：Seedream 出关键帧，Seedance 出片段，Seed-Audio 出配音
        ├─ 质控层：文件事实 + 像素测量，不合格回流重写
        └─ 合成层：ffmpeg 归一化、拼接、字幕烧录、封面
        │
        ▼
   workspace/projects/<id>/final/final_video.mp4
```

**三层职责分离**，替换任何一层都不影响其他层：

| 层 | 位置 | 职责 |
|---|---|---|
| 编排 | `agent/` | 状态流转、决策、重试 |
| 工具 | `mcp_servers/` | 外部能力调用，模型无关 |
| 呈现 | `backend/` + `frontend/` | HTTP API、本地存储、界面 |

Agent 节点永远不直接依赖某个模型供应商，只调用 `mcp_servers` 暴露的标准接口。

---

## 二、执行链路详解

### 阶段 1：目标解析

**`analyze_user_goal`**

把用户输入（创意、题材、镜头数、可选剧本）落成运行状态。这一步不调用模型，只做初始化和记录。

---

### 阶段 2：故事设定

**`plan_story`** → DeepSeek

产出一份「剧集设定集」：

| 字段 | 作用 |
|---|---|
| `logline` | 一句话故事，后面会烧到封面上 |
| `genre` / `tone` / `setting` | 题材、调性、场景，贯穿后续所有提示词 |
| `characters[].appearance` | **角色外观锚点**，全片一致性的命脉 |
| `outline` | 三幕梗概 |
| `hook` / `ending` | 开场钩子与结尾反转 |

`appearance` 的要求是「具体到可以被画师逐镜复现」——脸型、发型发色、体型、固定服装、显著特征。这个字段之后会被**逐字**塞进每一条图像提示词，是防人物漂移的第一道闸门。

---

### 阶段 3：分镜拆解

**`plan_shots`** → DeepSeek

产出镜头列表，每镜包含：

```
scene / shot_size / camera / action
dialogue / speaker / emotion / sfx
duration_s / transition / characters
```

两条硬性约束写进了提示词：

- **台词密度 ≥ 70%**：竖屏短剧靠台词推进，无台词镜头不得连续出现
- **相邻镜头景别必须变化**：避免连续同景别导致节奏拖沓

如果模型返回的镜头数和要求不符，会带着更强约束重试一次；仍不符则明确报错，不静默截断。

---

### 阶段 4：逐镜头生产（循环）

每个镜头依次经过以下节点，其中的条件边决定是否返工。

#### 4.1 `generate_image_prompt` → DeepSeek

写英文关键帧提示词，硬要求：

- 角色外观锚点**逐字**嵌入，不得改写
- 明确写出 `vertical 9:16 portrait framing`
- 同时产出负面提示词

#### 4.2 `_anchor_reference` → 关键帧记忆

从第二个镜头开始，读取关键帧记忆里的**锚点图**（首镜关键帧），作为参考图传给生成模型。

这是抑制跨镜漂移的主动手段：实测开启后色彩一致性 0.92 → 0.98，人物脸、制服、色彩被锁住，但场景可以自由切换。

#### 4.3 `generate_image` → Seedream 4.0

带参考图调用出图，保存为 1080×1920 PNG。

#### 4.4 `evaluate_image` → 文件级质检

核验：文件存在、可解码、分辨率正确、画幅比例正确、体积合理、提示词含锚点。

条件边 `should_retry_image`：

```
通过         → generate_video_prompt
不通过且有余量 → rewrite_image_prompt → 回到 generate_image
不通过且耗尽   → 放行，记录问题
mock 结果      → 直接放行（返工无法改变占位图）
```

#### 4.5 `generate_video_prompt` → DeepSeek

写英文运动描述，约束：只描述从单帧出发可行的连续运动、脸不能形变、不得出现切镜换景。

#### 4.6 `generate_video` → Seedance 1.0 pro-fast

关键帧以 base64 data URI 作为 `first_frame` 传入，异步任务轮询到完成再下载。带重试，但只对连接错误和 `429/5xx` 重试（不会产生双份计费）。

#### 4.7 `evaluate_video` → 文件级质检

核验：文件存在、可解码、画幅比例正确（±3%）、分辨率不低于 720p、时长是否接近目标。

注意这里判断的是**画幅比例**而不是精确像素——供应商的原生输出尺寸不同（Seedance 720p/9:16 给 704×1248），精确匹配会误判并触发无谓的付费重试。

#### 4.8 `evaluate_clip` → 七维评分 + 关键帧记忆

抽 8 帧做像素测量，按 Video-Bench 维度体系打分：

| 维度 | 权重 | 测什么 |
|---|---|---|
| imaging_quality | 0.15 | 锐度、对比度、曝光 |
| aesthetic_quality | 0.15 | 色彩饱和度、观感 |
| temporal_consistency | 0.20 | 闪烁、亮度抖动 |
| motion_effects | 0.15 | 运动幅度与连续性 |
| color_consistency | 0.10 | 与锚点的直方图余弦 |
| scene_consistency | 0.15 | 与锚点的整帧结构余弦 |
| object_consistency | 0.10 | 中心裁剪区域的主体一致性 |

**跨景别自动跳过**：景别映射为 特写/近景→close、中景/过肩→mid、全景/远景→wide 三组。当镜头与锚点不同组时，整帧结构本来就不可比，`scene_consistency` 标记为「不适用」不参与加权，但保留原始测量值供诊断。

评分完成后，抽出的关键帧写入记忆：

```
keyframes/
  shot_001.png       每个镜头的关键帧
  memory.json        锚点路径 + 每镜指纹与漂移量
```

首镜成为全片锚点，后续镜头的漂移量记录为 `drift_vs_anchor`。

#### 4.9 `synthesize_dialogue` → Seed-Audio 1.0

有台词的镜头送语音合成，返回 WAV。然后做**时长适配**：

台词长度由 LLM 决定，镜头时长由分镜决定，两者独立。实测一句 4.58 秒的台词配 3 秒镜头，直接合成会把语音砍断。

`fit_audio()` 的处理策略：

| 比例 | 动作 |
|---|---|
| 0.85 – 1.45 倍 | `atempo` 时间伸缩，保持自然语速 |
| > 1.45 倍 | 压到 1.45 倍后补静音 |
| < 0.85 倍 | 放慢到 0.85 倍后补齐 |

最后按字幕行数分配时间轴（按字数比例，跨镜头累加偏移）。

#### 4.10 `advance_shot`

推进到下一镜，或宣布所有镜头完成。

---

### 阶段 5：合成

**`compose_video`** → ffmpeg

```
每镜归一化（缩放到 1080×1920，统一 fps / 编码 / 采样率）
        ↓
concat 拼接
        ↓
ASS 字幕烧录
        ↓
封面生成（logline 烧录到首镜关键帧）
```

ASS 字幕使用 Noto Sans CJK，64px 白字黑边，底部居中留 260px 边距。

**`finish`**

汇总全片七维报告、LLM 用量、媒体调用成本，状态置为 `completed`。

---

## 三、Provider 架构

生成能力按名字从 `.env` 选择，代码里没有静默回退——名字写错直接报配置错误。

| 能力 | `mock` | 真实 |
|---|---|---|
| 图像 | PIL 绘制分镜卡 | `ark` / Seedream 4.0 |
| 视频 | ffmpeg zoompan 推拉 | `ark` / Seedance 1.0 |
| 语音 | 静音 AAC 轨 | `doubao_speech` / Seed-Audio 1.0 |

**mock 不是空壳**：图像是真实 PNG，视频是真实 H.264，音频是真实 AAC。整条合成、字幕、封面链路都真实执行，只是创意内容为合成品。用于零成本验证流程完好。

**付费保护**：Ark 按次计费，`ARK_ALLOW_PAID=1` 未设置时 provider 一律拒绝调用，即使 key 已配置。误跑一整部多镜短剧不会产生账单。

---

## 四、三套独立的鉴权

这是最容易踩的坑，三个服务互不通用：

| 服务 | 域名 | 鉴权方式 |
|---|---|---|
| DeepSeek | `api.deepseek.com` | `Authorization: Bearer <key>` |
| 火山方舟 Ark | `ark.cn-beijing.volces.com/api/v3` | `Authorization: Bearer <Ark key>` |
| 豆包语音 Speech | `openspeech.bytedance.com/api/v3` | `X-Api-Key: <Speech key>` |

方舟的 key 在豆包语音上会返回 401 `Invalid X-Api-Key`。

另外一个细节：豆包语音 `/tts/create` 的**模型名校验发生在鉴权之前**。模型名写错会返回 403 `fail to convert model to resource_id`，看起来像 key 问题，实际是模型名不对。正确模型名是 `seed-audio-1.0`，请求体字段是 `text_prompt` 而不是 `text`。

---

## 五、使用方法

### 5.1 首次安装

```bash
# 系统依赖
apt-get install -y ffmpeg fonts-noto-cjk

# Python 依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 前端依赖
cd frontend && npm install && cd ..

# 配置
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY / ARK_API_KEY / TTS_API_KEY
```

### 5.2 环境自检（零成本）

```bash
./run_demo.sh check
```

输出会列出 LLM 配置、三个 provider 的当前选择、模型名、分辨率，以及 Ark 模型的开通状态。不会产生任何费用。

### 5.3 跑测试

```bash
./run_demo.sh test
```

28 项测试，覆盖全链路、关键帧回灌、跨景别跳过、并发原子性、provider 付费保护等。

### 5.4 零成本试跑

```bash
./run_demo.sh simulate
```

用 mock provider 跑一部 3 镜短剧，约 30 秒。验证的是流程完好性，不是创意质量。

### 5.5 真实生成

```bash
# 单次最小付费验证（确认接口可用）
ARK_ALLOW_PAID=1 .venv/bin/python scripts/ark_smoke.py --kind image

# 全真实跑一部
IMAGE_PROVIDER=ark VIDEO_PROVIDER=ark TTS_PROVIDER=doubao_speech \
  ARK_ALLOW_PAID=1 .venv/bin/python - <<'PY'
from backend.services import project_store
from agent.graph import run_agent
from agent.settings import reset_settings
reset_settings()

p = project_store.create_project("深夜便利店，店员在监控里看到慢半拍的自己",
                                 "电影感竖屏短剧", 3, genre="悬疑")
state = project_store.load_state(p["project_id"])
state["status"] = "running"
final = run_agent(state)
print(final["episode_report"]["score"])
print(final["final_video_path"])
PY
```

3 镜约 200 秒。

### 5.6 网页界面

```bash
./run_demo.sh serve          # 后端 :8000
cd frontend && npm run dev   # 前端 :5173
```

界面上可以：填创意和剧本、选题材、设镜头数、实时看 Agent 的 Thought / Action / Observation / Decision 日志、看分镜卡片和七维评分条、切换历史项目、播放成片。

### 5.7 产物位置

```
workspace/projects/<project_id>/
  project.json          项目元数据
  state.json            完整运行状态
  events.jsonl          Agent 事件流（前端增量轮询）
  bible.json            设定集
  shots/shot_001.json   分镜
  images/shot_001.png   关键帧
  videos/shot_001.mp4   片段
  audio/shot_001.wav    原始配音
  audio/shot_001_fit.m4a 适配后的配音
  keyframes/shot_001.png 关键帧记忆
  keyframes/memory.json  锚点与漂移记录
  subtitles/subtitles.ass 字幕
  final/final_video.mp4  成片
  final/cover.png        封面
```

---

## 六、关键配置项

| 变量 | 说明 | 默认 |
|---|---|---|
| `IMAGE_PROVIDER` / `VIDEO_PROVIDER` | `mock` 或 `ark` | `mock` |
| `TTS_PROVIDER` | `mock` 或 `doubao_speech` | `mock` |
| `ARK_ALLOW_PAID` | 设为 `1` 才允许真实付费调用 | 未设置 |
| `ARK_VIDEO_RESOLUTION` | 视频分辨率，影响成本 | `720p` |
| `VIDEO_WIDTH` / `VIDEO_HEIGHT` | 画布尺寸 | `1080` / `1920` |
| `IMAGE_RETRIES` / `VIDEO_RETRIES` | 返工预算 | `1` / `1` |
| `IMAGE_REFERENCE` | 是否启用关键帧回灌 | `1` |

---

## 七、成本与耗时参考

| 项 | 单次 | 说明 |
|---|---|---|
| Seedream 出图 | 8.1k output tokens | 每镜 1 次 |
| Seedance 出视频 | 约 35-40k tokens | 每镜 1 次，按时长计 |
| Seed-Audio 配音 | 约 16 秒延迟 | 每句台词 1 次 |
| DeepSeek | 每个镜头约 3-4 次调用 | 规划 + 提示词 + 偶尔修复 |

3 镜短剧：约 200 秒，3 次出图 + 3 次出视频 + 若干次配音。

---

## 八、已知边界

| 项 | 状态 | 说明 |
|---|---|---|
| 多角色音色 | 未实现 | 所有台词用同一音色 |
| 视频侧参考图 | 不可用 | Seedance 1.0 不支持 `reference_image`，需 2.0+ |
| 视觉语义评审 | 未接入 | 质检是像素测量，抓不住「脸画歪」 |
| 任务取消/恢复 | 未实现 | `BackgroundTasks` 启动后只能等 |
| BGM 混音 | 未接入 | `compose_video` 已支持 `bgm_path`，无调用方 |
| 转场实现 | 未实现 | 分镜有 `transition` 字段，合成时全部按硬切 |
