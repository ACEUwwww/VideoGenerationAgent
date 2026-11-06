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
| 视频 | ffmpeg zoompan 推拉 | `ark` / Seedance |
| 语音 | 静音 AAC 轨 | `doubao_speech` / Seed-Audio 1.0 |
| 视觉质检 | 关闭，退回像素测量 | `ark` / doubao-seed-2-1-pro |

mock 用于零成本验证流程完好性，产出的是真实 PNG / H.264 / AAC，只是创意内容为合成品。mock 结果带 `mock: true` 标记，条件边直接放行，不触发注定无效的返工。

**付费保护**：Ark 按次计费，`ARK_ALLOW_PAID=1` 未设置时 provider 一律拒绝调用，即使 key 已配置。误跑一整部多镜短剧不会产生账单。

---

## 四、鉴权

三个服务用三套互不通用的凭据（DeepSeek / 方舟 Ark 用 `Authorization: Bearer`，
豆包语音用 `X-Api-Key`）。域名、易踩的坑与配置项见
[README 第 7 节](README.md#7-配置)。

---

## 五、使用方法

### 5.1 首次安装

**Linux / macOS**

```bash
apt-get install -y ffmpeg fonts-noto-cjk     # macOS: brew install ffmpeg
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd frontend && npm install && cd ..
cp .env.example .env
```

**Windows**

```bat
winget install Gyan.FFmpeg
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cd frontend && npm install && cd ..
copy .env.example .env
```

Windows 上与平台相关的只有三处，代码本身完全一致：

| 项 | 处理方式 |
|---|---|
| 中文字体 | 自动探测 `C:\Windows\Fonts\msyh.ttc`（微软雅黑）；缺失时用 `CJK_FONT_PATH` 指定 |
| ASS 字幕字体族 | 按字体文件自动映射（msyh→Microsoft YaHei）；可用 `CJK_FONT_FAMILY` 覆盖 |
| 控制台窗口 | ffmpeg 子进程带 `CREATE_NO_WINDOW`，不闪黑框 |

**ffmpeg 必须包含 libass**，否则 `ass=` 字幕滤镜不可用。Gyan 的 `full` 构建包含它，`run_demo.bat check` 会实测确认。

两个平台的启动脚本只是入口：

| 用途 | Linux / macOS | Windows |
|---|---|---|
| 自检 | `./run_demo.sh check` | `run_demo.bat check` |
| 测试 | `./run_demo.sh test` | `run_demo.bat test` |
| 零成本试跑 | `./run_demo.sh simulate` | `run_demo.bat simulate` |
| 启动后端 | `./run_demo.sh serve` | `run_demo.bat serve` |

实际逻辑在 `scripts/doctor.py` 和 `scripts/simulate.py`，Python 侧共用同一份代码。

### 5.2 环境自检（零成本）

```bash
./run_demo.sh check        # Windows: run_demo.bat check
```

检查项：ffmpeg 与 libass、ffprobe、中文字体文件与字体族映射、三套凭据是否齐全、三个 provider 的当前选择、画布与分辨率。不会产生任何费用。

### 5.3 跑测试

```bash
./run_demo.sh test
```

56 项测试，覆盖全链路、关键帧回灌、跨景别跳过、转场与字幕时间轴、视觉 judge 单次调用、并发原子性、provider 付费保护等。

### 5.4 零成本试跑

```bash
./run_demo.sh simulate
```

用 mock provider 跑一部 3 镜短剧，约 30 秒，零成本验证整条编排链路
（节点流转、条件边、合成、字幕、封面）完好，不用于评估创意质量。

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

目录树见 [README 第 8 节](README.md#8-产物)。前端通过
`GET /api/projects/<id>/assets/<path>` 读取这些文件，支持 HTTP Range，
播放器可拖动进度。

---

## 六、配置、成本与边界

配置项清单、各模型的成本量级、已知能力边界都维护在 README，避免两处各写一遍：

- [配置](README.md#7-配置)
- [成本与耗时](README.md#10-成本与耗时)
- [已知边界](README.md#11-已知边界)

---

## 七、运行态与故障恢复

### 心跳与僵尸运行回收

一次真实生成会占用进程几分钟（Seedance 单次渲染 60-90 秒，3 镜合计约 200 秒）。
进程被杀、机器重启、或用户中断之后，项目会停在 `running` 状态。

`claim_run()` 的处理规则：

| 情况 | 结果 |
|---|---|
| 没有运行 | 认领成功，写入 `run_started_at` 与 `run_heartbeat_at` |
| 运行中且心跳新鲜 | 拒绝，返回 `already_running` |
| 运行中但心跳超过 600 秒未更新 | 判定为僵尸，允许重新认领 |
| 旧版本数据，完全没有心跳字段 | 允许重新认领 |

心跳由两部分维持：`agent_runner` 里一个 45 秒周期的守护线程，以及 LangGraph
节点边界的显式调用。单步耗时（一次 Seedance 渲染）远长于心跳间隔，所以必须有
定时器，不能只靠节点边界。

重新认领时会清掉上一次的 `error` 字段，避免失败信息残留在新一次运行里。

### 需要人工干预的情况

- **付费任务卡住**：Seedance 轮询有 900 秒上限，超时会抛错并把项目置为 `failed`。
  这类项目可以直接重新运行。
- **同项目重复提交**：`/run` 在锁内完成认领，并发请求只有一个会真正启动。
- **主动取消**：目前不支持。正在执行的生成任务无法中断，只能等它返回或超时。

### 状态文件构成

`state.json` 只保存需要展示和恢复的内容。关键帧的 256 维指纹存在
`keyframes/memory.json`，因为它是漂移计算用的中间数据，体量远大于展示需求
（3 镜时曾占 state 的 55%）。3 镜项目压缩后约 16KB。
