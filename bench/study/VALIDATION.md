# 分层对比研究：验证记录

一条结论一个状态。没有证据的不写「已通过」。状态用：passed / failed / pending / partial / unavailable。

## 验证身份

| 项 | 值 |
| --- | --- |
| 仓库 | `/mdata/shiweisong/VideoGenerationAgent`，分支 `main` |
| 未提交改动 | `bench/backends.py`（OpenMontage 分辨率改读 `ARK_VIDEO_RESOLUTION`），以及 `agent/` 下此前的节奏/锚点改动 |
| 新增文件 | `bench/study/`（本目录，全部为新文件） |
| 基线 commit | `9034236` |
| 裁判 | `qwen2.5-vl-32b-local` @ `127.0.0.1:8791`（本地，不按次计费） |
| 硬件 | 1× NVIDIA B300 SXM6 AC，275 GB；torch 2.11.0a0 / CUDA 13.1 |
| 出片模型（付费侧） | `doubao-seedance-1-0-pro-fast-251015`，480p，9:16 |
| 出片模型（本地基线） | CogVideoX-5B、Wan2.1-T2V-1.3B、LongLive-1.3B（本地权重） |

## 已通过

| 项 | 证据 | 状态 |
| --- | --- | --- |
| 改动后既有 bench 测试未回归 | `pytest tests/test_bench.py -q` → **56 passed**（2026-11-17） | passed |
| 全量测试未回归 | `pytest tests -q` → 114 passed / 7 skipped（2026-11-17，275s） | passed |
| 分辨率对齐修复生效 | `bench/backends.py` 改读 `ARK_VIDEO_RESOLUTION`；两侧同为 480p 口径 | passed |
| OpenMontage 完整 workflow 能跑通并出片 | create→script→confirm→11 实体图→storyboard→15 镜→edit→export；成片 1080×1920/30fps/**60.0s**/15 镜/63.2MB | passed |
| 两侧同底模 | OpenMontage 每条 shot 的 `videos[].model` 均为 `doubao-seedance-1-0-pro-fast-251015`，与本项目一致 | passed |
| OpenMontage 成片身份稳定 | 镜头内 identity：min 0.951 / median 0.993 / mean 0.992，15 镜全部 ≥0.95 | passed |
| 本项目成片身份度量已排除片头卡假象 | 修复前 `identity_min` 0.369（shot 0 三窗全 0.37）→ 修复后 **0.588**，与 README 11.8 记录一致 | passed |
| 切镜实测而非按分镜表栅格 | 本项目 8 镜成片实测切点 4.8/8.83/12.83/16.87/20.87/24.9/28.7；OpenMontage 为 4.0/8.0/…/56.0 | passed |
| 分层 prompt 集可复现 | `select_prompts.py` 固定索引，重复运行集合一致（12 条：4/2/2/2/2） | passed |
| 本地基线矩阵已出齐 | `bench/study/gen_local.sh` → **180/180** 条（3 模型 × 12 prompt × 5 seed），全部 480×720、5.06–5.13s | passed |
| 本地基线可分段 | `segment_study.py --segment-seconds 1.0` → 180 条 × 5 窗 = 900 窗；每模型 identity 中位 CogVideoX 0.921 / Wan2.1 0.981 / LongLive 0.958 | passed |
| 四维裁判全部完成 | 240/240 组，每维 60 组 × 3 模型：成像 1.07/4.02/4.72、美学 1.32/3.90/4.25、时序 2.38/4.58/4.82、运动 1.50/2.87/3.37（CogVideoX / Wan2.1 / LongLive） | passed |
| 维度排序在模型间可分 | 四维一致排序 LongLive > Wan2.1 > CogVideoX；模型内极差（≤4 分）与模型间差（≥2.5 分）相比，排序可信。运动效果是最弱一维，三模型均未过 4 | passed |
| 三张图可直接生成 | `figures_study.py` → `docs/figures/study_{quality,identity,workflow}.png`，数值与 `judge/`、`segments/`、`compare_workflows/episodes.json` 逐一核对 | **已删除**（2026-11-19）：这些图把 workflow 与裸模型并列，口径不成立，见下节 |
| 身份指标已拆成「色彩 / 构图」两个可解释分量 | 逐镜 1s 切窗对比首窗：一条从背影全景推到正面特写的镜头，人物与服装未变，色彩 0.885 / 构图 0.694；另一条色彩 0.901 / 构图 −0.071。**结论：指纹的 `min(color, structure)` 测的是构图与机位，不是角色身份**，报告已按此改写措辞 | passed |
| 跨镜角色一致性有独立指标 | `character_consistency.py`：把同集各镜画面交给本地 Qwen2.5-VL 判定是否为同一人（脸/发型/体型/服装），明确排除机位、背景、打光。同一集重复问两遍，评分完全一致（3/3、4/4、5/5、4/4），说明问题稳定而非噪声 | passed |

## 本项目的失败样本（必须写进报告）

| 事件 | 证据 | 影响 |
| --- | --- | --- |
| 第 3 条成片因内容审核整集失败 | brief 0 / repeat 2，跑 1135s 后 `Ark video task cgt-20261117185942-mglj9 failed: OutputVideoSensitiveContentDetected`。当时的 `agent/` 与 `mcp_servers/` 中**没有**对该错误码的处理分支，失败直接冒泡到整集 | 本项目 9 条计划里已有 1 条作废，且该条的已付费渲染全部丢弃；OpenMontage 侧尚未出现同类事件（样本仅 1 条，不能据此说它更稳）。**已修复**（见下条） |
| **Ark 账号欠费，付费侧整体中断** | 2026-11-17 19:47 起，本项目批次连续 5 条在开工 29s 内返回 `Ark HTTP 403 {"code":"AccountOverdueError"}`（brief1 rep1/rep2、brief2 rep0/rep1/rep2） | **不是质量或编排问题，是账户余额问题**。本项目有效样本停在 3/9；OpenMontage 在同一账号下，剩余镜头预计受同一限制。需充值后才能补齐；本轮结论只能基于已有的 3 对 2 |
| 判定欠费发生的时点 | OpenMontage 已完成的两条任务时间戳为 19:44:58 / 19:45:39，早于欠费；本项目失败从 19:47 起。因此"OpenMontage 成功、本项目失败"的差异来自**时间先后**，不能读成两套 workflow 的可靠性差异 | 报告不得据此比较两侧成功率 |

这条不粉饰：单镜审核失败会毁掉整集，是本项目编排层的真实缺陷；欠费则是外部阻塞，与编排无关。两者性质不同，报告里分开写。本轮**不能**用成功率比较两个 workflow。

### 修复：审核失败隔离到单镜（2026-11-18）

`ark_media.py` 把 `SensitiveContentDetected` 识别为 `ContentModerationError`，`call_generate_video`
捕获后把该镜标 `video_skipped`，`clip_was_generated` 让状态机直接跳到下一镜，绕开质量评估
（没有像素可评）与配音（没有画面可配）。`compose_video` 本就丢弃没有 `video_path` 的镜头，
因此成片少一镜，而不是整集不产出。

不重试是刻意的：输出审核按像素判定，同一请求重发会再计费再被拒。

覆盖：`tests/test_pipeline.py::test_a_moderated_clip_is_skipped_instead_of_killing_the_episode`
（注入一条审核失败，断言其余镜头全部保留、成片仍产出）与
`test_the_skip_route_bypasses_qa_and_keyframe_extraction`。全量 `pytest tests` → 116 passed / 7 skipped。

**本轮数据早于此修复**，报告里记录的失败仍按修复前的行为解读，不能用修复后的能力去解释旧结果。

## 已定位的供应商缺陷（在 bench 层绕开，未改 vendored 源码）

| 缺陷 | 证据 | 状态 |
| --- | --- | --- |
| `auto-produce` 在分镜步骤丢自己的 job 记录 | `GET .../auto-produce` 返回 `job: null`，state 文件 `project.autoProduce` 消失；同时旧 state 回写，日志报「素材图达标 10/10」而实体只剩 1 个 `selectedImageId` | 已定位，改为逐阶段驱动 |
| 原 `run_pipeline.py` 缺定稿步骤 | 2026-11-16 日志：`5_storyboard → HTTP 500 请先定稿本集再拆分镜`，其后 6_voice `Edge 配音失败` | 已定位，新驱动先 `PATCH episodes status=confirmed` |
| 单集规划规模远超本项目 | 实测一次 brief 规划 **56 镜 / 176 秒**（本项目 15 镜 / 60 秒） | 已定位，按 `shotOrder` 裁剪并记录 |
| 出片轮询卡在 `generating` | 任务侧已 `succeeded`，本地仍 `generating`；用 `sync-video` 补齐 15/15。**seed 1 再次复现**：状态文件停滞 22 分钟，10 镜全部卡在 `generating`，而 Ark 侧任务早已 `succeeded`；另 1 镜（shot 3）从未提交（`providerTaskId` 为空）。`sync-video` 后 14/15，补交后 15/15 | 已定位，已恢复；**这是可复现缺陷，不是偶发** |
| 单集规划规模不稳 | seed 0 规划 15 镜/66 秒；seed 1 规划 13 个 3 秒镜头 = 48 秒。两侧"同为 15 镜"的口径在时长上并不可靠 | 已定位，逐条记录实际时长，不假设等长 |
| 剪辑草稿不随新镜头回填 | 15 镜成片首次只导出 **4 段** | 已定位，`action=save` 重建时间线后 15 段 |
| 实体图质检会阻断出片 | `entity_qa_blocked`：角色「林哲」wardrobe category 不符 | 已定位，用 `forceEntityQa`（studio 自带的强制开关） |
| `batch-video` 逐镜吞异常仍回 HTTP 200 | 第 2 条（`benchs0-2`）15 镜里 **2 镜从未提交到 Ark**：shot 7 `pending` 且 `providerTaskId=null`，shot 14 `generating` 且 `providerTaskId=null`。响应体 `started[]` 里带了错误，但 HTTP 状态是 200，驱动按状态码判定就漏了 | 已定位；`om_full.py` 改为检查 `started[]` 的 `error` 字段。补交后 15/15 |
| studio 成片质检会丢弃已付费的渲染 | 补交的 shot 14：Ark 任务 `cgt-20261117182015-759rk` 状态 **succeeded**、视频可下载，但 studio 的 `video_qa` 判定「质检未达标」标 `failed`，随后自动重试生成第二条并选中 | 已定位，属编排层行为；报告需记录「Ark 成功 ≠ 流程采纳」 |

## Tier 3a/3b 阶段 0：三个未知量（2026-11-18，零费用）

动代码前先测这三项，因为它们决定矩阵能不能做。

### 1. 本地权重的实际可用性

| 模型 | 本地路径 | 状态 |
| --- | --- | --- |
| CogVideoX-5B（T2V） | `bench/models/hf/zai-org/CogVideoX-5b` | 可用 |
| Wan2.1-T2V-1.3B | `bench/models/hf/Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 可用 |
| LongLive-1.3B | `bench/models/longlive`（Wan2.1 底座 + LoRA） | 可用 |
| **CogVideoX-5B-I2V** | 本地**不存在** | **需下载 21.6 GB，未获准，Tier 3b 因此暂不可执行** |

`zai-org/CogVideoX-5b-I2V` 在 HuggingFace 上存在且结构与 5B 一致（22 个文件，text_encoder 9.5 GB + transformer 11.3 GB + vae 0.86 GB）。`bench/fetch_models.sh` 是并行分段 curl，实测约 8 MB/s，下载预计 45–60 分钟。**本轮不下载**，等你点头。

### 2. 三模型能否共用同一分辨率 —— 可以，但只有一个方向

原生档位（读各自 `transformer/config.json` 与 `vae/config.json`）：

| 模型 | 原生输出 | 约束来源 |
| --- | --- | --- |
| CogVideoX-5B | 480×720 竖（= latent 60×90 × patch 2…VAE 8×） | `sample_height=60, sample_width=90` |
| Wan2.1-T2V-1.3B | 480×832 竖 | Wan 公开的竖版桶 |
| LongLive-1.3B | **832×480 横** | `inference.py` 里 latent 噪声写死 `[N, 16, 60, 104]`，`config.height/width` **从不被读取**（grep 计数 0） |

LongLive 的 `[16, 60, 104]` 是 480×832 的**横版**（60×8=480 高，104×8=832 宽）。它不是"支持 832×480 与 480×832"，是只有这一个形状；要改就得动 `bench/_vendor/`，而本轮禁止。

**实测探针**（`bench/.venv`，20 步，非生产的 40 步，单条 5s 量级）：

| 测试 | 结果 |
| --- | --- |
| CogVideoX @ 480×832 竖 | OK，22 s |
| CogVideoX @ 832×480 横 | OK，22 s |
| Wan2.1 @ 480×720 竖 | OK，12 s |
| Wan2.1 @ 832×480 横 | OK，14 s |

**结论**：三模型唯一能共用的档位是 **832×480 横版**——CogVideoX 与 Wan 都能出这个尺寸（已实测），LongLive 只能出这个尺寸（写死）。竖版 480×832 做不到，因为 LongLive 改不了。

这产生一个必须记录的配置差异：**Tier 3a 是横版 832×480，而已付费的 Tier 1 是竖版 480p。** 两层的分辨率口径不同，结论不能横向相加，也不能把 Tier 3a 的分数当作 Tier 1 的补充样本。

### 3. 判分吞吐（单锁串行，本地免费）

| 粒度 | 实测 | 说明 |
| --- | --- | --- |
| 片段级四维判分 | **31.7–53.5 s/组**（n=60 每组） | `motion_effects` 38.0 / `temporal_consistency` 31.7 / `imaging` 44.7 / `aesthetic` 53.5 |
| 成片级单维、单侧 | **139.9 s**（帧采样 9.1 s + 判分 130.8 s） | 刚实测：60 s 成片一条 |
| 成片级单维、两侧 | 229–580 s | 从 `episode_video_text.json` 读出的历史值 |

**本地渲染耗时**（从 180 条已有产物的文件时间戳推，不是生产口径，仅供排期）：

| 模型 | 相邻产物间隔中位 | 单集 12 镜估算 |
| --- | --- | --- |
| CogVideoX-5B | ~37 s | ~7.5 min |
| Wan2.1-T2V-1.3B | ~27 s | ~5.4 min |
| LongLive-1.3B | ~116 s | ~23 min |

据此，Tier 3a 每格 n=5 的估算：渲染 (7.5+5.4+23)×2×5 ≈ **6 h**；成片级判分 5 维 × 3 模型 × 5 对 × 约 350 s ≈ **7 h**。合计 **13–15 h 串行**，n=5 是可行上限，n=8 约 22–24 h。

**这三项已测完。分辨率一项需要你在开工前确认：接受 832×480 横版作为 Tier 3a 的统一口径，还是就此停住。**

## Tier 3a/3b 实现记录（2026-11-18）

### 本地 provider

`mcp_servers/providers/local_media.py`，`VIDEO_PROVIDER=local` + `VIDEO_MODEL=<名>` 选定。
权重从 `bench/models/hf/` 读，加载时不联网。

| `VIDEO_MODEL` | 状态 | 帧角色声明 | 实测单条 |
| --- | --- | --- | --- |
| `cogvideox5b` | passed | 空（T2V） | 832×480、41 帧、5.125 s、66.0 s |
| `wan21_1_3b` | passed | 空（T2V） | 832×480、81 帧、5.0625 s、54.7 s |
| `longlive_1_3b` | pending | 空（T2V） | 未在本轮实测（子进程路径已接） |
| `cogvideox5b_i2v` | **pending（权重未到位）** | `{first_frame}` | 未实测 |

单条渲染耗时是本层的真实成本，已记入元数据 `elapsed_s`。`usage` 为空字典、`billed: false`——不是"零 token"，是本层不计费。

### 编排层：能力查询取代模型名匹配

`VideoProvider` 契约新增 `frame_roles: frozenset[str]`。编排层读声明，不再匹配模型 id：

| provider | 声明 |
| --- | --- |
| ark（1.0 pro / 2.x） | `{first_frame, last_frame}` |
| ark（pro-fast） | `{first_frame}` |
| kling | `{first_frame}` |
| local（三个 T2V） | `∅` |
| local（I2V） | `{first_frame}` |
| mock | `{first_frame, last_frame}` |

`call_generate_video` 按声明分三支：空 → `join="none"` 且首帧为 None；有 `last_frame` → 编排关键帧；仅 `first_frame` → 尾帧接头。

### T2V 跳过整个图像阶段

`plan_shots` 后由 `needs_image_phase` 分流。T2V 下直接进 `start_video_phase`，不产出任何关键帧、不写 `image_prompt`。因此**所有依赖关键帧的指标在 Tier 3a 标为不适用**，不进聚合。

顺带修掉两个只有走到这条路径才暴露的缺口：

| 问题 | 证据 | 状态 |
| --- | --- | --- |
| `make_cover` 硬依赖 `images/shot_001.png` | T2V 下开跑第 76 秒抛 `FileNotFoundError` | passed——改为回落到首镜成片抽帧 |
| `tools.generate_video` 把空 `image_path` 当 Path 拼接 | `TypeError: PosixPath / NoneType` | passed——空串转 `None` |

单镜时长：本地模型固定约 5 s，`shot_window()` 在该层把窗口收成单值，分镜的「至少一个 8 秒长镜」「不得三连同长」两条硬约束在固定时长下自动停用，因为渲染器无法兑现。这是**配置差异**，不是放宽标准。

### OpenMontage 侧同模型接线

`bench/backends.py` 新增 `OpenMontageLocalBackend`（`openmontage_local`），调用 vendored 的
`generate_local_video`，但把变体表覆盖为我们自己的权重的绝对路径与 832×480。

**不硬编码**：形状与权重复用 `local_media.LOCAL_WIDTH/HEIGHT` 与 `MODELS[...]["path"]`，两侧由一个常量派生，不可能各自漂走。

实测：832×480、8 fps、5.125 s、57 s。与我们的 CogVideoX 侧同尺寸同帧率同帧数。

**范围**：这条路径仍只到 shot 阶段（prompt in、clip out），不含 storyboard/关键帧层——与 `openmontage_shot` 同一范围，只有底模不同。

### 测试

| 项 | 结果 |
| --- | --- |
| 全量 `pytest tests -q` | **128 passed / 7 skipped**（372 s） |
| 三 T2V 模型声明空帧角色 | parametrize 三例，passed |
| I2V 声明首帧角色 | passed |
| 各 provider 都有 `frame_roles` | passed |
| 各模型帧数契约（4k+1 / fps） | parametrize 两例，passed |
| T2V 跳过图像阶段 | passed——断言无 `image_path`、`image_prompt`，且 join 均为 `none` |
| 单镜 / 双镜 / 五镜边界 | parametrize 三例，passed |

### 已知失败与阻塞

| 项 | 状态 |
| --- | --- |
| I2V 权重下载 | **failed 两次**。`fetch_models.sh` 的并行分段 curl 在 4.7 GB 的 T5 分片上 `CHUNK FAILED`，两次都被删除。改用顺序 `curl -C -` 续传后速度约 500 KB/s，已下 449 MB / 21.6 GB，预计十余小时。**Tier 3b 因此 pending，不是 failed。** |
| `longlive_1_3b` 走本地 provider | pending——子进程路径已写，未实测 |

## 口径修正：删除把 workflow 与 model 并列的旧图（2026-11-19）

用户指出：本项目交付的是**工作流**，而 CogVideoX / Wan / LongLive 是**生成模型**，一次调用出一条片段。把两者放进同一张分数图并列，比的是两个不同层面的东西，差异无法归因到任何一方。

受影响的产物已删除：

| 删除项 | 问题 |
| --- | --- |
| `docs/figures/quality.png` | 同一张柱状图并列 `ours`、`openmontage_shot` 与三个裸模型 |
| `docs/figures/identity.png` | 同上 |
| `docs/figures/longform.png` | 把本项目 8 镜成片与 LongLive 30 秒单镜并列 |
| `docs/figures/capability.png` | 把工作流的交付能力当成模型的交付能力同表比较 |
| `docs/figures/study_{quality,identity,workflow}.png` | 同上，且与 `figures_study.py` 绑定 |
| `bench/results/compare/report.md` | 上述图的配套报告 |
| `bench/results/study/report_study.md` | 同上 |
| `bench/results/compare_workflows/report_workflows.md` | 同上 |

**保留下来的对比**（`tier3a_*`）不受影响，因为它们全部是同类对象之间的比较：

| 图 | 比较的是 | 是否同类 |
| --- | --- | --- |
| `tier3a_consistency.png` | 有状态传递 ↔ 无状态传递（同底模、同种子） | 是 |
| `tier3a_clip.png` | 三种提示词来源（同底模、同种子） | 是 |
| `tier3a_workflow.png` | 两组各自交付的事实 | 是 |

模型在本项目中只作为**实验介质**：固定它们，才能单独测出编排层的贡献。它们不是竞争对手。README §11.6 已写入这条口径。

三张新图的全部文字已改为英文。

## Tier 3a 正式报告（2026-11-19）

报告：`bench/results/tier3a_report.md`，图表 `docs/figures/tier3a_{consistency,clip,workflow}.png`。

### 两个结论，缺一不可

| 结论 | 数值 | 样本 | 显著性 | 状态 |
| --- | --- | --- | --- | --- |
| 跨镜角色一致性 | 2.00 → 4.17（+2.17，5 分制） | 6 组 × 5 镜（2 模型 × 3 brief） | 符号检验单侧 p=0.031 | passed |
| 单镜画质四维均值 | 4.175（brief）/ 3.675（完整规划）/ 3.750（截短消融） | 8 格（2 模型 × 4 维） | 单格 p≥0.33，brief 胜 6/8 | passed（**反面结果**） |

截短消融只补回 0.075，因此单镜落后**不是提示词长度造成的**，该解释已排除。

### 数字核对

报告内 11 项数值已用脚本对 `character_consistency.json`、`tier3a_ab/judge/*.json`、两份 `clips.jsonl` 逐一复核，全部一致。图表由 `tier3a_figures.py` 从这些文件读出，不硬编码。

### 新增/修改的代码

| 文件 | 作用 |
| --- | --- |
| `mcp_servers/providers/local_media.py` | 本地视频 provider（3 个 T2V + 1 个 I2V），声明 `frame_roles` |
| `mcp_servers/providers/base.py` | `VideoProvider.frame_roles`；`VideoRequest.image_path` 可为 None |
| `agent/nodes/video_nodes.py` | 按能力分派三条路径；T2V 走 `T2V_PROMPT_SYSTEM` |
| `agent/nodes/planner.py` | `shot_window()`：本地模型固定时长 |
| `agent/graph.py` | `needs_image_phase`：T2V 跳过整个图像阶段 |
| `mcp_servers/video_tools_server.py` | `make_cover` 无关键帧时回落到首镜抽帧 |
| `bench/backends.py` | `OpenMontageLocalBackend` |
| `bench/study/tier3a{,_ab,_story,_figures}.py` | 渲染、A/B、主实验、图表 |

回归：`pytest tests -q` → **128 passed / 7 skipped**。

### 未完成路径（如实记录）

| 路径 | 阻塞 | 状态 |
| --- | --- | --- |
| CogVideoX-5B-I2V | diffusers 对该 checkpoint 强制原生分辨率，`ValueError: ... different resolution that the defaults`；与 832×480 共用档位冲突 | unavailable（权重已备，代码已接） |
| OpenMontage 完整工作流（本地） | 需 Postgres/Redis/MinIO，且派发层不认本地 video tool；接入须改 vendor（本轮禁止） | unavailable |
| LongLive 的 OpenMontage 侧 | OpenMontage 无 LongLive 变体 | unavailable |
| 真实 TTS | 本轮 `TTS_PROVIDER=mock`，音频为静音占位轨 | untested |
| 9:16 竖版本地渲染 | 本层统一 832×480 横版 | untested |

### 数据来源对应

| 报告章节 | 原始产物 |
| --- | --- |
| §2 主结果 | `bench/results/tier3a_story/{clips.jsonl,character_consistency.json}` |
| §3 反面结果 | `bench/results/tier3a_ab/{clips.jsonl,judge/*.json}` |
| §4 交付事实 | `bench/results/tier3a_story/clips.jsonl` 的 prompt 长度与耗时字段 |

## 本轮（100 元充值后）的最终状态

| 项 | 状态 |
| --- | --- |
| 12 prompt × 5 seed × 3 本地基线（180 条） | **passed**（已完成） |
| 四维裁判打分（成像/美学/时序/运动） | **passed** — 240/240 组，每维 60 组 × 3 模型 |
| 付费成片对比（brief1，n=3 对 3） | **passed** — 本项目 rep0/1/2 vs OpenMontage seed0/1/2，同一 brief、同一底模、同一 480p 口径 |
| 跨镜角色一致性（VLM，1-5） | **passed** — 每集重复问两遍结果一致 |
| 视频-文本 5 维（本地基线，片段级） | partial — 第 1 维进行中，其余待跑；全部免费本地 |
| 视频-文本 5 维（成片级，评交付片是否兑现 brief） | partial — 刚启动，6 条成片 × 5 维 |
| OpenMontage 音频 | unavailable（edge-tts 在该环境未成功，报告按「无音轨」记录） |

### 本轮实际花销

| 项 | CNY |
| --- | ---: |
| 本项目 `cb6675e5`（brief1 rep1） | 27.74 |
| 本项目 `3a6dcd2d`（brief1 rep2） | 27.74 |
| OpenMontage seed2 补跑（按秒估算） | 22.32 |
| **合计 / 上限** | **77.80 / 100.00** |

余额 22.20，不足以再跑一条完整 episode（≈27.74）。因此 brief1 的 3 对 3 就是本预算下的终局。

### 本轮修掉的缺陷（每条都曾产出过一个可信但错误的数）

| 缺陷 | 后果 | 修复 |
| --- | --- | --- |
| 场景检测误检切点 | 64s/15 镜被当成 3 镜，identity 中位 −0.157 | 声明边界优先 + 逐点像素核验 |
| 合成片当测量单位 | 分镜 4.0s vs 实际 4.0417s + 交叉淡入，第 10 镜偏 0.4s | 改测源镜头文件（两侧规格完全一致） |
| `min(色彩,构图)` 当角色指标 | 运镜被误读成角色漂移 | 拆成色彩/构图两列并明确各自含义；角色交给 VLM |
| `batch-video` 吞异常返回 200 | 15 镜中 2 镜从未提交，却计入成功 | 检查 `started[]` 的 `error` |
| `--brief-index` 重编号 | 两条不同 brief 撞号，配对错误 | 切分前保留原始索引；历史行按 `project.json` 校正 |
| 预算脚本写死 episode id | 漏算 rep2 全部花费（20.47 元） | 改按目录 mtime 判定本轮范围 |
| 聚合表混入不同 brief | n 虚高至 6，结论被稀释 | `pair_by_brief` 只保留两侧都跑过的 brief |
| 交付镜数取自切点核验 | 15 镜报成 14 镜 | 改读源镜头文件数 |

## 不能声称的事

- 本轮 OpenMontage 侧只有 **1 条**成片，本项目侧也暂为 1 条；不足以支撑均值差异。
- 两侧时长口径**不同**（60s/15 镜 vs 32.7s/8 镜，匹配版 60s/15 镜在生成中）。
  时长不同的两条片不能直接比裁判分。
- 不把「有字幕 / 有配音 / 镜头更多」折算成分数再相加。三层刻度不同。
- 未做人工偏好对齐，与既有报告口径一致。
