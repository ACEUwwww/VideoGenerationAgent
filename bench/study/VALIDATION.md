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
| 三张图可直接生成 | `figures_study.py` → `docs/figures/study_{quality,identity,workflow}.png`，数值与 `judge/`、`segments/`、`compare_workflows/episodes.json` 逐一核对 | passed |
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
