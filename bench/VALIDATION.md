# Benchmark 验证记录

记录哪些结论已被实测证实、哪些还没验证。每一条都标明证据来源；没有证据的不写成已通过。

## 验证身份

| 项 | 值 |
|---|---|
| 仓库 | `/mdata/shiweisong/VideoGenerationAgent`，分支 `main` |
| 基线 commit | `f014cd7`（含未提交改动：节奏压缩、固定锚点、尾帧接首帧、删除片头卡） |
| 全量测试 | 106 passed, 2 skipped，2026-11-13（含 bench 41 项） |
| Video-Bench | vendored `d40b917c4238`（`bench/_vendor/`，不入库） |
| 裁判 | `doubao-seed-2-1-pro-260915` @ Ark，OpenAI-compatible |
| 硬件 | 1× NVIDIA B300 SXM6，275 GB；torch 2.11.0a0 / CUDA 13.1 |
| 生成侧 | `zai-org/CogVideoX-5b`、`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（本地权重） |

## 已验证

| 项 | 证据 | 状态 |
|---|---|---|
| 现有 pipeline 未回归 | `pytest tests -q` → 86 passed / 2 skipped（2026-11-12，308s） | passed |
| Ark 豆包可作裁判 | 单图 18.5s / 658 tokens，返回 `score: 4` | passed |
| 分数解析器覆盖三种回复格式 | `tests/test_bench.py` 6 个参数化用例 | passed |
| 报告分组不混算 1-5 与 1-3 | 合成数据跑通 `write_report` | passed |
| 抽帧载荷在 4 秒片段约 1.0 MB（640px） | 实测 `payload_bytes` | passed |
| CogVideoX 权重完整性 | `verify_models.sh` → 17 个文件 sha256 全部匹配 | passed |
| CogVideoX 产出真实画面 | 41 帧 / 480×720 / 1.03 MB，像素均值 113（非黑，无 NaN） | passed |
| CogVideoX 走 bench backend | 34s 含加载，产物可直接进判分 | passed |
| CogVideoX 两个 seed 均出真实画面 | seed 0/1 各 41 帧，均值 143.1 / 152.0，spread 10.7 / 10.3，无 NaN | passed |
| Wan2.1 权重完整性 | `verify_models.sh` → 19 个文件 sha256 全部匹配 | passed |
| Wan2.1 产出真实画面 | seed 0/1 各 81 帧 / 480×832 / 16fps / 5.06s；均值 98.7 / 124.1，帧间差 5.83 / 13.06，无 NaN | passed |
| 抽帧载荷在真实片段上可构造 | CogVideoX 4s 片段 → 11 帧 + 2 组拼图，0.26–0.58 MB | passed |
| 上传体积不是判分瓶颈 | 5.6 MB 请求体经代理打 ARK 仅 0.3s（无效 key，不计费） | passed |
| 流式调用可避开长静默断连 | 4/4 静态调用返回正文（12.3–61.6s），非流式同批仅 2/4 | passed |
| 同一片段重复判分一致（imaging_quality） | 两轮独立调用中重叠的 2 个片段分数完全相同（CogVideoX seed1 = 1/1，Wan seed0 = 3/3）；样本仅 2 条 | 弱证据 |
| 并行下载吞吐 | 12 路实测 8–10 MB/s；24 路反而降到 4.2 MB/s | passed |
| 权重内容完整性（修复前） | 5/5 文件 sha256 不匹配 → 全黑 NaN | failed（已修复，现已全绿） |

## 已定位并修复的缺陷

### 我们的下载器把分片写到错误偏移（已修）

`dd of="$out" bs=1M seek=$((s/1048576)) conv=notrunc` —— `bs=1M` 下 `seek` 以 1 MiB 块计并向下取整，除首块外每个分片都覆盖到错误位置。文件大小与远端一致，所以只看长度发现不了。

**可观察症状**：模型正常加载、正常跑完、输出**全黑**，像素为 NaN（`nan: True mean nan`），文本 embedding 全零（`mean 0.0000 std 0.0000`）。

**证据**：`dd bs=1M seek=999` 写入落在偏移 0；`oflag=seek_bytes` 落在 999。

**修复**：`bs=1M` + `oflag=seek_bytes`。校验脚本 `bench/verify_models.sh` 对 LFS 文件的 `x-linked-etag`（sha256）比对。

### tokenizer 词表缺失（已修）

下载过滤器只匹配 `.safetensors/.json/.txt`，漏掉 `tokenizer/spiece.model`，T5 无词表 → embedding 全零。

### transformers 5.x + sentencepiece 的 protobuf 版本冲突（已修）

sentencepiece 0.2.3 的 gencode 要求 protobuf 7.x，运行时是 6.33.5，加载失败后回退 tiktoken 解析器并崩在 `ValueError: Error parsing line`。venv 内升到 protobuf 7.36.2 后 T5Tokenizer 正常（vocab 32100）。

### diffusers 的 torchao 导入顺序 bug（绕开）

`quantizers/torchao/torchao_quantizer.py` 在模块级调用 `_update_torch_safe_globals()`（第 96 行），其异常分支引用第 109 行才定义的 `logger`。torchao 0.16 必然触发，导致 diffusers 0.33–0.36 无法导入任何 pipeline。留在 diffusers 0.42.1（已修）。

### 上游 Video-Bench 的缺陷（在 bench 层绕开，不改 vendored 源码）

| 缺陷 | 位置 | 后果 |
|---|---|---|
| `modelname` 赋值前被读取 | `VideoTextAlignment.py:327` | 5 个 video-text 维度全部 `UnboundLocalError` |
| 裁判模型名硬编码 5 处 | static/dynamic/VideoTextAlignment | 换 base_url 即 404 |
| `overall_consistency` 无路由 | `videobench/__init__.py` | 有 prompt 有 91 条标注题，调度器抛 `ValueError` |
| `video-text consistency` 键不存在 | `prompt_dict.py` | 按 README 维度名调用抛 `KeyError` |
| grid view 拼接块不带模型名 | `dynamicquality_gridview_customized.py` | 多模型同调无法归属分数 |
| 内容数组含裸字符串 | 各模块 | Ark 拒绝：`expected a string or array of objects` |
| 视频 prompt 从不传给裁判 | `VideoTextAlignment.py` | 一致性维度变成无意义打分 |

### 非流式判分在 153 秒处被断开（**已修**）

两轮独立实验各失败一次，且耗时几乎恒定：

```
pooled videos_0  OK   21.7s
pooled videos_1  FAIL 153.0s
fresh  videos_0  FAIL 153.8s
fresh  videos_1  OK   41.1s
```

新建连接与复用连接各挂一次，所以与 keep-alive 无关。非流式调用在裁判推理期间连接上没有任何字节流动（首 token 实测要等 10–58s），到点即断。改用流式后同一批 4 个调用全部返回。

**影响**：非流式下单维度 4 个片段有 2 个拿不到分数；这是判分可靠性的主要损失来源。

**修复**：`Judge.complete()` 改为 `stream=True` + `stream_options={"include_usage": True}`（Ark 支持，token 统计因此不丢）。

**验证**（修复后同批 4 片段 × 2 维度）：调用 8/8 均返回正文；`imaging_quality` 310.6s → **68.6s**，`aesthetic_quality` 656.1s → **144.4s**；出分 4/8 → **6/8**。

### 静态维度解析不到分数（**已修**）

`wan21_1_3b_1` 的原始回复：

```
- video: x ,score 2, because the video has prominent persistent motion blur ...
```

裁判保留了提示词模板里的字面 `x`，把真实分数写在后面的 `score 2` 子句里。`_loose_score()` 的四种形状都不匹配逗号结尾的 `score 2,`，于是返回 `None`——**实际分数是 2，只是没取到**。

根因之一在提示词：`_static_messages()` 把目标片段称为 `"the video you are evaluating"`，从不出现模型名（上游 `staticquality.py:65-70` 同此），解析器却按模型名匹配，只能退到 `_loose_score()` 兜底。

**修复**：`_loose_score()` 新增 `scores?(?![A-Za-z])\s*(?:is|:|=)?\s*["']?(\d{1,2})`，取**最后一处**匹配（结论在末尾），拒绝小数，保留 1-max 区间校验。修复后该片段解析为 **2**。

**仍存在的失败模式（未修）**：`aesthetic_quality` 上 `wan21_1_3b_1` 返回 `- video: x ,because the video has obvious severe flaws ...`——完全没给数字。请求里就没有数字，解析器无从救。根因是上游指令把分数槽位示范成字母 `x`。修法需要强化格式指令（改提示词语义），未在本次范围内。

### 下载失败会留下"看起来正确"的文件（已修）

`dl_ranges.sh` 用 `truncate -s "$size"` 预分配全尺寸再分片填充。任一 chunk 失败时脚本确实报 `CHUNK FAILED`，但**不删除输出**——留下一个长度正确、内部有零洞的文件。之后 `fetch_models.sh` 的 `[ -f "$target" ] && [ size 相等 ]` 判断会认为已下载完成，跳过重下。

这一轮实证了两个受害文件：

| 文件 | 大小 | 现象 |
|---|---|---|
| `models_t5_umt5-xxl-enc-bf16.pth` | 11.4 GB | `PytorchStreamReader failed reading zip archive: failed finding central directory` |
| `models/lora.pt` | 2.8 GB | 下载报 `CHUNK FAILED`，sha256 不匹配 |

**修复**：失败（chunk 失败、大小不符、sha256 不符）时 `rm -f "$out"`；新增可选第五参数传 sha256，下载后立即校验并在不符时删除。

**教训**：只比文件长度不能证明下载完整。这个坑在本项目已出现第二次（第一次是 `dd bs=1M seek` 的偏移错误，见上文）。

补充一条并发经验：`du -B1` 与 `stat -c%s` 的差值直接暴露"声明长度 vs 实际写入"。实测 `truncate` 预分配后，11.4 GB 的文件在只写入 2.2 GB 时两者的读数分别是 11.4 GB 和 2.2 GB——**用 `du` 检查比 `stat` 可靠**。

并发数也有反效果：同时跑 29 个 curl 分片时，T5 下载速率约 4 MB/s 且频繁 `CHUNK FAILED`；降到只剩它一个文件（9 片）后稳定在 12 MB/s。**大文件应串行下载，不要并行多个文件。**

## 长视频基线：LongLive 1.3B

接入方式：`bench/backends.py` 的 `LongLiveBackend`，走子进程调用 vendored 仓库（`inference.py` 在 import 时解析 argv 并硬编码路径，且其 requirements 锁 `diffusers==0.31.0`，与本 harness 的 0.42.1 冲突，不能在进程内导入）。

| 项 | 值 |
|---|---|
| 模型 | LongLive-1.3B（Wan2.1-T2V-1.3B + LoRA，`longlive_base.pt` + `lora.pt`） |
| 额外基础权重 | Wan 原生 `models_t5_umt5-xxl-enc-bf16.pth`（11.4GB）、`Wan2.1_VAE.pth`（508MB）、`diffusion_pytorch_model.safetensors`（5.7GB） |
| 输出 | 832×480 @ 16fps（横版，模型训练形状） |
| 渲染耗时 | 99–115s（含加载） |
| 许可 | Apache 2.0 |

### 已验证

| 项 | 证据 | 状态 |
|---|---|---|
| 全部权重 sha256 | 5 个文件逐个比对 hub 的 `x-linked-etag` 全部匹配 | passed |
| 后台可调用 | 经 `bench.runner` 生成成功，产物在两处（工作目录 + `videos/`） | passed |
| 产出真实画面 | 81 帧 / 832×480 / 16fps / 5.06s；均值 118.3，帧间差 6.31，无 NaN | passed |
| 画面内容正确 | 抽帧目视：人物背包沿小径前行，背景草地林荫，三帧连续推进 | passed |
| 时长换算 | 请求 5s → 实际 5.06s | passed |

### 已修：`num_output_frames` 是 latent 帧数，不是视频帧数

Wan VAE 时间维上采样 4 倍：`pixel_frames = 4 × (latent − 1) + 1`。实测 78 latent → 309 帧，完全吻合。

最初按视频帧数传参，导致**请求 5 秒实际产出 19.3 秒**。修正后按最近 3 的倍数取 latent，跨时长误差 ≤6%：

| 目标 | latent | 实际 |
|---:|---:|---:|
| 5s | 21 | 5.06s |
| 10s | 42 | 10.31s |
| 20s | 81 | 20.06s |

这与 CogVideoX 的 `4k+1` 规则是同一类问题：**每个模型的帧数语义不同，不能共用一套换算。**

## 未验证（阻塞中）

| 项 | 原因 |
|---|---|
| `ours` 后端真实出片 | 需要 `ARK_ALLOW_PAID=1`，会花 Seedance 钱，等用户批准 |
| 首次 9 维度真实打分 | 同上（裁判调用付费） |
| 多 seed 方差 | 冒烟规模不覆盖 |

## 下一步（需要用户批准付费）

```bash
bench/verify_models.sh zai-org/CogVideoX-5b
bench/verify_models.sh Wan-AI/Wan2.1-T2V-1.3B-Diffusers
# 单模型本地验证（不花钱）
bench/.venv/bin/python -m bench.runner --models cogvideox5b --stage generate --prompts 1 --seeds 0
# 付费对比（需要 ARK_ALLOW_PAID=1 BENCH_ALLOW_PAID=1）
```
