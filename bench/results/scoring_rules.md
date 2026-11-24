# 判分规则与公式（与代码逐行对应）

> 本文是 `mcp_servers/metrics.py` 与 `mcp_servers/videobench_eval.py` 的**规范说明**。
> 每个公式都标明实现位置，任何一处改动都必须同步改这里。
> 目的：让报告里出现的每一个分数都能被手工复算。

---

## 0. 三个层级的分数，不要混用

| 层级 | 函数 | 输入 | 输出 | 谁在用 |
| --- | --- | --- | --- | --- |
| 单镜像素分 | `score_dimensions` | 解码帧 | 7 维 0–1 | 流水线自动返工 |
| 单镜综合分 | `evaluate_clip` | 上述 + 可选参考帧 + 可选 VLM | 加权 0–1 | `PASS_SCORE` 判据 |
| 序列分 | `evaluate_sequence` | 全部关键帧 | 3 维 0–1 | 成片后复核 |

`bench/` 里的对照实验用的是**单镜像素分**（`imaging_quality` 等四维），不是综合分。
报告里引用分数时必须写明是哪一层。

---

## 1. 帧抽样与描述子

### 1.1 抽取（`metrics.extract_frames`）

```python
ffmpeg -vf "fps=4,scale=192:-2" -frames:v max(count*4, count)
```

先按 4 fps 抽，再把帧序列等间隔降采样到 `count=8` 帧。返回 `float32` RGB 数组。
**分辨率是 192 宽**——这是测量用的缩略图，不是给 VLM 看的图。

### 1.2 灰度

```python
gray = frame @ [0.299, 0.587, 0.114]
```

### 1.3 结构描述子 `_thumb`

```python
vector = resize(gray, (96, 170))          # BILINEAR
vector = vector - mean(vector)            # 去均值：亮度整体漂移不算结构变化
vector = vector / ||vector||
```

维度 96×170 = 16320，取前 256 维进 `keyframe_memory.fingerprint`。

### 1.4 颜色描述子 `_histogram`

```python
hist = concat([np.histogram(frame[..., c], bins=16, range=(0,255)) for c in R,G,B])
hist = hist / sum(hist)                   # L1 归一化
```

维度 48。

### 1.5 相似度 `_cosine`

```python
cos(a, b) = a·b / (||a||·||b||)   分母为 0 时返回 0.0
```

---

## 2. 原始测量 `measure(frames)`

令 `g_i` 为第 i 帧灰度图，`d_i = mean(|g_{i+1} − g_i|)` 为相邻帧差的均值。

| 字段 | 公式 |
| --- | --- |
| `sharpness` | `mean_i( LaplacianVar(g_i) )`，见下 |
| `contrast` | `mean_i( std(g_i) )` |
| `colorfulness` | Hasler–Süsstrunk，见下 |
| `brightness` | `mean_i( mean(g_i) )` |
| `brightness_jitter` | `std_i( mean(g_i) )` |
| `mean_frame_delta` | `mean(d_i)` |
| `delta_std` | `std(d_i)` |
| `flicker` | `max_i( |d_{i+1} − d_i| )`，即帧差的**二阶差分峰值** |

**LaplacianVar**（`_laplacian_var`，4 邻域核 `[[0,1,0],[1,-4,1],[0,1,0]]`）：

```
L(y,x) = g(y-1,x) + g(y+1,x) + g(y,x-1) + g(y,x+1) - 4·g(y,x)
sharpness_frame = mean( L ⊙ L )
clip sharpness  = mean_over_frames / max(H-2, 1)
```

**Colorfulness**（`_colorfulness`）：

```
rg = |R − G|,  yb = |0.5(R+G) − B|
C  = sqrt( std(rg)² + std(yb)² ) + 0.3·sqrt( mean(rg)² + mean(yb)² )
```

---

## 3. 两个映射函数

**线性压缩**，把原始值映射到 0..1，超出区间截断：

```
squash(v, lo, hi) = clamp( (v − lo) / (hi − lo), 0, 1 )
```

**区间打分**，区间内满分、两侧线性衰减：

```
band(v, low, ideal_low, ideal_high, high) =
    squash(v, low, ideal_low)                  if v < ideal_low
    1 − squash(v, ideal_high, high)            if v > ideal_high
    1                                          otherwise
```

用区间而不是单点，是因为按静态卡片校准的阈值会把正常运动判成抖动：实测真实
Seedance 片段帧间差 18.9、flicker 8.8，单点阈值下 `motion_effects` 得 0 分。

---

## 4. 七维分数（`score_dimensions`）

记 `sharp = squash(sharpness, 8, 90)`，`contrast = squash(contrast, 18, 62)`，
`colorful = squash(colorfulness, 12, 62)`，
`exposure = 1 − min(|brightness − 118| / 118, 1)`，
`motion = band(delta, 0.5, 8, 45, 70)`，
`smooth = 1 − squash(flicker, 12, 40)`，
`stable = 1 − squash(brightness_jitter, 6, 30)`。

| 维度 | 权重 | 公式 | 常数依据 |
| --- | ---: | --- | --- |
| `imaging_quality` | 0.15 | `mean(sharp, contrast, exposure)` | 1080×1920 清晰片段 sharpness 约 60–200 |
| `aesthetic_quality` | 0.15 | `mean(colorful, exposure, contrast)` | — |
| `temporal_consistency` | 0.20 | `mean(smooth, stable)` | 真实快动作 flicker 可到 ~10，频闪持续攀升 |
| `motion_effects` | 0.15 | `mean(motion, 1 − squash(delta_std, 8, 30))` | 2–6 近静止，8–45 正常，>70 退化为噪声 |
| `color_consistency` | 0.10 | `mean_i cos(hist(f_i), hist(ref))` | — |
| `scene_consistency` | 0.15 | `mean_i cos(thumb(f_i), thumb(ref))` | — |
| `object_consistency` | 0.10 | 见下 | — |

**`object_consistency` 是唯一有条件分支的维度**：

```
perfect_framing_comparable = mean( subject_consistency, subject_color_consistency )
                             where subject_* 用 center_crop(f, 0.5) 计算

otherwise                  = subject_color_consistency
```

即景别不同时，结构项不可比，只让颜色项独立承担。

**三个 `*_consistency` 的参考是谁**：全部与**全片首个通过质检的关键帧**比较，
不是与上一镜比较。链式比较会让误差沿镜头累积，这一点决定了长片是否会漂。

---

## 5. 加权与不适用（`evaluate_clip`）

```
overall = Σ(score_d · w_d) / Σ(w_d)      d ∈ {score_d ≠ None}
PASS_SCORE = 0.62
```

权重和为 1.00，但分母只累加**实际计分**的维度——所以一个维度被标为不适用时，
不是补 0，而是从分母里移除。

**景别可比性**（`FRAMING_GROUPS`）：

| 分组 | 景别标签 |
| --- | --- |
| `close` | 特写、近景 |
| `mid` | 中景、过肩 |
| `wide` | 全景、远景 |

两组都非空且不同 → `scene_consistency` 置 `applicable=False`、`score=None`。
依据：同一场景换景别时整帧结构相似度实测落在 0.05–0.28，那是构图变化，不是场景漂移。

---

## 6. VLM 判分的合并规则

### 6.1 单镜（4 维一次调用）

覆盖 `imaging_quality`、`aesthetic_quality`、`motion_effects`、`object_consistency`。
输入是**全分辨率关键帧**（1080×1920 原图），不是 192px 测量帧——后者看不出人脸形变。

映射与合并：

```
judge_score_0_1 = (s − 1) / 4         s ∈ {1..5}
final_score     = max(measured_score, judge_score_0_1)
```

**取 max，不覆盖**：原始分另存 `mllm_score_1to5`，理由存 `mllm_reason`。
judge 失败只记 `judge_error`，测量路径不受影响。

> 取 max 是一个有偏的选择：它让分数只升不降。报告引用含 VLM 的分数时必须说明
> 这一点，不能与纯测量分直接比大小。

### 6.2 序列（3 维一次调用）

`story_coherence`、`continuity`、`pacing`，各 1–5 分，同样映射为 `(s−1)/4`：

```
sequence_score = mean( (s_d − 1)/4 )     d ∈ 实际返回的三个维度
```

与单镜不同，这里**不做 max 合并**，因为序列级没有像素测量可合并。
无 judge 时返回 `configured=False`、`score=None`，绝不编造数字。

---

## 7. 复算检查表

给定一条片段和一个参考关键帧，按顺序可完全复算：

1. `extract_frames(192 宽, 8 帧)`
2. `measure` → 8 个原始量
3. `compare_to_reference` → 5 个相似度
4. `score_dimensions` → 7 个 0–1 分
5. 按第 5 节加权 → 综合分

任何一项对不上，说明代码与本文已经漂移，属于必须修复的缺陷。
