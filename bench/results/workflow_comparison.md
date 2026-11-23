# 三套工作流的对照：OpenMontage / DreamFactory / 本项目

> 写作口径：OpenMontage 与我们的部分**逐行读过源码**，DreamFactory 只有 arXiv
> 2408.11788 的论文（代码与数据集均未发布，摘要脚注写的是「接收后公开」），
> 所以它那一列的证据等级低于另外两列。
>
> 本地基模：`mcp_servers/providers/local_media.py` 里声明的 CogVideoX-5B、
> Wan2.1-1.3B、LongLive、CogVideoX-5B-I2V。

---

## 0. 先说这三样东西不是同一类对象

| | OpenMontage | DreamFactory | 本项目 |
| --- | --- | --- | --- |
| 是什么 | 可自部署的**产品级工坊** | 论文里的**方法** | 一个**图状状态机** |
| 代码可得 | 是（已 vendored） | 否 | — |
| 能否直接对比 | 能，已做过 shot 级对照 | **不能**，无实现 | — |
| 谁做一致性 | 参考资产（图） | 监控 Agent 写的文字 | 文字锚点 + 首镜锚点（图，视模型能力） |
| 成本控制 | 提案里写预算，$25 默认 | 未提及 | `ARK_ALLOW_PAID` 布尔闸门 + 用量记账 |
| 人在环 | 有，几乎每步一个 checkpoint | 对话式，但无闸门 | 无 |

把 DreamFactory 和 CogVideoX 画进同一张分数图，和把 workflow 与 model 并列是
同一类错误。它只能作为**方法论**参照。

---

## 1. OpenMontage：十段式生产线 + 每镜子图

清单在 `pipeline_defs/short-drama.yaml`，编排模式 `executive-producer`。

### 1.1 主链

`research → proposal → script → episode_breakdown → character_bible → storyboard → assets → edit → compose → publish`

十个串行阶段，每个阶段执行同一套协议
（`skills/pipelines/short-drama/executive-producer.md`）：

```
PREPARE → 跑该阶段的 director skill → REVIEW → GATE(pass / revise / send-back)
```

闸门参数写死在清单里：

| 参数 | 值 |
| --- | --- |
| `budget_default_usd` | 25.00 |
| `max_revisions_per_stage` | 3 |
| `max_send_backs` | 3 |
| `max_wall_time_minutes` | 180 |
| 人工审批默认 | 除 research / edit 外全部 `true` |

关键点：**send-back 是往回送**，不是就地重试。storyboard 不合格可以退回
script，这在单进程状态机里不容易做——它要求每一阶段的产物是可寻址的 artifact，
确实是（`research_brief`、`proposal_packet`、`script`、`character_bible`、
`storyboard`、`asset_manifest`、`shot_job_ledger`、`edit_decisions`）。

### 1.2 character_bible 是硬闸门

`character-bible-director.md` 的 Hard rule 原文：

> No character-bearing shot may proceed in assets without a bible entry + at
> least one reference image path (or explicit user waiver).

也就是说**一致性在进入出图之前就被强制**：没有角色参考图的镜头进不了 assets 阶段。
它的一致性不是生成后再纠，而是「先生成参考资产，再用参考资产生成镜头」。

配套的 `studio/server/entity_alpha.py` 负责把角色/道具**抠成透明底 PNG**
（"Punch seamless studio backdrops to PNG alpha for cast and prop cutouts"），
QA 侧还有 `character_empty_plate` 闸门专门拦「背景被画成风景」的空板。

### 1.3 每一镜是一个子图

`workflow_defs/subgraphs/keyframe-i2v-qa.workflow.json`，`max_reworks: 2`：

```
keyframe  image_selector(prompt=shot.keyframe_prompt, reference_images=bible_refs)
   ↓
route     subgraph_validate(video-router)
   ↓
i2v       video_selector(operation=image_to_video,
                         reference_image=keyframe.path,
                         prompt=shot.video_prompt)
   ↓
qa        drama_visual_qa(scope=shot, character_ids=shot.character_ids)
   ├─ critical_count > 0  → 回到 keyframe（最多 2 次）
   ├─ verdict == needs_review → human_gate
   └─ 否则通过
```

注意 i2v 那一步：`reference_image` 用的是**刚生成的关键帧**，而关键帧自己又是
用 bible 里的角色参考图生成的。所以它的像素身份信号是两跳传递：
`角色参考图 → 关键帧 → 视频首帧`。

assets 阶段另外三条硬规则（`asset-director.md`）：

- 禁止裸批量 `video_selector`（必须走 shot ledger）
- 暂停/恢复只通过 ledger
- 每次付费生成前先声明 provider

### 1.4 QA 形态

`studio/server/qa/strategies.py` 里是 VLM 策略，要求模型返回

```json
{"pass": boolean, "score": 0-1, "reason": string}
```

外加一串硬闸门（`qa/gates.py`，责任链式，按 `gate_ids` 组合）。

---

## 2. DreamFactory：对话式多 Agent + 关键帧迭代

证据全部来自论文正文与附录，**论文自身在 Agent 名单上是前后不一致的**，这里
如实记下来。

### 2.1 角色

§3.1 声明：CEO、movie director、film producer、screenwriter、filmmaker、
reviewer，外加 §1/§3.3 另起一个 Monitor。摘要又写成 "directors, art directors,
screenwriters, and artists"。

### 2.2 阶段

正文说「六个主要阶段」，接着只列了五个：

1. Task Definition
2. Style Decision
3. Story Prompting
4. Script Design
5. Key-frame Design

前四个是对话式的，第五个是视觉迭代。每个阶段有一份共享的 phase prompt
（角色、任务、目标结论、讨论格式、约束）+ 每个 Agent 一张 role card，最后必须
以 `<INFO>` + 结论收尾，结论作为本地记忆喂给后面的阶段。

### 2.3 Key Frames Iteration Design Method（§3.3）

```
Base:        painter P + director D + monitor M 循环，直到 script frame S1
             产出第一帧被接受 → Base Frame
Base Desc:   M 分析该帧，抽出风格/背景/角色特征 → B_D（长期记忆）
Iteration t: 生成器收到 p_t, d_t, S_t, C_{t-1}，并把 B_D 一并喂入
Extraction:  M 对新帧产出 C_t，再喂 p_{t+1}=P(S_t,C_t), d_{t+1}=D(S_t,C_t,p_{t+1})
```

**这里最值得注意的一点：一致性是文字承载的。** 上一帧不是作为参考图传下去，而是
被 Monitor **写成一段描述**再传。论文提到过一个「integrated image vector
database」用于稳定创作过程，但检索机制完全没有描述，无法复现。

循环的终止条件是 Monitor 的定性判断（运动连贯、角色统一、色彩/风格无突变、
角色细节——作者称角色细节最难控），**论文没有给任何数值阈值、评分细则或迭代
上限**。用 `BASE_DESCRIPTION` 这类产物做「同底模对照"的时候要记住：这个循环
无法被忠实复现。

### 2.4 出片与合成

关键帧交给第三方 i2v（Runway、SVD、DALL·E 3、Midjourney），每条约 4 秒。
**论文没有给任何拼接程序**——没有转场逻辑、没有对齐算法。限制章节自己承认，
成片可能"看起来只是片段的堆叠"（a mere assembly of clips），并把这一点留作
未来工作。

这是它和我们、和 OpenMontage 最大的差距：它交付的是关键帧方法与一致性论述，
不是一条能出成片的流水线。

---

## 3. 本项目：18 节点状态机，先全图再全片

`agent/graph.py`，18 个节点、6 条条件边。

### 3.1 拓扑

```
analyze_user_goal → plan_story → plan_shots
        │
        ├─ 模型不接图（T2V）────────────────────┐
        │                                       │
        ▼                                       │
   [关键帧阶段]  全部镜头先出图                  │
   generate_image_prompt → generate_image       │
        → evaluate_image ─(retry)→ rewrite      │
        → advance_keyframe ─(more/all_done)─────┤
                                                ▼
   [视频阶段]  全部镜头再出片               start_video_phase
   generate_video_prompt → generate_video
        → evaluate_video ─(retry)→ rewrite
        → evaluate_clip → synthesize_dialogue → advance_shot
                                                │
                            compose_video → finish
```

两段式而不是逐镜 image→video 循环，是一个有代价的选择：好处是第 N 镜的视频
可以用第 N+1 镜的关键帧当尾帧（首尾帧拼接），代价是在关键帧全部出完之前不能
开始任何视频生成，中断代价高。

### 3.2 一致性三层

1. **文字锚点**：`characters[].appearance` 逐字拼进每镜提示词
2. **首镜锚点**：第一个通过质检的关键帧成为全片锚点，后续每镜只跟它比，不链式累积
3. **参考图回灌**：`IMAGE_REFERENCE=1` 时把锚点关键帧作为 `image` 参数传给出图模型

第 3 层是唯一一个真正的像素信号，而且**只在出图模型接受参考图时才生效**。

### 3.3 能力声明驱动的分派

`needs_image_phase` 读 provider 声明的 `frame_roles`：空集就直接跳过整个关键帧
阶段，并在事件流里写明「本层不产出锚点，所有依赖关键帧的指标标记为不适用」。

这条设计对下面第 4 节至关重要。

---

## 4. 同一批本地基模下，三者的差异

本地可用基模（`local_media.py`）：

| 模型 | 模态 | fps | 帧规则 | `frame_roles` |
| --- | --- | --- | --- | --- |
| `cogvideox5b` | t2v 480×720 | 8 | 4k+1 | ∅ |
| `wan21_1_3b` | t2v 480×832 | 16 | 4k+1，≤81 | ∅ |
| `longlive_1_3b` | t2v 自回归（Wan2.1-1.3B + LongLive LoRA） | 16 | — | ∅ |
| `cogvideox5b_i2v` | **i2v** 首帧 | 8 | 4k+1 | `{first_frame}` |

### 4.1 三套工作流对基模的要求完全不同

| | 需要什么 | 三个 T2V 模型下会怎样 |
| --- | --- | --- |
| OpenMontage | 出图模型（角色参考图/抠像）+ **i2v 模型** | 一致性主干**整条不可用**：没有 i2v 就没有 `reference_image → 关键帧 → 视频` 这条链 |
| DreamFactory | 出图模型 + i2v 模型 | 同样不可用，它的关键帧迭代最后也要交给 i2v |
| 本项目 | 出图可选，i2v 可选 | **降级但可运行**：跳过关键帧阶段，退化为纯文字锚点 |

这是三者在工程上最实质的差别，而且不是设计品味，是**降级策略**：

- OpenMontage 的一致性写在硬闸门上（没有参考图不许进 assets），所以基模不够
  时它**拒绝运行**，而不是降级。这正是 `om_control_notes.md` 里记录的现象：
  本地 SD1.5 出不了纯白无缝板，5 个角色 0 个通过 0.7 闸门，参考图路径整条走不通，
  最后只跑通了它的 text-only 路径。
- DreamFactory 没有实现，无法验证；按论文描述，没有 i2v 就没有成片。
- 我们在 provider 上声明 `frame_roles`，**声明为空就跳过这一阶段**，并明确标注
  相关指标不适用。代价是测出来的东西变少，好处是测出来的东西是真的。

### 4.2 于是 tier3a 测到的到底是什么

这是最容易说错的地方。tier3a 用的是 `cogvideox5b` 与 `wan21_1_3b`，两者
`frame_roles = ∅`，所以：

- 关键帧阶段被**整个跳过**
- 第 3 层（参考图回灌）**从未执行**
- 第 2 层（首镜锚点）没有关键帧可锚
- **只剩第 1 层：文字锚点**

所以 2.28 → 4.61 这个数，准确的说法是：**在纯文字承载角色状态下，多镜人物
一致性显著提升**。它没有测到像素侧机制，也没有测到任何需要 i2v 的路径。
标题 "Identity as Text, Identity as Pixels" 里的 Pixels 那一半，在本地模型上
根本没有被执行过——OpenMontage 那边同样没跑通。**两边都缺像素臂。**

这与 `identity_crosscheck.md` 的结论是一致的、互相印证的：独立臂只有 4.4%
的镜头有可用人脸，所以连事后用像素指标去交叉验证都做不到。

### 4.3 各层贡献拆不开

`planned` 与 `independent` 之间差了三件事，不止一件：

1. 更长的提示词（实测 49 字符 → 704–879 字符）
2. 来自故事计划的动作与场景描述
3. 逐字复用的角色外形锚点

报告里对第 1 项做过消融（截断到 400 字符只回收 0.075），但第 2、3 项没有分开。
所以现在的因果结论只能到「携带计划状态有用」，还到不了「文字锚点是主因」。

---

## 5. 若要真正比较，缺的是哪一块

按代价排序：

1. **补像素臂。** 换成 `cogvideox5b_i2v`（已有声明，`roles={first_frame}`），
   三套工作流都能跑通参考图路径，才能回答「文字 vs 像素」。这是唯一能补上标题
   另一半的实验。
2. **拆开 planned 的三项差异。** 至少加一个「只换提示词、不带锚点」的臂。
3. **给 OpenMontage 一个能出板子的出图模型**（Seedream 或 Qwen-Image），
   让它的一致性主干真正跑起来，再做整剧对照，而不是只在 shot 级对照。
4. **DreamFactory 无法参与对照**，除非作者放出实现。在论文层面它只能当方法参照。

---

## 6. 结论

- OpenMontage 是**十段式、带人工闸门、以参考资产为一致性主干**的生产线，
  可靠性来自"不合格就不许往下走"。
- DreamFactory 是**对话式多 Agent + 关键帧迭代**，一致性靠 Monitor 把画面写成
  文字再传下去，无阈值、无上限、无拼接方案。
- 本项目是**两段式状态机**，一致性靠文字锚点 + 首镜锚点，按模型能力声明决定
  退不退化。

在本地 T2V 基模上，三者的真实差别不是"谁的工作流更好"，而是**当基模不具备
某个能力时各自怎么办**：OpenMontage 拒绝、我们降级并标注、DreamFactory 无实现
可查。而 `identity_crosscheck` 与 tier3a 两块证据共同指向同一件事——**像素侧
这个机制，在 1.3B–5B 这一层根本没有被验证过，因为素材里根本没有足够可检测的
人脸。**
