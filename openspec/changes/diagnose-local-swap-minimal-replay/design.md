## Context

主问题和范围见 [proposal.md](proposal.md)，验收契约见 [spec.md](specs/local-swap-minimal-replay/spec.md)。本文件是下游阅读入口。

历史输入根目录：`runs/lrs3_natural_to_tts_bridge_confirmation_20260904/`。

| 输入 | 用途 |
|---|---|
| `00_protocol/cohort.json` | 原始样本顺序、natural 和真实视频来源 |
| `01_audio/audio_manifest.json` | natural / LOCAL_SWAP 音频及 hash |
| `02_videos/videos_manifest.json` | 原生成视频、face box、驱动音频绑定 |
| `03_scores/scores_manifest.json` 与逐 cell 日志 | 历史 mux、评分与轨迹路径 |

2026-09-13 设计前只读复核：44 份 LOCAL_SWAP 日志均只有一组 Confidence / Min dist，均与 manifest 一致。22/22 条均为配 natural 更好；平均 C 增加 2.789、D 减少 2.994。这确认了历史记录中的现象，尚未完成独立前向复现或原因定位。

历史渲染采用动态原视频和 `constant_full_frame_fallback`，因此“生成视频保留原口型”是候选解释，不能先当成结论。历史 `CONTROL_FAILED` 也不是对 bridge 无效的证明。

## Goals / Non-Goals

分别回答：历史数字是否真实可追溯、重新评分是否复现、已知正确/错误的时序关系能否被区分、原生成视频在固定评分输入下偏向哪种音频。

只输出诊断。零次 Wav2Lip、零次 TTS、零次训练；不重跑 bridge，也不根据本轮分数挑样本或换裁剪。

## Decisions

### 1. 固定三条，历史检查覆盖全部 22 条

从原 cohort 取前三条，冻结后才读新分数：

1. `lrs3_6WeS1bXRBOk_00006`
2. `lrs3_6ul2TSvUDog_00007`
3. `lrs3_6wk4dkYSrV0_00006`

历史审计核对全部 22 条的 PCM 交换、文件 hash、生成命令音频绑定、两个 mux 的视频码流一致性、评分日志和算术。不是只检查 manifest 的 `verified=true`：重新解码与计算 hash。缺失或不一致逐项报告，不换样本。此阶段没有新神经网络评分。

### 2. 三个小检查，共 24 个新评分 cell

记 `G` 为旧 LOCAL_SWAP 驱动生成的视频，`N` 为原自然 PCM，`S` 为旧 LOCAL_SWAP PCM。

| 检查 | 每条的 cell | 数量 | 回答什么 |
|---|---|---:|---|
| A：原端到端重放 | `G/N`、`G/S` | 2 | 原现象是否能重新得到 |
| B：真实时序对照 | `R/N0`、`R/S0`、`Rs/N0`、`Rs/S0` | 4 | 已知匹配/错配能否被识别 |
| C：固定生成视频输入 | `Gc/Nc`、`Gc/Sc` | 2 | 去除每个 cell 独立检测/裁剪后，还偏向 natural 吗 |

总数 `(2+4+2)×3=24`。A、B、C 是不同端点，不把它们的绝对分数混合求一个均值。现有评分复算和从已保存距离矩阵派生统计，不计作新 cell。出错的尝试仍计入执行日志；没有按分数重试。

### 3. A 使用原媒体和原评分流程，重新产生所有中间文件

用旧 G 和原始 N/S 在新 run 中分别 strict mux。重新验证视频码流和 decoded PCM；使用原 SyncNet 权重、参数和原端到端 pipeline，独立工作目录和唯一 reference，禁止读旧 feature cache。

记录实际 Python 包版本、设备、权重、执行脚本 hash、命令、轨迹数和时间范围。历史代码版本缺失或与可核验绑定不同，标记 `HISTORICAL_RUNTIME_UNVERIFIED`，此时只能称“当前链路重放”，不得称完全相同环境复现。

逐条列旧/新 C、D、offset。历史数值近似复现阈值：每个 cell 的 C、D 差绝对值均 ≤0.100，offset 差 ≤1 帧。另行判断方向复现：3/3 条的新 `C(G,N)>C(G,S)` 且 `D(G,N)<D(G,S)`。方向复现和数值近似复现分别报告；阈值只是诊断公差。

### 4. B 的同步真值由真实音视频共同排列构成

先对真实源视频检测一次人脸轨迹，按官方 SyncNet 方法生成裁剪；在读取 B 的分数之前锁定轨迹。选择覆盖从原始时间零开始所需区间的最长轨迹，长度相同取编号最小者；不能按同步分数选择轨迹。若源视频不是 25 fps、无法确认 PTS 与 N 的共同零点、轨迹不覆盖起点，标记该项 `BLOCKED_INPUT`，不填充或自行修正 offset。

令可用完整视频帧数为 F，自然 PCM 样本数为 L（16 kHz）。取：

```text
k = floor(min(F, floor(L/640)) / 4)
R = 锁定裁剪后的前 4k 帧
N0 = N 的前 4k*640 个样本
Rs = R[0:k] || R[2k:3k] || R[k:2k] || R[3k:4k]
S0 = N0[0:640k] || N0[1280k:1920k] || N0[640k:1280k] || N0[1920k:2560k]
```

每帧 640 个样本，让音频和视频使用完全相同的交换边界。k 必须 ≥26 帧，否则该项无足够时长。只裁掉共同支持之外的尾部，保留裁剪计数；不补零、插帧、变速或重采样。

`N0/S0` 是本轮真实对照专用音频，**不等于旧 N/S**：旧四分点按完整 PCM 样本数计算，未必落在帧边界。A/C 必须保留旧交换，不能用 S0 代替。

预期：R 配 N0 优于 S0；Rs 配 S0 优于 N0。尤其 `Rs/S0` 使双方一起交换，仍保持原来的局部同步关系。这能检查“交换本身的接缝导致评分坏掉”的解释。

以解码后固定帧数组和 PCM 数组直接输入官方特征/距离计算；Rs 只重排同一组裁剪，不重新人脸检测。可播放视频从同一数组导出，采用统一无损中间格式及 PCM，视频导出后核验帧 hash。浏览用压缩预览不能作为评分输入。

### 5. C 固定同一份 G 裁剪和时间支持

从 A 的 `G/N` pipeline 输出取被历史评分规则选中的轨迹，锁定其帧数组、绝对时间范围、裁剪参数。将旧 N 和旧 S 按同一绝对时间范围切出 Nc/Sc，直接与相同视频帧评分。必须验证传到 MFCC 提取前的音频数组与预期切片相同。

不得各自重新检测、改变时长、归一化响度或搜索独立的音频起点。历史日志有多轨迹时，记录全部轨迹及历史实际取值规则；无法确定评分所指轨迹则标记定位不充分。

只比较 C 内两种音频的相对分数。A→C 同时涉及裁剪/媒体处理变化，若方向改变，结论为“预处理或配对链路可疑”，不能仅凭一次变化归罪于 face detector。

### 6. 输出全局分数，同时看交换段内部

每个新 cell 保存官方 C、D、best offset，以及逐时间窗×offset 的距离矩阵、offset 轴、视频帧索引、音频时间范围和 crop/PCM hash。偏移搜索保持官方旧配置并记录；B/C 直接计算须通过官方端到端结果或官方前向函数的数值 parity，不能另写一个未经核对的 Sync-C 公式。

B 的主要诊断是同一物理时刻（offset=0）的窗口距离，不依赖搜索最佳 offset。分别汇报前后未交换段、中间交换段；仅保留完整音视频感受野均落在段内、且窗口中心离段边界至少 5 帧的窗口。需要核验实际模型/特征感受野并据此剔除跨界窗口，不能仅凭中心位置。两音频条件使用同一窗口集合。每个中间四分段至少 10 个有效窗口，否则局部结论为 `INSUFFICIENT_SUPPORT`。

在相同支持上计算（D0 为平均 offset=0 距离，越低越好）：

```text
real_original_preference = D0(R,S0) - D0(R,N0)
real_swapped_preference  = D0(Rs,N0) - D0(Rs,S0)
generated_natural_preference = D0(Gc,Sc) - D0(Gc,Nc)
```

前两个量都为正才符合 B 的预期；分别报告中间第二、第三段，不能只看整段平均。使用固定数值平局公差 1e-6，3/3 条且两个中间段均方向正确，记为 `KNOWN_PAIRING_SENSITIVE`；否则如实记为 `MIXED_OR_UNRESOLVED`，不推导总体统计显著性。

A/C 的旧 S 分界使用原 PCM 样本边界和 crop 的绝对起点，不套用 B 的等长帧分界。B 的原配对若最佳 offset 绝对值 >1 帧或盲看不匹配，标记 `BASELINE_ALIGNMENT_SUSPECT`；不通过事后平移使其“通过”。保留全局最佳 offset 下的 C/D 作为辅助结果，局部 D0 不冒称官方 Sync-D。

### 7. 判定分层，保留不能定位的情况

| 证据 | 可以报告 | 不可以报告 |
|---|---|---|
| 旧日志、PCM 或绑定出现可复现不一致 | 具体的历史数据/实现问题 | 不定位就说 SyncNet 模型坏了 |
| A 3/3 条方向复现 | 旧反向现象在该子集可复现 | 22 条已全部重新推理 |
| B 两个真实配对方向正确，C 仍偏 natural | 评分可识别已知错配；G 在固定输入下更匹配 natural，支持驱动未充分转移的解释 | 已证明 mouth leakage 或 full-frame box 是唯一原因 |
| A 与 C 方向不同 | 优先排查预处理、裁剪及配对 | 已证明模型本身异常 |
| B 在同步配对有效、数组和 parity 均通过后仍失败 | 当前 SyncNet 端点对此扰动/样本的敏感性存疑 | 所有 SyncNet 评分都不可信 |
| B/C 不一致、缺样或局部支持不足 | `INCONCLUSIVE` 并列出缺失证据 | 用多数投票强行归因 |

技术状态为 `COMPLETE` / `BLOCKED_INPUT` / `IMPLEMENTATION_ERROR`；科学字段独立记录 `historical_record_verified`、`replay_direction`、`replay_numeric_agreement`、`known_pairing_sensitivity`、`generated_audio_preference`、`localization`。不使用一个笼统 GO/NO_GO 掩盖混合结果。

### 8. 可交接产物

新 run 至少包含 `inputs.json`、`historical_audit.json`、`scores.csv`（24 行）、每 cell 原始日志/距离矩阵、`support.json`、`final.json` 和 `report.md`。报告给出逐条矩阵、C/D/offset、上述偏好值及局部曲线。

为三条记录全部导出带原始时间码的播放页面：原真实配对、真实两种错配、双方一起交换、G/N、G/S；交换段显著标记。先用中性编号盲看，再提供条件映射。下游填写实际观察者及观察记录；无人观看就写“未人工核验”，不得把自动评分写成人工观察。

下游最终交接须能用一句话分别回答“旧记录对不对、重放是否复现、真实错配能否识别、G 更像跟随谁”，并给出下一项最小验证。若发现错误，本轮保留失败证据，修复与重新执行另用新 run；不覆盖旧分数。

## Risks / Trade-offs

- 三条样本 → 只做定位，不做总体效果检验；历史审计仍覆盖 22 条。
- 接缝可能影响短窗特征 → 共同交换对照加完整感受野内部窗口，不只看全局聚合。
- 原视频可能已有音画偏移 → 核查共同起点、基线 offset 和人工观察，失败就保留不确定结论。
- 新旧运行环境可能不同 → 保存代码与运行时绑定，区分方向复现与严格历史重放。
- 评分符合预期仍不能证明生成器根因 → 后续才考虑 mouth ROI、静态参考或新渲染对照，本轮不增加生成臂。
