## Context

动机见 proposal；规范性门槛见 [spec](specs/static-image-natural-to-tts-bridge/spec.md)。本次仅设计，尚无静态 bridge 实验结果。

历史输入并非全是视频：早期 AISHELL/Ditto 使用肖像 PNG；部分早期 Wav2Lip 使用 natural-driven Ditto 视频；本 bridge 的两轮使用 LRS3 动态视频。9 月 8 日已有 STATIC/DYNAMIC 的 ±200ms 实验，9 月 9 日已有固定帧参考诊断，但它们不能回答冻结 BRIDGE_075 的静态收益。

历史复核（自然音轨 replacement 口径，正值为改善）：

| 队列 | C 提升 | D 改善 | 两者同时 | mean ΔC | C 95% CI |
|---|---|---|---|---|---|
| discovery MAG_075，23 条 | 18/23 (78.3%) | 15/23 (65.2%) | 15/23 (65.2%) | +0.078 | [+0.016,+0.136] |
| confirmation BRIDGE_075，22 条 | 9/22 (40.9%) | 11/22 (50.0%) | 9/22 (40.9%) | +0.031 | [−0.034,+0.105] |

探索轮检验过多个强度，其区间未作多重选择校正；确认轮记录不同但共享来源组，不能称全新来源的独立确认。confirmation mel movement 22/22 通过只表示构造移动，不能代替评分改善比例。

## Goals / Non-Goals

**Goals:** 直接回答「同一张静态图 + 原 bridge 音频」是否比「同图 + 原自然音频」及「同图 + 纯重建音频」好；另回答局部交换驱动在静态条件下是否产生相应配对偏好。

**Non-Goals:** 本设计不启动生成，不换模型/候选，不重新搜索 alpha，不做静态/动态因果交互，不证明声音更好听或嘴部更自然，不授权训练。seen-fit 阳性只支持后续独立来源验证。

## Decisions

### 1. 复用波形，冻结新的静态视觉条件

父路径：

- `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json`
- 同 run `01_audio/audio_manifest.json`、`04_final/analysis.json`。
- discovery 数字来自 `runs/lrs3_phase_preserving_replacement_envelope_20260904/05_final/analysis.json`。

本 spec 编写时只读核实的文件 SHA256（实施时必须匹配；不是 JSON 内部自 hash）：

| 资产 | SHA256 |
|---|---|
| confirmation cohort.json | `b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b` |
| confirmation audio_manifest.json | `2c279dc5a74dbeb53ae547235347870c6664fd9caf656a594b48a8310d9bd288` |
| confirmation analysis.json | `0841a0e95efb8747516f866427bafa601c1fa5270ca4bcaaf6cfd396fc77556f` |
| discovery analysis.json | `965e4bbe34ab1538b0e806225faaedc42d67f2243e9c84df4266bd471461228f` |

实施 prepare 阶段从父清单按 key join 22 条，重新验证文件及 decoded PCM 身份；不得从多份历史 MAG 实现里挑一份近似波形。先冻结旧 B，再构造 RT。RT 复用旧 STFT 约定但 alpha=0；保存实现、库版本、floor/RMS/峰值/量化细节。RT 与 N 相同也照常保留独立臂。

使用实际 PNG 作为 `--face`，25 fps；检测只在第 0 帧运行一次。Wav2Lip `--box` 使用冻结的人脸框，禁止历史 `constant_full_frame_fallback`。评分框从未加 padding 的检测框中心得到边长 1.5×max(width,height) 的正方形，越界使用零 padding，固定映射至官方 224×224 图像 frontend；所有输出复用同一映射。保存检测原始结果、box 坐标约定与 crop 图。PNG/输入 RGB hash 和全零 source-index 序列证明输入确实静态。

此选择避免继续把整张场景缩为脸输入；代价是与旧动态实验还改变了框，因此只测新静态条件内 B−N，不做历史差分因果解释。

### 2. 五臂足以回答问题，不增加强度搜索

N 是完整自然基线；N_REPEAT 测重复性；RT 分离重建处理；B 是唯一候选；S 是时序诊断。B 的评分主音轨始终是 N，因此主差值不包含「直接把评分音频改得更讨 SyncNet 喜欢」的作用。V_B/B 明确为辅助项。

先 A：44 个视频和 66 cells。A 有效后 B：66 个视频和 132 cells。新输出总数 110/198（不含最多 4 次失败重试）。不因 S 质量不如正常语音而阻塞 bridge 计算；S 的局部偏好与 B 的收益是两个问题。

### 3. 固定评分输入与完整时间支持

使用官方 SyncNet 网络/frontend 的显式帧/PCM输入，固定 crop，复用同一视频 embedding 为不同音轨打分。禁止在 mux 后为每个 cell 独立检测 track。生成帧统一无损保存；需要包装时优先 FFV1+PCM MKV，浏览器播放另编码 MP4。

输入完整 L 样本参与生成。prepare 按冻结 Wav2Lip mel 分块规则确定可生成帧数及 frontend 感受野（含 STFT/MFCC 上下文）；固定 `W`、各 lag 的 audio/video 来源索引和有效掩码。实际不足是契约失败，不动态缩短 W。只裁尾部不可用窗口，不重采样时间轴，不拿填充样本当真实支持。

主统计：先在共同 W 平均距离再求 C/D/k；固定 k0 额外报 D_anchor，防止只看自由搜索后的同步。ND 延迟测试采用相同 W，并排除填零和前端上下文受填零影响的窗口；独立检查 +5 帧的符号。可以先用 feature-level 人工整数移位做索引单元测试，但正式延迟控制必须从实际 ND 波形经过官方 frontend 计算。

Wj 是两个中间段内完整感受野窗口，在固定 k0 下取 N/S、V_N/V_S 的交集；它不强求全 [-15,+15] 支持，以免把短四分之一段耗尽。保留每个原/交换采样来源区间，剔除拼接上下文，分段检查而非只算合并均值。

### 4. 明确何种结果可以叫什么

重复性和同视频延迟控制校准测量，不要求正常/交换语音的生成分数一样高。LOCAL_SWAP 用「N 画面偏 N、S 画面偏 S」的双向局部偏好判断响应；该诊断失败仍不唯一归因于生成器或评分器。

B 必须在自然音轨上同时胜过 N 和 RT，C 均值 >0.05 且 CI 下界 >0，并满足 D、固定 lag 距离与 offset 安全门槛。0.05 是预先选定的小效应下限，用来区分数值微涨与本实验认可的收益，不能说是人类感知阈值。要求两个比较共同成立，不从多个阳性路线选择，无需将未通过 swap 偷换成 B 无效。

最终允许的组合例如：

| measurement | bridge_gain | swap_transfer | 可说的结论 |
|---|---|---|---|
| FAIL | NOT_EVALUATED | NOT_EVALUATED | 测量未校准，bridge 未测 |
| PASS | GAIN_OBSERVED | OBSERVED | 静态条件内分数收益与交换响应均观察到 |
| PASS | GAIN_OBSERVED | UNRESOLVED | 有静态分数收益，时序解释仍未闭合 |
| PASS | NO_GAIN_ESTABLISHED | OBSERVED | 能检测交换响应，但该 bridge 没达到收益要求 |
| PASS | NO_GAIN_ESTABLISHED | UNRESOLVED | 本构造未建立收益，交换响应也未解决 |

表内缩写在 final 中使用 spec 的完整枚举；integrity 失败优先标记 INVALID，不能保留有效科学结论。对于 bootstrap、分母、并列和三位小数显示，validator 以未舍入矩阵结果重新计算。

### 5. 可复核交付和运行边界

未来实现目录为 `scripts/experiments/static_image_bridge/`；接口提供 prepare、stage-a、validate-a、stage-b、analyze、validate、report，不复用历史缓存冒充 fresh repeat。run 使用独立新目录。

必须保存：`protocol.json`、`inputs.json`、`audio_manifest.json`、`video_manifest.json`、`scores_manifest.json`、`matrices/`、`support/`、`per_record.csv`、`analysis.json`、`validation.json`、`final.json`、`report.md`、`playback/index.html`。报告提供历史两轮数字、当前分母 22、全量胜负及缺失列表；播放链接相对 run、导出后仍可用。

validator 从原资产和实际数组独立重算，不只验证字段存在或复用 runner 的最终 pass 函数。覆盖篡改静态帧、错误 box、容器/PCM hash 混淆、lag 符号、边界支持、重复视频被复制、臂错配和按三位小数误算胜负的负例。成功后只保留一次必要验证，不做科学重跑。

## Risks / Trade-offs

- 单张图仍带有嘴型/身份/姿态先验 → 明确只去掉原动态时序，没有消除所有视觉条件影响。
- 22 个来源组及其历史结果已见 → 只能作探索，不能凭 bootstrap 区间宣称泛化或新来源确认。
- 第一帧脸检测或支持不足 → 固定规则失败即阻塞，不补挑好看的图或更容易的样本。
- Sync-C 可能被最佳 offset 或距离曲线背景改变 → 同时报 D、D_anchor、offset、曲线和播放素材。
- 静态/历史动态同时改变 ROI 与评分 → 不计算有因果含义的历史差分；如需定位嘴型泄漏，另立同处理链 DYNAMIC/STATIC 交互 spec。
- 磁盘空间有限 → prepare 做存储估计，在新 run 内受控清理可重算临时文件，保留所有正式输入/矩阵/报告；不得清理历史 run 腾空间。

## Migration Plan

无迁移。仅新增独立实验目录和结果，不改历史产物。运行中断后仅当 spec、代码、模型、参数、PCM 和视觉输入 hash 全部一致时恢复；不同契约必须新建 run。最终按既有 BM 实验指令更新结果笔记，当前状态维持 planned。
