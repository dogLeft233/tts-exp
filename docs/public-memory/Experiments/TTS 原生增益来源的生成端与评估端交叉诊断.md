---
title: TTS 原生增益来源的生成端与评估端交叉诊断
type: experiment
permalink: tts-exp/experiments/tts-原生增益来源的生成端与评估端交叉诊断
status: concluded
date: '2026-09-15'
change_id: disentangle-tts-native-gain
hypothesis: TTS原生优势可能同时包含SyncNet声学/特征几何响应和TFG对较少干扰输入的偏好；用等时钟音频干预区分两条路径。
result: 正式续跑 A 完成并通过复验；B 固定分母 144+4 视频、336+8 评分已列账但因缺少完整 provenance-bound LEAPTALK_CONFIG
  实际完成 0；validator=valid，工程状态 PARTIAL/DEPENDENCY_BLOCKED。
conclusion: A 仍只定位到固定视频的评价端响应；B 尚未产生新生成视频和 fresh native 数据，因此不能解释 replacement 增益，也不能推出
  TTS 音质因果机制。
report: runs/tts_native_gain_attribution_completion_20260916_v1/report.md
inputs: review_v15；ID151–162；24历史LeapTalk视频/crop/矩阵及12真实视频；142个文件SHA-256绑定。
outputs: 父run A180+18，CPU396/396；B144+4视频/336+8评分仅固定台账，人工同步96+10题缺B媒体、音质48+5；旧validation
  valid/PARTIAL不代表B成功路径验收。
model: A：既有LeapTalk输出+SyncNet V2；B：待恢复/明确绑定的LeapTalk配置，两seed42/43；零新TTS/训练/云调用。
code_paths:
- openspec/changes/disentangle-tts-native-gain/
- scripts/experiments/tts_native_gain_attribution/
- tests/experiments/tts_native_gain_attribution/
tags:
- tts
- lrs3
- native-gain
- mechanism
- syncnet
- openspec
completion_change_id: complete-tts-native-gain-attribution
completion_status: implemented_b_blocked
engineering_status: PARTIAL
completion_run: runs/tts_native_gain_attribution_completion_20260916_v1
validation: runs/tts_native_gain_attribution_completion_20260916_v1/validation.json
---

# TTS 原生增益来源的生成端与评估端交叉诊断

用户要求回忆上次实验，进一步解释自然视频训练的TFG为何对TTS取得更高原生分数，并给下游编写实验spec。前轮已执行A、CPU诊断和有限状态验证，工程终态PARTIAL。2026-09-16已实现补全入口 `openspec/changes/complete-tts-native-gain-attribution/` 的代码、阻塞态全链路、CPU成功fixture和独立validator；正式 B 因缺少可核验 `LEAPTALK_CONFIG` 仍为 `DEPENDENCY_BLOCKED`。人工包已具备完整题目台账和隐私结构，但因 B 无视频而保持 `PARTIAL_MEDIA`；不能把本轮称为科学全量完成。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据用户请求建立机制假设、两阶段因果比较、资源与独立验收规范；只写spec | September 15, 2026 | user request |
| 实现并执行新spec：A/CPU/人工包/独立校验完成，B因LeapTalk依赖缺失阻塞；记录A结果与后续B入口 | September 15, 2026 | user request |
| 复核发现B评分占位、repeat覆盖、成功路径验收和人工媒体缺口；编写complete-tts-native-gain-attribution补全spec，未执行B | September 16, 2026 | user request |
| 实现补全 spec：A 只读续跑与复验完成，B 成功/阻塞路径均实现并通过独立验收；正式 B 因缺少 provenance-bound LEAPTALK_CONFIG 阻塞，BM 任务与看板已同步 | September 16, 2026 | user request |

上次有效结果为 `runs/lrs3_tts_gain_mechanism_review_v15`，本次重新运行独立validator仍valid。LeapTalk 12来源INTERIOR ΔSync-C=+1.253，最佳距离改善+0.286（校正区间跨零），背景项+0.967；历史50条记录Ditto +1.124、LeapTalk +1.389，正式历史统计按45来源等权。边界敏感性未显示明显主因，但不能把不显著当等效。人类同步尚未评估。

本次四个假设：H1评估器声学/特征几何贡献；H2生成器对干扰较少输入的偏好；H3节奏规则性（本轮仅描述）；H4训练筛选/模型选择与评价器共享偏好（本轮不识别其训练因果性）。Ditto论文报告广播视频清洗训练及lip-sync checkpoint选择，为H4提供线索；不能移植到所有checkpoint。训练于自然音频不等于天然最擅长每条自然录音，也不能断言全部预训练/蒸馏从未接触TTS。

A固定既有口型，逐一改变评分音频：ORIGINAL、共同headroom基线A0、−6dB、20dB有色噪声、固定谱抑制。12来源×3视频类型（自然驱动、TTS驱动、真实视频）×5音轨=180科学cell，另18控制；复用特征做396规划错来源内容配对。共享headroom避免加噪裁剪，ORIGINAL额外评分单独测量幅度缩放影响。

B固定同一LeapTalk家族，每录音A0/NOISE/DENOISE、两seed42/43，144科学视频+4重复视频，336科学评分+8控制。四格定义G=q10−q00、E=q01−q00、I=q11−q10−q01+q00，统一原音轨区分生成与评估路径；另检验本次配置fresh native是否复现。N/T各保留自己的时钟，不能直接交叉换未对齐音轨。B不能用Wav2Lip代替后称为完成；历史模型环境尚需恢复/绑定，资源不足应保留A并标PARTIAL。

主推断固定六对比、12来源等权、seed先平均、20000次共享bootstrap、99.166667%六项Bonferroni区间。raw C与单位范数诊断不同量纲，比较标准化效应和方向，不直接减分算衰减百分比。谱抑制不自动代表音质更好，不把代数分解或相关性当中介比例。

## Observations
- [completion] 2026-09-16：完成 `complete-tts-native-gain-attribution` 的实现审阅与阻塞态续跑；正式 run `runs/tts_native_gain_attribution_completion_20260916_v1` 保留父 A 产物并通过独立 validator（`valid`），A 的 180+18、72 个操纵检查、24 个重放和 396 个诊断均复验通过。
- [B-status] 正式 B 的固定分母为 144+4 个视频、336+8 个评分，实际完成为 0，`03_generation` 和 `04_crossed_scores` 均为 `DEPENDENCY_BLOCKED`；CPU-only success fixture 已覆盖七格交叉、共同支持、四格分解、repeat 和延迟控制，但不作为科学 B 数据。
- [blocker] 完整 B 仍缺少 provenance-bound `LEAPTALK_CONFIG`（官方 repo commit、各组件 revision/文件 hash、运行命令、frontend/RNG/PCM/chunk-to-frame 现场证明）；没有用 Wav2Lip 替代，也没有标记 `AUTOMATIC_COMPLETE`。盲评包保留真实分母但为 `PARTIAL_MEDIA`，人工评分仍未评估。
- [verification] 2026-09-16：19 个针对性 pytest、完整 ruff、py_compile、`git diff --check` 和 strict OpenSpec 校验均通过；GPU 共享锁、foreign PID 监控、独立 GPU/磁盘预算和中断隔离测试均已覆盖。
- [review] 2026-09-16：generation.py crossed_score_stage成功分支仍无条件抛ProtocolError；repeat与main使用同seed42.mp4；validator只对A独立重算主统计，B成功分支不能仅按计数验收。上述为未执行B路径缺口，不构成A数值已被污染的证据。
- [completion-spec] `openspec/changes/complete-tts-native-gain-attribution/`：保留父run，先CPU修复/fixture，再恢复同家族、全链smoke、144+4视频/336+8评分、96四格分解、六项统计、106/53可播放盲评题及独立验收；只有真实完整数据/控制通过才可AUTOMATIC_COMPLETE。
- [perception-correction] 旧人工同步包缺B媒体；成功路径需同一A0实际播放、隐藏重复身份不得公开、真实评分来源bootstrap及validator分支补全，不能仅以台账数量称包完整。
- [status] concluded
- [progress] 前轮A结果已记录，工程状态PARTIAL；B依赖与实现缺口待补，人工同步包为台账而非完整媒体；补全spec为spec_ready
- [implementation] 新增 `scripts/experiments/tts_native_gain_attribution/`，提供 audit、audio、fixed-video、generate、crossed-score、analyze、perception-pack、perception-analyze、report、validate 及 resume CLI；P0–P7 产物落在 `runs/tts_native_gain_attribution_implementation_20260915_v1/`
- [input] 固定 ID151–162、12 source groups；142 个 input-bindings、v15 父证据、24 个历史 N/T crop/matrix、12 个真实视频和肖像均完成 hash 核验。A 固定视频 36 个，像素/PTS/ROI 冻结；A ORIGINAL 重放 24 个矩阵最大绝对误差 `1.144e-05`，容差 `1e-04`
- [audio] 180 条音频记录完成 ORIGINAL/A0/GAIN/NOISE/DENOISE；共同 headroom、精确 float64 STFT/iSTFT、PCG64 有色噪声、量化后 GAIN/SNR 检查均通过（72/72）。GAIN 实际约 `-6 dB`，NOISE 实际约 `20 dB`
- [A] 完成 180 scientific cells + 18 controls；实际模型前向为 visual 36、audio 144、shifted audio 12，另有 36 个 ORIGINAL/A0 音频特征别名。控制为 IDENTITY_PASS=6、DELAY_DETECTED=12
- [result] `A_N_DENOISE_E` 均值 `-0.075`，六项 Bonferroni 校正区间 `[-0.161, 0.021]`，属于 SMALL_WITHIN_PREDECLARED_RANGE；`A_T_NOISE_E_HARM` 均值 `+0.571`，校正区间 `[0.158, 1.011]`，为 POSITIVE_EVIDENCE；`A_R_DENOISE_E` 均值 `-0.105`，校正区间 `[-0.213, 0.059]`，为 INCONCLUSIVE
- [diagnostic] 单位范数几何 180 行完成；错误内容检索 `396/396` 对、转写 VERIFIED；节奏摘要完成但只作 DESCRIPTIVE_ONLY
- [B] 本机没有可核验的 LeapTalk repo/checkpoint 和 provenance-bound adapter，generation 与 crossed scoring 明确为 DEPENDENCY_BLOCKED；仍保留 144 视频、4 repeat、336 scientific score、8 control score 的固定分母和 blocked rows，未用 Wav2Lip 替代
- [perception] 已构建 96 对同步包、48 对音质包、隐藏重复题、匿名映射和空模板；没有人工评分，状态为 PERCEPTION_NOT_ASSESSED / QUALITY_NOT_ASSESSED
- [validation] 前轮validator重新计算A的primary bootstrap并验证有限状态，结果valid/PARTIAL；2026-09-16确认其B成功路径主要检查计数，尚不具备完整矩阵身份/统计重算验收，需按补全spec修复，不能用旧valid宣称B已实现。
- [conclusion] A 只证明固定视频的评价端对特定音频干预有响应，尤其 V_T 在加噪评价音频下的 Sync-C 下降；它不能推出重新生成的口型会有同样变化，也不能在 B 未完成时解释 fresh native replacement 增益或作一般性的 TTS 质量结论。下一步需恢复同一 LeapTalk 家族的可核验 adapter、checkpoint、冻结 ROI 和成对 seed 后完成 B。
## Relations

- follows [[LRS3 TTS 原生优势的分数分解与曲线诊断]]
- relates_to [[跨数据集 TFG 测评（5×50 multiset）]]
- relates_to [[LRS3 bridge TTS quality comparison]]
