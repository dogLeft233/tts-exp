---
title: TTS 音高起伏交换的配对 Harvest 测量修复与 Wav2Lip 执行 Spec
type: research
permalink: tts-exp/research/tts-音高起伏交换的配对-harvest-测量修复与-wav2-lip-执行-spec
status: concluded
execution_status: complete
protocol: tts_f0_swap_v3_paired_harvest
date: '2026-09-18'
question: 在保留严格目标变化误差和共同支持门的前提下，用同一 Harvest 测量并以 ID 重合成为基线，能否解除 DIO 检测分歧造成的阻塞并完成 Wav2Lip/SyncNet？
tags:
- tts
- f0
- wav2lip
- repair
- measurement
run_id: tts_f0_swap_v3_audit2
result_status: INSUFFICIENT_SUPPORT
validation_status: PASS
---

# TTS 音高起伏交换的配对 Harvest 测量修复与 Wav2Lip 执行 Spec

## 目标与边界

修复 `tts_f0_swap_repair_v2` 的阻塞并执行原定 F0→Wav2Lip→SyncNet 实验。固定沿用 `runs/tts_f0_swap_f0spec_audit2/inputs.json` 的 4 个 pilot 和 24 个 formal pair，不按新分数重选样本，不修改旧 run。

v2 已证明普通平均恒等式正确，但 DIO 在 RAW 上就与 Harvest 有明显分歧，且 ID 重合成后又产生变化。DIO 不是可直接称为真值的独立标尺，因此 v3 改变测量契约：WORLD 的生产分析和候选验收统一使用 Harvest→StoneMask；DIO→StoneMask 仍对所有 WAV 重测并报告，只作为交叉诊断，不参与通过/失败投票。

本修复不改变 F0 目标公式、交换剂量、RMS/长度/PCM16 约束或候选目标误差门。正式结果仍只在通过音频门的样本上解释，缺失保留在分母记录中。

## 固定输入、产物和运行顺序

- 输入清单、natural/TTS WAV、MFA phone occurrence、portrait、box、模型和排序全部从 v1 parent inputs 原样复制并逐项 hash 校验。
- 新 run 使用 `runs/tts_f0_swap_v3_<run_id>/`，协议 ID 为 `tts_f0_swap_v3_paired_harvest`。
- 阶段顺序固定为 audit-copy → pilot → formal-audio → render → score → analyze → validate → report。
- pilot 固定 4 对，至少 3 对 N/T 双向通过才允许 formal-audio；否则保留音频诊断并停止 GPU。
- formal 使用冻结的 24 对，不因 pilot 或正式阶段失败补选。

## 音频操作

沿用 v2 的普通平均保持公式、phone-local taper、WORLD `harvest→stonemask/cheaptrick/d4c`、5 ms、Harvest 搜索范围 60–600 Hz、精确长度、一次 RMS 匹配、共同安全缩放、PCM16 重读和 32-bit/finite/clipping 检查。实际 `StoneMask` 会对少量低音帧做精修，使最终测得的正 F0 略低于传入 Harvest 的 60 Hz floor；v3 固定接受 50–600 Hz 的 post-StoneMask 验收范围，保留这些原始/目标值，不 clip、不把它们改成无声。50 Hz 是实现前冻结的容差，不随样本调节。不得插值跨无声、clip F0、改 ap/sp、调弱剂量或动态压缩。

每个接收方生成 RAW、ID、LEVEL、CONTOUR。ID 使用接收方原 F0/sp/ap 的 WORLD 重合成，是本接收方候选比较的载体基线。

## 配对 Harvest 验收

对每个接收方保存 H_source、H_RAW、H_ID、H_LEVEL、H_CONTOUR，以及 D_RAW、D_ID、D_LEVEL、D_CONTOUR，全部从最终 PCM16 WAV 重测，时间网格必须逐点一致，不插值。

1. 支持、目标数组、普通平均恒等式、波形长度/峰值/RMS/局部包络门沿用 v2。
2. 先检查 H_source→H_ID：H_ID 覆盖 H_source 原有声帧至少 0.80；共同有声帧半音误差 median≤1、p90≤3。mask 差异只作为诊断，不因 DIO/Harvest 的不同判定直接失败。
3. 候选检查以固定的 `S = valid & (weight≥0.5)` 与 H_ID 有声帧的交集为分母：H_LEVEL/H_CONTOUR 覆盖 `S∩H_ID` 至少 0.80；在 `S∩H_ID∩H_candidate` 上，实际 `H_candidate−H_ID` 与目标 `target_arm−target_ID` 的半音误差 median≤1、p90≤3。
4. 候选新增/丢失有声帧、全句 mask mismatch、DIO 相对 H 的覆盖和四格计数必须完整报告，但不把检测器分歧本身再当作候选无效。候选分母固定由 S 和 H_ID 预先定义，不能因失败帧临时缩小。
5. DIO 交叉诊断仍要求轨迹完整、finite、同网格；若 DIO 与 H 不同，只写检测差异，不改变 H 门。
6. pair 通过仍要求 N、T 两个接收方全部必需 H/波形门通过；pilot gate 固定为 3/4。

这个定义回答的是“在同一 WORLD/Harvest 载体中，实际候选相对 ID 是否实现了目标 F0 变化”。它不声称 DIO 与 H 哪个更接近听感真值，也不把 ID 重合成等同于原始 waveform。

## Wav2Lip/SyncNet 阶段

pilot gate 通过后，复用既有渲染和评分 worker，在冻结 portrait/box/checkpoint/seed/device 下完成 formal 24 对：

- 每个 pair、每个接收方四个生成音轨 RAW/ID/LEVEL/CONTOUR；
- 固定视频/固定音轨和 own-audio 的既有 10-cell 矩阵；
- 保存 features、distance matrices、scores.csv、analysis.json 和固定 speaker bootstrap；
- controls、视频/音频 hash、SyncNet 模型 hash 全部写入 manifest。
- 不能把同音轨替换分数当作生成端因果证据。

结果判定仍按原 Wav2Lip spec：分别报告 engineering、manipulation、native、scientific status；主统计为 g_N、l_T、h_N、h_T，背景为 G_raw/G_id。任何阳性只说明本实验的 F0 交换对该评价端点有响应，不自动解释全部 TTS 增益或授权训练。

## 独立验收

独立复算脚本不导入生产映射/QC/gate 函数，至少复算 formal score rows、speaker bootstrap、主统计区间和 pilot/final 固定分母。validation PASS 与 manipulation/scientific status 分开记录。

## 退出规则

- pilot <3/4：`MANIPULATION_NOT_VALIDATED`，不生成 formal 视频；
- pilot ≥3/4 但 formal audio/support 不足：保留完整缺失，scientific status 为 `INSUFFICIENT_SUPPORT`；
- render/score 资源缺失：工程状态明确为 `INPUT_UNAVAILABLE`，不伪造分数；
- 所有结果都不得把协议通过本身写成 TTS 机制证明。

## 执行结果

- run：`runs/tts_f0_swap_v3_audit2/`；冻结输入仍为 parent 的 4 pilot + 24 formal，没有重选样本。
- pilot：4/4 pair 通过；formal audio：24/24 pair manifest 完成，48 个 receiver QC 中 47 个通过。一个 T receiver 的 CONTOUR p90 目标误差为 3.650 半音，因此按预注册规则保留 RAW/ID/LEVEL，省略该 receiver 的 CONTOUR 视频。
- Wav2Lip：190 个 QC 条件视频完成；SyncNet：380/460 cells 完成。8 个 receiver 因共同 SyncNet 窗口少于 50 被保留为支持不足，另 1 个 pair 因音频 QC 排除。
- `analysis.json`：`complete_pair_count=16`、`main_pair_count=14`、6 speakers；`scientific_status=INSUFFICIENT_SUPPORT`，没有方向性因果结论。`validation.json=PASS`，`recompute.json=PASS`，bootstrap 索引一致。
- 阻塞修复：StoneMask 低音精修允许固定 50 Hz post-StoneMask 下限（Harvest 搜索仍为 60–600 Hz，不 clip）；分析器修复 raw replacement cell 的视频臂索引；v3 评分把已声明的短支持标为 `INSUFFICIENT_SUPPORT`，独立复算接受 v3 音频验收的 `NOT_APPLICABLE` 分支。

## Relations

- amends [[TTS 音高起伏交换的 WORLD 与测量链修复 Spec]]
- implements [[TTS 音高起伏双向交换的 Wav2Lip 因果诊断 Spec]]
- follows [[TTS 音高起伏交换 WORLD 与测量链修复结果 2026-09-18]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据 v2 的 RAW 检测分歧与 H 同方法轨迹，冻结 paired Harvest 基线和正式执行顺序 | September 18, 2026 | user request |
| 完成 v3 audit2；修复 StoneMask 下限、raw replacement 分析索引、支持不足状态和独立复算分支；记录 24 对正式运行结果 | September 18, 2026 | agent execution |
