---
title: Bridge 自然视频 SyncNet 强度扫描 2026-09-13
type: experiment
permalink: tts-exp/experiments/bridge-自然视频-sync-net-强度扫描-2026-09-13
status: concluded
---

# Bridge 自然视频 SyncNet 强度扫描 2026-09-13

固定真实自然视频，不经过 Wav2Lip，测 natural-phase bridge 的音频侧剂量响应。沿用确认实验22条样本；alpha不是实测mel移动比例。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial design | September 13, 2026 | user |
| 完成22条9臂，保存结果与核验 | September 13, 2026 | user request / agent execution |
| 补充官方 Wav2Lip mel progress：B025/B050 与静态11子集统计 | September 13, 2026 | user request / agent execution |

## Observations
- [status] concluded
- [inputs] runs/static_image_bridge_20260913/inputs.json 与 runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json；22条22来源组，已使用队列，无剔除。
- [protocol] N、alpha0 roundtrip、alpha 0.25/0.50/0.75/1.00、原始MFA-linear、LOCAL_SWAP、N延迟200ms。alpha1保留自然相位，不等于MFA-linear。
- [protocol] 固定每条自然视频的一条完整S3FD轨迹，所有音频共享视觉embedding与全lag有效时间支持；C为平均距离曲线中位数减最小值，D为最小值。offset采用SyncNet符号，单位帧40ms。当前共同支持C与历史含padding的全局C不直接比较。
- [result] N C=7.483；alpha0.25 ΔC=+0.001 CI[-0.077,+0.074]；0.50=-0.347 CI[-0.519,-0.188]；0.75=-0.783 CI[-1.052,-0.545]；1.00=-1.184 CI[-1.506,-0.900]；MFA=-1.250。配对样本bootstrap，探索性未校正多重比较。
- [diagnostic] 使用官方 `third_party/Wav2Lip/audio.py:melspectrogram` 对冻结落盘 WAV 重新计算 mel；22条记录均为80×482，N progress=0、MFA progress=1，B075均值=0.5118，与此前movement结果一致。
- [result] 官方mel投影progress在全22条上：B025均值=0.1313，median=0.1185，bootstrap 95% CI [0.1002,0.1634]；B050均值=0.3116，median=0.3007，CI [0.2620,0.3608]。
- [result] 静态图实际使用的前11条上：B025均值=0.1251，CI [0.0833,0.1720]；B050均值=0.3066，CI [0.2359,0.3800]。alpha 0.25/0.50均明显小于相应波形插值强度，且B050约为B025的2.45倍。
- [boundary] mel progress只证明bridge输入在官方Wav2Lip特征空间中朝MFA-linear移动，不证明生成视频或replacement分数会改善。
- [result] B075、B100均22/22 C下降；仅3/22最佳offset相对N增加1帧。固定N最佳offset距离分别增加0.650、0.984。11/22从N到B100逐级C不升，平均单调不等于个体单调。
- [control] LOCAL_SWAP使22/22 C下降，平均ΔC=-4.639。延迟200ms使22/22最佳offset精确移动-5帧；固定N offset距离由7.169变15.002，但重新搜索offset后ΔC=+0.057。因此搜索C会掩盖可补偿的全局延迟。
- [validation] 22/22 RT与N WAV/PCM一致；22/22 B075复现历史PCM；198个矩阵有限且重算曲线一致；3单元测试通过，ruff通过。
- [conclusion] 较强bridge在真实自然口型上的SyncNet匹配变差，整体offset仍大体保留；不能单凭此区分局部音素时间变化与频谱/发音表征变化。该结果不等于历史生成实验有bug，也不授权audio head训练。
- [report] runs/natural_video_bridge_sweep_20260913_v2/report.md
- [report] runs/bridge_mel_progress_official_20260913/report.md；分析 JSON 只使用score-free inputs/cohort/protocol/analysis_scope与冻结WAV，不读取SyncNet scores。
- [outputs] runs/natural_video_bridge_sweep_20260913_v2：protocol/results/summary JSON、逐样本9臂WAV与距离矩阵。初始无后缀目录因权重路径检查失败，未评分；v2使用配置中的正确权重路径。
- [code_paths] scripts/experiments/natural_video_bridge_sweep.py；tests/experiments/test_natural_video_bridge_sweep.py；scripts/experiments/bridge_mel_progress.py

## Relations
- relates_to [[LOCAL_SWAP 最小重放诊断 2026-09-13]]