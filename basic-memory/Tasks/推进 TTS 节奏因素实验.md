---
title: 推进 TTS 节奏因素实验
type: task
permalink: tts-exp/tasks/推进-tts-节奏因素实验
status: done
priority: high
current_step: 1
steps:
- 固定并审计 train/valid/heldout split 与 MFA 全 tier 数据
- 提取 global rate、phone duration allocation、pause、boundary/coarticulation 因素
- 与已有 valid S0765 的 paired SyncNet 结果做关联分析
- 构造 global rate、pause、phone allocation 的 exact-N 反事实音频及 no-op/shuffle controls
- GPU 空闲后先跑小规模 valid smoke，再扩展到 50 条 valid
- 根据因果结果决定 timing side-channel、latent duration modeling 或停止节奏方向
- 冻结策略后再考虑一次性 heldout S0770 blind evaluation
assigned_to: claude
tags:
- task
- tts
- rhythm
- syncnet
---

# 推进 TTS 节奏因素实验

## Status

本任务已完成。train/valid 节奏因素审计、global/phone-duration 反事实、pause-allocation 反事实以及统一 Wav2Lip/SyncNet valid-only 评估均已结束。heldout S0770 全程未用于设计、选择或验证。

## Completed steps

1. [x] 固定 train/valid/heldout split 与 MFA full timeline
2. [x] 提取 global rate、phone duration、pause 和 boundary displacement 因素
3. [x] 与既有 valid SyncNet 做关联分析
4. [x] 构造并评估 global length 与 phone-duration exact-N controls
5. [x] 识别 sample-rate/AAC/mux 与 Wav2Lip shared-temp 混杂并修复协议
6. [x] 构造 sample-conserving pause-only controls
7. [x] 完成 clean isolated valid-50 pause matrix（250/250）与 bootstrap analysis
8. [x] 更新项目实验诊断笔记

## Final conclusions

- Naive global TTS-to-natural length conversion strongly harms standardized SyncNet: ΔC=-1.16528, 95% CI [-1.33156,-1.00674].
- Relative phone-duration transfer has no stable positive dose response; alpha 0.5/1 versus alpha 0 had Sync-C CIs crossing zero.
- Pause allocation toward natural does not improve Sync-C and mildly worsens Sync-D: alpha 0.5 ΔD=+0.08882, CI [+0.00980,+0.17250]; alpha 1 ΔD=+0.12782, CI [+0.03432,+0.22192].
- Internal-pause edge/reverse controls show no stable positive effect.
- No tested rhythm factor supports adding a duration predictor, length regulator, or timing side-channel to the current enhancer.
- Local phone/viseme boundary acoustics remain a separate possible mechanism, but are not justified as the next model change by the current evidence.

## Calibration finding

此前 raw-TTS 优势被音轨链混杂: 同一 16 kHz 写回链下 natural 与 TTS identity 基本无差异 (ΔC=-0.069, CI [-0.25562,+0.11558])。样本 1 的 24 kHz TTS 复现旧高分 8.134/6.369, 逐样本等价的 16 kHz identity 得到 6.221/7.914; 两次 Wav2Lip 视频帧 frame-MD5 完全一致; 将 raw-TTS 视频流 mux 到 identity 音轨后分数精确变为 6.221/7.914 — 证明旧 raw-TTS 优势来自 MP4/AAC/sample-rate 音轨链, 不是生成唇形。所有未来 downstream 对比必须统一 sample rate、PCM/mux/AAC 参数并保留 processed identity control。

## Artifacts

- `runs/rhythm_timing/20260812_factor_audit/`
- `runs/rhythm_timing/20260812_counterfactual_valid50/`
- `runs/rhythm_timing/20260812_syncnet_smoke10/{summary,analysis}.json`（完整 valid 50×6 矩阵, 300/300 成功）
- `runs/rhythm_timing/20260812_pause_counterfactual_valid50/summary.json`
- `runs/rhythm_timing/20260812_pause_syncnet_valid50/{summary,analysis}.json`
- `runs/rhythm_timing/20260812_mux_calibration1/`
- 脚本: `scripts/42_extract_rhythm_factors.py`、`scripts/43_build_rhythm_counterfactuals.py`、`scripts/44_eval_rhythm_tfg_matrix.py`、`scripts/45_analyze_rhythm_matrix.py`、`scripts/pnp_feature_controls.py`

## Observations

- [status] done
- [result] Valid rhythm matrix completed 300/300 with zero failures
- [result] Global TTS-to-natural length conversion strongly worsens standardized SyncNet
- [result] Relative phone-duration transfer has no stable positive dose response
- [result] Pause allocation toward natural mildly worsens Sync-D and does not improve Sync-C
- [problem] Earlier raw-TTS SyncNet advantage is confounded by 24 kHz versus 16 kHz MP4/AAC audio paths
- [decision] Do not add duration or timing modules based on these rhythm experiments
- [decision] Standardize sample rate and mux/audio encoding before further downstream claims
- [constraint] heldout S0770 remains excluded from all protocol and model selection

## Relations

- relates_to [[tts-exp/docs/experiments/tts-enhancer-feasibility-diagnosis]]
- relates_to [[Scale 0.5 enhancer retrain and downstream evaluation]]
- depends_on [[MFA Alignment]]
- relates_to [[SyncNet Evaluation]]