---
title: TTS 声学变化重组轨迹与剩余项干预：实现与结果
type: experiment
permalink: tts-exp/experiments/tts-声学变化重组轨迹与剩余项干预实现与结果
status: concluded
hypothesis: 共享 phone 轨迹 P 与剩余项 R 的等幅干预是否解释 TTS 的 TFG 优势
result: S=+0.014、A=+0.132；b_N=-0.004、h_N=-0.003、h_T=-0.020、J=-0.024，六项中 S、b_N、h_N、h_T、J 的 Bonferroni
  区间均跨 0
conclusion: 表征诊断阳性，功能干预未支持声学重组机制
report: runs/tts_acoustic_reorganization_20260916/report.md
inputs: runs/mfa_linear_trajectory_ablation_20260916；5 donor/10 evaluation；S0765
outputs: runs/tts_acoustic_reorganization_20260916/
model: bshall/knn-vc c616845c4e309e24d5927f15adbdf277a3d65358；WavLM-Large L6；prematched
  HiFi-GAN；Wav2Lip；SyncNet V2
code_paths:
- scripts/experiments/tts_acoustic_reorganization.py
- scripts/experiments/check_tts_acoustic_reorganization_independent.py
- tests/experiments/test_tts_acoustic_reorganization.py
- tests/experiments/test_check_tts_acoustic_reorganization_independent.py
tags:
- tts
- mechanism
- trajectory
- intervention
- syncnet
---

# TTS 声学变化重组轨迹与剩余项干预：实现与结果

## Observations

- [status] concluded
- [hypothesis] 在固定跨句 phone 模板方向 P、等幅削弱剩余项 R/P 的干预下，检验 TTS 是否通过减少无益剩余变化并保留可重复发音动态而获得 TFG 优势。
- [protocol] `tts_acoustic_reorganization_v1`；5 条 donor（sample 1–5）只建模板，10 条 evaluation（sample 6–15）统计；17 个共享 N/T phone 模板；每句有效干预 occurrence 为 9–20 个，覆盖门全部通过。
- [implementation] 新增 `scripts/experiments/tts_acoustic_reorganization.py` 与 `tests/experiments/test_tts_acoustic_reorganization.py`。实现 prepare/features/audio/render/score/analyze/validate；父 Z_N/Z_T 与 N_100/T_100 conditioning 逐元素核验；原生 TTS 重新提取 Z_T_RAW；固定父 score_box 作为 Wav2Lip box；支持 resume、3 位 CSV 分数和 raw/曲线精度。
- [result] 工程完整：70 条音频、70 个视频、210 个科学评分、3 个控制；音频无 clipping，所有特征/音频/视频/曲线和统计独立复算通过。SyncNet 使用绑定的 `[redacted-local-path]`，因运行时 CUDA 驱动不可用而采用 CPU；模型与 checkpoint hash 未改变。
- [result] 六个预注册主统计（均值；Bonferroni 99.1667% 区间）：S=+0.014 [-0.005, 0.032]；A=+0.132 [0.097, 0.161]；b_N=-0.004 [-0.044, 0.035]；h_N=-0.003 [-0.040, 0.034]；h_T=-0.020 [-0.048, 0.020]；J=-0.024 [-0.067, 0.042]。只有 A 的区间下界为正，S、b_N、h_N、h_T、J 均跨 0；剩余项削弱的自然收益 b_N 均值略为负。
- [control] ID6 的 N_RAW/T_ID 独立重复曲线最大绝对误差均为 0；N_RAW 音频右移 3200 samples 得到 k*=2→7，符合预期 +5 帧。
- [quality] 未进行人工听检，状态为 QUALITY_NOT_ASSESSED；因此不能排除声码器/音质混杂。
- [conclusion] 模板正确减错标签的语义诊断 A 支持“共享轨迹在表征中可重复”的描述；S 的均值略为正但校正区间跨 0，不能据此确认原生 TTS 的剩余比例更低。P/R 等幅功能干预也没有显示削弱 R 对自然音频有利、削弱 P 对两来源有害，或自然相对 TTS 的剩余项额外帮助 J。按预注册规则，不允许写成该声学重组机制得到共同支持；本实验只把机制定位收窄为“模板语义诊断阳性，来源残差与当前功能干预均未判支持”。
- [boundary] 结果条件于 S0765 单说话人、5 条 donor、历史 TTS/声码器/Wav2Lip/SyncNet 链和非真人 SyncNet 评价；不能直接推出泛化或启动可部署增强头。

## Relations

- validates [[TTS 声学变化重组的轨迹与剩余项干预 Spec]]
- follows [[MFA-linear 轨迹增益的匹配项与背景项分解 2026-09-16]]
- relates_to [[TTS 原生增益来源的生成端与评估端交叉诊断]]

- [audit] 复核后补齐逐句 b_T/J、来源优势、replace 参照、D 与固定自然 lag 距离改善、C 的 B/D 分解，以及 `diagnostic_plots.png`（b/h、六主统计区间、f_R 来源对比）；同时锁死父输入顺序并检查 float32 干预预算/均值误差。
- [validation] 最终 `validation.json` 21 项检查（含独立短脚本复算）全部 PASS；ruff、py_compile 与 10 个 CPU 契约测试通过（pytest 10 passed）。