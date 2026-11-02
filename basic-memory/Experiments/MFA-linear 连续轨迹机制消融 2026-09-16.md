---
title: MFA-linear 连续轨迹机制消融 2026-09-16
type: experiment
permalink: tts-exp/experiments/mfa-linear-连续轨迹机制消融-2026-09-16
status: concluded
date: September 16, 2026
protocol: mfa_linear_trajectory_ablation_v1
run_path: runs/mfa_linear_trajectory_ablation_20260916
result_status: incomplete
tags:
- mfa-linear
- trajectory-ablation
- syncnet
- mechanism
- incomplete
---

依据 [[MFA-linear 连续轨迹机制消融实验 Spec]]，固定 S0765 的 15 条历史样本，冻结 WavLM-Large L6、prematched HiFi-GAN、Wav2Lip 和 SyncNet，执行八臂音频、固定 crop 与共同支持评分流程。无训练、无新 TTS 请求、未换样本或模型。

## Observations
- [status] concluded
- [progress] 运行已结束；协议验收状态为 incomplete
- [protocol] prepare 15/15、audio 120/120、render 120/120；15 个固定 score box 文件、输入/模型 SHA 和 occurrence mask 均已保存。
- [quality] 120 条音频全部有限、越界样本数为 0，重合成臂均精确到自然音频长度；smoke 八臂和 15 cell 连通性通过。
- [result] SyncNet 完成 120/225 个 cell，来自 sample 1、4、6、7、10、12、14、15；成功曲线全部含 31 个 lag 点且 C/D 可由曲线重算。
- [blocker] sample 2、3、5、8、9、11、13 按固定规则 `range(15,F-15)` 的共同支持分别为 35、31、41、41、45、33、47，低于 spec 要求的 50（对应 F=65、61、71、71、75、63、77）。因此没有生成完整 225 cell，也没有计算阳性参照、R/E/I 或剂量统计。
- [review] 自审修复了 venv 启动器路径解析、Wav2Lip 独立 temp、可中断 render 恢复、score box 独立文件和 incomplete analyze；修复后 py_compile、ruff、6 个协议测试和严格产物审计均通过。
- [conclusion] 本轮不能对“重合成、TTS 来源或音素内动态”作机制结论；120 个成功 cell 只作为技术诊断保留。降低支持阈值、改选更长样本或换协议都必须另立实验，不能把本轮标为完整。
- [report] 详细失败清单与产物指针：`runs/mfa_linear_trajectory_ablation_20260916/report.md`、`analysis.json`、`scores_manifest.json`。
## Relations
- implements [[MFA-linear 连续轨迹机制消融实验 Spec]]
- extends [[MFA-linear 连续轨迹机制消融 support30 疏通 Spec]]
- relates_to [[MFA-linear 连续轨迹机制消融 support30 2026-09-16]]
- relates_to [[TTS 原生增益来源的生成端与评估端交叉诊断]]
## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 执行固定 S0765 轨迹消融协议；发现 7 条短样本不满足共同支持门槛，按 spec 保留 incomplete 和失败清单 | September 16, 2026 | user request / agent execution |