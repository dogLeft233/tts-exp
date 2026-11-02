---
title: Codex checkpoint - 2026-10-26T04-00-00Z - Full fit OOV repaired and three reconstruction
  failures isolated
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t04-00-00-z-full-fit-oov-repaired-and-three-reconstruction-failures-isolated
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T04:00:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a175f1-7164-7510-ae4b-f5af8e9d0fde
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-26T04-00-00Z - Full fit OOV repaired and three reconstruction failures isolated

## Summary
完整拟合集MFA零OOV通过；普通HiFiGAN候选较prematched显著改善重建，但300句中3句仍低于原STOI门，用户授权继续fit-only修复与全量复验。

## Story
原目标是验证逐音素静态、动态和联合TTS差方向能否带来生成端响应，并区分评分音轨偏好；11臂及联合唯一主检验已确认。当前用户连续收窄为先做好拟合集技术校准，不降门、不删样本，开发和确认均不动。设计与执行事实分别见 [[逐音素差方向算法与验证方案]] 和 [[逐音素静态动态差的拟合集技术校准与40人验证]]。

固定20句同一L6特征只换声码器，prematched/regular STOI中位为0.809534/0.900331；加16帧上下文未救回prematched。上游普通权重用于直接L6，prematched用于近邻平均特征，支持训练条件错配为主要因素。普通权重仍仅候选，未正式切换共享适配器。完整100fit/300自然重建中位0.898819，S0168W0257、S0356W0277、S0039W0124分别0.744469、0.736995、0.745960，低于原单句0.750；中位门0.850及其余门不变。S0168仅恢复非语音区原波形后诊断STOI0.967398，其余两句同类诊断仍低；这些混合波形不作合格重建。固定−16dB编码前RMS只使后两句过门，未使三句全部通过；单位方差更差，改训练补零未解决。用户最新授权继续定位并尝试修复后完整拟合集复验，尚未执行新修复。

MFA原缺涯/撰/膛扩展为全fit26缺字，独立带来源overlay修复；完整MFA又修綠别名及显式繁简位置映射。两条40.88秒TTS尾部孤立重复词标签只有在SQLite明确sil证据、全区静音、重复末字且恢复全文精确一致时派生去标，不改音频或phone时钟。完整1200条通过零unknown/丢失/非单调/文字位置错误，38377严格同位置整字phone序列配对尚非动态覆盖率。原词典SHA不变。

## Working State
权威run为 runs/phoneme_direction_20261026_repair01/；本地37份交付文件SHA读回一致，WAV、特征、TextGrid、数据库与日志仍在Ubuntu同名目录。原attempt01和部分失败JSON保留。当前状态HOLD_RECONSTRUCTION_THREE_FULL_FIT_FAILURES，numeric_protocol_frozen=false，无新TFG/SyncNet或模板估计。固定20句配对ASR及自动MFA时间一致性通过，不等同全量内容/边界真值。

机器上大量其它聊天脏文件存在，不可混提交。当前HEAD仅本地验证，当前分支无PR。Ubuntu约52GB可用；其它聊天的数据生产器正在运行development batch032，这是独立既有任务，本聊天不干预、不将其说成自己的开发评价。

## Changed Files
scripts/experiments/phoneme_direction_20261026/ 下新增 diagnose_reconstruction.py、resume_diagnostic.py、repair_dictionary.py、audit_repaired_alignment.py、summarize_reconstruction_probes.py、calibrate_regular_fit.py、check_reconstruction_content.py、repair_alignment_normalization.py、canonical_alignment.py、audit_final_alignment.py、measure_full_fit_reconstruction.py、check_roundtrip_timing.py、probe_remaining_reconstruction.py。这些仍未跟踪。共享 wavlm_knn_vc_adapter.py 未改。新增两组OOV/canonical回归测试；更新方案、执行实验笔记、研究地图和run报告。

## Verification
本轮先前实际运行14个phoneme unittest与compileall通过；37交付SHA通过。全300候选STOI实际失败3句，所有长度/RMS/clipping/全局包络lag通过。固定20×3离线ASR共60次，A/prematched/regular CER为0.069164/0.057637/0.051873，regular无单句新增参考编辑错误。固定20句A/R共40MFA，phone匹配100%，中心绝对差中位0ms、p95 15ms。未完成全300ASR/重编码模板保留门或人工边界验收。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前本地HEAD `2b71639c3b555293375d248350999377168d515d`，未验证GitHub可解析，不构造提交链接。
- runs/phoneme_direction_20261026_repair01/report.md、regular_full_fit_quality.json、alignment_final_fit.json、regional_counterfactual_diagnostic.json为复核入口。

## Observations
- [result] MFA全1200条OOV结构修复完成，原词典和音频不改；重建仍有3/300低于原门。
- [decision] 候选改动只在fit验证，沿用全部原门，保留全部样本；开发和确认不启动，无自动恢复40dev授权。
- [blocker] 直接L6普通声码器仍有非语音区和语音内失真，未形成统一且通过全量的修复规则。
- [next_step] 对3个失败及固定控制做频谱/局部时钟/底噪定位，测试统一候选规则，再用全部300句原指标复验。

## Relations
- continues [[Codex checkpoint - 2026-10-26T03-17-02Z - Phoneme calibration held on reconstruction and OOV]]
- relates_to [[逐音素差方向算法与验证方案]]
- relates_to [[逐音素静态动态差的拟合集技术校准与40人验证]]
