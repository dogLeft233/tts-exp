---
title: Codex checkpoint - 2026-10-16T07-22-01Z - Proposal centered on generator mechanisms
type: coding_session
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c6c1e7bd005abeaa4a2cc52991391ef2aaec5223
started: 2026-10-16 07:22:01+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a139e9-67ce-7b41-90f2-5b7019492a69
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t07-22-01-z-proposal-centered-on-generator-mechanisms
---

# Codex checkpoint - 2026-10-16T07-22-01Z - Proposal centered on generator mechanisms

## Summary

中文会议 proposal 已围绕生成端幅度通路与谱形迁移检验重写，编码损伤和条件分类不再承担核心创新。

## Story

用户希望把既有大量实验组织成清楚的投稿贡献，持续否定“换轨不对齐会降分”“音频损伤会降分”等显然控制作为创新。最新授权是调整论文思路，让各节明确解决问题，同时保持中文、流程图、精炼单句的投屏形式。

本次回读研究地图与正式报告25、43，并检索 Wav2Sem 正式论文，将核心问题收敛为“哪些 TTS 变化真正影响生成器，哪些能移到自然语音”。正文保留原生优势作为动机，以报告25的40说话人干预作为主要贡献：幅度经未截断 log-mel 坐标和编码方向产生正向固定音频评分响应；保留受体时刻、音高和能量的联合谱形移植相对自身重建仍降分。这两类干预使用各自端点和支持集，不相加、不换算总TTS优势比例，也不宣称整个机制已跨模型确认。

编码审计压缩为附录的处理路径换位结果，清楚区分处理路径与发声内容。预测失败作为选择直接干预的辅助证据；交叉发声控制和小型增强头试验移出本proposal。研究地图增加投稿主线索引，保留原来的科学状态与历史建议。

## Working State

持久产物为 proposal 和研究地图；没有新GPU实验、训练、模型前向或代码改动。增强头仍暂停。主文包含3个Mermaid流程、3张结果表和两项具体贡献；未来验证仅保留一小节。

机器本地工作树已有大量无关改动，检查时9个修改项、1004个未跟踪项，未清理或提交。Basic Memory MCP读写接口未暴露，因此交接记录使用本地记忆目录落盘，未声称MCP索引成功。GitHub只读API返回403限流且gh不可用；PR及远端提交可访问性未验证，SHA只作本地锚点。

## Changed Files

- `basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md`：改为“从 TTS 高分到可控驱动：唇形生成的输入机制与迁移边界”。
- `basic-memory/Research/研究主线与证据地图.md`：增加2026-10-16投稿叙事与证据入口；未修改实验结论。

## Verification

逐项核对报告25的+0.171、−0.009、+0.161、+0.020及直接效应差，与两方向谱形移植−0.197／−0.275及区间。编码路径表与`runs/tts_clock_cross_20260926/report.md`一致。proposal本地证据链接检查无失效；3个Mermaid块的6个围栏、4组details均闭合。未运行代码测试或重新计算实验，未进行浏览器渲染验收。

## References

- `basic-memory/docs/experiments/25-tts-tfg-mechanism.md`：主要机制与谱形移植证据。
- `basic-memory/docs/experiments/43-processing-chain-predictive-audit.md`：辅助预测证据。
- [Wav2Sem正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Wav2Sem_Plug-and-Play_Audio_Semantic_Decoupling_for_3D_Speech-Driven_Facial_Animation_CVPR_2025_paper.html)：相关研究已通过检索核验。
- 本地Git提交：`c6c1e7bd005abeaa4a2cc52991391ef2aaec5223`，未验证远端可解析。

## Observations

- [decision] 投稿定位为机制实证分析；现象复现建立动机，幅度通路定位和指定谱形迁移边界承担核心贡献。
- [decision] 不把编码敏感性、普通换轨损失、条件可分类性或未通过的增强原型包装为独立方法创新。
- [result] 当前机制证据主要限定Wav2Lip；没有建立通用自然音频增强方法，也没有充分独立感知同步确认。
- [blocker] Basic Memory MCP未暴露，使用本地文件作为交接产物，索引状态未确认。
- [next_step] 围绕新proposal审阅两项机制贡献是否足以支撑投稿定位；只有另获实验授权后才开展第二生成器复核或独立盲评。

## Relations

- continues [[Codex checkpoint - 2026-10-16T07-02-18Z - Proposal evidence and codec interpretation]]
- relates_to [[研究主线与证据地图]]