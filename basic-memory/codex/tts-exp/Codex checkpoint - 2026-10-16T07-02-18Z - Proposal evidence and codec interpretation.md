---
title: Codex checkpoint - 2026-10-16T07-02-18Z - Proposal evidence and codec interpretation
type: coding_session
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c6c1e7bd005abeaa4a2cc52991391ef2aaec5223
started: 2026-10-16T07-02-18Z
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a139e9-67ce-7b41-90f2-5b7019492a69
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t07-02-18-z-proposal-evidence-and-codec-interpretation
---

# Codex checkpoint - 2026-10-16T07-02-18Z - Proposal evidence and codec interpretation

## Summary

中文投稿 proposal 已整理成投屏用流程图与短表格；当前核对明确，高频修复实验解释特定编码链损失，不证明高频比低频更重要。

## Story

用户要求根据已完成实验和领域文献形成中文 proposal，突出问题、假设、研究路线和贡献，未来计划仅保留短节。后续要求去掉冗长解释，改用流程图和精简结论，删除将不同发声实例的交叉换轨低分作为独立贡献的段落。当前正式文档为 `basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md`，已采用会议展示版结构，评分音频控制 A 在左、生成输入控制 B 在右。

主线为原生评分现象→定位评价链与生成端响应→检验样本外解释能力→检验自然音轨下迁移。区分 SyncNet 高分、生成端响应、真实视觉/感知改善，不将它们等同。LibriSpeech 按用户指出的泄漏排除独立英语与机制推断。不同发声实例局部时钟失配保留为附录归因边界，不作为独立创新。通用增强头仍暂停，不恢复训练或服务器任务。

最新用户质疑“去高频降分、补回恢复有什么意义，损伤低频是否也降分”。核查 canonical run 报告后，正确回答应承认缺少等强度低/高频损伤比较；编码音频的低频恢复小，不证明低频损伤不重要。更有研究意义的是固定视频、去编码延迟、共同 INTERIOR 支持下，交换自然与 TTS 的评价编码链使相对优势从 +1.308 反转到 −0.418；统一源 PCM 为 +0.287，95% CI [−0.090,+0.648]。高频修复只是定位该编码链影响的机制证据。此批结果不能推广为全部 TTS 优势由编码造成，另一 Wav2Lip/云 Qwen 队列仍有 PCM 剩余差。

## Working State

proposal 是已保存的本地文档，本轮最新问题尚未请求修改其表 2；下一回答应先澄清科学含义。仓库存在大量其他任务的修改与未跟踪文件，不清理、不提交。当前 Git SHA 为本地核验值，未通过 GitHub 验证；gh 命令不可用，因此无已确认 PR 身份。

Basic Memory MCP 本会话未提供 read_note/search_notes/write_note；本 checkpoint 用 CreateNew 写入配置的本地项目记忆目录，防止覆盖已有 handoff。未声称 MCP 创建成功或云端索引已同步。对项目笔记与同 session 旧 checkpoint 的检索使用本地全文搜索；同 session 无命中，未添加推测的 lineage。

## Changed Files

- `basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md`：本会话此前正式化并压缩为会议展示版；删除独立有效性约束节，控制图 A 左 B 右。本轮最新质疑未改正文。
- 本 checkpoint 为新增不可覆盖 handoff；未改研究地图、实验报告或代码。

## Verification

此前文档检查包括本地证据链接、数值与 Markdown 围栏配对；展示版有 8 个 Mermaid 图和 4 个折叠详情块。最新核对实际读取 `runs/tts_clock_cross_20260926/report.md` 与 `runs/syncnet_codec_mechanism_20260926/report.md`：编码链交换 +1.308 / −0.418、统一 PCM +0.287 区间跨零；特征恢复对照低频 +0.111/+0.102/+0.122，高频 +0.967/+0.829/+0.889，分别为自然47/TTS47/真实12。低频恢复并非等损伤频带重要性比较。未执行新实验或代码测试。

## References

- `basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md`
- `basic-memory/Research/研究主线与证据地图.md`
- `runs/tts_clock_cross_20260926/report.md`：固定视频评价编码路径反转与 PCM 对照。
- `runs/syncnet_codec_mechanism_20260926/report.md`：码率、频带、MFCC 与真实波形修复。
- `basic-memory/docs/experiments/43-processing-chain-predictive-audit.md`：来源 AUC 0.972，样本外分差 R² −0.140，非因果解释比例。
- `basic-memory/docs/experiments/44-repeated-take-crossover.md`：普通发声实例失配足以产生交叉降分。

## Observations

- [decision] proposal 主要呈现已完成结论，未来验证短写；优先图表与单句，避免无信息的归因警告成为独立章节。
- [result] 具体评价编码链能在视频不变时反转历史自然/TTS相对评分；这比“破坏频带导致降分”更直接对应比较有效性。
- [decision] 高频修复属于特定编码损失的定位证据，不证明高频普遍优于低频，不换算 TTS 总优势解释比例，也不单独证明评价器错误。
- [next_step] 回答用户低频质疑，说明贡献在编码链如何影响相对比较，并建议将编码链反转前置、频带修复作为支持证据；收到修改要求再改 proposal。

## Relations

- relates_to [[研究主线与证据地图]]