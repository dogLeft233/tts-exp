---
title: Codex checkpoint - 2026-10-16T07-52-12Z - Negative results and transfer paper
  framing
type: coding_session
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c6c1e7bd005abeaa4a2cc52991391ef2aaec5223
started: 2026-10-16 07:52:12+00:00
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
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t07-52-12-z-negative-results-and-transfer-paper-framing
---

# Codex checkpoint - 2026-10-16T07-52-12Z - Negative results and transfer paper framing

## Summary

历史证据梳理把论文问题从单个谱形失败扩展到“TTS原生优势能否迁移为保留自然音轨的驱动收益”，并区分可诊断的负结果与工程失败。

## Story

用户持续要求中文会议proposal有明确核心贡献、短句与流程图，拒绝将换轨失配和编码损伤的显然结果包装成创新。在上一交接之后，按用户要求将3.3的单一谱形核心结果改为候选解释的辅助检验，并同步地图。随后广泛检索两类实验目录，清点309份实验Markdown的标题与摘要，精读关键正式报告及后续修正；不是逐篇全文复核309份。

用户最新问题是这么多失败究竟能说明什么、是否应提出领域警示、论文应怎样组织。本轮讨论尚未再次改写proposal。拟回答：失败数量不是贡献；可发表价值来自明确假设、有效干预、具体失效定位、修复或适用边界。主线候选是TTS优势的可迁移性，不把当前幅度正结果说成唯一机制，也不将代理指标失败说成整个领域错误。

## Working State

当前proposal仍以幅度机制为核心，3.3包含分离度、稳定性、局部特征长度和谱形辅助证据；尚未纳入新梳理的ASR、MFA-linear、输出时间定位和独立VSR链条。研究地图投稿段同样仍反映此前幅度聚焦版本。用户此轮是讨论，不是明确授权新一轮改稿。

主要证据入口：
- `basic-memory/docs/experiments/25-tts-tfg-mechanism.md`：40说话人幅度干预，未截断坐标+0.171、编码方向+0.161，有99%区间；受体时刻保留的谱形移植负结果仅限所测方法。
- `basic-memory/docs/experiments/39-native-input-factor-mechanism-screen.md`：2个复用说话人的3模型筛查，0/9组两人同时通过，不是普遍无效证据。
- `basic-memory/docs/experiments/LRS3 ASR 错误与局部同步低分相关性实验.md` 与 `ASR-targeted local replacement prototype.md`：自然语音ASR错误与低同步弱相关rho0.1448，低同步格中13.71%落在ASR错误区；13个fit-only来源局部替换median Sync-C -0.0341、95%CI[-0.2092,-0.0054]。未证明干预改善整体可懂度，不得宣称提高ASR却损坏唇形的普遍规律。
- `basic-memory/docs/experiments/33-mfa-linear-content-pause-prosody-mapping.md` 与 `36-mfa-linear-heldout-speech-residual.md`：缺失停顿定位及修复；新17条5说话人中6个缺失停顿案例修复均改善、均值+0.337，但全17条修复后仍比自然基线低0.674。修复也改变部分停顿外内容，不能全部归因单一操作。
- `basic-memory/docs/experiments/42-output-owned-native-coordinate.md`：修正输入行与输出帧映射后，2句预设视觉功能响应从0/2变2/2；原自然音轨正式评分未改善。约61.3%扰动能量违反共享前端切片一致性，是原生坐标不等于波形可实现的具体示例，尚非跨模型定律。
- `basic-memory/Experiments/无人工条件下的独立纯视觉内容与时间校准审计 2026-09-26.md`：42评估对视觉内容相对指标TTS正向；同帧数后不稳，直接视频时钟迁移仍负。不能把全部TTS现象说成评分假象。

机器本地状态仍9个修改项、1004个未跟踪项，未清理或提交。增强头仍暂停；本轮没有GPU实验或代码变更。Basic Memory MCP未暴露，checkpoint通过本地目录独占创建，索引未确认。GitHub API此前403限流且gh缺失，PR和远端SHA可解析性仍未确认；SHA只作本地证据。

## Changed Files

此前用户授权编辑：`basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md` 与 `basic-memory/Research/研究主线与证据地图.md`。当前讨论仅增加本交接，不修改上述文档。

## Verification

此前改稿已检查相对证据链接、3个Mermaid块、4个details及git diff --check。本轮重新核对ASR原报告数值、MFA新说话人结果、报告42输出映射和前端一致性数字；未运行测试，因为无代码变更。已检索官方论文Wav2Sem与THEval，并读THEval arXiv HTML；没有据此确认本项目新颖性或投稿可接受性。

## References

- [Wav2Sem](https://arxiv.org/abs/2505.23290)：3D语音面部动画的语义解耦有效结果，不可用本项目冻结2D模型负结果概括反驳。
- [THEval](https://arxiv.org/abs/2511.04520)：质量、自然度、同步的多维评测；一般指标局限已不是本项目独有贡献。

## Observations

- [decision] 不将工程阻塞、控制失败或未充分实现目标的试验归入科学阴性；小样本探索与独立确认分开。
- [decision] 不把“很多失败”累计为“幅度唯一原因”；论文按问题和排除的解释组织，避免实验日记。
- [result] MFA-linear提供比单独谱形失败更完整的诊断—修复—残差证据链，ASR适合作为代理目标的辅助边界。
- [blocker] 尚未形成统一端点下跨模型确认的通用增强方法，也没有足够依据声称SyncNet与人类感知矛盾。
- [next_step] 回答用户如何将负结果写成有边界的实证论文；如后续要求改稿，以TTS优势迁移问题组织，保留幅度正向机制、MFA具体修复与有限负结果，不扩大已证结论。

## Relations

- continues [[Codex checkpoint - 2026-10-16T07-22-01Z - Proposal centered on generator mechanisms]]
- relates_to [[研究主线与证据地图]]