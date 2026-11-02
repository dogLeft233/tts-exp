---
title: Codex checkpoint - 2026-10-04T10-36-05Z - Qualified V4 native results and portable
  method continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t10-36-05-z-qualified-v4-native-results-and-portable-method-continuation
started: '2026-10-04T10:36:05.5741797+00:00'
branch: codex/cross-tfg-audio-head-20261003
username: dogle
trigger: compact
agent: codex
hostname: LAPTOP-E133GSLE
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
status: open
cwd: E:/Documents/tts-exp
project: tts-exp
git_sha: 7088c1e1a563a9b627946f313cbc324b135736ad
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T10-36-05Z - Qualified V4 native results and portable method continuation

## Summary
第四版增强头的三模型开发结果已独立复算，均低于 RMS 控制；完成的原始矩阵与视觉证据已在本地保存，主线转向已冻结的三个跨前端候选。

## Story
用户要求将 TTS 增强机制做成适配多个 TFG 的自然音频增强头，驱动视频后换回原始自然音轨，持续推进到可用方法。根 agent 维护理论、协议和 Basic Memory，代码与实验由 GPT 6 Luna max 子 agent 执行；新授权允许用 Qwen ASR 校验内容。实际20次ASR、80份音频与180个实际渲染音频的完整关联已经通过内容核验，297字零错误。

V4 在20段、10位说话者、3种 seed 的完整开发支持集上，相对 identity 的模型等权 ΔCbest 为 +0.109，但相对 RMS 为 −0.044（Wav2Lip −0.040、MuseTalk −0.029、Ditto −0.064）。额外增强未超过响度控制，不能据此定稿。FULL 主端点保留完整10位说话者；GUARD20 有27个不合格行，完整cohort统计为空，只报告9位完整子集的描述性结果。现阶段不用 guard 子集替换主端点，也没有访问80段确认集。

根 agent 从4个实际桥接响应独立恢复18,784,301字节 raw 联合归档，复核全部1721成员/1700资产/540原始行、PCM/时钟/lag映射，以及42171个数值字段（最大误差5.722e−6，在声明的1e−5范围内）。又独立复算20份原音频与540份视觉数据的34264个语音运动字段（最大误差7.77e−16）：V4−RMS aperture-envelope零lag相关变化约−0.000952。嘴更张开并不自动带来更好的自然语音运动对应。下一步先看实际 mask μ/δ 拆解，再运行已冻结的多前端共享谱形候选，而非继续同一8段单模型拟合。

## Working State
持久设计在 openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md 与 transfer-iteration.md，任务和实验快照已更新到39/38。完整 joint 原始归档及根复核在 runs/cross_tfg_audio_head_20261003/v4_join_root_independent_readback_attempt01/；语音运动独立复核在 runs/cross_tfg_audio_head_20261003/speech_motion_association_attempt01/root_independent_numeric_qc_attempt01.json。联合证据已解除旧 Muse/Ditto run 的分析依赖。

机器本地状态：私有 PowerShell PTY 50274 当前空闲，凭据只在内存。第一棵旧 recovery04 run 已在 C/F 全归档、严格恢复与两次完整树复核后精确删除；2189字节实际receipt SHA9c8873794ed186ec30b122a60a2bf32ca50c87850ef0f569980b1e8bd8a4bee7。其原始远端归档、code/control/计划仍保留，B盘副本hash不匹配，不计入恢复依据。新stat02只读路径检查尚未运行。

Muse 源保留归档实际完成：517027505字节，SHA2b08aacdd723fc9096735c433736fad5591861c24132dbabef0930b85d7f61ed，worker已退出。其 C/F/F严格恢复 packet 已就绪未运行。根盘剩约1.72GB、SHM剩约2.43GB；shared540启动需5GiB SHM，须继续依次归档 Muse/failed shared/Ditto 并有恢复证明后回收精确旧run。不得删除输入、权重、代码、失败记录或最初1.28GB归档。

μ/δ CPU packet 已就绪未执行：.codex/scratch/cross_tfg_audio_head_20261003/v4_logmask_decomposition_attempt01/requests_attempt01/request_index_attempt01.json，绑定28段（20dev+8历史fit）及20段逐样本 bit-exact replay。三个shared候选的540行 plan 已冻结并完成stage00–03，packet07的04CPU preflight/05GPU launch尚未执行。SadTalker实际attempt06已2/2 engineering PASS，最终只读原始 pair 归档packet05待审阅。LatentSync历史92query identity/RMS pair已经独立复算，仅是工程控制，不是候选泛化证据。

## Changed Files
CONTEXT.md 已更新并保持93行；openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md 记录实际 V4 数值和下一步。BM Task39/Experiment38 已写入并精确读回。HANDOFF.md 尚需同步当前结果。本地历史 dirty、未跟踪源码/运行资产均保留；没有暂存或提交本轮改动。

## Verification
实际运行的验证包括：whole-archive/member/source seals；FULL/GUARD支持集和整数lag映射；540原始评分独立复算；20原音与540视觉的语音运动关联独立复算；20次Qwen ASR及180渲染器PCM内容关联。未运行 pytest、未新增测试。本地传输packet AST检查通过但尚未执行传输，不把准备完成写成归档恢复成功。

## References
[Repository](https://github.com/dogLeft233/tts-exp)。当前 HEAD `7088c1e1a563a9b627946f313cbc324b135736ad` 为本地未发布；已验证远端 [commit c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前分支无PR。

## Observations
- [result] V4完整开发三模型均低于RMS控制，已有独立原始数据复算支持该候选结果，整个可迁移增强方法任务仍未完成。
- [decision] 保留完整FULL主端点，GUARD缺失不静默删人；确认集保持封存；继续多前端候选和局部机制驱动方法。
- [decision] 回收研究输出只依据可信C/F完整归档与严格恢复，B盘当前不可信；保留源码、计划、原输入、权重和失败证据。
- [blocker] shared540实际启动受5GiB SHM gate阻塞；源保留归档已在顺序释放空间。
- [next_step] 运行μ/δ CPU28精确回放与 Muse C/F/F恢复，完成有证明的旧run回收，然后启动已冻结的shared540开发比较。

## Relations
- continues [[Codex checkpoint - 2026-10-04T09-56-54Z - Qualified raw matrices and common visual support before transfer effects]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]