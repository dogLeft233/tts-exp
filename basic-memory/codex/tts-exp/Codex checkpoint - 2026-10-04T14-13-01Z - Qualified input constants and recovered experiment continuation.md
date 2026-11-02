---
title: Codex checkpoint - 2026-10-04T14-13-01Z - Qualified input constants and recovered
  experiment continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t14-13-01-z-qualified-input-constants-and-recovered-experiment-continuation
username: dogle
started: '2026-10-04T14:13:01.2247588+00:00'
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
git_sha: 10bbea2f3808c31d0fd36bd30d00f81268ca9221
model: gpt-6.1-sol
repository: dogLeft233/tts-exp
project: tts-exp
agent: codex
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
branch: codex/cross-tfg-audio-head-20261003
hostname: LAPTOP-E133GSLE
status: open
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T14-13-01Z - Qualified input constants and recovered experiment continuation

## Summary

扩展训练音频与 Ditto 固定输入仿射常数已独立核验；三模型候选验证处于待空间执行阶段，完整恢复后的旧输出现在具备精确回收条件。

## Story

用户最初要求还原 TTS 改善 TFG 的研究路线，随后把目标推进为可用于多种 TFG 的自然音频增强头：增强音频只负责驱动模型，交付视频始终换回原始自然 PCM 和原始时钟。主代理负责机制、实验设计和独立复核，实验与代码由 GPT 6 Luna / max 子代理负责。最新授权允许在需要核验转写时使用阿里云 Qwen ASR；密钥仅留在私有运行时，不进入文件或记忆。

V4 原生验证已经完整覆盖 20 段、10 位说话人、三模型和三个种子；相对 identity 的平均 Cbest 为 +0.109303，但相对 RMS 对照为 -0.044300。因此后续选择必须在 RMS 基线上取得跨模型收益。已冻结的三个共享局部谱候选等待 540 行开发集原生生成；与归档的 identity/RMS 合并后统一评价 900 行。确认集 80 段、40 位说话人仍密封，选定方法前不打开。

本轮把扩展拟合所需的 40 个正式端点核验到原始归档成员、PCM16 与规范 float32 PCM：20 个官方说话人归档全部匹配，6,944 个 WAV 的合格成员清单重建了 320 个历史槽位，40 个端点没有替换或缺失。读取部署 ONNX 中两个标量 Constant 后，确认 Ditto 输入计算为 (x - 2.586069967946969e-05) / 0.20523720979690552，实际 4 字节 float32 载荷也独立复现。这个结论只描述输入仿射层；编码器内部归一化和生成效果仍由后续实验测量。输入响应探针正按 Wav2Lip mel、MuseTalk Whisper log-mel、Ditto 固定仿射分别准备，不把输入张量变化写成编码器效果。

存储恢复保留了旧实验的全部可追溯证据。失败拟合的 1.291 GB 压缩包已传到 C 与 F，323 个范围和完整文件 SHA 均通过；F 完整恢复 46,089 个文件。新的只读 audit03 和主代理对其成员清单的独立核验均通过，精确回收 worker 已准备但尚未执行。旧 Ditto 输出还需要单独恢复并回收，随后满足共享生成所需的 5 GiB SHM 空间门槛。内容 guard04 的完整源代码与所有 31 个包成员已静态核验；它会核验每个实际生成音频路径，再按精确 PCM16 WAV 文件字节去重调用 Qwen ASR。540 行终态尚未生成，因此没有运行新 API 检查。

## Working State

持久状态以现有 Task、实验笔记、ADR011/012/013 和冻结协议为准。Task snapshot43 / experiment42 已写入且读回通过；其中回收状态早于最新 audit03，通过后应更新下一版本。HEAD `10bbea2f3808c31d0fd36bd30d00f81268ca9221` 为本地未发布提交，包含 CONTEXT、transfer-iteration 和 constants05 采集器。历史五个 dirty 文件、未跟踪研究源和测试文件保持原状；没有执行 pytest 或添加单元测试。

机器本地：私有 PTY 50274 保存 SecureString 凭据，bridge06 在端口 10887 / PID 252640；恢复后先验证它仍可用。服务器地址 root@connect.weste.seetacloud.com:12349，密码不入文件。SHM 工作根 /dev/shm/cross_tfg_audio_head_20261003；项目执行根 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003。正确旧 Ditto 目标是 stageb_ditto_attempt03_retry04/run，不能沿用旧快照中的简写。C/F 归档目录和 .codex/scratch 均属本地状态，应按封存 receipt 验证后使用。

## Changed Files

- CONTEXT.md 与 openspec/changes/cross-tfg-audio-head-20261003/transfer-iteration.md：已提交的扩展拟合与输入层证据入口。
- scripts/experiments/cross_tfg_audio_head_20261003/inspect_ditto_hubert_input_constants_attempt05.py：已提交的有界 ONNX 标量读取器；实际部署模型 SHA 为 04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697。
- .codex/scratch/failedfit_output_retirement_exec_attempt01/request_index_attempt01.json：未派发的精确目录回收包，必须完整核验 source / requests 后执行。
- .codex/scratch/ditto_retry04_archive_inventory_attempt03/request_index_attempt03.json：未派发的旧 Ditto 只读 inventory 包。

## Verification

- runs/cross_tfg_audio_head_20261003/v5_fit_expansion_member_scan_attempt02/member_scan.json 与 .codex/scratch/v5_fit40_root_raw_member_qc_attempt01.json：官方 20 个 tar、320 个槽位、40 个端点独立 PASS。
- .codex/scratch/ditto_hubert_onnx_input_constants_attempt05/readback_input_constants_audit.json 与 root_constants_numeric_qc_attempt01.json：实际远程 PASS，原始报告 SHA、部署权重绑定和两个 float32 载荷重编码 PASS；没有模型推理。
- .codex/scratch/failedfit_output_retirement_audit_attempt03/remote_audit_receipt_attempt03.json：21769 B / SHA 4fc41b5b2878461a040a45b9c6696b275abe9405da501a87d06943cddaf86f7a，全部只读 gates PASS，活动和待执行依赖均为零。root_audit_manifest_qc_attempt01.json 独立匹配 50,635 条归档成员记录；目录 inode544823 / device118，成员规范摘要 c84c4afcd83ee3a9b08d72f6f2a0f90524f31682515b33b15cb7fbaa81b3116c。
- shared_profile_content_guard_attempt04：73309 B 源与 31 成员包静态核验 PASS；无新 API 调用。缺少完整 20 段或 10 位说话人时主要平均值留空，完整或可用子集单独描述，60 个内容单元全部保留。
- 新共享 540 行尚未生成；900 行 join 已实际完成代码/输入部署，分析阶段尚未执行。所有工程失败和修复 attempt 均保留。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current local unpublished commit: `10bbea2f3808c31d0fd36bd30d00f81268ca9221`; branch codex/cross-tfg-audio-head-20261003 has no PR in the direct GitHub query.

## Observations

- [result] Ditto 当前部署输入为固定标量仿射，音量扰动会到达第一层卷积；未据此推断后续编码器或原生效果。
- [decision] 维持 RMS 对照、原自然 PCM 换轨、完整说话人分母和三前端模型迁移目标；确认集保持密封。
- [decision] 回收精确旧输出前，必须 C/F 完整 SHA 与成员恢复通过，并立即重查树摘要、inode 和活动/排队依赖；外部代码、协议和封存证据保留。
- [blocker] SHM 尚未满足共享生成 5 GiB 门槛；失败拟合 run 的精确回收已具备只读证据，旧 Ditto 恢复回收随后执行。
- [next_step] 完整复核并派发失败拟合 exact-run 回收包，取得实际回执和外部保留项核验，再单独回收重复压缩包并推进旧 Ditto 恢复，使共享 540 行生成可启动。

## Relations

- continues [[Codex checkpoint - 2026-10-04T13-36-34Z - Qualified fit audio and recoverable shared-model execution]]
- advances [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]