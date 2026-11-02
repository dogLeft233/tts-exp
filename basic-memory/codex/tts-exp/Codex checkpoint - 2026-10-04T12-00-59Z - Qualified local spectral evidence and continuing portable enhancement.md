---
title: Codex checkpoint - 2026-10-04T12-00-59Z - Qualified local spectral evidence
  and continuing portable enhancement
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t12-00-59-z-qualified-local-spectral-evidence-and-continuing-portable-enhancement
status: open
agent: codex
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
hostname: LAPTOP-E133GSLE
started: '2026-10-04T12:00:59.8366789+00:00'
repository: dogLeft233/tts-exp
project: tts-exp
branch: codex/cross-tfg-audio-head-20261003
capture: deliberate
model: gpt-6.1-sol
username: dogle
trigger: compact
repo_root: E:/Documents/tts-exp
git_sha: 769d5358305dca224924d83f6a12f614e1c9f302
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T12-00-59Z - Qualified local spectral evidence and continuing portable enhancement

## Summary

跨 TFG 增强头继续研发：V4 的完整开发集结果未超过 RMS 控制，原数组与原音独立重算已将下一轮机制候选收敛到局部谱变化；可信双盘恢复后 Muse 旧输出已精确退休，正在释放共享候选实际运行所需空间。

## Story

用户最初要求恢复 TTS 为何增强 TFG 的研究路线，随后明确要求产出适配多种 TFG 的自然音频增强头，并持续迭代至方法完备。增强音频只驱动视频，最终恢复原自然 PCM 和原时钟再评分；阿里云 Qwen ASR/TTS 已获授权，凭据只在运行时传递。root 维护理论、协议、Basic Memory 和独立核验；模型代码与实验由 GPT-6-LUNA/MAX 子代理承担。

V4 在 20 clip/10 speaker ×3 seed、三个前端、540 行九路配对 FULL 支持上，相对 identity 的 Cbest 等权 +0.109，相对 RMS 等权 −0.044，三个模型均为负。这给出明确工程方向：不能继续把单个 Wav2Lip 代理上的收益当作跨模型收益。冻结的三个共享频谱候选来自完整历史 fit360（8 clip/4 speaker ×单个共享 seed ×15 methods ×3 models），尚未运行其 dev540，因此未据此选择新系数。

局部掩膜 CPU28 和 dev20 精确重放实际完成。47MB NPZ 经完整范围恢复后，root 直接验证 196 数组和 180 字段，并从 28 段原音独立重算 340 个关联字段。时间变化部分 δ 中，dev 的频率共同项 a 占 29.658%，其余频谱重分配 r 占 70.342%（均为未加权 logmask 范数）。a 与原音局部 logRMS 的相关 −0.082，与正向 logmag flux 的相关 +0.434。这支持继续研究频谱变化门控，但仍须实际视频和匹配物理帧功率的消融来检验因果；稀少 quiet 帧不能支持静音机制结论。

存储恢复遵循 ADR012：C/F 两份全包 hash、F 严格恢复、最新全树和依赖审计后，只退休具体旧 run。Muse 9644 文件恢复合格，旧 run 本次实际删除并生成固定收据；代码、控制、计划和归档仍保留。下一步用 allocated bytes +1GiB 的保守预算重新归档此前中止但源完好的 failed-fit，再处理旧 Ditto 输出，使冻结 shared540 达到原定 5GiB SHM 入场条件。

## Working State

Durable anchors: `openspec/changes/develop-cross-tfg-audio-head-20261003/`、现有 BM Task40/Experiment39、ADR009/011/012、完整报告25；报告26和最终可用方法尚未完成。确认集80 clip/40 speaker及保留模型 LatentSync 继续封存，禁止因开发失败打开确认集。

Machine-local state: HEAD `769d5358305dca224924d83f6a12f614e1c9f302` 本地未发布，无 PR。CONTEXT 和 transfer-iteration 有新的未提交研究/资源说明，历史 dirty 未改变。忽略的 HANDOFF、scratch、runs 和归档需要保留。私有 PTY50274 的 bridge06 端口10887/PID252640 修复远端 stdout EOF 截尾；旧 bridge05 空闲。服务器为 root@connect.weste.seetacloud.com:12349，密码与 API key 不入文件。

## Changed Files

- `CONTEXT.md` 与 `openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md`：实际 V4 结果、方法继续迭代及已测资源预算；新的局部机制段待补入。
- `scripts/audio_enhancement_eval/decompose_v4_logmask_attempt03.py`、`decompose_v4_delta_common_mode_attempt04.py`：子代理生成的实际 CPU 分解、关联统计和描述图；第四版图仍待 root 视觉核验。
- `.codex/scratch/remote_bridge_20261001_attempt06.py`：等待 EOF 并排空接收缓冲，流不完整时失败；实际 3MB 原数据读取已合格。

## Verification

V4 joint `runs/cross_tfg_audio_head_20261003/v4_join_root_independent_readback_attempt01/archive.tar.gz`：18,784,301 B，SHA349c5b62f09c1a775013039d2abb3bd72d7c6b6233531fb7f87e6398fff981d4；root 恢复1721成员/1700资产/540原矩阵，最大数值差5.72e−6。20实际 ASR、297字0错误、原音 PCM/clock及视觉支持均已独立合格。

Mask `runs/cross_tfg_audio_head_20261003/v4_logmask_decomposition_attempt02_arrays_readback_attempt05/v4_logmask_arrays.npz`：46,753,644 B，SHA69925e7659e8a3e2defd5acb55a7aab5e6b92660fcbd8f9fc377aa9c1e1982be。原数组最大差2.13e−14，原音关联最大差1.33e−15。失败的传输尝试保留，新桥真实范围读取也验证了完整内层 JSON。

SadTalker archive06：3,085,913 B，SHA3a2d9f79c072757818f263b135511da4ff81f7bc8c88e3aa5c3698746d83bb83；260成员/14数组/精确35帧25fps/原音 PCM独立合格，root10字段最大差1.91e−6。n1 RMS−identity Cbest −0.044，仅模型工程资格，不是候选迁移结果。

Muse exact retirement actual PASS：2819 B收据 SHA087b7278b43caafe81c0ac164d1dac4b7bc64cdabf2ce513f1181b1dc64923fe；释放683,323,392 allocated bytes。收据下载、保护路径复核和临时远端 tar 单文件回收待执行。没有运行 pytest 或新增单元测试。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)

## Observations

- [result] V4 已有完整三模型开发集科学阴性结果；整个可用增强头目标仍在进行，下一轮采用共享前端证据和局部谱结构。
- [decision] primary 为原音换轨后的 Cbest；D0/Canchor/B/offset 分解和独立视觉作为控制，不能用动作幅度或 fit-only 结果替代迁移收益。
- [decision] phase/clock 保持、原音 RMS 控制及 ADR011 的共同 κ/帧功率控制继续执行；确认集保持封存。
- [blocker] shared540 已 stage00–03，04CPU/05GPU 尚未运行；入场要求 SHM≥5GiB，不能降低原门槛。
- [next_step] 审核并执行新的 failed-fit 保守归档与可恢复旧输出回收，然后实际启动冻结 shared540。

## Relations

- continues [[Codex checkpoint - 2026-10-04T11-18-27Z - Local mask evidence and recoverable storage for portable validation]]