---
title: Codex checkpoint - 2026-10-07T19-37-14Z - Running fixed120 scores and repairing
  native render policy
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t19-37-14-z-running-fixed120-scores-and-repairing-native-render-policy
trigger: compact
hostname: LAPTOP-E133GSLE
started: '2026-10-07T19:37:14.8622116+00:00'
capture: deliberate
repo_root: E:/Documents/tts-exp
username: dogle
repository: dogLeft233/tts-exp
git_sha: cd6039cdeca1a7ee3b3ca2e6efd2de2731cc4ef9
branch: codex/cross-tfg-audio-head-20261003
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
agent: codex
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T19-37-14Z - Running fixed120 scores and repairing native render policy

## Summary
固定 Adam120 增强头已实际完成训练和单音频 CLI 核验；完整40条冻结权重评分已启动，三模型迁移验证正修复渲染控制链。

## Story
原目标是解释中文 TTS 对 TFG Sync-C 的增强机制，用户将目标扩展为适配较多模型的自然音频增强头，并要求持续完成失败分析与方法迭代。增强音频只用于生成视频，成片保留精确原自然 PCM 和源时间轴。根 agent 维护理论、协议、Basic Memory、私有 SSH 和独立核验；GPT-6 Luna/max 子 agent 执行代码与实验。

当前方法用32ms/8ms STFT、四频带静态项与正负谱通量门控的相加掩码、逐帧去公共频谱偏移、全局 RMS 归一化和3dB有界约束。Fit40/20说话人 Adam120/3遍训练实际完成，完整120次更新账本和末步 theta 已独立重算。五个原始候选 metadata 文件的字节/SHA、checkpoint参数、constructor身份和七模块源码 pins 均实际核验通过。单条 Fit40 CLI 保留62639样本并生成浮点 WAV；它尚无 Dev/Confirm 评分或选择含义。

固定权重重评分09–12的失败均保留。12已到达实际模型评分，却在持久化首行的 NumPy int32 诊断字段处失败，0条已写结果。新13仅将该字段转为普通整数，不改变训练、权重或评分公式；实际预检、binding、启动已通过，PID60015/start_ticks325613575。下一步读取实际状态和40条原始评分数组，独立重算端点。

Wav2Lip/Ditto repeat05实际终止为4失败/计划6；Wav2Lip遇到librosa关键字接口变化，Ditto的包装器给原生子进程传了错误的策略脚本路径。Muse02实际终止为2失败/计划6，同样缺原生策略收据。两个Ditto视频虽然字节相同，仍不能替代原生控制链门槛。子 agent 正准备新的统一 bootstrap/包装器与Wav2Lip兼容修复，后续全部6条重新运行，保持旧失败分母。

生产归档全4979成员已核验，不能重新打包消除历史失败。实际单文件probe02确认 selected-track 是精确原音区间[0,97)的62080样本，full source62639样本的559样本尾部在指定ROI之外。ADR016明确分析切片和完整交付PCM的范围；完整成片仍要求原源全部样本，训练/重评分使用full source/cache。rawdecode04修正审计器的切片预期，实际CPU预检通过，尚未启动。

## Working State
持久任务为 [[开发跨 TFG 的自然音频增强头]]（Task89 active）和 [[跨 TFG 自然音频增强头的训练与原音换轨验证]]（Exp88 implemented）。冻结 Dev20/10说话人×3种子×Wav2Lip/MuseTalk/Ditto，360 identity/RMS对照及180候选；要求各模型正增益、等模型额外+0.050 over RMS，并核验D0/Canchor/content/visual/GUARD20和完整失败分母。Confirm80/40未见说话人与LatentSync仍封存。

当前服务器 root@connect.westd.seetacloud.com:18625，GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea；地址保存在 basic-memory/docs/reference/server-access.md。密码和API key只在根 agent运行时变量，永不入文件。远端持久namespace为/root/tts-research-system-20261001/cross_tfg_audio_head_20261003，临时输出/dev/shm/cross_tfg_audio_head_20261003。机器局部易失状态：PTY57588 bridge07/12249，rescore13可能已经终止，须读取实际状态；E盘约0.9GiB空闲，避免本地复制263MB归档。

## Changed Files
CONTEXT.md未提交修改为93行，HANDOFF.md为被忽略的22行快照；五个既有portable_source_axis/overlay/tests修改未动。HEAD `cd6039cdeca1a7ee3b3ca2e6efd2de2731cc4ef9`为本地未发布提交。

可恢复请求索引为 `.codex/scratch/root_rescore13_status_index_attempt01.json` 和 `.codex/scratch/root_rawdecode04_launch_index_attempt01.json`。评分包在 `.codex/scratch/fit120_final_rescore_packet_attempt13_revision01/packet_revision01/`，binding SHA为31b15751e47bd482e75d88b8b48bc9da0faed8623546c7d9122fd4d1d01e4862。候选原始metadata与独立核验在 `.codex/scratch/root_candidate_original_metadata_readback_attempt01/`。Dev180 metadata01因缺legacy builder在产物前失败，新02请求已准备但根agent尚未完整复核/执行。实际Dev540的正确路线必须含原生sitecustomize与逐行候选收据门槛。

## Verification
实际PASS：完整Adam120账本、候选五原文件SHA与theta/constructor/七源码、单Fit40 CLI、Muse9资产/6别名、Ditto资产/运行依赖、selected ROI单文件原PCM切片、rawdecode04 CPU预检、rescore13预检/binding/启动。实际失败保留：12 JSON序列化、repeat05全部四行、Muse02两行、rawdecode03切片预期。13完整端点评分、rawdecode04终端与新的六行严格重复尚未读回；没有新增Dev/Confirm结果或报告26。未运行通用pytest。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
本次HTTP200确认当前分支无匹配PR；当前HEAD未发布。

## Observations
- [result] 固定120步候选已通过原始metadata身份、参数与单音频执行核验，完整迁移收益仍待实际Dev评分。
- [decision] 维持冻结末步、同GPU对照、精确原音和完整失败分母；工程错误作为控制失败保留。
- [decision] 分析ROI要求精确源区间PCM；完整交付仍要求全源PCM和源时间轴，遵循ADR016。
- [blocker] 原生策略脚本路径与Wav2Lip接口需统一修复后重跑全部6条，不能以包装器收据替代原生收据。
- [next_step] 读取rescore13实际40条终端产物并独立重算，同时启动CPU rawdecode04，随后执行新严格重复和Dev540迁移验证。

## Relations
- continues [[Codex checkpoint - 2026-10-07T19-06-14Z - Materialized fixed120 candidate and continuing cross model execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[016-source-clock-roi-and-full-delivery-pcm]]
- relates_to [[015-same-gpu-deterministic-controls]]