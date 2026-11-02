---
title: Codex checkpoint - 2026-10-08T08-35-52Z - Sealed failed-run evidence and actual
  offline CLI
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t08-35-52-z-sealed-failed-run-evidence-and-actual-offline-cli
agent: codex
capture: deliberate
hostname: LAPTOP-E133GSLE
git_sha: 7746b01f34b5850f11e322a0abc6934095f95f64
project: tts-exp
repo_root: E:/Documents/tts-exp
model: gpt-6.1-sol
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
started: '2026-10-08T08:35:52.9932872+00:00'
status: open
username: dogle
trigger: compact
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T08-35-52Z - Sealed failed-run evidence and actual offline CLI

## Summary
修复后的三模型 Dev 对照已实际启动并取得 34 个全部成功的结果，冻结增强头离线包在隔离 CPU 环境复现了相同 WAV，旧失败实验的紧凑快照已完整保存并读回。

## Story
本线程最初恢复 TTS 增强 TFG 的历史路线，用户随后把目标扩展为适配多个 TFG 模型的自然音频增强头：增强 PCM 只用于驱动视频，交付时恢复完整原始自然 F32 PCM 与原始时钟。用户要求持续推进到可用方法，允许重跑、下载、Qwen ASR/TTS 与网盘；ROOT 维护理论、协议、独立核验和 BM，代码与实验交给 GPT-6 Luna/max 子代理。

当前冻结的是 Fit40/20 说话人 Adam120/3 passes 候选。Fit 结果和门控消融提示主要 Cbest 收益来自频谱/校准，正确门控另有固定时钟 D0 收益；这属于 Fit 证据，不能代替跨模型 Dev 验证。Dev 使用 20 句/10 说话人、3 seeds、Wav2Lip/MuseTalk/Ditto：360 identity/RMS 对照后运行 180 候选。预设要求三个模型方向为正，等权模型平均相对 RMS 至少 +0.050，并核验 D0/Canchor、内容、完整原音换轨、视觉与 GUARD20。Confirm80/40 新说话人及 LatentSync 仍封存。

旧 controls17 的 360 行保留 120 成功、240 原生适配失败；这些是工程控制失败。已修复 MuseTalk/Ditto 环境覆盖后入口恢复，生成实际 binding09、plan09、controls18 preflight09，ROOT 与独立子代理核验输入、依赖、闭包及完整计划。新 attempt10 在新命名空间运行，快照03实际 34/360 全 PASS，六个模型×方法单元都有生成和评分。没有据此计算或宣布新完整效应。

为交付可用工具，已制作含源码、冻结参数、配置和来源证明的离线包02。ROOT 独立核对 13 个 ZIP 成员及来源；实际隔离 CPU CLI 对 Fit 单句生成 62,639 个采样点、250,600 B WAV，整文件 SHA 与原实现完全相同，约 2.389 秒。实际输出 WAV/metadata 的两次下载准备就绪，尚待 ROOT 逐字节 RIFF/PCM 核验。旧失败实验紧凑 copier03 实际封存 1,966 个数据对象/80,023,309 B，随后独立全目标读回实际 PASS；ROOT 与子代理仍需本地逐项对照回包和原始索引。完整视频/JPEG 尚在 SHM，不能当作完整归档或删除。

## Working State
本地仓库 E:/Documents/tts-exp，分支 codex/cross-tfg-audio-head-20261003，HEAD `7746b01f34b5850f11e322a0abc6934095f95f64` 为本地未推送提交。已提交的十个修复源与 source_snapshot_attempt07 是当前稳定实现。九份新的小型运行/离线包证明已复制到 openspec 的 reproducibility/，README_fit_step120_candidate.md、CONTEXT.md 与 .gitattributes 尚未提交；README 的隔离 CLI 状态仍需随实际 raw QC 更新。五个不相关的已跟踪改动保留，不批量 add/reset。

用户最新迁移服务器为 root@connect.westd.seetacloud.com:18625。密码/API key 只在私有运行时，永不写入文件或 BM。GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea；远端根 /root/tts-research-system-20261001，研究命名空间 cross_tfg_audio_head_20261003。私有 PTY 57588 与 bridge07 port12249 是易失状态，恢复时先检查。

新 GPU run_id 为 cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt10；PID344830/start_ticks329924069。实际 receipt raw SHA cd5e1fa2c37d615bb3427947fa1e9a2948af56dced57c5d3d1b6e10b404a02b5；计划 SHA 8a592cc54dc981bec5e2398108e6873190a54c7c1ed2bb7901cd31aad6cc86eb；binding SHA dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5。新 receipt.output_root 是 controls360 物理叶目录；旧版本该字段指父命名空间，未来归档 consumer 必须区分实际来源语义。

快照03位于 .codex/scratch/dev360_detached_status_attempt08/requests_attempt03/，回包 raw SHA 8e4f9c6ee8c88c40f0fdbab1ec65dfe8c9cab5630e6b9b7e2de4ee34406ea499，观测 2026-10-08 07:52:04 UTC，34 行全部 PASS。snapshot04 请求已审阅但未执行；索引是单数 request 字段，使用 base dispatcher02 并先检查无旧 response。只读 detached observer 曾在 ROOT 审阅过程中被原位更新，尚未启动；子代理正在冻结不可变 attempt02，修复嵌套行数 high-watermark 与明确 clean-envelope 校验后再由 ROOT 私有 PTY 启动。

## Changed Files
- scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt09.py 及 HEAD7746 的其余九个修复文件，已提交；具体闭包见 reproducibility/source_snapshot_attempt07.json。
- openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/v3_step120_fit_candidate_offline_bundle_attempt02.zip：44,326 B，SHA 32145ea20d18602d2fda2b66ca6fcc7256c215ad9fd70ca23da5a91534a33700，未提交候选包，NOT_SELECTED。
- .codex/scratch/v3_step120_bundle_cpu_repro_attempt03/requests_attempt03/03_extract_and_run_fit_only_cpu_cli_attempt03.json.response.json：实际隔离运行证明，raw SHA 27a295f78011f27f77a739ad04b2fadc6f1a4e2dbdde94de89c9da3b83fcd3c3，易失本地证据。
- .codex/scratch/dev360_compact_snapshot_packet_attempt05/copier_attempt01/readback_attempt01/：实际 snapshot_receipt.json 与完整目标读回回包，易失本地证据；原始索引在 index_readback_attempt02/source_index_reassembled_attempt02.json，6,430,409 B/raw 2817fc317971a7404b8014a1b5fb56d6d9ad0f281196cfbb460ad3c7a98fe675。

## Verification
实际 CPU09 的输入/来源/依赖/360 行序列及 runtime/plan 闭包已由 ROOT 与独立子代理核验。实际 GPU launch07/status08 的源、diff、请求、Bash 和回包已核验；34 行快照为完整 ledger 工程计数，尚未媒体/最终效应核验。离线包全部成员由 ROOT 实际读取并比对，隔离 CLI 已成功执行；下载后的 RIFF/PCM 复核待完成。紧凑索引的全 1,966 对象、5,073 source paths、5,413 aliases、排除类、元数据链及完整失败分母已 ROOT 核验，封存和远端完整读回已实际成功，本地逐项 receipt/QC 待完成。没有新 pytest、环境安装、push 或 PR。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)
- [已确认历史提交 c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 当前 HEAD `7746b01f34b5850f11e322a0abc6934095f95f64` 为本地未推送提交；GitHub 分支 PR 只读查询没有匹配项。
- openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md：既有跨模型失败路线与候选资格协议。

## Observations
- [result] 隔离离线包 CLI 实际输出整 WAV 精确复现既有实现，62,639 samples/16k/mono/F32；工程可执行性已成立，候选仍 NOT_SELECTED。
- [result] 修复后的新三模型 Dev 对照实际运行且 34/360 全 PASS；旧完整失败360分母仍保留。
- [decision] 算法与选择协议保持冻结；不以局部通过、Fit 结果或成功子集替代完整 Dev/确认集。
- [decision] 增强 PCM 只驱动生成；最终 mux 使用完整原自然 F32 PCM 和原时钟。生成输入与交付音轨分别追踪实际 SHA，避免增强两遍。
- [blocker] 全360对照、180候选、实际 generator PCM 内容守卫、原音换轨及视觉/GUARD20 尚未完成；完整视频归档也待完成。
- [next_step] 读取最新 Dev snapshot04、完成已执行紧凑快照与离线 CLI 的本地逐项 QC，再推进完整360→180和所有内容/时钟/视觉资格。

## Relations
- continues [[Codex checkpoint - 2026-10-08T07-48-32Z - Actual repaired Dev run and offline candidate delivery]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]