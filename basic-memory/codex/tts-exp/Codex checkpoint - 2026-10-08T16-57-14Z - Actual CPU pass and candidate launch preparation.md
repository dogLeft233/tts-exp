---
title: Codex checkpoint - 2026-10-08T16-57-14Z - Actual CPU pass and candidate launch
  preparation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t16-57-14-z-actual-cpu-pass-and-candidate-launch-preparation
username: dogle
project: tts-exp
git_sha: 82a77586da5bffc059173ce68d289c46f03735f8
capture: deliberate
started: '2026-10-08T16:57:14.6137945+00:00'
cwd: E:/Documents/tts-exp
trigger: compact
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
repo_root: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
agent: codex
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T16-57-14Z - Actual CPU pass and candidate launch preparation

## Summary
严格 CPU 预检已确认修复后的增强音频保留冻结候选的数值输出，下一步将以新目录实际运行三模型 180 条候选。

## Story
用户希望利用 TTS 的局部声学机制做一个适配多种 TFG 的自然音频增强头：增强音频驱动视频，交付时恢复完整自然 PCM 和原始时钟。ROOT 维护理论、冻结资格和实验主线，GPT-6 Luna/max 子代理负责代码与实验；需要持续推进至可用方法。

360 条同机 identity/RMS 对照已全部完成并独立复核。旧 180 条候选在音频生成前统一失败，原因是严格启动器已经设定线程，CLI02 又无条件设置 interop 线程；这属于执行失败。CLI03 改为只在需要时设置线程。实际 CPU03 回执覆盖 20 个 Dev 音频和一个历史参考，全部通过；历史参考完整 WAV 和 PCM 逐字节一致，Dev 的 N、时钟、RMS 和 Wav2Lip 帧数均符合冻结计划。候选权重和科学参数保留不变。

这些实际证据已进入本地提交 82a77586da5bffc059173ce68d289c46f03735f8，并复制至 F、B 两处，117 文件均完成两次全文哈希验证。现在使用 guard15、builder03 的新元数据 plan14/manifest15 和补充 binding10：保持原 runtime09 与 360 对照只读，候选进入全新 namespace。补充 binding10 使用实际持久 CPU 回执路径；错误路径的 metadata01/02 保留为历史，不能启动它们。

同时准备直接优化零延迟 D0/单位特征方向的新 Fit40+GateDC 方法。此支线尚无 GPU run，不能替代冻结 step120 候选。17 个 B 盘历史压缩包已完成外层头/名称检查，未找到 264 个合格未曝光说话人的训练音频；ROOT 已授权子代理从官方 OpenSLR33 下载完整压缩容器至 B，下载和校验阶段不解包、不解码排除或 Confirm 音频。

## Working State
冻结 Fit40/20sp Adam120；Dev20clips/10sp×3seeds×3models，每模型 60 条候选。晋级需三模型均正、等模型平均超过 RMS +0.050，并通过原时钟 D0/Canchor、内容、视觉和完整分母门控。Confirm80/40sp 与 LatentSync 保持封闭。

当前服务器为 root@connect.westd.seetacloud.com:18625，ROOT=/root/tts-research-system-20261001，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码/API 只存在 ROOT 私有运行时，不写文件。私有 PowerShell PTY 57588 和 bridge07 属于脆弱本机状态；失联时先读状态，不重发模糊启动。

机器本地未执行的候选 packet 位于 .codex/scratch/candidate_v3_cli03_runtime_repair_attempt03；实际 metadata 输出位于 metadata_output_attempt03。新的 run_id 为 cross_tfg_audio_head_20261003_v3_fit40_dev180_cli03_runtime_repair_attempt03，输出 namespace 为 /dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cli03_runtime_repair_attempt03。启动前整个新 namespace 必须不存在或为空，日志与启动回执应放独立 JOB 目录。此时尚未执行新 180 条候选的远程 staging 或 GPU launch。

## Changed Files
已提交 CLI03、严格 CPU 实际证据和 CONTEXT.md；权威复制证据在 openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/cpu03_actual_strict_preflight_attempt01/。guard15 和新 candidate packet 仍是未提交本机文件。HANDOFF.md 为忽略文件。5 个无关 dirty 文件保持原状；完整 git 状态另存 checkpoint_GL_git_status.txt。

## Verification
CPU03 回执 raw SHA 8c44b1c80a08e48465f25672ef6b990002653ff7d250ede7369be13dab3a3b09；ROOT 全文核验和独立 Signal QC 均通过。历史参考 WAV/PCM 完全一致、maxabs/RMSE 为零；CPU admission 不代表 GPU qualification。

HEAD82a 的 68 个复制文件和 HEAD29488 的 52 个复制文件均完成 Git blob/工作树/来源全文哈希核验。F/B 新备份 117 文件 2,864,150 B 两次全读均通过；百度云同步确认尚未取得。未新增或执行 pytest、未推送、未创建 PR。

guard15 与 builder03/metadata03 已独立审阅通过。DirectD0 的 launcher/status 仍在修复并封装；ROOT 要求检查 runner02 的线程初始化顺序，避免重复设置 interop。代码审阅不作为运行结果。

## References
- Repository: [dogLeft233/tts-exp](https://github.com/dogLeft233/tts-exp)；当前提交为本地未推送 `82a77586da5bffc059173ce68d289c46f03735f8`，分支查询未发现 PR。
- CPU 原始证据：.codex/scratch/candidate_v3_cpu_preflight_attempt03/downloads_attempt02_revision02/cpu_preflight_receipt_attempt03.json。
- ROOT QC：.codex/scratch/root_candidate_cpu03_actual_receipt_qc_attempt01.json。
- 新元数据：.codex/scratch/candidate_v3_cli03_runtime_repair_attempt03/metadata_output_attempt03/。
- [官方 AISHELL-1 数据源](https://www.openslr.org/resources/33/)。

## Observations
- [result] 360 条对照全部 PASS；旧候选 180 条在音频生成前失败；新 CPU03 的 20 个 Dev 音频及一个历史参考全部 PASS。
- [decision] 新候选保留冻结权重、原 runtime09 与对照，使用独立新目录；补充 binding10 只证明 CPU admission，GPU 门控继续保留。
- [decision] DirectD0/GateDC 支线单独训练与报告，不将新目标混入冻结候选结果。
- [blocker] 新 180 条候选仍需完成具体 staging/metadata/launch/status packet 的审阅并实际启动。
- [next_step] 审阅 Dev 的完整新候选 packet，执行唯一一次实际 metadata 预检，绑定其回执后启动三模型 180 条候选并读取实际状态。

## Relations
- continues [[Codex checkpoint - 2026-10-08T16-43-24Z - Actual CPU parity and fresh candidate preparation]]