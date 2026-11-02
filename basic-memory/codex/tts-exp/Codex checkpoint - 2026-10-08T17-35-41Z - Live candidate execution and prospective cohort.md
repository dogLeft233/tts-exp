---
title: Codex checkpoint - 2026-10-08T17-35-41Z - Live candidate execution and prospective
  cohort
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t17-35-41-z-live-candidate-execution-and-prospective-cohort
status: open
started: '2026-10-08T17:35:41.3997529+00:00'
git_sha: 82a77586da5bffc059173ce68d289c46f03735f8
model: gpt-6.1-sol
trigger: compact
repo_root: E:/Documents/tts-exp
agent: codex
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
branch: codex/cross-tfg-audio-head-20261003
hostname: LAPTOP-E133GSLE
username: dogle
project: tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T17-35-41Z - Live candidate execution and prospective cohort

## Summary
冻结 step120 的三模型 180 条候选已实际启动，首份只读快照为 5 PASS、0 FAILED；新选定的独立自然音频 20 段已下载，DirectD0/GateDC 下一轮仅完成部分暂存。

## Story
用户希望从 TTS 增强效应中做出可用于多种 TFG 的自然音频增强头：增强音频驱动原生视频，交付恢复完整自然 PCM 与原时钟。ROOT 维护理论、协议、BM 与证据；GPT-6 Luna/max 子代理执行代码与实验，并持续推进可用方法。

同机 identity/RMS 360 条对照已经全部完成。旧候选的 180 条失败发生在音频生成前，严格启动器与 CLI02 重复设置 interop 线程是共同原因。CLI03 修复已通过 20 Dev 音频及一份历史参考的严格 CPU 预检，历史参考 WAV/PCM 完全一致。随后实际 metadata03 预检通过，guard15 用补充 binding10 只改变输出根，保留 runtime09、权重、科学参数与完整 180 条计划的科学投影。新候选已唯一一次实际启动，父进程 PID164225/start_ticks333464927；首份 status13 快照覆盖 5 条完整通过的原生生成/换轨/评分执行，Wav2Lip 2、MuseTalk 2、Ditto 1。运行通过不等于方法晋级；必须完成全体分母及原时钟、内容和视觉门控。

ROOT 已将实际 metadata、launch、first-status 请求与响应及源码/QC 复制成 84 文件、5,967,842 B 的新 Git 证据包，全文哈希均匹配；该包目前尚未提交。下一轮 DirectD0 直接奖励零延迟距离与单位特征方向，并配对 GateDC 以分离平均谱形与时间变化。它保持独立科学身份。stage-only q00–q09 已实际成功，q10 因裸 python 不存在而 code127，未执行 geometry 动作；q11 未执行、GPU 训练未启动。保留失败响应，由 Signal 准备绝对解释器的新恢复索引。

为了后续独立验证，先按名单和格式/时长筛选新 10 位说话人的各 2 段 4.04–7.37 秒音频。第三方 HF 镜像固定 commit2724409d538167445e43ebf846990319f12a1cbf，40 个既有 Fit 文件的 LFS 大小/SHA 全部匹配原始记录。20 个新文件已由子代理完整下载到 B，共 3,274,068 B，全部大小/SHA 匹配，规范 F32 保留 N 与 16kHz 时钟。ROOT 尚待独立复核其实际回执和完整音频；未来 Dev20 不替换原 Dev 的分母。旧慢速 OpenSLR 下载已只停止核实的进程，114,159,616 B partial 保留。

## Working State
冻结 Fit40/20sp Adam120；原 Dev20clips/10sp×3seeds×3models。晋级需三模型均正、等模型平均高于 RMS +0.050，并满足 D0/Canchor、内容、视觉、GUARD20 和完整失败分母；Confirm80/40sp 与 LatentSync 仍封闭。

当前服务器 root@connect.westd.seetacloud.com:18625，ROOT=/root/tts-research-system-20261001，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。凭证仅 ROOT 私有运行时内存。机器本地脆弱状态包括私有 PTY57588/bridge07 与 /dev/shm 活跃实验；遇到模糊失败先只读核实，不重放启动。活跃 run_id=cross_tfg_audio_head_20261003_v3_fit40_dev180_cli03_runtime_repair_attempt03；candidate 输出位于 /dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cli03_runtime_repair_attempt03/candidate180。

## Changed Files
本地未推送 HEAD82a77586da5bffc059173ce68d289c46f03735f8 已保留严格 CPU 实际证据；新未跟踪包为 openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/candidate03_metadata_and_live_launch_attempt01/，其 manifest SHA5bd495e14a70e1be95a54f9e80fa678ab78b3c7b80ef90f08348b70e76a58889。guard15 未提交。HANDOFF.md 被忽略，已更新实际运行位置。5 个无关 dirty 文件保持原状；完整状态另存 checkpoint_GM_git_status.txt。

## Verification
ROOT 已验证 actual metadata03 的 11 个响应、22 个源码 pin、唯一 output_root 投影和完整科学计划；launch12 的 5 个实际响应成功。实际 launch receipt 为 11,668 B/raw fe8fc0e82cf15fe971ca8d16d69d4961f5186ea12a93eb157c26845dee36e268，self a7f88f5e2e960516ee5864be7677df55b7524e666abc327d5db95fd9cc66996e。status13 第一实际响应 8,581 B/raw ca39fc2d100e3d928c0d2a569df6855bf6461ebe898fab0ac9fa9f90538173d1；5 条恰为计划前缀，无失败或缺失。进行中 startup attestation 尚未写入外层 execution，需终局核验。

新 84 文件包已全文复制核验。已有 HEAD29488/82a 证据已 F/B 双份全文验证；云同步确认未取得。D0 stage 失败原文为 bash: line 2: python: command not found，不能记作科学阴性。未新增或运行 pytest，未推送或创建 PR。

## References
- Repository: [dogLeft233/tts-exp](https://github.com/dogLeft233/tts-exp)；当前 SHA 是本地未推送 `82a77586da5bffc059173ce68d289c46f03735f8`，GitHub 查询无本分支 PR。
- 已执行 launch packet：.codex/scratch/dev360_candidate180_detached_launch_attempt12/request_index_attempt03.json。
- 实际第一快照：.codex/scratch/dev360_candidate180_status13_readback_attempt01/requests_attempt01/01_read_candidate_status13_once.json.response.json。
- 待执行第二只读快照：.codex/scratch/dev360_candidate180_status13_readback_attempt02/request_index_attempt01.json。
- D0 失败响应：.codex/scratch/fit40_d0_gatedc_training_packet_attempt02/requests_attempt04/10_stage_exact_geometry_manifest_create_or_identical.json.response.json。
- 新数据计划与回执：.codex/scratch/dev20_future_cohort_metadata_attempt06/；完整下载包由 ASR 子代理准备，尚未远程上传。

## Observations
- [result] 新 180 条候选已实际运行，第一快照 5/180 PASS；360 条对照已完成。
- [decision] 保持冻结候选与 DirectD0/GateDC 为独立实验；新数据只作未来独立 Dev20，原分母与 Confirm 封闭规则继续生效。
- [decision] 任何输入暂存和监测命令使用绑定环境的绝对解释器；已经成功的请求不重放。
- [blocker] DirectD0 stage q10 由裸 python 缺失阻塞，恢复包待准备；当前候选仍需完整终局证据。
- [next_step] 执行已审阅的单次 candidate status13 snapshot02，并核对活跃进程身份与完整计划前缀；同时完成新 Git 证据提交和 D0 暂存恢复。

## Relations
- continues [[Codex checkpoint - 2026-10-08T16-57-14Z - Actual CPU pass and candidate launch preparation]]