---
title: Codex checkpoint - 2026-10-08T18-10-36Z - Live native candidate and verified
  staging
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t18-10-36-z-live-native-candidate-and-verified-staging
started: '2026-10-08T18:10:36.4576693+00:00'
repo_root: E:/Documents/tts-exp
git_sha: 4887c362dc69a1d43e625a259eb0bd2a8dddec84
project: tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
agent: codex
username: dogle
status: open
repository: dogLeft233/tts-exp
model: gpt-6.1-sol
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
hostname: LAPTOP-E133GSLE
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T18-10-36Z - Live native candidate and verified staging

## Summary
三模型候选已实际完成 64/180 条且均 PASS；DirectD0 的 23 项源码与几何输入已完成暂存及独立回执核验，新增独立长音频 20 条也已远端完整暂存。

## Story
用户要求把 TTS 对 TFG 的增强机制做成适配多模型的自然音频增强头，并持续迭代到可用方法。增强音频用于驱动原生视频，交付恢复完整自然 PCM 和原时钟；ROOT 负责理论、协议、证据、BM 与 Git，GPT-6 Luna/max 子代理负责代码和实验。

同机 identity/RMS 360 条对照已全部完成，RMS 的等模型平均 Cbest 改善为 +0.152，但单模型联合区间与 D0 仍要求后续验证。此前候选在生成音频前因重复 interop 设置全体失败；严格 CPU 资格修复与历史 PCM 完全一致核验通过后，冻结 step120 的新候选已唯一一次启动。当前父进程 PID164225/start_ticks333464927；最新实际只读 observation 为 64 PASS、0 FAILED、0 INCOMPLETE，Wav2Lip22/MuseTalk21/Ditto21。全部执行与科学晋级仍需终局核验。

ROOT 已提交实际 metadata、launch、first-status 与完整源码/QC 证据到本地 HEAD4887c362，87 个 Git 路径逐字节匹配，89 项证据已 F/B 双份各两次全文核验。没有推送或 PR。只读 observer10 已启动，ROOT 本机 PID117048，每 60 秒核对现有运行身份与完整计划前缀，绝不重启候选。

下一轮 DirectD0 直接奖励自然音轨上的零延迟距离与单位特征方向，同一冻结权重配对 GateDC。裸 python 暂存失败已保留，并通过绝对 Wav2Lip Python 的新请求恢复：几何 manifest 与全部 23 项源码实际暂存完成；ROOT 全文核验实际 receipt。launcher03/status03 仅修复真实工作目录和回执 pin 比较，已经上传；训练仍未启动，等待当前候选终局。

局部机制解释还需拆开投影的作用：现有 head 对完整频率×时间 mask 使用一次全局径向缩放，GateDC 会改变最大值及缩放比例。因此新增 source-only 投影后每频点均值、±64 帧循环移位对照，保持冻结 theta、原投影、ISTFT、N、时钟和 RMS；检查每频点统计并披露循环接缝。另准备能读取 DirectD0 checkpoint 的交付 CLI，避免实验权重与旧 V3 接口不兼容。科学训练 runner/head 保持原身份。

独立未来 Dev20 在分数前冻结为 S0102–S0111 各两段 4.04–7.37 秒长音频，固定第三方 HF AISHELL-1 commit。20 WAV 共 3,274,068 B、规范 F32 共 6,546,376 B；ROOT 全量 RIFF/PCM/时钟/ZIP 与子代理独立复核均通过。远端 packet04 五个实际请求成功，worker 将 20 WAV、20 F32、7 证据和 manifest 共 48 项 create-only 写入。实际 producer receipt 已公布 25,886 B/raw0896932659649d3328d9986329c9bddfa4fd73f3b7b3c6f4268dba44441ff379，尚待下载全文复核。packet03 因迁移后旧 Ditto Python 路径不存在在 mkdir 前失败，记录保留。

## Working State
冻结 Fit40/20sp Adam120；原 Dev20clips/10sp×3seeds×3models，360 controls +180 candidate。晋级要求三模型均正、等模型平均比 RMS +0.050，并通过 D0/Canchor、内容、视觉、GUARD20 与完整失败分母。未来 Dev20 单独验证，原 Dev 分母不变；Confirm80/40sp 和 LatentSync 仍封闭。

当前服务器 root@connect.westd.seetacloud.com:18625，ROOT=/root/tts-research-system-20261001，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码/API 仅 ROOT 私有运行时内存。机器本地脆弱状态是私有 PTY57588、bridge07、observer10 和 /dev/shm 活跃实验。candidate run_id=cross_tfg_audio_head_20261003_v3_fit40_dev180_cli03_runtime_repair_attempt03。D0 output 尚不存在；候选不得重放。

## Changed Files
本地未推送 HEAD4887c362dc69a1d43e625a259eb0bd2a8dddec84 保留 candidate03_metadata_and_live_launch_attempt01 证据包、guard15、gitattributes 与 94 行 CONTEXT.md。5 个无关 tracked dirty 文件保留，详情见 checkpoint_GN_git_status.txt。ignored HANDOFF.md 此轮尚未成功更新，恢复时以本 checkpoint 和实际产物为准。新 readback05 仅将候选 guard13 pin 改成实际 guard15，尚未 ROOT 审阅/执行。

## Verification
ROOT 已完整核验 metadata03 11 个实际响应和科学投影、唯一 launch 的 5 个响应、完整 launch receipt raw/self 与身份，连续完成的只读快照为计划前缀。observer10 启动及三份首轮快照已 ROOT/子代理独立核验；最新64条记录全文已 ROOT 读取。

D0 实际 source receipt 为 9,594 B/rawd4a2a5d8e6b8c4edd9c57e13f53bee6d727d77b518bea1311ad8be87896c5161，23 source 字节/SHA、ZIP、manifest、自哈希与 no-GPU 范围全部 ROOT PASS。未来20音频 ROOT 独立重建所有 F32，保持 N/16kHz 时钟；远端完整暂存回执待下载。未新增或运行 pytest，未推送、未创建 PR，未运行新 ASR 或 DirectD0 GPU。

## References
- Repository: [dogLeft233/tts-exp](https://github.com/dogLeft233/tts-exp)；SHA `4887c362dc69a1d43e625a259eb0bd2a8dddec84` 本地未推送，GitHub 本分支无 PR。
- 实际候选计划与启动：.codex/scratch/dev360_candidate180_detached_launch_attempt12/request_index_attempt03.json。
- 最新全文快照：.codex/scratch/dev360_candidate180_status_observer_attempt10/polls/20261008T180837Z_poll_8784446e/observation.json。
- D0 暂存回执及 ROOT 核验：.codex/scratch/fit40_d0_gatedc_training_packet_attempt02/source_receipt_readback_attempt01/；.codex/scratch/root_d0_actual_source_stage_receipt_qc_attempt01.json。
- 新数据实际暂存：.codex/scratch/dev20_future_cohort_upload_packet_attempt04/；本地全量核验 root_future_dev20_full_pcm_qc_attempt01.json。
- 新 joint540 终局契约：.codex/scratch/dev360_candidate180_terminal_contract_attempt05/candidate180_terminal_transfer_contract_attempt05.json（source-only，未来结果 pin 为 null）。

## Observations
- [result] 当前候选64/180全部执行通过，RMS/identity360已完成；尚未作科学晋级。
- [decision] DirectD0、投影后局部机制对照与未来长音频 Dev20 保持独立科学身份；完整自然音轨、时钟及未暴露 Confirm 规则继续生效。
- [decision] 已成功的远端请求不重放，失败数据包/解释器绑定保留，后续恢复使用新 create-only packet。
- [blocker] 当前候选仍运行；DirectD0 GPU 启动必须等待其完整终局证据和 GPU 空闲，新20远端 receipt 尚待下载全文核验。
- [next_step] 下载实际 future Dev20 producer receipt，按原始响应 pin 核验全文及全部20行/48输出，再更新 Task/HANDOFF；同时观察现有候选至终局并推进已冻结机制对照。

## Relations
- continues [[Codex checkpoint - 2026-10-08T17-35-41Z - Live candidate execution and prospective cohort]]