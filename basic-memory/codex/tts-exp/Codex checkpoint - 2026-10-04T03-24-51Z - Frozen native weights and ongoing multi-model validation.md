---
title: Codex checkpoint - 2026-10-04T03-24-51Z - Frozen native weights and ongoing
  multi-model validation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t03-24-51-z-frozen-native-weights-and-ongoing-multi-model-validation
username: dogle
status: open
repo_root: E:/Documents/tts-exp
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
project: tts-exp
git_sha: 6c6cb1c6509c0d7fb035ebc163d3437829721b64
started: '2026-10-04T03:24:51.9183720+00:00'
model: gpt-6.1-sol
trigger: compact
repository: dogLeft233/tts-exp
capture: deliberate
hostname: LAPTOP-E133GSLE
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T03-24-51Z - Frozen native weights and ongoing multi-model validation

## Summary
跨 TFG 自然音频增强头的 v4 原生映射训练实际完成 96 步，最终权重冻结并取回核验；两个新架构继续推进真实推理，方法验证仍在进行。

## Story
用户最初要求解释中文 TTS 为什么增强 TFG，随后要求做成适配多个模型的自然音频增强头：h(x) 仅驱动视频生成，成片播放及主 SyncNet 评分均使用原始自然 PCM。报告 25 的机制研究已结束，当前目标是交付可复现的完整方法。root 维护理论、协议、证据、Git 和本地 BM；实验实现与运行由 gpt-6-luna/max 子代理完成。最新用户再次授权使用 Qwen ASR，凭据只在运行时使用，原音历史 ASR 通过不能替代新候选的内容检查。

v4 在八条历史 fit 上使用逐片原生帧对应、全图贴回和固定原音 MFCC 查询时钟，先验证 RMS baseline 与零初始化恒等，再训练置信度/远滞后对比，并约束 D0/Canchor 退化。预声明的 96 步、12 epoch、固定种子和最终 checkpoint 已实际执行完成。原 preflight01 因 Python 导入失败停止；窄修复 attempt02 没有改变计划，真实通过后训练 exit0。最终权重 317195B/SHA 3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4，禁止再按 dev 选择 epoch。21 个终态文件现已取回，本地逐项长度和 SHA 全部通过；训练完成不等于跨模型增强成立。

Ditto recovery05 已实际 240 PASS、零失败/缺项且 fresh repeat PASS，但全原始矩阵的独立 consumer 仍待完成。共享十二谱形网格最近核验为 293/360 全 PASS，必须完成全部 360 后再按 ADR009 选择最多三个共享候选。v4 独立候选可以直接进入冻结 dev20×3 seeds×3 frontends 的 180 项评估，不必等待谱形候选；已有对照只能在原 PCM、portrait、模型、seed、评价器、时钟协议和来源 pin 完全一致时加入共同支持集。

Latent CPU04 已真实完成 1246/1246 UNet meta key/shape/numel coverage，零缺失或随机初始化。官方 buffalo_l 在本地完整下载核验，通过 <=6MB 分片传到服务器后，FaceAnalysis 真实 CPU 前向检测到一张脸，106 个二维关键点全部有限，两个 ONNX session 均仅 CPU provider。下一步是历史 fit-only 原音/RMS native 视频生成，必须记录 stock CUDA FaceAnalysis 到 CPU 的显式 runtime adapter。SadTalker 17 项必需 native 导入通过后，两臂 fit-only 试跑已实际启动；fresh status09 显示两个官方 inference subprocess 都 exit1，worker exit2/PARTIAL_FAILURE，未进入 SyncNet。保留失败，取 stderr 定位后窄修复同计划重跑。

## Working State
持久入口：openspec/changes/develop-cross-tfg-audio-head-20261003/、任务“开发跨 TFG 的自然音频增强头”、实验“跨 TFG 自然音频增强头的训练与原音换轨验证”和 ADR009。八条历史 fit、已暴露 dev20、封存确认 40 speaker/80 clip 分离。确认集仍关闭，目标 ACTIVE；没有报告 26 的最终结论。

本地分支 codex/cross-tfg-audio-head-20261003、HEAD 6c6cb1c6509c0d7fb035ebc163d3437829721b64，未推送、无 PR。v4 四个源文件在本地 236c201fff6dac789980d06cd36be76749542e00；6c6cb1c 只提交 CONTEXT.md。CONTEXT.md、ignored HANDOFF.md 和 Task31/Experiment30 仍有训练前/人脸前向前的快照，需要更新。不要 stage-all 混入旧 dirty/untracked 文件。

机器本地易失：私有桥 PTY50274/端口2133；服务器 root@connect.weste.seetacloud.com:12349，公钥 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。密码/API key 永不入文件或 BM。NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003；SHM=/dev/shm/cross_tfg_audio_head_20261003。共享 PID202805/start_tick764677412。Sad 最初实际 parent516943/child516949，需以 fresh status 为准。unused Muse training SyncNet 精确单文件 SHA 校验/无 open FD 后已删除，释放1488023552B；Sad preflight SHM free3056979968B。upload/download 都有8MiB限制，大文件 <=6MB 分片。

## Changed Files
- 已提交 v4 runtime、fit CLI、native geometry helper 和 fullcontext plan builder；v4-only dev builder 及 candidate-family Qwen ASR guard 新增源码仍未提交/完整审核。
- 终态本地目录 runs/cross_tfg_audio_head_20261003/fit_v4_native_fullcontext_readback_attempt02/artifacts/；索引 .codex/scratch/cross_tfg_audio_head_fit_v4_fullcontext_readback_attempt02/request_index_attempt02.json。
- Latent native FaceAnalysis 实际报告 .codex/scratch/latentsync_faceanalysis_only_attempt01/actual_faceanalysis_cpu_report_attempt01.json；Sad fresh status09 请求 .codex/scratch/cross_tfg_sadtalker_attempt01/fit_smoke_attempt03/requests_attempt05/09_read_status_attempt09_request.json。

## Verification
v4 实际 status 为 FIT_PROCESS_EXITED_WITH_OUTPUT_GATE_PASS，96 步、14 output gates 全 true、8 RMS cache 和8零初始化 parity PASS。fit_report 602452B/SHA9cb7bf9c3cbc5796f3284cdfcb733617a8734d2f68b5d7a65b6b9d0e408e6c38；steps 233410B/SHA16ab21144b113f5187f9707149a9e5e8ec397431aa7cef63a73dcc2b871ea3e2。root 独立取回21文件/3189031B长度与SHA检查通过，receipt .codex/scratch/v4_fit_root_readback_qc_attempt01.json；报告内部和NPZ独立数值复核仍要继续。

Latent CPU face report4600B/SHA0b25ebc7a0d494cb5e34f11ab1c45c27af94455c60869d44d13e066447034f13：1 face、det_score0.8376072645、106×2 finite landmarks、CUDA hidden/uninitialized、8.465s。CPU04完整UNet meta report SHA9df329aa8106c5b36601e7022be462ff2c109494ef166b5fb2c4b691cca54261；该次整体 FAILED 的原因只是下载 RemoteDisconnected，后续 face 分片 continuation02 全11项实际通过。

Sad namespace/upload/extract/preflight 00–06 实际 PASS（15 bundle members、4官方权重、Torch2.5.1/NumPy1.26.4、CUDA隐藏）。root 在派发前发现 outer logs 占据 launcher 要求空目录，fresh launch05 只改 dispatch_control，保留科学 bundle/config；07/08 实际 PASS 启动，status09 实际 PARTIAL_FAILURE，两个 inference exit1/worker exit2/没有 SyncNet结果，stderr 细节待取回。未运行 pytest 或新增单元测试；检查属于用户授权的实际实验、原始产物核验和代码审核。

## References
[Repository](https://github.com/dogLeft233/tts-exp)；[已验证的远程基准提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前 HEAD 与236c201均为本地未推送提交。

## Observations
- [result] v4 固定96步训练真实完成，最终权重冻结，21个终态文件取回长度/SHA全部PASS。
- [result] Latent 全1246个UNet meta参数覆盖与真实CPU FaceAnalysis前向通过；尚无完整native视频结果。
- [decision] v4 独立180项dev立即推进；共享谱形必须完整360后筛选，对照仅精确join，确认集继续封存。
- [decision] 每个新增强候选独立 Qwen ASR 内容检查，并进行独立视觉口部 guard，主评分始终原音换轨。
- [blocker] dev180、完整共享筛选、Ditto独立consumer、第四第五模型完整生成、候选guards和封存确认尚未完成。
- [next_step] 核验v4终态报告与NPZ并启动180项dev，同时完成Sad/Latent真实fit-only推理、完整共享selector和独立consumer，随后执行guards与封存确认直至方法交付。

## Relations
- continues [[Codex checkpoint - 2026-10-04T02-37-35Z - Native objective implementation and completed Ditto recovery]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]