---
title: Codex checkpoint - 2026-10-04T02-37-35Z - Native objective implementation and
  completed Ditto recovery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t02-37-35-z-native-objective-implementation-and-completed-ditto-recovery
trigger: compact
username: dogle
cwd: E:/Documents/tts-exp
capture: deliberate
git_sha: 236c201fff6dac789980d06cd36be76749542e00
repo_root: E:/Documents/tts-exp
project: tts-exp
repository: dogLeft233/tts-exp
started: '2026-10-04T02:37:35.1426855+00:00'
agent: codex
branch: codex/cross-tfg-audio-head-20261003
hostname: LAPTOP-E133GSLE
status: open
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T02-37-35Z - Native objective implementation and completed Ditto recovery

## Summary
原音换轨的跨 TFG 增强头继续开发：v4 原生映射训练已实现并提交，Ditto 240 项及新重复检查实际通过，共享筛选和第四、第五模型的真实推理仍在推进。

## Story
用户最初要求解释中文 TTS 的局部声学增强机制，随后要求交付可用于多个 TFG 模型的自然音频增强头。报告 25 的机制研究已结束；当前目标是 h(x) 驱动视频生成，而原始自然 PCM 用于播放和主 SyncNet 评分。root 维护理论、冻结协议、证据和本地 BM；实验实现与运行由 gpt-6-luna/max 子代理完成。最新授权允许对新增强音频使用 Qwen ASR；原音的历史 ASR 通过结果不能代替候选头的内容检查。

v4 现在实际实现并提交为本地 236c201fff6dac789980d06cd36be76749542e00。它使用八条历史 fit 的逐片原生帧映射和全图贴回裁剪，固定原音 MFCC 查询时钟，先验证 RMS baseline 与零初始化恒等，再优化 Cbest/远滞后对比，并以 D0/Canchor 单侧损失约束退化。96 步、12 epoch、固定种子及最终 checkpoint 已事先声明。root 完整审核运行时代码、CLI、几何和计划；新远程执行包刚准备好，尚未实际启动训练。

Ditto fresh recovery05 真实结束于 2026-10-04T02:16:59Z：240 PASS、零失败、零缺项，重跑失败行和新重复检查均 PASS。此前 attempt03/04 的失败是前置检查错误，旧记录 false 为未执行的占位标志，不能解释为测得不确定性。共享十二谱形筛选的最近实际只读记录为 278/360、全部 PASS，仍禁止从部分网格选择候选。SadTalker fresh native_import04 的 17 个必需导入及真实官方 CLI 导入通过；可选训练用 LMDB 在 SHM noexec 下失败并如实保留，对默认推理不构成必需依赖。尚无 SadTalker 模型前向。

LatentSync 官方 UNet 完整 SHA 及 1246 个 FP16 tensor 派生、CPU 重载已核验，官方 InsightFace 原生扩展也已真实编译。CPU03 UNet meta/FaceAnalysis 包刚实际执行并 exit1；4359B 失败报告 SHA 6745d9654d824a31f132868ceb92dc9bd525197b162a678ff8a8a794f62f47a3，下载请求已完成，下一步必须按实际错误窄修复。不能把导入成功或派发包称为 native 推理成功。

## Working State
持久入口是 openspec/changes/develop-cross-tfg-audio-head-20261003/、任务“开发跨 TFG 的自然音频增强头”、实验“跨 TFG 自然音频增强头的训练与原音换轨验证”及 ADR009。八条历史 fit、已暴露 dev20、封存确认 40 speaker/80 clip 分离。共享 360 项完成后依冻结规则选择最多三个共享候选，再做 dev；确认集保持关闭。目标仍 ACTIVE，最终报告 26 尚未形成。

本地分支 codex/cross-tfg-audio-head-20261003 未推送，无 PR。CONTEXT.md 和 ignored HANDOFF.md 有本地快照；Task30/Experiment29 已写入读回，但后续需更新 Ditto COMPLETE、Sad native import PASS、CPU03 FAILED 和共享 278 状态。无关旧 dirty/untracked 文件不能 stage-all。

机器本地易失状态：私有桥 PTY 50274、端口 2133，凭据仅在运行环境。服务器 root@connect.weste.seetacloud.com:12349，公钥 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；禁止将密码或 API key 写入文件或 BM。共享 Python 202805/start_ticks764677412，临时根 /dev/shm/cross_tfg_audio_head_20261003/，持久代码根 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003/。最近 root 可用 3651878912B、SHM 1976332288B；精确 1.488GB 公共 Muse 训练 SyncNet 单文件清理包已准备但未执行，需先审核并释放训练空间。

## Changed Files
- 本地提交 236c201 包含 scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py、fit_cli_v4_attempt01.py、build_native_geometry_manifest_v4_attempt01.py、scripts/experiments/cross_tfg_audio_head_20261003/build_fullcontext_fit_plan_v4_attempt01.py。
- v4 执行入口 .codex/scratch/cross_tfg_audio_head_fit_v4_fullcontext_attempt01/packet_attempt01/request_index_attempt01.json，8541B/SHA 2415aa936bb471220b1d552b987609ec94f921b7b438e7ed5dbad8f2d6d0b48d；计划 b66b0addc25ed6c2b7f7831f8643ba4f35bc227a35ce9e4dd5c4faa0b540801b，几何 1fd5d97d977bb541ef9ebfdd467d066c9f2e678ae5e0fc5f5d96c8b555aecd43。
- Latent CPU03 入口 .codex/scratch/latentsync_cpu_face_unet_attempt03/packet_attempt03/packet_index_attempt03.json；Sad 实际 receipt .codex/scratch/cross_tfg_sadtalker_attempt01/native_import_attempt04/import_qualification_actual_attempt04.json。
- 精确清理入口 .codex/scratch/latentsync_syncnet_retirement_attempt09/packet_attempt02/packet_index_attempt02.json，SHA 96a027e0dc2fb7d5839aee8c9650b524ef26c2d494925d6dcbb4c57fe62b94f6。

## Verification
Ditto 实际终态 .codex/scratch/ditto_recovery05_status_root_attempt02.json.response.json：worker exit0、240/240 PASS、新重复 QC PASS、stderr 空；manifest SHA 4b6d1d432569413f66cb40417f6278cd41ab9b08b8d979b9786faea30a36b5d2。全矩阵与原始 NPZ 的独立 consumer 复核仍待执行。

Sad 实际原始 receipt 11708B/SHA 4b22aa2246237f10dc994150605fbfed7f34e415791aae7167caa4695c27a8bb；17 必需模块来源已记录，官方 inference.py 未替换，CUDA 未初始化。兼容层只指向 torchvision 自有 rgb_to_grayscale 的同一函数对象，不伪造推理逻辑。CPU03 实际失败记录保留；不能用此前 CPU02 的猜测字段错误解释当前失败。

v4 源码及目标、全部八条几何映射已完整审核，40 个 fit 源资产此前实际 SHA/长度/非符号链接检查全部 PASS。尚无 v4 训练结果。未运行 pytest 或新增单元测试；已做检查属于授权的实验运行、原始产物复核和实现审核。

## References
[Repository](https://github.com/dogLeft233/tts-exp)；[已验证的远程基准提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前 236c201 为本地未推送提交。

## Observations
- [result] Ditto recovery05 实际 240 PASS，重跑失败行及新重复检查均通过。
- [result] SadTalker 默认推理的 17 项必需导入通过，真实模型前向尚待执行。
- [decision] v4 固定原音时钟、逐片映射、96 步最终 checkpoint；训练前必须通过 RMS/恒等控制。
- [decision] 完整共享网格后选候选，增强音频另做 Qwen ASR 及独立视觉 guard，再开放封存确认。
- [blocker] 共享筛选未结束，v4 未启动，Latent CPU03 实际失败，两个新架构的模型前向和最终确认尚未完成。
- [next_step] 审核并执行精确空间清理及 v4 执行包，按 CPU03 实际报告修复 Latent，并继续原始矩阵复核、候选 dev guards 和封存确认直至方法交付。

## Relations
- continues [[Codex checkpoint - 2026-10-04T02-03-11Z - Native delivery verification and actual model dependency recovery]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]