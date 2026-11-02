---
title: Codex checkpoint - 2026-10-04T02-03-11Z - Native delivery verification and
  actual model dependency recovery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t02-03-11-z-native-delivery-verification-and-actual-model-dependency-recovery
started: '2026-10-04T02:03:11.6466439+00:00'
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
capture: deliberate
agent: codex
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
username: dogle
repository: dogLeft233/tts-exp
git_sha: f2e04c4dee2eb6ea5efb37efe91ac6a497e7db27
status: open
project: tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T02-03-11Z - Native delivery verification and actual model dependency recovery

## Summary
原音换轨的跨 TFG 增强头仍在开发：已核验真实视频交付链的帧映射并编译 LatentSync 人脸依赖，正在完成共享谱形筛选、Ditto 工程补跑和新版原生裁剪训练。

## Story
用户最初要求解释中文 TTS 提高 TFG 同步分数的局部声学原因，随后明确要求产出适配多个 TFG 的自然音频增强头。报告 25 的机制研究已结束，当前目标是 h(x) 驱动生成、原始自然 PCM 同时用于播放和主评分，交付可运行方法及跨模型独立验证。root 管理理论、协议、证据和 Basic Memory；实验代码与运行由 gpt-6-luna/max 子代理完成。失败尝试仅用于修复工程，不代替科学结论，也不结束目标。

当前训练修复处理了真实 FFmpeg JPEG、Wav2Lip 全图贴回、原生检测裁剪之间的帧对应关系。历史 fit 的八条音频已有逐片映射；正式 W2L 样本的实际交付重放也完成，但其静态画面最小 MAE 对应关系并非全片固定减二，不能把 fit 的局部映射盲目推广。新版 v4 固定原音查询时钟，使用真实原生裁剪和逐片映射优化 Cbest 与远滞后对比，并以零容差 D0/Canchor 单侧损失控制退化。96 步、12 epoch、固定种子和正则已事先声明。v4 源码、八条几何清单和计划刚准备好，尚未远程执行或训练。

共享十二谱形筛选恢复运行的最新实际记录为 233/360、233 PASS、零失败。Ditto 恢复 attempt03 在生成之前因错误要求旧失败重复记录的布尔字段全部 null 而退出；实际旧记录是三个定量字段 null、四个 false 占位标志，不能解释为测得不确定性。旧文件未改，fresh attempt04 已准备好。另一路已真实编译官方 InsightFace 原生扩展并通过全部 15 项 CPU 导入，Latent 官方 UNet 的完整 SHA 验证及 FP16 派生已完成；真实 native UNet 加载和推理尚待执行。

## Working State
持久锚点是 openspec/changes/develop-cross-tfg-audio-head-20261003/、任务“开发跨 TFG 的自然音频增强头”、实验“跨 TFG 自然音频增强头的训练与原音换轨验证”和 ADR009 的候选选择规则。八条历史 fit、已暴露 dev20 与封存确认集 40 speaker/80 clip 分离；确认集仍未开放。当前本地提交 f2e04c4dee2eb6ea5efb37efe91ac6a497e7db27 未推送，无 PR。大量未跟踪尝试文件及旧的无关 dirty 文件仍存在，不能 stage-all。

机器本地、脆弱运行状态：SSH private bridge PTY 50274 仍活动，环境内持有凭据，任何密码/API key 都不得写文件或 BM。服务器 root@connect.weste.seetacloud.com:12349；公钥 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。共享恢复 Python 202805/start_ticks764677412；临时运行根 /dev/shm/cross_tfg_audio_head_20261003/，持久代码根 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003/。最近 root 剩余 3.652GB、SHM 2.395GB。恢复 UV 缓存已实际完成；Muse 不用于推理的 1.488GB training SyncNet 仍存在，仅完成精确只读引用扫描，未删除。

## Changed Files
- CONTEXT.md 和 scripts/audio_enhancement_eval/map_formal_w2l_delivery_clock_attempt03.py 已提交在本地 f2e04c4。
- HANDOFF.md 为 ignored 本地快照；Task29/Experiment28 已写入并直接读回核验，但下一次更新需补充 Ditto 旧 false 占位字段的纠正及 fresh04 状态。
- 待执行包：.codex/scratch/latentsync_cpu_face_unet_attempt01/packet_attempt01/packet_index_attempt01.json；.codex/scratch/cross_tfg_audio_head_20261003/ditto_recovery_attempt04/packet_attempt04/request_index_attempt04.json；.codex/scratch/cross_tfg_sadtalker_attempt01/native_import_attempt02/request_index_attempt02.json。均刚准备，不能称已执行。
- v4 待审核：scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py、fit_cli_v4_attempt01.py、.codex/scratch/fit_v4_native_geometry_manifest_attempt01.json 及 .codex/scratch/cross_tfg_audio_head_20261003_fit_v4_attempt01/plan.json。

## Verification
正式 W2L 时钟包 attempt03 实际六请求全部 PASS；完整 receipt .codex/scratch/root_formal_w2l_clock_actual_attempt01.json 为 245517B/SHA eb2b0a17db71da23841041cd98a2c00f70c39a17205327cb607ac3e1f1a55d9f。85 MP4 帧、87 重放 JPEG/原生 crop、82 查询；35/82 五帧窗口全部减二，其余具有零至减三的估计对应。重放 JPEG 不是历史原始 JPEG；编码 MAE 非零，未宣称位级相等。40 个 fit 源资产实际 SHA/长度及非符号链接检查全部通过。

InsightFace 官方源码真实编译与导入 receipt 为 1195216B/SHA 5845745f71b1528766cfaf43df1e280b44eac6817237820f2cd4efd33bb230f8；overlay 6737 文件/37749758B/tree c2374866314b8b94936e8b79f972812cc70fe10fd1d5274af43ed9e7f42a40b9。CUDA 未初始化，尚无模型加载或人脸前向。Latent FP16 文件 2536318892B/SHA 243be78daffff133a10d8c191f999db996a31a25011ffea7554d40705e18e9de，1246 tensor 全部有限、CPU 重载验证通过。

Ditto attempt03 启动请求通过后真实 worker exit1、未生成新视频；.codex/scratch/ditto_frozen_failed_repeat_schema_root_attempt01.json.response.json 核验历史失败重复记录 1049B/SHA 283bb3ff849b2a666bcbc91b3d3e5d31bf27111c74e6b26cd87821eda2e4b9ea。没有运行 pytest 或新增单元测试，已执行检查是授权的实验、原始产物核验和工程验证。

## References
[Repository](https://github.com/dogLeft233/tts-exp)；[已验证的远程基准提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前 f2e04c4 为本地未推送提交。检查点和临时 packet 只指向实际运行产物，不替代最终报告 26。

## Observations
- [result] 原生视频交付时钟已实际重放核验；fit 八条源资产全部可用。
- [result] InsightFace 真实编译及全部导入成功，Latent FP16 权重已验证；native 加载和推理仍待执行。
- [decision] 主评价固定原始自然 PCM；不改变时长/时钟，以完整共享筛选结果选择候选，封存确认集继续关闭。
- [decision] v4 采用逐片原生映射、全滞后 Cbest/far 目标与零容差 D0/Canchor 退化损失，固定 96 步最终 checkpoint。
- [blocker] 共享筛选未完成，Ditto 工程补跑、v4 实际训练、第四/第五模型真实推理和最终确认尚需完成。
- [next_step] 审核并实际派发已准备的 Latent CPU UNet/FaceAnalysis 包和 Ditto fresh04，随后执行 v4 原生 baseline 与固定训练；继续完成跨模型 dev、独立 guard 和封存确认后交付方法。

## Relations
- continues [[Codex checkpoint - 2026-10-04T01-20-14Z - Shared recovery execution and verified Latent runtime weights]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]