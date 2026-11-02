---
title: Codex checkpoint - 2026-10-04T03-58-36Z - Verified native training and pending
  cross-model execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t03-58-36-z-verified-native-training-and-pending-cross-model-execution
status: open
model: gpt-6.1-sol
git_sha: 6c6cb1c6509c0d7fb035ebc163d3437829721b64
started: '2026-10-04T03:58:36.8092916+00:00'
hostname: LAPTOP-E133GSLE
capture: deliberate
trigger: compact
repository: dogLeft233/tts-exp
project: tts-exp
username: dogle
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
repo_root: E:/Documents/tts-exp
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T03-58-36Z - Verified native training and pending cross-model execution

## Summary
自然音频增强头已冻结 v4 权重并完成训练数值复核，跨模型开发验证与两个新增原生模型的运行恢复正在继续。

## Story
原目标是恢复 TTS 增强 TFG 的研究路线并找出局部声学机制；已完成的报告 25 将评价器响应、真实视频运动和原生模型条件效应分开。用户随后要求产出能适配较多 TFG 的自然音频增强头，并明确要求失败后继续迭代。当前目标是 h(x) 只驱动视频生成，播放和主要 SyncNet 评分换回逐样本相同的原自然音。阿里云 Qwen ASR 可用于候选音频内容保真检查，凭据只存在运行时环境。

采用两条互补路线：共享频谱 profile 在三个 frontend 的历史 fit 集上按冻结的 lag-contrast 规则筛选；可训练 v4 则在完整原生 Wav2Lip 时序和编码后的真实画面几何上优化。冻结的 96 步训练已实际结束，最终 checkpoint 不按 dev 分数挑选。root 维持理论、协议、Basic Memory 和独立复核，实验实现与运行交给 gpt-6-luna/max subagents。dev 为 20 音频、10 位男说话人、3 paired seeds、3 frontend；后续仍需封存确认集、内容与独立嘴部运动 guard、CLI/权重/延时和最终报告。训练通过仅表示实现和权重可用，方法收益须以实际原音换轨实验为准。

## Working State
Git HEAD `6c6cb1c6509c0d7fb035ebc163d3437829721b64` 在 `codex/cross-tfg-audio-head-20261003`，是本地未发布提交，没有该分支 PR。Task snapshot 32 和 Experiment snapshot 31 已写入本地 tts-exp 并直接读回；其共享行数 337 和部分准备状态已被下面新证据超越，下一次实际里程碑应修订已有 typed notes。

v4 的本地 21 文件权威回读为 `runs/cross_tfg_audio_head_20261003/fit_v4_native_fullcontext_readback_attempt02/artifacts/`。最终权重 317195 B、SHA `3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4`。root 复核结果分别在 `.codex/scratch/v4_fit_root_readback_qc_attempt01.json` 与 `v4_fit_root_numeric_qc_attempt01.json`。新的 180 行 plan attempt02 已纠正旧 stagea runner pin，必须用 dispatcher→core→adapter 路由；不得仅调用 core 把 Muse/Ditto 当 Wav2Lip。fit-only 的公开 CLI/dispatcher PCM 等价 smoke 已形成 17 请求 packet，但尚未实际运行，index 位于 `.codex/scratch/cross_tfg_audio_head_20261003/v4_dispatch_smoke_attempt01/packet_attempt01/request_index_attempt01.json`。

共享 fit 的最近实际只读回执为 349/360 PASS、0 FAILED，worker PID202805；完整终态、重复 QC、consumer 和最终 selector 尚待执行。只读新 request 为 `shared_frequency_fit_recovery_attempt05/16_compact_status_root_attempt01_request.json`。Ditto recovery05 已封存 240/240 PASS 和真实重复 QC PASS；consumer03 实际于 03:35:35 UTC 启动 PID575259/start tick765673830，03:43 仍 CPU 工作、无输出、stderr 空。resource probe06 已准备，须读取而非重启这个 consumer。

LatentSync 已实际通过完整 1246 UNet key/shape 检查、官方 FP16 导出、CPU FaceAnalysis 检出和 VAE 复用核验。完整 62639-sample 历史 fit 音频的官方 identity 原生渲染 packet03 已由 root 阅读 source/launcher，index SHA `311aef0b05ce54d67723f5205e79b32424b3dddd0523795111c307793727ca66`，位置 `.codex/scratch/latentsync_fit_inference_attempt03/packet_attempt02/packet_index_attempt02.json`；仍未 dispatch。worker 两次间隔 5 秒验证 GPU 无其他 compute PID/空闲至少 18432 MiB，并在 CUDA init 前重新验证。只改 FaceDetector CPU provider，不改 UNet/Whisper/VAE 数学。原生 AAC 将丢弃，再封入原自然音 PCM；此步没有 dev/confirmation 或 candidate 分数。

SadTalker smoke03 两臂实际在导入 safetensors 时因 SHM noexec 失败。新的 executable Muse safetensors 树已实际核验。attempt04 的 namespace/四上传成功，但 ZIP 缺根 config.json，在 extractall 前失败，故 code root 尚未创建且模型未运行。fresh continuation06 为 `.codex/scratch/cross_tfg_sadtalker_attempt01/fit_smoke_attempt04/requests_attempt06/request_index_attempt06.json`，加入完全相同 config 字节并维持原科学源码，改用 dispatch_control 日志避免空 control 矛盾；尚待 root 阅读和执行。失败回执保留，不能当作方法阴性。

本机 fragile 状态：私有 PTY50274 和 bridge2133 持有运行时密钥；凭据不写文件/BM。服务器地址 root@connect.weste.seetacloud.com:12349，host key 已有公钥 pin 回执；AGENTS 中 westc 是旧地址。远端 NS `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`，运行产物 SHM `/dev/shm/cross_tfg_audio_head_20261003`，GPU 工作使用两读准入、不终止其他任务。此前仅清理经精确 SHA/无引用确认的缓存；不重复删除研究数据或推理权重。

## Changed Files
本轮未新提交源码。CONTEXT.md 是本地未提交的 93 行入口修订；HANDOFF.md 是忽略的本机快照，均需在实际新里程碑后更新。ASR family guard 和多个 frozen experiment runner 是未提交文件；旧机制实验及 tests 的既有脏文件保持原状，不使用 git add 全仓库。

## Verification
21 回读文件的字节长度和 SHA 全部通过。root 独立数值重算实际 96 rows/steps、8 fit×12 epoch、8 NPZ/32 finite arrays：canonical curve 最大误差 9.54e-7，reward 最大误差 3.87e-10，loss 最大误差 2.18e-8，identity RMS parity 最大误差不超过 1e-7，所有 head gradient/update 有限且非零。读取了 ASR family guard 全部源码；实际 API 尚未调用。它需 Wav2Lip 单 frontend merged ledger 含 identity 和候选，直接喂 3-model v4-only ledger 会拒绝。未运行 pytest/unittest。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified remote baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- `openspec/changes/cross-tfg-audio-head-20261003/spec.md` 与 `transfer-iteration.md`：确认集和跨模型接受规则；路径以已读协议为准。

## Observations
- [result] v4 最终权重与原生训练的完整日志/数组已实际回读并独立重算通过。
- [decision] 原自然音及其时钟为播放和评分锚点；训练损失或未封存子集不替代跨模型确认收益。
- [decision] 三 frontend 共享 profile 只能按完整 360-row fit 与 ADR009 的规则筛选，失败行不得删除，候选 family 最多三个 profile 加一个 v4。
- [blocker] v4 dev180、共享全量 consumer/selector、候选专属 ASR 和视觉 guard、确认集及交付报告均未完成；Sad/Latent 为继续修复的原生工程依赖。
- [next_step] 顺序实际执行已审查的 Latent03 和新的 shared/Ditto 状态探针，修复 Sad continuation06，同时通过 v4 fit CLI smoke 后启动冻结的 180 行开发验证。

## Relations
- continues [[Codex checkpoint - 2026-10-04T03-24-51Z - Frozen native weights and ongoing multi-model validation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]