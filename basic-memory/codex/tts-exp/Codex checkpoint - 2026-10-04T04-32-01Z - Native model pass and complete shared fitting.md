---
title: Codex checkpoint - 2026-10-04T04-32-01Z - Native model pass and complete shared
  fitting
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t04-32-01-z-native-model-pass-and-complete-shared-fitting
branch: codex/cross-tfg-audio-head-20261003
repo_root: E:/Documents/tts-exp
agent: codex
status: open
repository: dogLeft233/tts-exp
username: dogle
model: gpt-6.1-sol
cwd: E:/Documents/tts-exp
git_sha: 6c6cb1c6509c0d7fb035ebc163d3437829721b64
started: '2026-10-04T04:32:01.8665972+00:00'
trigger: compact
capture: deliberate
hostname: LAPTOP-E133GSLE
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
project: tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T04-32-01Z - Native model pass and complete shared fitting

## Summary
增强头 v4 的最终权重已完成原生训练和独立数值复核，共享频谱 fit 的 360 行已生成完毕；Ditto 的完整开发结果支持放弃 v3 作为跨模型候选，而新增 LatentSync 已实际完成原生渲染。

## Story
原目标是恢复 TTS 增强 TFG 的研究路线并解释局部声学机制，报告 25 已区分评价器响应、真实视频运动和模型条件效应。用户进一步要求产出适配较多 TFG 的自然音频增强头，并要求失败后持续迭代。h(x) 只驱动视频生成，播放及主要 SyncNet 评分使用相同原自然音及原时钟。root 负责理论、协议、BM、传输和独立复核；实验代码与运行由 gpt-6-luna/max subagents 完成。阿里云 Qwen ASR 已获授权，用于新候选内容检查；密钥和服务器密码仅存运行时环境。

当前并行推进共享频谱 profile 与原生时序训练 v4。前者按 ADR009 的完整三 frontend fit 规则冻结最多三个 profile；后者采用固定 96 步后的最终权重，不能按 dev 分数选 epoch。dev 固定 20 音频、10 位男说话人、三 paired seeds；确认集、独立嘴部运动 guard、候选 ASR、部署 CLI、延时和报告 26 尚未完成。训练通过说明实现可用，方法收益仍由原音换轨实验确定。

## Working State
Git HEAD `6c6cb1c6509c0d7fb035ebc163d3437829721b64`、分支 `codex/cross-tfg-audio-head-20261003` 是本地未发布状态，无该分支 PR。已有 Task snapshot33 与 Experiment snapshot32 已写入并读回；其中部分运行状态已由本 checkpoint 更新。CONTEXT.md 的 93 行修订尚未提交，HANDOFF.md 是忽略的本机快照，下一次实际里程碑后更新。

v4 权威回读为 `runs/cross_tfg_audio_head_20261003/fit_v4_native_fullcontext_readback_attempt02/artifacts/`。最终权重 317195 B、SHA `3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4`；root 的长度/hash 和独立重算 receipts 为 `.codex/scratch/v4_fit_root_readback_qc_attempt01.json` 与 `v4_fit_root_numeric_qc_attempt01.json`。dev180 plan attempt02 SHA `70701ac4a83e5ea2abd737708aba50b5c5ed2a0d51c18bad38ae84b4480ef28c`，尚未 launch。必须 dispatcher→core→adapter 路由。公开 CLI smoke continuation05 实际失败：outer exit1、CLI exit2，内层 stderr SHA `9bc9b756043429d05813851d523c59892b977d5c06999d3606beaeea154f69b1`；机制 agent 正准备只读真实 CLI 日志和窄修复，dev180 完整 closure packet 同时构建，不能停留在 prepared。

共享 fit recovery05 已实际 COMPLETE360/360 PASS，0 FAILED/incomplete；ledger SHA `e67c753a2c29048757f629d18ac259b9f146f980cfaf94959d17048874fb0e0d`。空 repeat ledger 对应冻结计划 generation_repeat_qc=[]，不得报告重复 QC PASS。consumer01 实际于 04:08:15 UTC 启动 PID826164/start tick765869808，输出在 SHM/shared_frequency_fit_recovery_attempt05/consumer_attempt01/output。只读 status02 packet 已准备，完整 consumer/raw QC/selector 尚待执行。纯 fit consumer 的 NO_EVALUATION_ENDPOINTS/exit2 若出现是无 summary endpoints 的真实状态，须与行审计分开。

Ditto producer 完成240/240及同 seed engineering repeat PASS。consumer03 的全部六个 FULL endpoints COMPLETE；仅预设 GUARD20 因一条短音频不足41 queries 而 INCOMPLETE，不能缩小分母。FULL dev 描述均值：RMS−identity +0.168718；local−RMS −0.028273；v3−RMS −0.170123，10/10 speaker 均负。9 回读文件长度/hash root PASS。raw241 NPZ archive 实际已下载，2587997 B、SHA `1f250930fe709e36fa4660327e0fbf9d01bdbf5a672612743bf9be462aa26f16`，位于 `runs/cross_tfg_audio_head_20261003/ditto_recovery_attempt05_raw_matrix_readback01/`。agent receipt 通过241文件及1687数组hash；root 的共同支持集/float32 curve/说话人汇总数值独立复核仍待做。

LatentSync native03 已实际 exit0/PASS_ENGINEERING_HISTORICAL_FIT_ONLY_NATIVE_LATENTSYNC，使用官方 source/config、FP16 UNet 全1246 key/shape、Whisper tiny、VAE，只有 FaceDetector 设为 CPU。完整历史 fit S0052/W0121 输入为 PCM16 WAV、62639样本；canonical F32 SHA `f4e611e9f816665ad7c4dce4980f461c4d9af4fe6fe9678c8f756bd03fa6b36c`。静态128-frame/25fps源经官方流水线输出99-frame/25fps，无音频 retime；丢弃 pipeline AAC 后封装原音 pcm_f32le MKV。5个本地回读的长度/SHA 和 root 实际 FFmpeg 解码原音 PCM hash PASS，contact sheet 已查看有真实口部变化。路径 `.codex/scratch/latentsync_fit_inference_attempt03/readback_attempt01/`，root receipt `root_local_readback_qc_attempt01.json`。这不是同步/方法收益结论。agent 正准备完整音频 RMS engineering pair，转换 packet 位于 `latentsync_rms_audio_attempt01/packet_attempt01/packet_index_attempt01.json`，尚未执行，无 dev/候选/确认。

SadTalker smoke04 两臂实际失败于 facexlib 的 np.float（NumPy1.26.4），前一次 SHM noexec 已修复并验证。fresh attempt05 只添加经 pinned awing_arch.py SHA 验证的 numpy.float=builtins.float 兼容 shim，GPU 两次5秒准入，模型/输入/科学设置不变；index `.codex/scratch/cross_tfg_sadtalker_attempt01/fit_smoke_attempt05/requests_attempt01/request_index_attempt01.json` SHA `e3e9d5dff9d20cb042c256628db21f59d35701400e35bf28228a6b30671a7f82` 已准备，尚未 review/dispatch。所有失败控制和日志保留，不写成科学阴性。

本机 fragile：私有 PTY50274/bridge2133 持运行时凭据；服务器 root@connect.weste.seetacloud.com:12349，公钥 pin 有只读回执。远端 NS `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`，SHM `/dev/shm/cross_tfg_audio_head_20261003`。传输单文件上限8MiB，bulk分片。GPU 工作两次稳定空闲准入，不终止同行任务。现有旧实验/tests 脏文件不 stage-all，未运行 pytest/unittest。

## Verification
v4 21 文件及96 rows/steps、8fit×12 epoch、8NPZ/32 arrays 已独立验证；canonical curve最大误差9.54e-7，reward3.87e-10，loss2.18e-8，所有梯度/更新有限非零。Ditto9回读以及Latent5回读/PCM解码 root PASS。ASR family guard 已阅读全文，实际新候选 API 尚未调用；需单 frontend merged ledger 含 identity/各候选，禁止旧 v3 CER0 替代新候选内容证据。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified remote baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md` 和 `transfer-iteration.md` 是真实协议路径；上一个 checkpoint 漏写 develop- 前缀，已在 typed notes 修正，不编辑旧 handoff。

## Observations
- [result] 完整 Ditto dev 的 v3 相对 RMS 为−0.170123且10位说话人均负；共享profile与v4继续验证。
- [result] LatentSync 官方完整历史 fit 原生渲染和无损原音换轨已实际通过。
- [decision] 保持原自然音/原时钟、全部失败行、完整共同支持集与确认集封存；候选须同时过跨模型效果、内容和运动 guard。
- [blocker] v4 CLI 内层 exit2 待真实日志定位；共享360 consumer/selector、dev180、候选guards、确认和交付未完成。
- [next_step] 读取 v4 内层 CLI 实际 stderr 并修复等价 smoke，随即启动已冻结 dev180，同时完成 shared360 selector 和 Sad/Latent engineering pairs。

## Relations
- continues [[Codex checkpoint - 2026-10-04T03-58-36Z - Verified native training and pending cross-model execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]