---
title: Codex checkpoint - 2026-10-07T17-29-10Z - Completed Fit40 Adam120 and preparing
  transfer evaluation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t17-29-10-z-completed-fit40-adam120-and-preparing-transfer-evaluation
model: gpt-6.1-sol
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
git_sha: d5fafb0c71c2d5ba3a5dca694852b9ab19a7ebd0
started: '2026-10-07T17:29:10.1138287+00:00'
repository: dogLeft233/tts-exp
username: dogle
trigger: compact
agent: codex
status: open
project: tts-exp
repo_root: E:/Documents/tts-exp
capture: deliberate
hostname: LAPTOP-E133GSLE
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T17-29-10Z - Completed Fit40 Adam120 and preparing transfer evaluation

## Summary

严格完整反向资格化已经通过，九参数局部谱形增强头真实完成 Fit40 的冻结 Adam120；现在进入终端独立评分和迁移后同 GPU 三模型验证。

## Story

用户从恢复未同步的 TTS/TFG 研究路线，推进到解释局部声学机制并产出跨模型自然音频增强头，要求持续执行。增强音频只驱动视频，最终换回原始自然 PCM 和时钟，再评估同步。root 维护理论、冻结协议、BM、私有 SSH 和独立复核；GPT-6 Luna/max 子代理实现实验。历史静态共享谱形未达到相对 RMS 额外增益门槛，当前九参数头针对稳定、上升、下降谱形变化乘三个参考编码频带，固定时钟、RMS、平滑和 ±3 dB。

上一轮实际 training10 的完整 loss backward 因 CUDA reflection_pad2d 严格确定性退出，更新零。此次 v7 在 head 模块局部 F 引用中保留原始 CUDA 前向，使用 CPU ATen 反向，未改全局 functional、STFT 或冻结裁剪源码。root 全文审阅源码差异和运行包后实际派发 attempt11：40 零头不变，122 次原生前向逐位相同，72 次 CPU 反向重复逐位相同，独立 pool serial scatter 相同，完整 objective 梯度九个标量均非零。资格化通过后才实际启动冻结训练。最新只读状态证明 PID50473 已退出，120 条步记录连续完成且无解析错误，三遍40行/20说话人，没有打开 Dev 或 Confirm。训练完成不等于效果已经合格；下一步必须对最后一步参数做独立终端评分与三模型迁移。

producer05 原始树仍在 SHM。root 发现旧 packet02 几何 schema 和嵌套缓存字段不匹配，保留失败响应，完整审阅 revision03 三处修复后实际预检并启动 CPU 解码和完整持久归档 PID50723。归档/解码结果尚未回读。Ditto 十二权重已完整恢复；MuseTalk 九权重恢复仍在下载，最新已读 unet 为1,087,496,192/3,400,074,924字节。Wav2Lip/Ditto 先行 repeat pilot 的十源闭包和 detached launcher 已审阅，但 q00 无 JSON stdout、q13 混用 canonical 与 raw manifest SHA 在派发前被发现；子代理已经给出新 stage-only packet04，尚待 root 核验和实际执行。

## Working State

冻结 Fit40/20 speakers、Adam120/三遍/seed20261003。迁移后同 GPU Dev20/10 speakers×三seed×Wav2Lip/MuseTalk/Ditto，fresh identity/RMS360 + candidate180。接受条件仍为三个模型均正、等模型相对 RMS 额外至少 +0.050，并通过 D0/Canchor、内容、视觉、GUARD20 和完整失败分母。Confirm80/40未见说话人与 LatentSync 密封。当前没有新 Dev 或 Confirm 效果。

持久源码 HEAD `d5fafb0c71c2d5ba3a5dca694852b9ab19a7ebd0` 本地未推送，branch codex/cross-tfg-audio-head-20261003；保存 v7/runner11/binder11 和94行 CONTEXT。Task86/Experiment85 最近记录仍为资格化通过、训练待启动，应在独立终端读回后更新。HANDOFF 为 ignored 本地快照；五项无关旧修改保留。basic-memory/ 与 runs/ 为本机忽略产物，不能视为已推送。

服务器 root@connect.westd.seetacloud.com:18625，SSH 指纹 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码和 API key 只走私有运行时，不入文件。持久命名空间 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003，易失树 /dev/shm/cross_tfg_audio_head_20261003。私有 PTY57588/bridge07:12249 是机器本地易失状态。所有 SSH 由 root 唯一派发，GPU 工作串行，CPU 归档/下载可并行；不在小空间 autodl-tmp 写大产物。

## Changed Files

- scripts/audio_enhancement/v3_v4_native_runtime_adapter_v7_attempt01.py：35785字节，SHA256 c307d0b7f363fb5a0805ee7466a470f19a416960793452afb704ec9d95bdb51c。
- scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_attempt11.py 与 late_bind_fit40_native_transition_head_v3_attempt11.py：实际训练源码，已提交。
- .codex/scratch/fit40_transition_head_v3_attempt11_reflection_pad_attempt01/：actual 资格化 raw 回读及 training120_attempt03 的 actual 绑定、预检、启动、最新终端状态响应。
- .codex/scratch/fit40_native_rms_geometry_attempt05_root_raw_readback_attempt03/：真实启动的解码/全树备份修复包；status/verify 子索引位于 .codex/scratch/root_producer05_rawbackup_revision03_*_index_attempt01.json。
- runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_subset_attempt03_packet_attempt04/：修复后的 stage-only 请求，尚未派发。

## Verification

root 直接下载并核验资格化 completion f76fa4b272d786cf7ab9da7845f2a6f4c45a7dde8f6963075e3ca6fcbbe0a346 与 strict record b36a34cb3ecdebe237b531704f79577cc45a98d1072d432b77ce70f447d286a8；独立 receipt 为 root_independent_actual_qualification_readback_attempt01.json。零头 NPZ c4e3723500b1618e8d8f026b71d8144170e7ea37d807335999e78efc612eb5eb 与09/10相同，资格化完整梯度 d774aadb4a9c3efe2a35a30e219953072d3a75f1574fe941ebff311374630752。

training120 实际 binding fe80a1d920a039d48581d8f63a1b3fdf4eb30e122ffe2fbc5ed8715df0b6f199、preflight85d5ada6776642d132926d5cccf9d4f6c79652015e83cf1f3c217bd74f51bc5c、launch4ace173bf82f6fc356f315d210b17f96e6798b675a6cf975be06b9593d86afd6。最新实际只读 status revision02 response SHA256 cae4dac857f85fb19b9757e6a5929cd9f12f0fb91875cdbdbbb3476e9987e2c53：无截断/timeout/stream error，进程已退出，completion COMPLETE_FIT40_V3_120_STEPS，耗时196.215秒，步数120/120，零头40通过，无 Dev/Confirm。该响应中 completion3558字节/3535908e79bea9854b1d30c6af214b68735d6db2d20e11d3b2111d07a58f6a1c，checkpoint1802字节/8d5eb607f77963291372b33245f13f9b9578ebf71657acf1b43243746c538eb8，training_record5628字节/c2ed183ae5894bd3c42dc56c0242539f1f8290df1230d7b1aa4df8b7584a7e67；独立 raw 下载和终端 rescore 待做。

producer03实际 preflight 检查4,979文件/287,367,187字节通过，无GPU/noTorch；launch response d57216e64c86e2ae6a9c17a7cb1399369c6418424b4407380a8208511be32a76，PID50723/tick324833840。尚未把归档或解码记为通过。未运行通用 pytest；已执行授权实验资格化和独立源码/实际响应检查。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)；当前 d5fafb0 为本地未推送。
- 当前 branch PR 的 GitHub 只读查询无匹配。
- .codex/scratch/fit40_transition_head_v3_attempt11_reflection_pad_attempt01/training120_attempt03/13_read_training_status_attempt11_request_revision02.json.response.json 是最新实际训练状态锚点；不要重复派发已完成请求。

## Observations

- [result] 严格完整反向资格化通过，随后真实完成 Fit40/20说话人/Adam120；尚未得到独立终端或跨模型效果结论。
- [result] producer05 revision03 CPU 原始解码和全树备份已经真实启动；结果未回读，Muse06 仍下载。
- [decision] 坚持冻结科学协议，最后一步参数独立评分后进入同 GPU fresh360控制与180候选；工程修复不改变效果门槛。
- [decision] 仅在 head 模块局部用原CUDA前向加CPU反向，严格确定性全程打开且 warn_only=False。
- [blocker] MuseTalk 资产与新推理资格化未齐；旧 rescore 包绑定 producer04/runner02，需改成实际 producer05/runner11/v7。
- [next_step] 独立下载核验120步终端/checkpoint/训练记录并实际执行 fresh no-grad 终端评分，同时完成原始备份和后端重复资格化，再启动三模型同 GPU Dev。

## Relations

- continues [[Codex checkpoint - 2026-10-07T16-57-01Z - Verified zero head and repairing reflection backward]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]