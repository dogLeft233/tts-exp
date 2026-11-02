---
title: Codex checkpoint - 2026-10-07T16-03-27Z - Verified zero head arrays and repairing
  strict training backward
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t16-03-27-z-verified-zero-head-arrays-and-repairing-strict-training-backward
repo_root: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
hostname: LAPTOP-E133GSLE
project: tts-exp
cwd: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
git_sha: 428a782d61d50ae6d389ca07cb3cdd7df17c7792
username: dogle
capture: deliberate
status: open
started: '2026-10-07T16:03:27.5484842+00:00'
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T16-03-27Z - Verified zero head arrays and repairing strict training backward

## Summary

独立核验了 Fit40 零头与 RMS 的全部评分和视觉数组，保留首次训练反向失败的完整证据，并完成 Ditto 权重恢复；当前正修复严格确定性反向计算以完成可训练的跨 TFG 自然音频增强头。

## Story

研究由中文自克隆 TTS 对 TFG 的增强现象发展到可使用的方法：自然音频经局部谱形增强只用于生成视频，成片恢复原自然 PCM 与时钟，再以原音评分。用户要求持续推进、遇到失败继续迭代，并由 root 维护理论与实验协议，GPT-6 Luna/max 子代理实现代码和实验。历史共享谱形方案未超过 RMS 的预设增益，因此现在学习九参数局部过渡增强头；幅度界限 ±3 dB、固定时长、RMS 与内容约束保持冻结。

服务器迁移后必须重建同 GPU 控制。Fit40 的 40 个 RMS 几何缓存已真实完成。training09 通过绑定与预检，完成全部 40 个零头门控，随后首次 loss.backward 在 CUDA MaxPool3d 的严格确定性限制处失败，优化更新数为零。root 下载原始 NPZ 和两份逐样本账本，独立逐值比对全部 1,240 个 lag 曲线值与 1,105,920 个视觉 float32 值，完全相等。这确认了当前零头计算路径的复现性，尚未产生学习参数或方法效果结果。

失败输出的 87 个文件已在服务器持久目录归档，并由收集器逐成员回读哈希；root 已独立核验下载的六个产物。GPU 前向和已冻结模型必须保持原样，修复仅将 MaxPool3d 和双线性裁剪缩放的反向累加移到确定性 CPU ATen 路径。先资格化真实激活尺寸、保存索引和梯度，再运行一次完整目标反向且不做优化，成功后才执行原定 120 步。

模型恢复并行推进：Ditto 的 12 个固定权重真实下载完成，总计 2,314,719,638 字节，worker 逐文件重算哈希全部通过。MuseTalk 原启动在发现 gdown 缺失时退出，尚未下载新资产；旧失败保留，新的 attempt05 仅向隔离目录安装 SHA 固定的 gdown 与缺失纯 Python 依赖，同时修复不受支持的命令行选项。Dev 执行器正修复三个后端子进程内的严格策略、逐行随机种子、六个已资格化符号链接的实际目标校验，以及原始媒体和策略回执收集。此处源代码准备不能替代实际 Dev 运行。

## Working State

冻结 Fit40/20 speakers、Adam 120 步/三遍；Dev20/10 speakers × 三种子 × Wav2Lip/MuseTalk/Ditto，必须在迁移后的同 GPU 上新做 360 个 identity/RMS 控制和 180 个候选。接受条件为三个模型方向均正，等模型增益相对 RMS 额外至少 +0.050，并同时通过 D0/Canchor、内容、视觉、GUARD20 和完整失败分母。Confirm80/40 未见说话人与 LatentSync 留作密封确认，未开启。

本地 HEAD `428a782d61d50ae6d389ca07cb3cdd7df17c7792` 尚未推送。该提交保存 CONTEXT 和已审阅的 training09 runner/binder/launcher。五个原先存在的无关修改仍保留。HANDOFF 被 Git 忽略，scratch、未提交源与进程均为机器本地状态。Task 第 84 次更新、Experiment 第 83 次更新已真实写入并读回，阶段为 `zero40_independent_pass_training09_deterministic_backward_repair`；Muse03 的 gdown 缺失是在此次更新之后发现的，需要在下一实际阶段同步。

当前服务器为 `root@connect.westd.seetacloud.com:18625`，指纹 `SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0`，GPU UUID `GPU-d09622d0-3535-e662-bb45-3df2b049fcea`。密码和云 API key 只存在私有运行时，永不写文件。实验命名空间 `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`；临时数据在 `/dev/shm/cross_tfg_audio_head_20261003`。root 私有 PTY 57588、bridge07 端口 12249 是脆弱状态，需要实际存活核验。持久项目空间在 training09 归档后约 7.529 GB；临时空间约 30.583 GB。Fit40 producer 的 287 MB 完整树尚待持久备份，不能仅依赖 SHM。

## Changed Files

- `CONTEXT.md`：94 行，记录真实零头 PASS 与首次反向失败。
- `scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_attempt09.py`、`late_bind_fit40_native_transition_head_v3_attempt09.py`、`launch_fit40_transition_head_v3_attempt09.py`：已审阅来源逐字节复制后提交，旧 attempt 不覆盖。
- `runs/cross_tfg_audio_head_20261003/fit40_transition_head_v3_attempt09_recovery_attempt01/`：六个真实下载产物与独立核验回执，当前未提交。
- `.codex/scratch/root_training09_zero40_readback_attempt01.py`：root 独立数组核验实现，机器本地。

## Verification

CPU source closure attempt11、training09 实际绑定和 GPU/容量预检通过；训练真实 PID 47690 已退出，status FAILED/optimizer_step/completed_steps=0。独立零头核验回执 `root_independent_zero40_readback_attempt01.json` SHA256 `1c0a48159c5f997ace11c018edfd0fd24724622ccb18db5d48144d2901102562`；原始 NPZ SHA256 `c4e3723500b1618e8d8f026b71d8144170e7ea37d807335999e78efc612eb5eb`。40 个零头门控记录的波形一致性全部通过，但 root 尚未独立重新解码波形张量。

服务器 `fit40_transition_head_v3_attempt09_recovery_attempt01/training_evidence_attempt01/` 的归档含 87 文件/29,758,336 字节，tar SHA256 `05d9c2da8a71cb4e4e30b5a2399f9e07e753f52544955871dca9eaf0c0833a04`，状态 `BACKUP_CAPTURED_NONCOMPLETE_RUN`。收集器已独立读回全部成员，root 尚未重新读取完整 tar。

Ditto03 rc=0、12 项终端完整重算通过；Muse03 rc=1、completed_assets=[]，失败原因为 gdown missing。Muse04 预检实际重算八个源文件 SHA 通过，三项固定 revision 的 HTTP HEAD 实际 200；下载前仍需 attempt05 隔离依赖安装成功。没有新 Dev、终端 120 步独立评分或 Confirm 结果。没有新增或运行通用 pytest。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)；当前 `428a782d61d50ae6d389ca07cb3cdd7df17c7792` 为本地未推送提交。
- GitHub 当前分支 PR 搜索实际无匹配；无 PR。
- `.codex/scratch/root_training09_failure_download_index_attempt01.json`、`.codex/scratch/root_ditto03_status_request_attempt03.json.response.json`、`.codex/scratch/root_muse03_status_request_attempt01.json.response.json` 是本次实际运行的本地证据入口。

## Observations

- [result] Fit40 零头与 RMS 的 1,240 个 lag 值和 1,105,920 个视觉值经 root 独立逐值比较完全一致；训练尚未发生任何优化更新。
- [decision] 严格确定性策略不放宽，保留 CUDA 前向，仅资格化 CPU 固定累加反向；失败记录保留在原 attempt。
- [decision] 新 GPU 的 360 控制不可复用旧 GPU 分数；Wav2Lip 的三臂重复资格化不代表三种 TFG 均已资格化。
- [blocker] 当前完整训练反向受 MaxPool3d 与 bilinear CUDA backward 限制，joint 修复与完整单次 backward gate 尚未执行。
- [blocker] MuseTalk 缺少 gdown 与部分纯 Python 依赖；隔离修复正在准备，完整资产恢复尚未完成。
- [next_step] 审阅并执行 joint 确定性反向修复的真实激活资格化与完整目标单次 backward gate，通过后运行冻结的 Fit40/120 并独立重评分，再进入同 GPU 三模型 Dev。

## Relations

- continues [[Codex checkpoint - 2026-10-07T15-31-56Z - Qualified forty sample terminal caches and restoring training execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]