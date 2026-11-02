---
title: Codex checkpoint - 2026-10-08T01-33-13Z - Durable gate mechanism and identified
  Wav2Lip startup policy mismatch
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t01-33-13-z-durable-gate-mechanism-and-identified-wav2-lip-startup-policy-mismatch
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
capture: deliberate
repo_root: E:/Documents/tts-exp
started: '2026-10-08T01:33:13.8017629+00:00'
git_sha: 42e124806900b8e4e7951a31d513efeed8f0f573
project: tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repository: dogLeft233/tts-exp
trigger: compact
username: dogle
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
status: open
agent: codex
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T01-33-13Z - Durable gate mechanism and identified Wav2Lip startup policy mismatch

## Summary
160 个冻结 Fit 门控干预已完成独立数组复核与持久保存；Wav2Lip 重复生成的失败已定位为启动校验混淆协议 ID 与实际策略摘要，正在通过新版本修复后重跑。

## Story
原目标是查清 TTS 为什么改善中文 TFG，用户随后要求产出适配多种 TFG 的自然音频增强头：增强音频只驱动生成，成片恢复完整原始自然 PCM 与原始时间轴。主 agent 维护理论、协议、Basic Memory、私有 SSH 和独立复核，GPT-6 Luna/max 子 agent 负责代码与实验。服务器已迁移至 root@connect.westd.seetacloud.com:18625；密码仅在私有运行时，不写文件。

冻结 Fit40/20 speakers、Adam120/3 passes 后，用同一权重比较 actual/static_only/±64 帧门控。四臂共 160 行原始 NPZ、原始自然 PCM、查询支持集与时间轴均经 root 独立核对。actual 的代理 ΔCbest=+0.057816875、ΔCanchor=+0.148420191、ΔD0=−0.046865845（相对 RMS）；门控置零仍有 ΔCbest=+0.048502278，两种错时滚动约 +0.0575。正确门控主要提供额外 D0 改善，当前候选的大部分 Cbest 增益与门控的精确时序关系较小。原始嵌入角度分解也支持 D0 改变涉及角度，并非仅改变范数。

进一步直接从原始 160 条完整距离曲线验证 Cbest=B−Dmin 的恒等式：actual 相对 RMS 的 ΔB=+0.101554346、ΔDmin=+0.043737471，二者之差才是 +0.057816875。17/40 个样本 Dmin 改善、23/40 恶化，39/40 最佳 lag 不变。当前训练对 D0 只有不恶化的平方惩罚，后续可研究中心化门控与直接 D0/角度奖励，但应先完成当前冻结候选的跨模型 Dev 评价。这是冻结 Wav2Lip 可微代理的 Fit 结果，尚非官方编码视频分数或跨模型泛化结论。

Wav2Lip strict11 实际 CPU preflight 通过，两次原生生成完成 141 帧后均因缺少构造策略 receipt 被拒收。完整 stderr 揭示 sitecustomize 提前抛错，但 Python 继续了原始推理。Root 独立 AST 验证 bootstrap03 的 POLICY 与 runner12 STRICT_POLICY 完全相同，实际 canonical payload SHA 为 54d853a6b6ce09cc8c757297f9617b5ecf4e0d01458a0b2e33dcf0316a4417f5；wrapper03 错把它和 QC06 协议 ID 196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc 比较。原生模型策略钩子因此未安装。这是工程失败，不是增强方法的科学阴性。

## Working State
持久权威：Basic Memory Task94 active、Experiment93 implemented 已写入并直接读回；《冻结增强头的门控时序与频谱贡献》concluded 笔记已写入读回。Fit120 的 39 文件持久快照与 gate diagnostic02 的 41 文件 39,432,642 B 持久快照均经独立逐文件全 SHA 读回。后新增 confidence decomposition receipt 尚未加入 41 文件快照。

机器局部：私有 PTY 57588/bridge07 仍可用于 root SSH；strict11 PID93956 已退出，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea，最后读取空闲 32230 MiB。源码包、下载和本地校验位于未追踪 .codex/scratch；禁止重复提交已有 response 的请求、覆盖消耗过的源码，或把本机 Linux 路径解析为 Windows 路径。

dev_method_resume 正准备新 wrapper04/runner13/strict attempt12：分开协议 ID 与实际 policy payload SHA，CPU AST preflight 验证原始 POLICY，启动失败写有界证据后 os._exit，保留 bootstrap03、原生 inference、M/D 与编码器协议。signal11_resume 独立审查。fit08_resume 已提供 Dev20 CPU 原生前端帧数包 attempt03，尚未由 root 完整审查和执行；它只从 exact source mel-loop 推导帧数，不能称为视频观测帧数。

Dev360 identity/RMS + Dev180 candidate 尚未运行。既有 MuseTalk/Ditto strict09 的原始 16 文件 root QC03 PASS 仍有效。Confirm80/40 未见 speakers 与 LatentSync 保持封存。正式准入需三个模型全为正、等模型平均超过 RMS +0.050，结合 D0/Canchor/content/visual/GUARD20 与完整失败分母。

## Changed Files
本轮仅提交 CONTEXT.md 的机制结果与持久证据入口，当前本地未发布提交 42e124806900b8e4e7951a31d513efeed8f0f573。HANDOFF.md 为忽略的本机快照。五个既有 dirty 文件未动；大量未追踪实验源码与 scratch 未提交。本次未运行通用 pytest。

## Verification
Root 独立原始 NPZ 数值复核、12 primary family 的 speaker-paired simultaneous99 区间、secondary angular 分解、confidence 恒等式和计数均已实际执行。静态四臂支持集、原始 PCM、geometry、查询和滚动规则通过；41 文件快照复制/fsync/read-only/all-target SHA 实际通过。strict11 CPU 原生依赖与 torch CUDA 未初始化检查实际通过；GPU 两行构造 receipt 失败，原始 manifest/stdout/stderr 五文件已下载并全 SHA 校验。Task94/Experiment93 更新脚本最终返回 verified=true，分别 7509/7519 字符。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 当前 HEAD `42e124806900b8e4e7951a31d513efeed8f0f573` 为本地未发布；GitHub 当前 branch 查询没有 PR。
- `.codex/scratch/root_gate_diag02_whole_readback_attempt01/`：original NPZ、numeric/angular/confidence independent receipts。
- `.codex/scratch/fit_only_gate_timing_diagnostic_durability_attempt01/`：41 文件 persistent snapshot 的 sealed copier、index、实际 responses。
- `.codex/scratch/root_strict11_failure_readback_attempt01/`：原始 failed final manifest 和两行 stdout/stderr。
- `.codex/scratch/cross_tfg_task_snapshot_receipt_attempt94.json` 与 `cross_tfg_experiment_snapshot_receipt_attempt93.json`：实际 Task/Experiment 更新读回。
- `.codex/scratch/dev20_w2l_native_frame_binding_attempt03/requests_attempt03/request_index_attempt03.json`：待 root 执行的 20 音频 CPU frame map。

## Observations
- [result] 冻结 Fit 的 Cbest 增益主要由背景距离 B 的增加贡献；正确门控额外改善 D0，精确时序对 Cbest 的额外收益较小。
- [decision] 完成当前冻结候选的 Dev 后再研究中心化门控与 D0/角度奖励，不使用 Fit diagnostic 重新选择当前候选。
- [decision] 已消耗 wrapper03/runner12/strict11 全部保留；工程修复使用新 wrapper04/runner13/strict12，模型、bootstrap 与原生入口不改。
- [blocker] Wav2Lip strict11 启动策略摘要比较错误造成钩子未安装；新的重复性资格尚未取得。
- [next_step] Root 审查新 wrapper04/runner13 strict12 封包，先实际 CPU preflight 再执行同样的两次原生 GPU 生成，并从原始 receipt/数组/媒体独立核验重复性。

## Relations
- continues [[Codex checkpoint - 2026-10-07T21-25-24Z - Independent gate mechanism results and native forward policy repair]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]