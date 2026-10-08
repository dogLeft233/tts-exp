---
title: MFA-linear 视频重定时 Sync-C 约束工程 smoke 结果 2026-09-23
type: experiment
permalink: tts-exp/experiments/mfa-linear-视频重定时-sync-c-约束工程-smoke-结果-2026-09-23
status: concluded
date: '2026-09-23'
hypothesis: 受约束的视频重定时能否使 MFA-linear 驱动视频更好匹配原自然音频。
result: ENGINEERING_PASS；identity fallback；官方 ΔSync-C=0.000。
conclusion: Smoke 验证实现和评分链可复算；未找到可接受 warp，且科学状态仍 NOT_RUN。
report: runs/mfa_linear_video_retiming_v1/smoke_20260923_stageb256_rounds4/09_report/report.md
inputs: AISHELL-1 clean-MFA sample 1，paired_key=aishell1_test_400__BAC009S0765W0122；portrait
  3。
outputs:
- runs/mfa_linear_video_retiming_v1/smoke_20260923_speccomplete2/
- runs/mfa_linear_video_retiming_v1/smoke_20260923_stageb256/
- runs/mfa_linear_video_retiming_v1/smoke_20260923_stageb256_rounds4/
model: Wav2Lip GAN checkpoint ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8；SyncNet V2 checkpoint 961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442。
code_paths:
- scripts/experiments/mfa_linear_video_retiming/
- scripts/configs/mfa_linear_video_retiming_v1.yaml
- scripts/configs/mfa_linear_video_retiming_v1_stageb_budget256.yaml
- scripts/configs/mfa_linear_video_retiming_v1_stageb_budget256_rounds4.yaml
- tests/experiments/mfa_linear_video_retiming/
tags:
- mfa-linear
- wav2lip
- syncnet
- video-retiming
- engineering-smoke
---

# MFA-linear 视频重定时 Sync-C 约束工程 smoke 结果 2026-09-23

## Purpose

按《MFA-linear 视频重定时与自然音频同步 Implementation Spec》完成 Wav2Lip 首轮工程 smoke：MFA-linear 音频 M 驱动固定肖像生成视频，搜索仅重采样视频时间轴以匹配原自然音频 N；随后运行仓库 SyncNet 官方整链并由独立 checker 复核。此 run 仅覆盖 sample 1 / portrait 3，用于验证模型调用、评分、媒体合同、搜索和恢复链，不用于科学结论。

使用固定 clean-MFA 配对 `aishell1_test_400__BAC009S0765W0122`。搜索尝试 183 个候选后达到 Stage B 候选预算；没有通过接受门的非 identity warp，最终安全回退 identity，最大位移为 0 帧。因此本次只证明流程可运行、证据可复算，不能据此判断 Sync-C 引导重定时是否有效。

## Results

- 固定裁剪搜索基线：Sync-C=5.484，Sync-D=7.638，D0=14.372，offset=-3。
- 搜索状态为 `BUDGET_LIMITED`，停止原因 `STAGE_B_CANDIDATE_BUDGET`；候选数 183，selected=identity，搜索前后 Sync-C 变化为 0。
- 官方 SyncNet 8/8 个 smoke cell 完成。主配对 M/N 与 R/N 使用相同 128 列支持：M/N 的 Sync-C/D/D0=6.013/8.185/13.891，R/N 同为 6.013/8.185/13.891；两者 offset 均为 -2，官方 ΔSync-C=0.000、ΔD0=0.000。
- 自然驱动参考 N/N 的官方 Sync-C/D=7.012/7.326。搜索与官方评分使用同一冻结 SyncNet 权重，官方复评不是独立模型验证。
- 最终视频保留自然音频 N；PCM 解码逐样本一致，视频像素与 PTS 在 mux 前后逐一一致。

## Engineering validation

所有阶段 audit、render、calibrate、search、seal、transfer、official、check、report 均为 PASS。独立 checker 重新验证 Wav2Lip checkpoint/参数与 N/M 调用、从 knots 重建 map、从候选 embedding 重算分数和选择、官方 activesd/曲线/日志及 PCM/像素/PTS；validation 为 PASS / engineering GO。report binding 通过。Fresh SyncNet 在独立进程中重载并前向 M、R，各自 embedding 和距离矩阵相对搜索缓存的最大差均为 0。

阶段运行中将 GPU 利用率采样区分为外部 compute 进程与本 run 刚结束前向的短暂峰值；外部进程仍触发 RESOURCE_WAIT，不终止任何进程。有限冷却轮询结果随 cell/checker 记录。

## Budget expansion replay

用户要求增加评分预算后，保留原 run 并进行了两次独立扩展 smoke。第一版把 Stage B 上限从176提到249，但仍为每步2轮；坐标轮次自然结束于197次总评分（Stage A 7、Stage B 190），最佳候选与首轮相同。随后将每步轮数提到4，仍保持总评分上限256、单样本20分钟上限、±3帧映射与全部接受门不变。

最终扩展 run `smoke_20260923_stageb256_rounds4` 实际完成256次评分（Stage A 7、Stage B 249），在 Stage B 候选预算处停止。固定裁剪搜索的 identity 基线为 C/D/D0/offset=5.484/7.638/14.372/-3；最佳非 identity 候选为 6.177/7.257/13.068/-2，最大位移1帧。Sync-C、D0、D及局部指标均有改善，但 offset=-2 未达到最终要求的 |offset|≤1，故 selected 仍为 identity。

官方 SyncNet 8/8 cells 与独立 checker 均通过；identity 回退使官方 M/N 与 R/N 的 ΔSync-C、ΔD0仍为0.000。fresh M/R 前向及距离矩阵最大差为0，report binding PASS。该最佳尝试是固定裁剪搜索分数，未作为封存 R 进入官方评分；同一 SyncNet 被用于搜索和复评。科学状态仍为 `NOT_RUN`，人工画质评估仍为 `HUMAN_NOT_ASSESSED`。

## Interpretation and limits

工程 smoke 完成，科学状态仍为 `NOT_RUN`。单一样本、单肖像和 identity 回退不支持任何同步增益或泛化主张；人工画质盲评为 `HUMAN_NOT_ASSESSED`。同一 SyncNet 被用于搜索和官方复评，存在评价器过拟合风险。需另外按冻结 3 样本 × 3 肖像协议运行正式队列，并继续保留未达门的负结果。

## Observations

- [status] concluded
- [progress] 结论范围仅为工程 smoke，科学状态 `NOT_RUN`。
- [result] 搜索预算内未找到可接受的非 identity warp；selected identity，search ΔSync-C=0.000。
- [result] 官方主配对 M/N 与 R/N 的 ΔSync-C=0.000、ΔD0=0.000，支持列数均为 128。
- [validation] 最终扩展 run 的独立 checker PASS；2 个 fresh forward、8 个官方 cell 均已复核。
- [budget] 总评分预算扩至256（7 Stage A + 249 Stage B），最佳固定裁剪候选 C=6.177、offset=-2；最终接受门未通过，官方 ΔSync-C=0.000。
- [tests] compileall 通过；`pytest -q tests/experiments/mfa_linear_video_retiming`：67 passed。
- [quality] `HUMAN_NOT_ASSESSED`；没有人工盲看结论。
- [boundary] 此次工程结果不构成正式实验、科学阴性结论、跨肖像泛化或增强器可蒸馏性证据。
- [outputs] run=`runs/mfa_linear_video_retiming_v1/smoke_20260923_speccomplete2/`；报告=`runs/mfa_linear_video_retiming_v1/smoke_20260923_speccomplete2/09_report/report.md`；独立验证=`runs/mfa_linear_video_retiming_v1/smoke_20260923_speccomplete2/07_check/validation.json`。

## Relations

- part_of [[MFA-linear 视频重定时与自然音频同步 Implementation Spec]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 完成 Wav2Lip 单样本工程 smoke、官方 SyncNet 评分与独立 checker；科学状态保持 NOT_RUN | September 23, 2026 | agent |
| 按用户要求扩展 Stage B 评分预算并完成 256 次候选评分、官方复评和 fresh checker；科学状态仍为 NOT_RUN | September 23, 2026 | user（扩展请求）；agent（实施与验证） |