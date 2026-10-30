---
title: Wav2Lip face-ROI replacement pilot 2026-09-06
type: experiment
permalink: tts-exp/experiments/wav2-lip-face-roi-replacement-pilot-2026-09-06
tags:
- wav2lip
- roi
- replacement
- control
- failed
- engineering-fixed
---

# Wav2Lip face-ROI replacement pilot 2026-09-06

本实验按 OpenSpec `validate-wav2lip-face-roi-replacement` 验证“官方 SFD 人脸 ROI + 冻结 Wav2Lip + replacement bridge”是否具备继续实验的工程与控制前提。正式结果已封存；本轮没有进入 bridge。

## Protocol and execution

- 数据为 22 个 seen-fit source groups；控制阶段按协议生成 66 个视频并产生 198 个 SyncNet score cells。
- GPU 必须在宿主命名空间执行：`[redacted-local-path]` 在宿主机验证为 torch 2.6.0+cu124、CUDA 可用、Tesla V100-SXM2-16GB、kernel test 通过。普通 Codex sandbox 的 `/dev/nvidia*` 透传失败是执行环境限制，不是 wav2lip 环境或驱动故障；无需重装驱动。
- 首次宿主运行暴露并修复了三处工程问题：视频 arm 到音频 arm 的映射、inherited gate 的聚合 `passes` 字段、validator 对 media/control-analysis 实际 schema 的读取。修复后 focused pytest 为 9 passed，ruff clean。
- GPU 产物由 `runs/wav2lip_face_roi_replacement_20260906_host_fix1/` 生成；最终在 `runs/wav2lip_face_roi_replacement_20260906_host_fix5/` 重新分析并完成独立 validation。fix5 复用了不可变的 GPU 产物，没有重复生成视频。

## Result

- 工程状态：GO。
- validation：valid，status=complete。
- scientific decision：**CONTROL_FAILED**。
- repeatability：PASS（22/22）。
- baseline：PASS（R_N=22、G_N_N=22，均达到要求）。
- generated-repeat：PASS（offset=22，CI open）。
- inherited：FAIL，仅 C 条件为 14/22，低于 18/22；A/B/O 均 PASS。
- own-audio：FAIL（Sync-C 95% CI=`[-0.170397, 0.186682]`，非劣性下界 `-0.10` 未通过；CI 跨 0 本身不是失败规则；Sync-D CI=`[0.000202, 0.287374]`，offset 合格 `22/22`）。
- replacement-damage：PASS（22/22 为正）。
- bridge：未运行（0 个 bridge video、0 个 bridge score），因此不能声称存在 replacement effect。

## Conclusion and next step

GPU 已恢复且可用；本轮真正的阻塞是科学控制条件未通过，而不是设备问题。当前只能得出：在这组 seen-fit Wav2Lip 控制数据上，inherited C gate 与 own-audio gate 不满足预设阈值，不能进入 BRIDGE_075，也不能据此宣称 replacement 效应或泛化结论。

下一步已完成固定父 run 的 CPU 离线诊断：198 个 score cells 全部独立复算，C 的 8/22 失败记录均由 `offset_error` 构成，baseline/边界峰/模糊峰均为 0；历史派生值逐字段一致。own-audio 的失败来自 Sync-C 非劣性下界低于 `-0.10`，不是“跨 0”。诊断产物在 `runs/wav2lip_roi_control_diagnostic_20260906_audit4/`。下一步只建议对 C 失败记录做一次独立 SyncNet 局部峰复核；本轮不自动运行 bridge，也不把结果解释为“Wav2Lip 泛化失败”。

## Observations

- [result] 宿主命名空间可稳定运行 GPU；普通 sandbox 的设备隔离是已确认的工程边界。
- [result] 控制阶段完整产出 66 个视频、198 个分数，正式 validation 通过。
- [problem] inherited C 仅 14/22，8 条记录均为 offset residual 超过 1 帧；own-audio Sync-C 非劣性下界为 `-0.170397`，低于 `-0.10`，导致 CONTROL_FAILED。
- [decision] 在控制修复前不运行 bridge，不宣称 replacement 效应。

## Relations

- 验证规范：[[openspec/changes/validate-wav2lip-face-roi-replacement]]
- 失败诊断：[[Wav2Lip ROI control failure diagnostic 2026-09-06]]
- 项目入口：[[CONTEXT]]
