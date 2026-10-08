---
title: GRID aperture 轨迹形状相关开发门 2026-09-27
type: experiment
permalink: tts-exp/experiments/grid-aperture-轨迹形状相关开发门-2026-09-27
status: concluded
---

# GRID aperture 轨迹形状相关开发门

## Observations
- [status] concluded
- [question] 固定时间配对下 aperture 轨迹形状相关的测量开发；不解释为纯时序误差、幅度或完整嘴型保真。
- [result] 冻结门一次计算7/8通过；唯一失败s5 FIXED SCALE1.1 r=.9497408965<.95。运动/空间残差门24/24，空间r门191/192，repeat24/24，reverse/warp72/72，self shift96/96、relative shift32/32通过。固定分母8，不删源。
- [validation] 独立complex LS从432数组重建，864文件hash PASS，3296标量最大差2.4158453015843406e−13，7/8一致。旧cal-only全局lag−3未重选，RAW lag0与lag−3逐条并列。
- [boundary] 规则受旧cal结果启发，这8条是开发集，不是独立验证；旧完整几何3/8 FAIL保持concluded。未计算cal RAW→FIXED差、Sync-C，未读eval/媒体或新forward。后续eval需另run封存。
- [report] runs/grid_shape_correlation_development_20260927/report.md；protocol SHA c81ddee283d0aed7a92fc4695611e51516af03550460dde6ebda2ce0bb007793；独立复核 runs/grid_geometry_protocol_audit_20260927/r_development_review/independent_gate.json。
## Relations

- follows_up [[GRID 固定电平连续口部几何校准 2026-09-27]]

## Changelog

- September 27, 2026：按 root 授权创建 planned；新聚合尚未计算，待独立协议审核。

- September 27, 2026：静态reviewer PASS后一次CPU计算，7/8；独立数值复核PASS后concluded。保留旧3/8 FAIL及开发适配边界；root另授权新16eval协议，未在本run执行。