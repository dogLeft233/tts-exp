## Why

Wav2Lip face-ROI pilot 已完成控制阶段，但 inherited-C 仅 14/22（要求 ≥18/22），own-audio ΔSync-C 的 95% CI 为 [-0.170397, 0.186682]（要求下界 >−0.10）。现有 198 份距离矩阵足以先分辨统计/配对偏差、局部峰不清晰与预期响应偏差；直接再生成一轮视频缺少明确干预依据。

## What Changes

- 新增一次 CPU 离线诊断，固定复用 `20260906_host_fix5` 及其引用的 `fix1` 资产，零新生成、零新 SyncNet forward。
- 独立重建门禁，逐条解释 C 的失败条件，并把 own-audio ΔSync-C 分解为距离曲线中位数变化与最小距离变化。
- 输出可复核的差异清单、22 条诊断表和一个有证据支持的下一步建议。发现历史实现偏差时提供最小复现，本轮不修改历史实现或运行 bridge。
- 明确纠正旧记忆的措辞：own-audio 是非劣性检验，失败原因是 CI 下界 ≤−0.10，不能用“跨零”判定。

## Capabilities

### New Capabilities

- `wav2lip-roi-control-failure-diagnostic`: 对已封存 ROI 控制实验进行固定输入的独立复核和失败分解。

### Modified Capabilities

无；父实验的门槛、科学终态和 bridge 准入规则保持原定义。

## Impact

预计仅新增 `scripts/experiments/wav2lip_roi_control_diagnostic/`、对应测试及新 run。复用 NumPy、现有只读媒体/hash 工具，无新依赖、GPU、租卡或模型部署。实验执行后按 BM 指令记录诊断；本次交付仅为 spec，不宣称实验已运行。
