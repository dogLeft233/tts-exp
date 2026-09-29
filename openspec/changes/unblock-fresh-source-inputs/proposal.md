## Why

上一轮 `runs/fresh_source_cross_generator_20260910/` 在 P 阶段得到 6 个 fresh groups，其中 3 个有完整时长候选，但 `visual_audit=null`、fully-screened=0，未达到 12 正式 + 2 smoke。独立审计的 `integrity=GO` 只表示阻塞产物一致，不能启动科学实验。

## What Changes

- 补齐可追溯的新 pretrain 来源和真实 MediaPipe 输入审计，复用现有 P 包。
- 修复视觉 FAIL 记录处理，增加 cohort-only 验收，明确获取资产失败与质量不合格的区别。
- 为下游提供两个可并行实施的任务和一个集成验收任务；达到输入就绪后接回原 A/B/C/D 协议。
- 本补充协议仅将原设计“不自动下载新包”扩展为：下游执行本修复时，可按预先记录的来源和有限预算补充公开可访问或已有授权的 LRS3 pretrain；科学门槛保持原值。

## Capabilities

### New Capabilities

- `fresh-source-input-unblock`: 有界来源补充、可复核视觉审计和分阶段输入验收。

### Modified Capabilities

无。原 change 的 A/B/C/D 科学规则继续适用。

## Impact

改动限于 `scripts/experiments/fresh_source_inputs/`、对应测试、新 run、本 change 和原入口链接。复用 `lrs3_tts_visual_advantage/video_features.py` 的 MediaPipe 加载；不修改旧 run。实现已落地并完成本地 CPU 演练；由于完整历史排除后无可用新组，当前仍未取得视觉 PASS 或恢复科学实验。
