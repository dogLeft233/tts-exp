# Residual local-response audit

## Why

残差只写入部分mel列，但父主评分对固定U整体平均。需要区分“影响窗中也没有正向响应”和“局部有响应但未形成全局收益”。该问题可用既有mask、chunks、embeddings和距离矩阵回答，无需重新生成。

## What Changes

- 按Wav2Lip与SyncNet真实时间支持计算每评分行的残差曝光比例。
- 使用冻结全U natural anchor，检验曝光比例与CORRECT/N行级收益的组级关联。
- 只作回顾性定位诊断，不挑窗口重新宣称旧实验成功。

## Impact

CPU-only独立包 `wav2lip_residual_local_response`，遵循公共并行契约。视频/评分/训练预算均0；不依赖视觉教师审计的landmarks，也不改父完整U阴性。
