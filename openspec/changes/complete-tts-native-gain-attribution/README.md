# 补全 TTS 原生增益归因实验

状态：🔄 SPEC_READY，2026-09-16。本 change 交付下游执行规范，尚未部署模型或生成 B 数据。

## 项目和问题

TFG 根据肖像和音频生成说话视频，SyncNet 对视频和音频评分；Sync-C 越大代表该指标下的同步置信度越高。历史上某些 LRS3 英文实验中，TTS 音频也有原生优势。原生差 `C(G(T),T)−C(G(N),N)` 同时改变了生成输入与评价输入，因此单看分数无法判断口型是否真的更好。

上一轮将视频固定，仅改变评分音频（A）；还计划重新生成视频并交叉评分（B）。本任务补全 B，并修复 B 尚未走通时隐藏的实现和验收缺口。不能把此任务简化成设置几个环境变量。

## 已经知道什么

父 run：`runs/tts_native_gain_attribution_implementation_20260915_v1/`。A 有 180 个科学配对和 18 个控制；24 个历史矩阵重放最大误差 `1.144e-05`，6 个恒等控制和 12 个延迟控制通过。以下区间均为六项 Bonferroni 校正的 99.166667% bootstrap 区间：

| 固定视频实验 | ΔSync-C | 校正区间 | 能说明什么 |
|---|---:|---|---|
| TTS 驱动视频：原评分音频减去加噪音频 | +0.571 | [0.158, 1.011] | 加噪降低了固定视频的评分 |
| 自然音频驱动视频：谱抑制减去基线 | −0.075 | [−0.161, 0.021] | 没有改善证据；效应区间落在预定 ±0.200 内 |
| 真实视频：谱抑制减去基线 | −0.105 | [−0.213, 0.059] | 方向不确定，不能认定等效 |

这支持“评价输入会影响分数”这一局部结论。它尚未证明 TTS 优势来自更低噪声、SyncNet 偏爱 TTS，或较高 TTS 质量带来更好口型。N 的谱抑制和 T 的加噪是不同操作，不能直接把它们比较成 TTS 特异性。人类同步和音质尚未评估。

## 为什么没完成

1. 本轮未绑定可验证的 LeapTalk repo、完整权重和推理 adapter；历史部署笔记指向已关闭的远端，历史视频文件不能代替生成环境。
2. `generation.py:crossed_score_stage` 的成功路径仍无条件抛出 `ProtocolError`；设置 `LEAPTALK_CROP_COMMAND` 也不能执行交叉评分。
3. `_run_one_generation` 未将 repeat 纳入输出路径，4 个重复控制会覆盖基线；现有验证器未完整验收 B 成功路径，`valid/PARTIAL` 只说明上一轮有限产物通过了已有检查。
4. 人工同步包目前是配对台账，缺 B 视频；后续还必须保证双方实际播放同一 A0、隐藏重复题不泄漏、真实评分分析可用。

更多经源码确认的缺口及测试见 [design.md](design.md)。不能将所有失败归为外部依赖。

## 推荐修复顺序

1. CPU 上补齐 adapter 契约、repeat 隔离、冻结 ROI、完整评分、独立验证和续跑；合成 fixture 先验证成功路径。
2. 优先复用已获授权的现成 LeapTalk 环境或有空间的本地盘；缺失时按官方固定 revision 部署同一家族。历史权重无法恢复时采用 `NEW_LEAPTALK_CONFIGURATION`，重新测 fresh native。禁止替换成 Wav2Lip。
3. 先测资源与完整小样本链，再冻结正式配置运行 144+4 视频、336+8 评分。完整复用 A 的证据，不修改父 run。
4. 补齐可实际播放的盲评包、六项分析、独立验收和中文最终报告。无人评分时如实保留未评估状态。

2026-09-16 快照：本地 V100 16 GiB，显存已用 5 MiB、利用率 0%；根盘空闲约 7.1 GiB。历史部署记载权重约 6.5 GB，另需环境、缓存和产物，不能据此认定本地可部署。下游必须实时核算。优先更大已授权磁盘；若必须新付费租机，先提交目标、报价、总预算及收尾方案，并检查现有授权。不能因缺租机授权停止 CPU 实现。

## 下游入口与验收

依次阅读 [proposal.md](proposal.md)、[design.md](design.md)、[规范](specs/tts-native-gain-completion/spec.md)、[tasks.md](tasks.md)、[evidence-bindings.json](evidence-bindings.json)。原科学协议继承 `../disentangle-tts-native-gain/protocol.md`，本 change 补充工程契约和完成定义，不修改原统计假设。

正式新 run 建议 `tts_native_gain_attribution_completion_<UTC时间>_v1`。新增 CLI 的目标接口写在 design，实施前并不存在，不得将命令示例当作已交付功能。

成功必须是：完整数据、有效控制、六项重算、可用盲评媒体、独立 `validation=valid` 和 `engineering_status=AUTOMATIC_COMPLETE`。结果阳性不是完成条件。部署成功、空表、mock 通过或只有 A 均不算完成。
