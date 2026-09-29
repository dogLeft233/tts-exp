## Why

ROI 控制实验为 `CONTROL_FAILED`：C=14/22，own-audio Sync-C 非劣性下界 −0.170397 < −0.10。CPU 诊断已从历史距离矩阵独立复现全部门禁，8 条 C 失败均为 offset_error。再次读取相同矩阵不能检查产生矩阵的评分实现，因此下一步从固定媒体重新提取 SyncNet 特征，独立计算距离和局部峰。

## What Changes

- 一次小型 seen-fit 诊断：8 条失败记录 + 按父顺序固定的前 2 条通过记录，每条仅 G_N/N、G_W/N，共 20 个新评分 cell。
- 独立评分入口复用官方模型结构/权重及明确的预处理契约，不调用历史 evaluate、calc_pdist 或评分汇总函数。
- 保存新旧曲线、峰、响应残差和独立离线验证；由执行 subagent 更新同一份 BM 实验笔记。
- 此实验只检查评分可复现性；历史 own-audio 失败仍独立存在，不解锁 bridge 或训练。

## Impact

新增 capability `wav2lip-roi-local-peak-recheck`；实现限定在 `scripts/experiments/wav2lip_roi_peak_recheck/`、对应 tests 和新 run。父实验、历史评分、模型权重保持只读。无需云实例、下载模型或训练。
