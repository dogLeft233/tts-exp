---
title: Ditto-50 VSR 与 Sync-C 增益关联验证结果 2026-09-18
type: experiment
permalink: tts-exp/experiments/ditto-50-vsr-与-sync-c-增益关联验证结果-2026-09-18
status: concluded
experiment_date: '2026-09-18'
tags:
- vsr
- ditto
- syncnet
- tts
- linkage
---

# Ditto-50 VSR 与 Sync-C 增益关联验证结果 2026-09-18

- implements [[Ditto-50 VSR 与 Sync-C 增益关联验证 Spec]]
- run: `runs/vsr_ditto50_linkage_v1/`；smoke: `runs/vsr_ditto50_linkage_smoke/`
- cohort: Ditto-50 natural_raw/tts_raw，单说话人 S0765；VSR 只读去音轨视频帧，SyncNet 使用原始 MP4 音轨。

## Observations
- [status] concluded

- 审计候选为 50 对；47 对进入冻结的 source-eligible 集合。ID 2 因历史 `tts_meta` SHA 与当前成对清单不一致而排除；ID 42 同因 SHA 不一致且目标含 CMLR OOV 字符“蒲”排除；ID 44 因目标含 OOV 字符“仇恨”排除。排除发生在推理前，未按分数挑样本。
- 47 对均完成五视图（native/frozen/reversed/matched/matched_frozen），共 470 个 VSR 视图；两臂 SyncNet 共 94 个 cell，0 失败；严格完整交集 J=47。
- 工程状态为 `PASS`。独立验证重新读取保存的 logp、目标/decoy 和 SyncNet receipt，检查 2,820 个 CTC 候选 loss；最大 CTC 绝对误差 `8.88e-15`，统计重建最大误差 `6.22e-15`。冻结输入、帧/PTS、音频隔离和重复前向检查均通过。
- VSR 测量校准未完全通过：natural native 严格 top1 为 25/47=`0.532`，低于预注册的 `0.60` 门槛；TTS 为 30/47=`0.638`。两臂的 `Q>0` 和平均 Q 均为正，但这不能替代 strict top1 校准。因此 `calibration_status=INCONCLUSIVE_VSR_VALIDITY`，`visual_status=INCONCLUSIVE_VSR_VALIDITY`，`association_status=NOT_INTERPRETABLE`。
- 未经校准门控的探索性 VSR 数值：`G=Q(TTS)-Q(natural)` 均值 `+0.432`，95% bootstrap CI `[+0.169,+0.697]`，32/47 为正；`B` 均值 `+0.416`；同长度量 `Gmatched` 均值 `+0.117`，CI `[-0.124,+0.365]`。由于 natural strict top1 校准失败，不能把这些数值写成“嘴型更准确”的证据。
- 同批次重新解析的 Sync-C 差值为 `+1.068`，95% CI `[+0.873,+1.257]`，44/47 为正；`sync_status=EXPLORATORY_SYNC_GAIN`。这确认本 cohort 的 Sync-C 增益在新评分链上仍存在，但不说明增益来自视觉或音频。
- 主要关联 `Spearman(G, ΔC)=-0.179`，20,000 次置换 `p=0.228`，相关 bootstrap CI `[-0.425,+0.091]`，没有清楚的逐样本关联。2×2 表为 G>0/ΔC>0=`30`、G>0/ΔC≤0=`2`、G≤0/ΔC>0=`14`、两者均非正=`1`；独立基准下期望交集 `29.96`，所以 30/47 的同时为正不能单独说明两个指标有关联。

## Interpretation

本实验同时得到“Sync-C 在该 cohort 上稳定上升”和“VSR 结果不能通过预设校准门槛”。因此没有足够证据区分“Ditto 生成的嘴型更准确”与“SyncNet 更容易利用 TTS 配对音频”。它也没有证明嘴型没有改善；它只表明当前 CMLR VSR 视觉指标尚未提供可校准、可关联的独立视觉证据。结果仍受单说话人、固定历史脸和单一 VSR 模型限制，不能做因果分解或声称跨模型泛化。

## Artifacts

- `scripts/experiments/vsr_ditto50_linkage.py`
- `scripts/experiments/vsr_ditto50_metrics.py`
- `tests/experiments/test_vsr_ditto50_linkage.py`
- `runs/vsr_ditto50_linkage_v1/manifest.json`、`analysis.json`、`pair_rows.json`、`validation.json`、`report.md`、`linkage_scatter.png`
- `runs/vsr_ditto50_linkage_v1/features/`（470 个视图）和 `syncnet/`（94 个 score receipt）

## Changelog

| Date | Change |
|---|---|
| September 18, 2026 | 完成 smoke、Ditto-50 全量提取、同视频 SyncNet 重评分、关联分析和独立验证；工程 PASS，VSR 校准不足，未形成视觉机制结论。 |
| September 18, 2026 | 自审补充独立 2×2 统计复算与 `sync_status` 字段；测试 17 passed。 |
| September 18, 2026 | 修正 smoke 分支将 `sync_status` 标为 `SMOKE_TECHNICAL_ONLY`，避免小样本输出被误读为科学支持。 |
| September 18, 2026 | 自审补强缺失 VSR/SyncNet cell 检查，并在 validation 中写出 expected/completed/failed/source_excluded 数量；formal 与 smoke 仍 PASS。 |