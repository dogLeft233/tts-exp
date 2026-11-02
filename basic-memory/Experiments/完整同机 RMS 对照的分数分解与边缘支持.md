---
title: 完整同机 RMS 对照的分数分解与边缘支持
type: experiment
permalink: tts-exp/experiments/完整同机-rms-对照的分数分解与边缘支持
project: tts-exp
repository: dogLeft233/tts-exp
status: concluded
scientific_status: complete_controls_rms_score_margin_and_support_decomposition
execution_phase: actual360_terminal_and_independent_100000_cluster_recompute_pass
git_sha: 0d3d675ce397718506801a984ad31408759464c7
scope: Same-host Dev20 clips, 10 speakers, 3 seeds, 3 TFG models, 360 controls and
  180 paired contrasts; descriptive post-experiment analysis
---

# 完整同机 RMS 对照的分数分解与边缘支持

## 问题与设计

本实验描述同机自然音 identity 与固定 RMS 驱动、最终恢复完整自然 F32 PCM 后的 SyncNet 响应，为局部声学增强头提供正确的比较基线。360 条对照全部 PASS；20 clips/10 speakers，每人两条，3 seeds，Wav2Lip/MuseTalk/Ditto，各模型方法单元60条。180 个配对全部保留，配对键为 split/sample/speaker/model/face/seed。

使用已经冻结并产出的 full 支持与31-lag距离曲线，不重跑评分。先在说话人内平均六条配对，再平均10说话人；总体目标对三模型等权平均。分析协议在本次复算前锁定，但属于实验完成后的描述分析。100,000 次 speaker-cluster bootstrap，seed20261003，同一抽样保留每位说话人的全部配对。对3模型及等模型目标×7指标构造28项联合99%区间；固定 bootstrap SD 标准化的最大绝对中心偏差，nearest-rank临界值3.4416474004008264。只有10簇，区间是近似描述，不作为新候选选择规则。

## 实际结果

| 模型 | RMS−identity ΔSync-C | 联合99%区间 |
| --- | ---: | --- |
| Wav2Lip | +0.238 | [−0.051,+0.527] |
| MuseTalk | +0.053 | [−0.006,+0.111] |
| Ditto | +0.165 | [−0.041,+0.372] |
| 三模型等权 | +0.152 | [+0.002,+0.302] |

三模型点估计均正，逐模型联合区间均跨零。等模型的 +0.152 是本次实际数据的均值；不能用历史其他 GPU 的 +0.154 替换。

| 等模型指标 | RMS−identity均差 | 联合99%区间 |
| --- | ---: | --- |
| Cbest | +0.152 | [+0.002,+0.302] |
| B（31-lag曲线中位距离） | +0.155 | [+0.025,+0.286] |
| Dmin | +0.003 | [−0.055,+0.062] |
| D0（固定零延迟距离） | +0.052 | [−0.016,+0.120] |
| Canchor | +0.103 | [−0.027,+0.233] |
| signed best-lag frames | −0.011 | [−0.125,+0.103] |
| absolute best-lag frames | −0.056 | [−0.153,+0.042] |

Cbest=B−Dmin。RMS 的 Sync-C 增益主要由 B 上升构成，Dmin 点估计反而略升；D0 点估计也上升，区间跨零。结论是本批生成输出与原音的 SyncNet 分离度提高，并没有固定零延迟距离改善的结果。它为直接优化局部时间对齐与编码器方向提供依据，不等同人类同步质量的改善。

## 边缘支持

full 180/180 配对合格。GUARD20 171/180 合格，9 个失配配对全部是 BAC009S0022W0123，覆盖3seeds×3模型：Ditto/MuseTalk full query75，guard query35；Wav2Lip full query74，guard query34，低于规定41。保留这个样本及其说话人在完整分母内。当前冻结增强头 Dev 仍继续完成；另行建立仅按元数据筛选、时长满足边缘支持的新验证批次，不更改 GUARD20、不以九说话人子集替代当前主端点。

## 复核与来源

完整ledger 11,940,846字节，SHA `0ec1d6043f8e3c8ebce4dfdd17c95418239c663bdf8d24a47cdcbfde0e793cb2`。独立终端QC `0733a4c0...6c09c` 核对360行及14源输入。分析目录 `.codex/scratch/dev360_controls360_rms_diagnostic_attempt01/` 保存6218字节协议、22622字节分析源码、149096字节effects表及自哈希manifest。

ROOT在独立实现中从原ledger重新配对、汇总、抽样并复算全部28项区间，receipt `.codex/scratch/root_controls360_rms_numeric_qc_attempt02.json` 为 PASS_ROOT_INDEPENDENT_NUMERIC_RECOMPUTE。720个Cbest/Canchor恒等式与原始F32舍入逐项相符，最大双精度重算残差4.76837158203125e−7；先前以双精度严格相等检查的失败保留为attempt01，并没有改动原数据。没有GPU、API、SSH或实验重放。

## Observations
- [result] 同机完整360对照中，等模型RMS相对identity ΔSync-C=+0.152，近似联合99%[+0.002,+0.302]，全部10说话人和180配对保留。
- [result] 分解显示B+0.155、Dmin+0.003、D0+0.052；Sync-C主要来自距离曲线分离度变化，固定零延迟距离没有改善的实测结果。
- [decision] 局部增强头必须相对当前RMS对照验收，并同时检查D0/编码器方向、内容与独立视觉；冻结step120候选与确认集不改动。
- [decision] 原GUARD20短片段9配对失配保留完整分母；新增时长合格批次须仅按元数据选取并独立冻结。
- [next_step] 完成冻结180候选实际生成与原音换轨端点后，用当前360对照接续540联合内容/视觉/边缘支持验证，并推动更直接的局部时间对齐方法。

## Relations
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]