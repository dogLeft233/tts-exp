---
title: 004-canonical-float32-score-field
type: decision
permalink: tts-exp/docs/adr/004-canonical-float32-score-field
repository: dogLeft233/tts-exp
implementation_status: complete_distance_audit_full_11776_reconstruction_restore_and_figures_pass
decided: '2026-10-02T14:27:32.8077445+08:00'
project: tts-exp
status: open
tags:
- decision
- tts
- tfg
- numerics
- reproducibility
---

# 004-canonical-float32-score-field

## Decision and current implementation

独立embedding重建完整距离场并通过原CROSS_ATOL/RTOL=1e−4后，观测、100完整A时间置换和局部anchor贡献共同使用冻结producer float32距离场。saved cross/zero→saved lag逐值精确gather；serialized曲线/固定anchor/offset/派生metric仍按原METRIC_ATOL=2e−6核验。该选择在mu/rho效应读取前作出，保持物理波形、800模型视频、1600评分、估计量和科学freeze。

实施已完成：consumer19/namespace16实际退出0、family11完整645输出/11776规划端点独立PASS，完整712文件恢复、160phone map镜像以及ROOT recon11全部局部/联合统计复算PASS。原先的17/14 int(None)、18/15 metadata层级错误和13数值失败保留；它们是执行修复出处，当前实现状态由完整实际结果确定。status保持open，表示仍适用的决策。

## Context and rationale

Producer使用CUDA pairwise_distance float32累加；旧consumer用NumPy float32平方和。两者独立full-cross比较合格，但旧reader把NumPy场派生曲线与CUDA场派生曲线按更严格metric门比较，又将NumPy cross传入null/local，混合了两个浮点场。固定anchor、31列lag、full-A索引和support均值轴定义一致。

原运算由冻结float32场确定；独立重建验证该场，后续曲线/null/bootstrap由同一个合格原场重算。这样保存评分定义，并让观测与null采用相同浮点值/边界规则。

## Alternatives and consequences

不提高METRIC_ATOL，不改float64求均值，不重跑模型来处理读取累加差异。继续保存全3200 cross/zero/lag及全部nullable metadata独立审计、11776端点、共享100置换和20000×40联合99%族；短guard20按原40分母保留INCOMPLETE。新源码/独立family/ROOTreconstruction各有fresh pins，原失败完整保留。

## Verification and authoritative pointers

全1600评分/3200几何cross/32400480元素原1e−4门内0miss，max1.9073486e−6/2ULP；zero298580值max9.5367432e−7/2ULP。3200 savedlag与producerloop/consumergather逐位一致，12780非空组savedlag/scalar残差0，20empty保留。NumPy场派生curve370/396180值超原2e−6/max4.7683716e−6，故未把它换作新评分定义。

输入930028745B/160context/1600score NPZ全量SHA、ID、float32、finite、shape验证，432nullable项与160phonepipeline完整重建。数值诊断 `.codex/scratch/world_phone_shape_mean_dynamic_consumer_attempt13/lag_numerical_parity_audit_attempt01.json`；授权 `world_phone_shape_mean_dynamic_consumer_attempt14/root_canonical_field_release_readback_attempt01.json` 保留。

完整输出 clean目录 `runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_consumer_attempt16_full_output_restore_attempt01/consumer_output/`；manifest raw7c65c8e58ff73ee3747789f19731f6a3d76a39960c478fb872510bc4dd7439da，11776JSONL rawc47cb2e0fb5b6eb014a54ae9b13fc86774cf8ce983753495e8a020e5a819de1c。ROOT本机recon11比较665600局部值max4.44e−16并重建共同bootstrap/CI，receipt `.codex/scratch/world_phone_shape_mean_dynamic_local_finalize_attempt02/root_reconstruction_receipt.json` rawdea04cb3e2479605681f97f72b3a7319296b5832d3670334d1df9054c1ed79c4。图表全族读回与ROOT审阅PASS。

## Observations

- [decision] 独立embedding距离核验后，观测/null/local共用冻结producer float32场；strict gather与原数值门保持。
- [rationale] 合格原场保持CUDA float32评分定义，独立NumPy重建负责核验，严格派生量由同一场重算。
- [alternative] 不放宽容差、不改float64定义、不重跑模型。
- [result] 完整11776族、恢复和独立ROOT重建已实际PASS，数值决策完整实施。

## Relations

- affects [[音素平均谱形与内部轨迹的因果分解]]
- affects [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[003-immutable-execution-repair-provenance]]
- relates_to [[25-tts-tfg-mechanism]]