---
title: local-clock-crossover-20261016
type: note
permalink: tts-exp/codex/tts-exp/local-clock-crossover-20261016
status: done
phase: concluded_bounded_pilot
current_step: 4
priority: medium
project: tts-exp
date: '2026-10-16'
head_research: paused
tags:
- local-clock
- repeated-takes
- controlled-pilot
---

# local-clock-crossover-20261016

## Objective

执行用户授权的固定视频局部时钟小实验，先4来源及已知错位/身份/endpoint-affine sham，控制有效才补固定后4人；不恢复跨TFG增强头。完整冻结设计见 [[固定视频的局部时钟错位正控与交叉校正 SPEC 2026-10-16]]。

## Current State

已完成，正式结论 [[45-local-clock-crossover]]。首4来源S0003/S0006/S0007/S0008，控制28forward经ROOT实际QA达到3/4；主比较8forward+浮点上界遗漏补充2forward，共5/24合格配对（来自2来源），Sync-C全部提高+1.770～+5.495，实际MFA中心MAE从241.8～594.7ms降至7.7～16.3ms。其余19对真实局部伸缩越界，完整类内/跨类来源覆盖0/4，按预设规则停在4人，不扩8、不恢复增强头。

总38audio forwards，0新TFG/TTS/ASR/安装/tests；实际original/status/拒绝/缺失均保存，S0008正控R grid因ENOSPC缺失不计控制通过。原RUNNING receipt保持历史原件，所有记录DSP/MFA CLI/scorer PID已针对/proc确认不存在。source-only及rootQA/完整计数、单次数值边界精度修复说明见报告45。

权威路径：`runs/local_clock_crossover_20261016_attempt02/provenance/score_recovery/`；`runs/local_clock_crossover_20261016_main_attempt01/`；`runs/local_clock_crossover_20261016_boundary_completion_attempt01/`。ROOT控制QA381bd121…；主/补充30数组及实际MFA QA98e78a93…，指标/clock重建误差0。24行派生合表与图在main run的analysis_delivery_attempt01，包含5合格/19未执行，不填完整队列均值/CI。可执行源码5文件已本地提交 `c6c1e7bd005abeaa4a2cc52991391ef2aaec5223`；未push/PR、旧dirty保持。

本任务MFA缓存390文件/963886811B仅移到SHM并核对，科学原件未改；SHM易失，实际科学subset已本地读回，不称云备份。下一机制候选为原4来源上的受速率约束平滑/软时钟映射，尚未执行；当前无待运行实验。
## Requirements

### Requirement: 以正控资格决定有界局部时钟实验的终局

SHALL 保存首四来源真实波形/MFA/评分原数组、控制门、条件扩容决定与最终解释，并维护研究地图。

#### Scenario: 控制通过

- **WHEN** 首四来源至少3人的冻结正控/身份门通过
- **THEN** 执行首批主比较；仅实际clock/coverage门满足才扩到固定后4人，最多8人，保存全表及原数组，不按主分数方向选择。

#### Scenario: 控制不通过

- **WHEN** 实际正控或主clock扩容门未通过
- **THEN** 保留失败输出，停止本小方案扩容，报告失败具体性质，不把控制失败写成主机制阴性或恢复增强头搜索。

#### Scenario: 终局归档

- **WHEN** 达到8人或门失败的规定停止点
- **THEN** 一次ROOT实际QA后生成正式报告45、更新地图/Task/HANDOFF与CONTEXT<=100行，全部运行退出，阶段成本与解释边界可读回。

## Observations

- [status] done
- [result] 3/4正控通过；5/24合格配对均有C增益+1.770～+5.495，19真实映射越界、来源完整覆盖0/4，未扩8。
- [verification] ROOT实际WAV/114新数组/输出MFA数学通过；指标与clock误差0，所有记录实验PID已退出。
- [close_date] October 16, 2026
- [decision] 当前固定视频评价端clock试验收口；增强头继续暂停，后续soft-clock另立小方案。
## Relations

- completed [[45-local-clock-crossover]]
- project [[固定视频的局部时钟错位正控与交叉校正 SPEC 2026-10-16]]
- follows [[repeated-take-crossover-20261015]]
- relates_to [[研究交接总览]]