---
title: 040-synctalk-recording-role-fallback
type: decision
permalink: tts-exp/docs/adr/040-synctalk-recording-role-fallback
status: open
project: tts-exp
decided: 2026-10-17 12:34:07.846728+00:00
---

# 040-synctalk-recording-role-fallback

## Decision
在本轮TFG更新仍0、生成视频仍0时，只有原录制配对未通过已冻结的预处理资格，才允许将这两段录制反转用途一次，保持四身份和原预算。已经通过的配对不重排；反转仍失败则保留两次失败并停止该候选，不扫描片段或择优SyncNet。每次尝试独立目录和data version，原失败证据不可覆盖。

## Context and rationale
Debbie原main=录制0/test=录制1的全量v2测试flow median4.48346127047664/p956.69030642173168px，超过冻结2/5；val与训练恢复控制通过。独立NumPy投影复现，尚未确认根因。完整61成员失败包已F逐成员SHA验证，归档SHA796861f6aaedeabcf168c9d41a262cd22857093b7b2e7a86e47d657518f72734。继续相同不合格数据不能满足用户的先核验要求。

原录制1实际290秒，原0实际310秒，来源SHA未变，已有材料只保留310+110。候选改录制1为main290秒、录制0为test130秒，仍420秒和两独立原录制。全部材料时钟/RGB/PCM/上下文/身份/stage1/全量几何资格须重做；不是交换已拟合的姿态标签。训练6000、val750、test1500与六个10秒位置、整录制音频上下文不变。反转是在不同训练域重新估计原训练形状/BA，再以同一冻结v2参考方法资格heldout，不改变v2数学或阈值。

## Alternatives
保留不合格配对直接训练违反准入；放宽阈值或改几何实现会扩大方法自由度；立即换人增加身份选择变动。单次反转以最少新增素材保持原四身份，但不能保证通过。不能按分数选反转方向。

## Consequences
这属于观察到几何资格失败后的数据适配，最终报告须披露原失败与所有尝试，结论仅适用于通过冻结几何资格的保留录制配对，不声称随机选择录制或对不合格域泛化。旧source-owned评分参考只用于旧数据诊断；候选test变更后必须重建全部评分参考/音频forward/独立QA，禁止复用旧分数。

F dataset_revisions/DebbieDingell_recording_roles_v2/revision_contract.json与selection.json在任何候选准备前封存；ROOT对应独立目录。尚未准入训练，所有4×2×3 FP16/65536ray/完整102000attempts与实际更新控制不变。

## Observations
- [decision] 冻结门失败的录制配对允许一次独立版本化角色反转，取首个合格方向，禁止SyncNet择优或删帧。
- [limitation] 几何资格后适配限制总体外推，所有失败与版本须披露；当前没有TFG处理效应。

## Relations
- relates_to [[039-synctalk-shared-training-flow-geometry]]
- affects [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- affects [[准备并核验四个新身份的 SyncTalk 完整训练数据]]