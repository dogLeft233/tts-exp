---
title: 041-synctalk-eligible-recording-fallback
type: decision
permalink: tts-exp/docs/adr/041-synctalk-eligible-recording-fallback
status: open
project: tts-exp
decided: 2026-10-17 13:06:34.951827+00:00
---

# 041-synctalk-eligible-recording-fallback

## Decision
仅当原录制配对不合格且ADR040角色反转因长度无法满足冻结train/val/上下文时，允许一次下一合格独立原录制作为测试。候选按此前已下载、同官方身份标签、元数据录制编号顺序选首个长度合格且非训练原录制者，不按SyncNet或TFG结果排序，不扫描片段。失败保留，不能连续搜索最佳分数。

## Context and rationale
Cathy原main录制1/test录制0的全量冻结v2测试flow median2.372264813512586px超过2px；p953.936536358146548px通过5px，仍整体失败。原train controls/val通过，完整60文件26443101原B失败包已独立F封存，SHA624a46a0f1135e819b7238f1c8282f4f004bdea40b37b8d7b481de475227b9e5。原test视频实长243秒，不能作为需≥278秒的main，不能缩监督/拼不同录制或放宽门以作角色反转。

已下载的下一独立原录制2实长110秒；官方WRA_video_url.txt映射21mNN49Pp34，WRA_annotion_time.txt为00:05–01:55，与训练yDi_pmbanwc不同。下载SHA核验，五均匀帧只作单人清晰自然讲话/场景检查，身份标签来源于官方元数据；完整重复/人脸/音画资格仍待独立运行。原main295秒不变，新test110秒，总405秒=6分45秒，满足用户6–8分钟与原spec360–480秒；不把目标420秒冒称实有420。

## Alternatives and consequences
不做243秒不足长度的强行反转，不缩6000/750/1500与六固定10秒，不因flow略超限改2px阈值，不把同一原录制不同文件当独立test。直接换身份会扩大cohort变化；保留原四身份而只换不合格测试录制更小。原训练JPEG/FAN/parser/BFM/BA/flow可在原始来源、主录制RGB/PCM与每项SHA精确相同后复用，不重复估计训练状态以引入额外自由度。新test全部重新预处理，冻结v2所有训练恢复/全750val/1500test、6条dense重复与点不变控制仍实际重算；loader/rays/完整资源与评分参考须重新独立验收。

此选择在观察到几何资格失败后、仍0新TFG更新/视频时作出，不能追称原始预注册。最终报告披露原失败、长度排除与唯一新候选；结论条件于合格录制配对，SyncNet不能直接解释为感知改善。405秒原料与其他人420不同，但实际train240/val30/test60监督、24模型、两个seed、初始化、65536rays、FP16/原optimizer/scaler/EMA、60000/102000尝试预算不变。

## Affected artifacts
F dataset_revisions/CathyMcMorrisRodgers_recording2_v2/selection.json、revision_contract.json在新候选准备前固定。原选择/失败数据与F原材料保留。新test需新source-owned ROI与三评价音频forward/独立18矩阵参考，不能继承旧test评分。

## Observations
- [decision] 角色反转长度不合格时只允许按官方元数据顺序一个新独立test候选，保持原训练录制和全部监督/训练预算。
- [limitation] Cathy候选实际405秒原料，尚未全门通过；资格后适配须披露且限制外推，0新TFG处理效应。

## Relations
- refines [[040-synctalk-recording-role-fallback]]
- relates_to [[039-synctalk-shared-training-flow-geometry]]
- affects [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- affects [[准备并核验四个新身份的 SyncTalk 完整训练数据]]