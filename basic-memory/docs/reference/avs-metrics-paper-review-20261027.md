---
title: AVS_u、AVS_m、AVS_v原论文与本项目适用性核对 2026-10-27
type: reference
status: reviewed
verified: 2026-10-27
permalink: tts-exp/docs/reference/avs-metrics-paper-review-20261027
---

# AVS_u、AVS_m、AVS_v原论文与本项目适用性核对

判断：AVS_u值得作为本项目下一项独立验证指标；AVS_m/v需要真实参考口型视频，当前静态image3驱动的AISHELL自然/TTS配对实验不能直接照用。这里的“更合适”指任务匹配，尚未证明它在本项目中文/TTS域更可信。

## 原定义

论文为Yaman等CVPRW 2024《Audio-Visual Speech Representation Expert for Enhanced Talking Face Video Generation and Evaluation》。令E为AV-HuBERT transformer表示，V为生成口型，R为真实参考口型，A为音频：

| 指标 | 论文式4–6 | 依赖 |
|---|---|---|
| AVS_u | cos(E(V,0),E(0,A)) | 无真实视频参考；两模态分别编码，另一模态补零 |
| AVS_m | cos(E(V,A),E(R,A)) | 真实参考视频与同一音频 |
| AVS_v | cos(E(V,0),E(R,0)) | 真实参考视频；不读音频 |

它们共享唇读专家，不是三项相互独立的证据。[原文§4.1](https://arxiv.org/html/2405.04327v1#S4.SS1)

## 证据与边界

原文提取最后层T×768表示，使用唇读微调版AV-HuBERT；公式写序列余弦，但未在已核对正文中明确逐帧余弦平均、展平余弦或先池化的归约选择。不能把任意AV-HuBERT checkpoint加任意池化称为严格官方复现。[原文§3.4–3.5](https://arxiv.org/html/2405.04327v1#S3.SS4)

论文的鲁棒性图5测试脸在画面内的平移和旋转；**不是音轨提前/延迟**。实验评估LRS2、LRW、HDTF，不能据此认定中文TTS偏好问题已解决。[原文图5及§4](https://arxiv.org/html/2405.04327v1#S4)

用户研究只有10段HDTF视频、10位参与者；且AVS_u的TalkLip高于作者模型，人类同步评分却相反。因此有人类证据，但不能称所有指标排名与人完全一致，也未给出本项目所需的中文TTS配对验证。[原文表2–3及§4.4](https://arxiv.org/html/2405.04327v1#S4.SS4)

## 官方实现与权重可获得性

作者公开仓库[AVTFG](https://github.com/yamand16/AVTFG)的递归树SHA为`7e8cffa4804e14f03a839db239a79e5673935a6a`，本次API读取仅见空README、项目展示HTML/CSS/JS和媒体，没有评分Python、checkpoint文件或下载说明。展示页中的GitHub按钮仍是被注释的模板占位。因此当前只能确认论文公式，不能声称已取得三指标的官方可执行评分器。[当次固定树](https://github.com/yamand16/AVTFG/tree/7e8cffa4804e14f03a839db239a79e5673935a6a)

基础AV-HuBERT本身有[官方代码](https://github.com/facebookresearch/av_hubert)和[预训练/唇读微调权重表](https://facebookresearch.github.io/av_hubert/)。该表列有多种Base/Large、LRS3及VoxCeleb2(En)组合；并未替AVS论文唯一指认checkpoint。模型可获得不等于原评分协议完整可复现。

## 对本项目的推论与建议

以下为结合项目设计的推论，非论文已验证结论：

- AVS_u可以对自然音频驱动视频配自然音频、TTS驱动视频配TTS做原生评价，与现有Sync-C问题对应；它仍依赖音频特征，不能先验排除对某些TTS音色、清晰度或内容编码的偏好。
- AVS_m/v需要与目标发音时间结构对应的真实口型参考。AISHELL音频加任意静态image3没有这样的GT；将自然驱动的生成视频当GT，会使自然臂自比接近1，形成循环基准。TTS与自然音频时长、停顿不同，直接比也混入时间轴差异。DTW后再比测的是重对齐后的内容/口型相似性，不能代替原生同步评分。
- AVS_m两侧还共享音频，存在共同音频贡献掩盖视频差异的可能；这是需要消融验证的风险，非本次实证结果。AVS_v不含音频，因此不能对“仅换音轨”干预作出响应，这属于定义限制。
- 建议先实现并标注“论文定义复现版AVS_u”，冻结明确checkpoint、预处理与归约；先测试已知±80/160/400ms时移、局部错位、错内容音频，以及不改时间的EQ/响度干预。通过后再跑现有20来源原生配对，按来源统计，并用少量盲评核验冲突案例。不要因其同意或反对Sync-C就判定可靠。

本次未安装模型、下载权重、执行GPU评分或修改现有评分代码。