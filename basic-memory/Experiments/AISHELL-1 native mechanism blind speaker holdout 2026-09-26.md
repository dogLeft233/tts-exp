---
title: AISHELL-1 native mechanism blind speaker holdout 2026-09-26
type: experiment
permalink: tts-exp/experiments/aishell-1-native-mechanism-blind-speaker-holdout-2026-09-26
---

# AISHELL-1 native mechanism blind speaker holdout 2026-09-26

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 冻结盲选规则并开始历史输入排查 | September 26, 2026 | user |
| 完成 40×2 原始数据盲选及独立完整性验收 | September 26, 2026 | user |

## Observations

- [status] concluded
- [result] 完成 S0002–S0041 共40位×2句=80条，全部原始16kHz mono PCM_16，单句3.0359375–7.919秒，总长376.825375秒。40个speaker包下载1,512,686,963 bytes，人工转写10,091,431 bytes。发布者AISHELL/AISHELL-1 revision=bbe295d530192a4cd41644b711c9aecd087df653，所有archive通过发布者LFS SHA256校验。
- [conclusion] 40位在可核验历史实验输入记录中未出现；16位历史排除集覆盖旧n100及400-pair，包含S0770。已核对1,142份现存输入/条件音频的1,126个精确PCM指纹，80条零历史/内部重复。90个旧原始路径缺失且旧数值别名未完全恢复说话人映射，因此不声称绝对从未使用；完整语料400位名单不被当成实验暴露。
- [validation] 独立重算首两句规则、原archive字节、人工转写、时长与PCM；validation.json passed，2项契约测试通过。9条因时长排除，未发生说话人递补；不生成TTS/TFG或任何评分。
- [manifest] runs/tts_native_holdout_inventory_20260926/manifest.json；原音频 data/native_mechanism_holdout_20260926/audio/。数据准备已完成，机制确认协议及实验由主agent另行冻结。
- [protocol] 目标为40位在可核验历史实验输入未出现的AISHELL-1说话人，每人2句；按canonical Sxxxx字典序选择说话人，按原utterance ID字典序选择首2条3–8秒、有非空官方人工转写、可解码finite PCM的语音；不使用ASR、TTS、TFG或同步评分筛选。
- [history] 排查全部可核验项目输入及说话人映射；完整语料转写目录的出现不视为历史实验使用。核对可用历史实际输入PCM hash以排除别名重复。
- [report] runs/tts_native_holdout_inventory_20260926/report.md
- [data] data/native_mechanism_holdout_20260926/

## Relations

- relates_to [[tts-exp]]