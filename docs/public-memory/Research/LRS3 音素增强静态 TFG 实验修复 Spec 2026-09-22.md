---
title: LRS3 音素增强静态 TFG 实验修复 Spec 2026-09-22
type: research_topic
permalink: tts-exp/research/lrs3-音素增强静态-tfg-实验修复-spec-2026-09-22
status: planned
spec_status: ready_for_implementation
question: 纠正生成和测量错误后，冻结DIRECT及MFA条件增强是否有音素和TFG收益
protocols:
- phone_gain_static_tfg_mfa_remeasure_v2
- phone_gain_static_tfg_mfa_repair_v2
tags:
- lrs3
- tfg
- syncnet
- mfa
- repair-spec
- implementation-architect
---

# LRS3 音素增强静态 TFG 实验修复 Spec 2026-09-22

修复合同全文：`openspec/changes/repair-phone-gain-static-tfg-mfa/spec.md`。使用 implementation-architect 对原spec、实现、产物和CPU反例逐条审查，给出F01–F17问题、具体代码锚点、两条执行路径与验收矩阵。本次仅编写spec，未修改实验代码或启动GPU重跑。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 审查并制定修复合同，区分固定音频重测与原协议补齐 | September 22, 2026 | user request；agent design |

## Observations

- [status] planned；ready_for_implementation。
- [question] 正确生成/评分后，规则DIRECT是否提高自然音轨上的TFG同步；是否达到TTS级可分度；MFA训练与推理条件是否有收益？
- [finding] `_portrait_meta`使用整幅图作为Wav2Lip generation_box及fixed score_box，违反原静态检测框要求。单样本同音频同官方评价器只改生成框，Sync-C 0.521→7.444，Sync-D 11.246→6.921。此为诊断，不能外推完整40句效应。
- [finding] natural_primary错误包含TTS并混用N时间；无实际frame_indices冻结；official/direct弱缓存可能复用旧分数；checker未独立复算完整性。
- [finding] DEV选型65句代替pilot24、跨seed择优；FIT normalization/taper/SIL冻结及训练resume合同偏差；DIRECT质量漏评、ASR硬编码blocked、MFA edit rate漏插入；多项原定校准/敏感性/ABX/消融未接入。
- [verification] CPU反例复现：仅N/N可被标COMPLETE；两个缺失结果被解释NEITHER_CLEARLY_UP；零残差step0在同分时输给step25；[a,b]→[a,x,b] edit_rate=0。旧7项pytest仍全部通过。
- [decision] A=frozen_audio_remeasurement：保留OLD六臂PCM，纠正几何/校准/评分/统计；旧A/B/C标LEGACY_SELECTION。B=protocol_repair：修复原合同并重新训练、按seed选型及补齐全分支；各有独立run/protocol。
- [boundary] OLD所有TFG阴性推论对预期正确生成链路无效，保留原分数作错误配置诊断；修复前不能断言规则音频无TFG增强。未发现实际视频泄漏，现有访问隔离验收不足。
- [requirement] 固定E_SEEN曝光说明；官方端到端与固定窗并列，不强求绝对C一致；不能要求所有样本C≥5；GPU阶段先检查占用和资源，缺失资源走已授权7890代理。
- [artifact] `openspec/changes/repair-phone-gain-static-tfg-mfa/spec.md` 是完整实施与验收合同；OLD=`runs/phone_gain_static_tfg_mfa_lrs3_phone_gain_static_tfg_mfa_v1_20260922b`。

## Relations

- repairs [[LRS3 音素增强静态肖像 TFG 与 MFA 条件实验 2026-09-22]]
- extends [[LRS3 音素增强的静态肖像 TFG 验证与 MFA 条件增强 Implementation Spec]]
