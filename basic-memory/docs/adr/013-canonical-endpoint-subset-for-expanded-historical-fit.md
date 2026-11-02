---
title: 013-canonical-endpoint-subset-for-expanded-historical-fit
type: decision
permalink: tts-exp/docs/adr/013-canonical-endpoint-subset-for-expanded-historical-fit
git_sha: 0597b3dfbfd6bd5df93c96d7cb70be12ce476e59
project: tts-exp
decided: '2026-10-04T12:27:16.1285553+00:00'
status: open
repository: dogLeft233/tts-exp
tags:
- research
- protocol
- audio-enhancement
---

# 013-canonical-endpoint-subset-for-expanded-historical-fit

## Decision

为下一版局部频谱门控预先固定一个额外的历史拟合子集：采用原data-protocol.json中的前20位fit speaker（S0052–S0071），每位保留原16等间距规则的首尾槽位j=0与j=15，规划40个独立clip。先生成完整320槽位的目录/资格清单，再导出这个明确命名的端点子集；原50位fit池及16槽位规则继续作为权威全目录。此次决定只固定取样和预算，不选择V5参数，不改变已冻结shared540。

## Context and rationale

现有实际历史fit360使用8clip/4speaker/一个seed；其fit-only跨模型优选还须实际dev540验证。Wav2Lip代理训练的V4在完整三模型开发上均低于RMS，说明继续在少量声音及单一代理上增加训练步数不足以形成可迁移方法。用户已授权root维护理论和实验主线并持续迭代。扩展说话人覆盖与保持完全未读确认集相结合，可以检验下一版局部谱门控的稳定性。

元数据准备已核对三个speaker名单互不相交，但尚未扫描新的音频成员、验证40个合格clip确实存在。选择首尾canonical槽位是运行成本与同一说话人文本覆盖之间的预先折中；它不是随机总体样本，也不替代原320槽位目录。资格沿用mono16k、51200–128000样本、finite、speaker/archive一致及跨split重复PCM拒绝规则。缺失、重复或不合格槽位保留在规划分母，不根据得分替换。

## Selection and resource boundary

- 20位speaker沿冻结fit_speaker_ids顺序选择，禁止访问dev/confirm音频决定拟合成员。
- j=0/15在原16等间距eligible utterance规则上选取。若可用数不足或首尾重复，记录该槽位不合格；独立clip不足40时不能宣称达到40clip实验资格，也不临时换规则。
- 拟合统一使用一个共享seed20261003、三已部署前端及同一波形参数。最多6个新的固定候选/消融配置，identity与RMS为额外控制；完整一次最多40×3×8=960个原生行。实际配置仍须等shared540结果后冻结。
- 5fold按speaker分组，4位speaker/8个规划clip每fold；分配用sha256(UTF8("cross-tfg-v5-fit-fold-v1|speaker_id"))升序后依次循环分配0–4。候选在另外4fold选择并在余下fold描述性评估；同一个已生成视频可复用其冻结原始矩阵，不能为fold重复生成相同波形来扩大独立单位。
- 最终全部fit上的选择与fold稳定性分别报告；扩展fit仍是训练/选型证据，不能充当未见speaker确认。选择utility、Cbest原音端点、B/Dmin/D0/Canchor/offset分解沿ADR009与总spec；不额外引入未预设的零容忍门槛。
- 权重/配置最终冻结后才读取原80clip/40speaker确认及reserved模型。当前不新增确认读取、训练、ASR或生成。

## Alternatives and consequences

继续仅用旧8段声音成本最低，但说话人覆盖不足。一次使用完整320slot（6候选+2控制、3模型）最多7680行，保留为后续扩大拟合选项，当前先准备覆盖更多speaker的40clip子集。每位speaker的lexical-first2另成一个新规则，无法保持原16slot的覆盖方式；本次采用canonical首尾槽位。根据新分数或失败情况换音频会引入选择偏差，故保留原规划缺口。

该选择增加新声音源文件核验和三前端原生生成成本，并需要稳定存储。先完成可恢复旧输出归档与shared540实际开发结果，之后冻结V5配置并确认资源。源音频与原始分数、失败候选保持可恢复。参数、模型、种子与确认家族均不能从这个取样决定中推断为已冻结或已验证。

## Affected work and references

- `openspec/changes/develop-cross-tfg-audio-head-20261003/data-protocol.json`：4335B，SHA256cf615529edc8ae0b6ffff0a34246db71b3ec8ce5ac17d9f0daf2618ac624cf87。
- `runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt01/v5_fit_expansion_roster.json`：19455B，SHA2564c107253e979fa76b8158c33637100ceaf651e976a7799798165b96da37f0f6b；仅元数据准备，实际成员扫描待执行。
- `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md`与`transfer-iteration.md`。
- [Repository](https://github.com/dogLeft233/tts-exp)；HEAD0597b3dfbfd6bd5df93c96d7cb70be12ce476e59为本地未发布文档提交。

## Observations

- [decision] V5额外40clip历史fit子集采用前20位冻结fit speaker和每位canonical16slot的首尾j=0/15，保留完整320slot目录及资格缺口。
- [rationale] 扩展speaker覆盖并限制每轮最多6新候选的三前端成本，同时维持开发/确认分离。
- [alternative] 完整320slot拟合保留后续选项；lexical-first2与根据得分换样本不用于本子集。
- [consequence] 尚未验证40个合格源clip存在；V5配置与实际运行仍须后续独立冻结。

## Relations

- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]
- relates_to [[011-common-scale-mask-and-envelope-ablation]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]