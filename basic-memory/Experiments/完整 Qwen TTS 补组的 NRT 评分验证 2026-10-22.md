---
title: 完整 Qwen TTS 补组的 NRT 评分验证 2026-10-22
type: note
permalink: tts-exp/experiments/完整-qwen-tts-补组的-nrt-评分验证-2026-10-22
status: concluded
scientific_status: NO_CONFIRMED_TTS_ADVANTAGE_SUPPORT_SENSITIVE_NOT_LM_OBJECTIVE_IDENTIFIED
execution_status: COMPLETE_WITH_TOKEN_CAP_AND_DISCLOSED_GPU_ORDER_DEVIATION
hypothesis: 完整文字条件TTS相对自然N及单独codec重建R是否出现更高自配Sync-C
model: Qwen3-TTS-12Hz-0.6B-Base full self-ICL FP32; frozen Wav2Lip GAN and SyncNet
  V2
inputs: parent qwen_codec_roundtrip_20261022_attempt01的10个历史AISHELL开发来源及固定转写
outputs: runs/qwen_full_tts_extension_20261022_attempt01; C盘完整副本
report: runs/qwen_full_tts_extension_20261022_attempt01/report.md
code_paths:
- scripts/experiments/qwen_full_tts_extension_20261022/prepare.py
- scripts/experiments/qwen_full_tts_extension_20261022/tts_stage.py
- scripts/experiments/qwen_full_tts_extension_20261022/pipeline.py
- scripts/experiments/qwen_full_tts_extension_20261022/independent_downstream_audit.py
tags:
- qwen
- tts
- codec
- syncnet
- no-training
---

# 完整 Qwen TTS 补组的 N/R/T 评分验证

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

用户授权复用刚完成的N/R，补完整TTS组，重点比较T−N、T−R，不训练。已执行10人×3预定种子的30条TTS、30个新增Wav2Lip视频、60份新SyncNet特征。原N/R音频、视频、特征直接复用；原2740文件的完整seal SHA未变。

## Material Passport

Origin Skill: academic-research-suite；Origin Mode: experiment-agent/run with numerical validation；Origin Date: 2026-10-22；Verification Status: EXECUTED_AND_NUMERICALLY_AUDITED；Version: qwen_full_tts_extension_v1。没有全流水线独立再次合成、视频重生成或人工内容/感知验证。Basic Memory MCP本轮不可用，本笔记经本地文件写入并读回，未声称MCP索引写入或验证。

## 主要结果

**本队列未确认稳定的完整TTS评分优势，尚不能把既有TTS优势定位到文字生成环节。**

| 支持规则 | N / R / T平均C | T−N，95%CI | T−R，95%CI |
|---|---|---|---|
| 五条件共同绝对q窗口，主要 | 6.263 / 6.310 / 6.417 | +0.154 [−0.171,+0.483] | +0.107 [−0.088,+0.279] |
| 各自全部真实有效窗口 | 6.272 / 6.256 / 6.178 | −0.094 [−0.418,+0.179] | −0.077 [−0.428,+0.217] |
| 相同窗口数量，均匀覆盖各自时间轴 | 6.281 / 6.246 / 6.174 | −0.107 [−0.430,+0.150] | −0.072 [−0.360,+0.178] |

两项主比较的97.5%分位bootstrap区间分别为T−N [−0.210,+0.522]、T−R [−0.113,+0.302]，对应预设两比较Bonferroni口径，均跨零。10人先各自平均三次T，再做10000次说话人bootstrap，seed20261022；30次生成与视频帧不算独立人。共同窗口下T−N有6人正/4人负，T−R有7正/3负。

取窗规则改变两项均值方向，三个种子的自配C变化均未建立稳定收益。覆盖整段下，T1/T2/T3的两项点估计都为负；不能把这一变化直接全部归给异常T3，也不能把负点估计写成TTS普遍有害。完整表、全部曲线、逐人/逐种子及辅助端点见正式report和analysis.json。

## 结构控制与边界

T是官方full self-ICL：目标与参考文字为同一固定语料转写去空白，参考音频为原同一句自然录音；文字条件生成新语音编码，再将参考编码与新编码拼接decode，最后按官方比例裁切。三个固定seed、不选最佳，temperature0.9/top_k50/top_p1/max_new_tokens512。新旧全部下游共同RMS0.0376838172675746、相同Wav2Lip权重/源码/脸图与SyncNet源PCM；无削波或时间拉伸。

- 10/10参考编码逐元素等于原R；另直接decode原R编码，10/10原始FLOAT逐元素复现。wrapper请求eager，但codec内部encoder/decoder实际config为sdpa；不能把请求参数误写为所有内部层实现。
- 30/30官方concat重复decode exact；独立CPU读回核验codes、完整FLOAT、官方裁切、seed/EOS通过。两条比例裁切少1样本，原样保留。
- R加入参考解码上下文后10/10波形改变，重叠MAE0.001001–0.004440；官方T与newcodes-only也30/30不同。**T−R仍同时包含新编码和解码上下文变化，并非单独文字LM或其学习目标的因果效应。** 未为上下文诊断额外生成TFG视频，不能从波形差断言评分效应。
- S0011 seed3达到生成长度上限，511新codec步、40.88秒、无EOS；其余29条有EOS。按协议全部保留、不重采样；输出没有人工内容认证，不能保证每条正确读完整句。
- 主TTS最后一条生成期间，codec数值桥接被错误地提前启动，约10.020秒的桥接日志窗口与其重叠。两进程exit0、无OOM/failure，参数/种子未变，数值控制通过；准确进程退出时刻未捕获。执行偏离完整保存在execution_deviation.json，不能声称严格串行或数值审计证明并发不影响随机生成。
- 共同有效q数17–125，S0006只有17个、低于预设25警示；未删除。均保证±15来自真实特征、不用边缘补零。旧N/R按原q精确复现，新主要分析按五条件共同q，故N/R汇总值不必等于旧报告。

Sync-C未建立稳定优势；Dmin、D0、B的变化也不能替代主端点或真实口型准确度。无上游训练重合排除、无独立视觉真值/盲评；10位历史开发来源、单固定脸、单TFG，不推广为所有TTS结论。

## 核验与产物

12项pytest通过；独立TTS读回PASS。另一实现从原embedding复算50组距离，NumPy/Torch最大误差3.8147e−6；150支持×条件端点和全部135组95%/97.5%bootstrap误差0。30canonical PCM精确、30视频绑定、60新特征SHA、2740原文件seal全部通过。审计状态PASS_WITH_DISCLOSED_EXECUTION_DEVIATION，errors=[]；这不是独立再次生成30条TTS或视频。

- protocol SHA `63c095d8e74488614f82dfb51cfb709ed4dd40671f9ce6a6adc2c771b9371724`。
- TTS manifest SHA `58cf39bfde3ce89a7c368b0b40dd3fe0bcf482c4148cbb10de0dc8df518ef71a`。
- 独立下游审计SHA `f3a39b357685a5e28e3f5d2e3a057647b863c0813b5699c5e79832cbb566eabb`；源码SHA `1979d96d9242698337cb91ed32003a748abac4c0bbe603ba078946c2c6b3391b`。
- 报告：`runs/qwen_full_tts_extension_20261022_attempt01/report.md`；图/全部指标CSV：`report_assets/`；原始计算与核验后状态分别保留在analysis.json与final_status.json。
- 远端run：`/home/wjj/tts-audio/tts-exp/runs/qwen_full_tts_extension_20261022_attempt01/`。
- 本地完整副本：`${USER_HOME}/.codex/tts-exp-artifacts/qwen-full-tts-extension-20261022/qwen_full_tts_extension_20261022_attempt01/`；封存与本地逐文件核验以artifact_manifest.json和local_readback_verification.json为准。parent完整副本已位于相邻qwen-codec-roundtrip-20261022目录。
- 本地完整读回PASS：4333个封存文件、186427012字节逐文件SHA/大小一致，144个JSON解析通过。artifact_manifest SHA `680399a642b6be7067e672a63ce949040275d1a6c8740ca9893b93007a27c3ba`；本地读回记录为额外侧记，不冒充封存时已有的文件。

## Observations

- [status] concluded：30条T与视频已执行，原N/R复用，全部数值审计通过并披露执行偏离。
- [result] 主要T−N +0.154、T−R +0.107，主比较区间均跨零；覆盖整段的两种预定规则点估计为负，结果对支持敏感。
- [limitation] 一个无EOS上限输出、一个低共同支持来源、一次GPU顺序偏离；均保留，不能升级为确认性收益或文字LM机制证据。
- [decision] 本次到零训练验证与报告为止；先稳定复现完整TTS优势，再讨论模块归因或训练方案，不据当前局部正均值启动方法训练。

## Relations

- qualifies [[Qwen 自身语音编解码的时间保持与生成评分拆分 2026-10-22]]
- advances [[研究交接总览]]
- relates_to [[电平鲁棒零训练筛查资产核查 2026-10-22]]

## Changelog

- 2026-10-22：完成授权完整TTS补组、全部预定种子/支持规则、独立数值复核；保留token上限异常和GPU编排偏离，没有训练。
