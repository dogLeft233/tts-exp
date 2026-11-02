---
title: 自然音频与双规模TTS的SyncNet偏好探针 2026-10-24
type: note
permalink: tts-exp/experiments/自然音频与双规模-tts-的-sync-net-偏好探针-2026-10-24
status: concluded
scientific_status: EXPLORATORY_PROBE_NO_INCREMENTAL_VALUE_PARENT_REPLAY_PASS
generator_training: false
analysis_model_training: completed-small-ridge-heads
executor_model: gpt-6.1-sol
executor_reasoning: medium
tags:
- syncnet
- preference
- probe
- natural-audio
- qwen
- exploratory
---

# 自然音频与双规模TTS的SyncNet偏好探针 2026-10-24

用户明确要求加入自然音频，因个人难判断归一化后的视听优劣，先解释SyncNet排序；指定GPT-6.1-sol medium子代理执行。/root/syncnet_preference_probe已完成冻结HuBERT提取与小型Ridge奖励探针；父侧独立闭式拟合与指标重算通过。没有训练TTS/TFG、新合成、视频生成、新SyncNet前向、SAE或人评。

**本轮未通过复杂解释模型推进门。** 共同HuBERT表示没有超过原声学/流程基线；资格敏感性下，声学模型的排序正确率也几乎不超过训练折多数标签规则，尚未建立稳定可迁移的偏好解释。

## 权威资产与冻结设计

本机 .、Ubuntu /home/wjj/tts-audio/tts-exp，同relative run `runs/syncnet_preference_probe_20261024_attempt01/`。report.md、parent_validation.md、input_manifest.json、full/qualified_results.json、full/qualified_oof.json、folds/、majority_control.json、parent_independent_audit.json。115文件9,598,753字节已在本机逐SHA运输核验，缺失0/错配0；final_manifest.json不含自身。不声称网盘归档。

protocol.json SHA a03bf68967cbd3029cd74c3197667310e27536c2cf032838e88ddb08b83768ea，提取和拟合前冻结。140canonical norm WAV=N20+L06/L17各60；20来源、20speaker、20不同去空格文本；首track全部唯一。N是自然N不是codec R。RMS目标0.0376838172675746；实测偏差最大6.89037e-7；mono16k PCM16，原全长不截尾。由原float32的31lag距离恢复C=median(mean distance)-min(mean distance)，与3位打印值最大误差.0004893265。

每source9主pairs：N-L06三draw、N-L17三draw、L17-L06同draw三次；另两TTS各3个within-draw对，合计300pairs/180主pairs。N复用不产生独立样本。共5family，每source/family等权。连续分差为主，符号为辅助。共同资格去S0011/S0016两模型同d3、N和20来源保留，再完整拟合286pairs/174主pairs。

M0=arm身份；M1=M0+既有6声学特征+log1p官方支持行数；M2=M0+HuBERT PCA8+log时长/支持；M3=M1+HuBERT PCA8，预指定主比较M3对M1。C/D/B/offset/距离摘要禁止作输入；支持行数为明确的流程混杂基线。HuBERT base-ls960 snapshot dba3bb02fda4248b6e082697eee756de8fe8aa8a共同冻结final layer，FP32/eval/no_grad、真实有效帧mean/std1536维，无层/模型搜索；处理器自带音频归一化已记录。外层每次留一speaker/text组，所有draw同折；inner GroupKFold5每折重新fitunique训练clip的scaler/PCA8，full SVD，Ridge alpha仅1/10/100，无截距r(A)-r(B)。

## 结果

| 模型 | 全量主MAE | 全量排序准确率 | 共同资格主MAE | 共同资格排序准确率 |
|---|---:|---:|---:|---:|
| M0 | 0.710 | 51.111% | 0.628 | 59.167% |
| M1 | 0.581 | 65.556% | 0.586 | 61.111% |
| M2 | 0.600 | 61.667% | 0.627 | 58.333% |
| M3 | 0.598 | 65.556% | 0.617 | 63.889% |

全量M1−M3 MAE改善−0.01691966，source bootstrap10000/seed20261024描述性95%[−0.04770504,+0.01479378]；old/new分别−0.027662/−0.006178；资格重拟合−0.03061684，[−0.06583202,+0.00323864]。绝对分差>=.1的排序正确率M1/M3为67.778%/66.111%。预定门（MAE+.05、准确率+5pp、两批不负、资格方向保持）四项均不满足，停止复杂RM/SAE，不扩层/模型搜索。

全量M1三主family准确率：N-L06 60.000%，N-L17 73.333%，L17-L06 63.333%；资格后54.167%、72.500%、56.667%。M3分别63.333%、68.333%、65.000%；资格后64.167%、67.500%、60.000%。不同辅助端点有变化，不替代预设主MAE结论。within比较结果全部保留文件。

父侧为避免将连续均值拟合M0当最强分类基线，**事后补充**每outer训练集family多数符号比较器（不替换M0或更改gate），子代理另独立复算一致。全量60.556%，共同资格60.833%；M1资格只高0.278个百分点。这削弱把全量65.556%视为稳定规律的说法，不能仅与50%比。

## 独立核验与执行边界

父侧脚本 `scripts/experiments/syncnet_preference_parent_audit_20261024.py` 不导入runner；140音频/视频/score/activesd四类SHA与原float32数组核验。40外层×4模型160次保存结果从训练数据与数值处理器闭式独立重新求解，maxcoef误差2.524e-7、OOF误差1.888e-7，主指标/阈值敏感性/source bootstrap误差0。全部分组/配对target/资格数量检查通过。未独立重复HuBERT前向或内层alpha选择，这两部分为源码/记录审查。3个pytest兼容测试直接断言调用PASS，环境没有pytest，不称pytest执行。父侧看图通过，parent_validation.md覆盖11/11统计谬误检查。

原audit/extract成功；analyze已保存完整科学结果后因matplotlib缺失包装exit1。失败receipt和source快照保留，report_attempt02用已有PIL修复报告exit0，不重新提取/拟合，科学SHA不变。代码 `scripts/experiments/syncnet_preference_probe_20261024.py` 及相应tests、独立report修复脚本均新增，不改原实验。

## 结论边界

标签为native各自音视频的官方SyncNet排序；audio-only probe针对现有固定TFG链，不隔离评价器纯偏好与生成响应，不证明人类质量。20来源先前均已看过，只是探索性组外预测，不是新确认集；共享训练折bootstrap仅描述性。HuBERT池化/PCA不是音素概念，负结果只限制此固定表示和线性探针的增量价值，不能证明其他时间/视听结构中不存在机制。当前不扩大模型训练，未启动人评或因果干预。

## Observations
- [result] 自然N/0.6B/1.7B联合140音频的小型偏好探针完成，原始评分/160模型独立重算PASS。
- [result] HuBERT增量未过门；M3 MAE全量.598、资格.617，均比M1差。
- [limitation] M1资格排序61.111%仅略高多数规则60.833%，不能称已学到稳定的SyncNet偏好机制。
- [decision] 按冻结推进门停止复杂偏好模型/SAE扩展；不由此否定所有可解释机制。

## Relations
- follows [[偏好模型解释TTS同步分差的定向文献检索 2026-10-24]]
- follows [[Qwen双规模实验独立复核与来源差异诊断 2026-10-24]]
- informs [[研究交接总览]]
