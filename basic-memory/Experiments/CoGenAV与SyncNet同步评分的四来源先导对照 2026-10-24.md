---
title: CoGenAV与SyncNet同步评分的四来源先导对照 2026-10-24
type: note
permalink: tts-exp/experiments/co-gen-av-与-sync-net-同步评分的四来源先导对照-2026-10-24
status: concluded
scientific_status: EXPLORATORY_METRIC_DISAGREEMENT_KNOWN_SHIFT_PASS
training: false
tags:
- cogenav
- syncnet
- metric-validation
---

# CoGenAV 与 SyncNet 同步评分的四来源先导对照

| Notes | Modification Date | Approved By |
|---|---|---|
| Initial bounded diagnostic | 2026-10-24 | user requested code/weight feasibility and synchronization-score comparison |

## 问题与范围
用户怀疑Sync-C具有输入偏好，授权检查CoGenAV权重/代码，并用其方法评价已有音唇同步度、比较排序。仅探索性先导，不判定人类感知真值或纯评价器偏好。没有训练、新TFG、新SyncNet评分或人评。

## 可复现资产与实现
本机.，远端/home/wjj/tts-audio/tts-exp，run均为runs/cogenav_sync_audit_20261024_attempt01。report.md、protocol.json、comparison.json、pairs.json、source_family_means.json、selected_inputs.json、112项input_verification.json、embeddings/、crops/、final_manifest.json、execution_receipt.json。117科学/源码/配置/日志文件34,759,795字节已逐SHA回传核验，缺失/错配0；权重本体留远端，不声称网盘归档。

CoGenAV官方GitHub dbd3f835a2cc859766c421aaf3918cda2246008d；作者HF detao/CoGenAV revision3c85c000c60fe7ffb3aaa3c63c96ba33ad8ce42c，Base权重494562616bytes，SHA256 90f8c5165282c190c81c23d8fe92db2aa700862e021b1312fb96779caf9cfe61。strict加载通过。独立venv cogenav-sync-20261024复用既有torch2.5.1+cu121运行库，不修改既有环境。fairseq PyPI0.12.2源码分发缺C++文件，最终以官方v0.12.2完整源码构建成功；hydra1.0.7/omegaconf2.0.6。合并Wav2Vec2Config默认值解决pos_conv_depth等缺字段；不改权重/模型源。

采用AutoAVSR官方MediaPipe detector/VideoProcess、96x96对齐RGB口部ROI，随后遵循CoGenAV原推理的加权灰度/Base mean0 std1/30秒padding。分别A-only/V-only、upsampler前25fpsFP32表示，只评分真实有效长度。28条口部采样总览实际查看，无明显裁剪失位；不称视觉同步人工核验。112模型forward=28A+28V+56真实波形shiftA。

## 冻结协议与队列
前次140输入清单按sample_id字典序取前4来源（S0002/S0003/S0004/S0005），不按分数选样。每来源N1、L06/L17各3draw，合计28视频，独立语音来源4，固定生成头像；三个family各12pair/4source。canonical RMS mono16k PCM和视频/官方评分/activesd四类输入共112SHA全部复核。N复用不扩增独立样本。

protocol.json在任何forward前冻结，SHA b0e79739417d554e1cfb5787275286cce8c120ad8d791f81521f30415453621c。主量为论文Eq1 mean ReLU cosine；派生量max/zero_excess/peak_excess单列。31lag±600ms在每clip内部统一视频中心支持；不声称跨臂同一发音事件支持。真实波形±160ms后重新计算mel与A表示，不是平移缓存表示；峰移动误差容差40ms，预定>=80%通过，实际56/56通过。

## 结果
|比较|平均ΔSync-C|平均ΔEq1|平均Δmax|Eq1冲突pair|来源均值冲突|
|---|---:|---:|---:|---:|---:|
|L06-N|-0.114|-0.013975|+0.012782|7/12|3/4|
|L17-N|-0.029|+0.040937|+0.050962|10/12|4/4|
|L17-L06|+0.085|+0.054912|+0.038179|6/12|1/4|

合计Eq1符号冲突23/36，搜索max仍17/36。pair分差Spearman=-0.163，source×family均值Spearman=-0.098，均描述性，不计算伪独立p值。事后近零敏感性|ΔC|>=.1且|ΔEq1|>=.005保留32pair，冲突19。

均值受到S0004大幅正CoGen差影响，不把均值当跨来源稳定优势：S0003 L17-N ΔC+0.596但ΔEq1−0.034892；S0004 ΔC−0.612但ΔEq1+0.218005。N/L06/L17平均Eq1=.303/.289/.344，而lag0减外围lag均值=.034/.022/.019，原始跨模态相似度升高不自动等于时间选择性升高。N峰均+2帧，可能为流程/模型系统偏移，不能直接称物理延迟80ms。

## 验证与边界
独立float64逐窗点积/曲线重放最大误差6.944872810166203e-8。3个科学契约函数（lag符号/共同支持、尺度/ReLU、source分组/draw配对）直接执行PASS。本地pytest缺numpy收集失败，远端无pytest，不能声称pytest通过。保存安装失败与成功日志及源码/config。GPU任务已终局退出，回到322MiB桌面占用。

只表明两套评价表示存在排序分歧，不能判定哪一个符合人类、不能单独证实SyncNet偏好。英文训练表示的中文泛化未校准；已知160ms定位控制不证明微小误差辨别或所有扰动有效。只有4独立来源，36pair、56control不是独立样本数。原生音轨/时长/语速与内容支持不同；当前比较仍联合生成与评价，不隔离纯评价器贡献。

## Observations
- [result] 官方Base权重可获取且strict加载，112个实际前向完成。
- [result] 56/56真实音频错位定位控制通过；CoGenAV Eq1与Sync-C排序冲突23/36，max冲突17/36。
- [limitation] 四来源探索性、固定头像、无感知ground truth，不从评价器分歧判定SyncNet错。
- [next_step] 若继续，先扩大独立来源并考察预处理/固定偏移及时间选择性稳定性；不能将原始Eq1直接升级为主同步质量终点。

## Relations
- follows [[自然音频与双规模TTS的SyncNet偏好探针 2026-10-24]]
