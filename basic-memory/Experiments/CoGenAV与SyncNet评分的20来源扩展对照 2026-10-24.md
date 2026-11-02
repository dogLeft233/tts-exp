---
title: CoGenAV与SyncNet评分的20来源扩展对照 2026-10-24
type: note
permalink: tts-exp/experiments/co-gen-av-与-sync-net-评分的-20-来源扩展对照-2026-10-24
status: concluded
scientific_status: EXPLORATORY_METRIC_DISAGREEMENT_EXPANDED_20_SOURCES
training: false
tags:
- cogenav
- syncnet
- metric-validation
---

# CoGenAV 与 SyncNet 评分的20来源扩展对照

| Notes | Modification Date | Approved By |
|---|---|---|
| Initial expanded comparison | 2026-10-24 | user requested more data |

## 设计与资产
用户要求在四来源先导后扩大数据。沿用同一官方CoGenAV Base权重、代码、fairseq默认配置合并和AutoAVSR口部预处理，论文Eq1及派生31lag±600ms诊断定义不变。SyncNet复用上次canonical PCM官方流程float32结果，本次没有重评分/TFG生成/训练/人评。是已有队列的探索性扩展，不是全新盲法确认队列。

本机.、远端/home/wjj/tts-audio/tts-exp，同相对run runs/cogenav_sync_full20_20261024_attempt01。先导28clip/84缓存文件按原manifest SHA复用；新16来源计划112clip，成功110。全20来源计划140，成功138（N20/L06 58/L17 60）。S0011/S0016 L06 d3各40.88秒，超过冻结的原生30秒输入设置，明确UNMEASURABLE；未截断或改变上下文，原qualified均false。首次遇长clip停止、保留日志，冻结eligibility_protocol.json后继续，成功缓存未重做。

协议继承先导科学设置；analysis_protocol.json SHA12fbf01828be1d92594718a1ac91bed186592a2708d8180b5ba590ef0381342d，在扩展完成前冻结。报告拆full20、additional16、common_qualified_full20。独立单位20语音来源，固定头像；每来源三个family，draw先平均，再等权来源。描述性source bootstrap10000/seed20261024，保留每来源全部family/draw，边际95%区间，不做p值或多终点确认推断。

## 结果
实际全量176pair；新增16来源140pair。共同资格按原探针协议从两种TTS同时剔除失败draw，保留N和来源，174pair。原始逐clip qualified只标出坏L06，不能直接当成共同资格；首次统计已纠正并保留comparison_before_qualification_fix.json与对应源码，所有评分数组不变。

|端点|全20来源同向/反向|新增16同向/反向|全量来源等权同向率95%描述区间|全量source×family均值分差Spearman|
|---|---|---|---|---:|
|Eq1全长|71/105|58/82|40.3% [33.6%,47.2%]|-0.196|
|zero_common|76/100|64/76|43.3% [35.0%,52.2%]|-0.217|
|max_score|94/82|75/65|53.9% [46.7%,61.7%]|+0.198|
|zero_excess|72/104|52/88|41.1% [32.5%,50.3%]|-0.270|
|peak_excess|107/69|77/63|61.1% [48.3%,73.3%]|+0.317|

共同资格174pair的Eq1/zero/max反向数仍105/100/82。沿用先导的近零敏感性|ΔC|>=.1且|ΔCo|>=.005：全量Eq1保留147pair/90反向，max保留144/64；新增16分别115/71、112/50。阈值仅敏感性，不是显著性界限。

全量source等权均值差：L06-N ΔC+0.059/ΔEq1−0.013758/Δmax+0.001103；L17-N +0.167/+0.003482/+0.016734；L17-L06 +0.096/+0.015057/+0.012858。均值方向与逐来源/配对一致性分开，不用小均值优势盖过异质性。

138clip中124峰lag+2帧，14峰+3帧；不当作物理延迟80/120ms真值。真实波形±160ms重新mel/音频encoder，276/276峰移动控制通过（容差40ms）。这是粗错位定位资格，不证明细小感知同步或中文域通用有效。

## 工程与核验
新增440实际模型forward，原28缓存不重复。全140计划输入的audio/video/score/activesd共560SHA验证通过。全部138原始及276shift曲线从已存特征独立float64逐点重放，最大误差1.1142744088532552e-7。8个科学契约测试函数直接执行PASS（无pytest环境，不声称pytest运行）；包含来源等权、缺draw不造pair、共同资格对称剔除、缓存协议兼容性。5张带标识的口部采样总览实际查看，无明显ROI失位；不是人工同步评判。

460科学/源/配置/日志文件176423655bytes已本机逐SHA回传核验，缺失错配0。完整embedding、rawshift表示、JSON、报告与故障/修复证据保留；权重本体及依赖build body留远端，不声明网盘归档。GPU科学进程已退出，回到322MiB桌面占用。继承dirty工作树保留，未commit/push。

## 结论与边界
增加到20来源后，原始CoGenAV Eq1与Sync-C仍明显排序分歧；新增16来源也出现类似分歧。相同支持下搜索最佳lag能提高一致性，但只有约54%来源等权同向，不能称与Sync-C一致。减外围参照的peak_excess描述性更接近Sync-C，但区间宽且包含50%，不能将其事后升级为主指标或证明更正确。

CoGenAV是官方模型+论文Eq1的自写评估，不是作者完整官方同步benchmark；max/excess是自定义诊断。固定头像、中文域未人评校准、不同原生语速/时长/发音时钟仍可能影响分数。模型分歧不能单独判定SyncNet偏好或人类嘴型准确度。当前未自动启动新语料/新生成/新模型训练或Large版本比较。

## Observations
- [result] 全20来源138可测clip完成，另外2超长输入明确不可测；276/276真实错位控制通过。
- [result] Eq1与Sync-C反向105/176；同支持零时移100/176，搜索最佳时移82/176；新增16来源分别82/140、76/140、65/140。
- [limitation] 最大时移分数不是Sync-C；来源等权同向约54%，不宣称两者一致或任何一方为感知真值。

## Relations
- extends [[CoGenAV与SyncNet同步评分的四来源先导对照 2026-10-24]]
- follows [[自然音频与双规模TTS的SyncNet偏好探针 2026-10-24]]
