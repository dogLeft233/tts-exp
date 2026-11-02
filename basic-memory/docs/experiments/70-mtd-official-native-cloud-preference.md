---
title: 70-mtd-official-native-cloud-preference
type: experiment
permalink: tts-exp/docs/experiments/70-mtd-official-native-cloud-preference
status: concluded
scientific_status: OFFICIAL_RETRIEVAL_DESCRIPTIVE_COMPLETE_LOW_NATIVE_ACCURACY
---

# 70 — MTDVocaLiST官方检索指标重评云TTS原生视频

✅ 2026-10-28完成80/80原生视频、六项官方K指标与独立复算。主K5：T胜21/60=35%，平均T−N−0.565642个百分点，与Sync-C冲突33/60=55%。这是官方检索口径，替代实验68自定义logit作为本次任务结果；不覆盖旧文件。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 官方K5–K15原生重评完成 | 2026-10-28 | 用户明确要求官方方法重评 |

## Cohort与冻结设计

沿用之前中文云Qwen qwen-audio-3.1-tts-plus的20来源自然视频各1条、TTS各3条，共80媒体、60配对。不是新EARS。无API调用、TFG重生成或交叉音轨，继续使用已有规范音轨及电平处理，不额外音量归一化。原Sync-C逐次配对值来自cloud_av_preference20_20261027_attempt05，经cloud_mtd_synchformer_20261028_attempt01冻结inputs/sync_pairs复用。

主指标评分前固定为K5，辅K7/9/11/13/15全列，不按结果挑K。原生全片，不应用审计发现的lag+3，不逐条最佳时移校正，不拉伸、填充、剔除边界窗口。模型仍要搜索31个候选时移用于测量；不把其最佳时移用来修改输入或作为新零点。

## 官方方法执行

官方源https://github.com/xjchenGit/MTDVocaLiST commit5795416f9277094d3ae6215596a436a0a01f79ad；test_stu.py https://github.com/xjchenGit/MTDVocaLiST/blob/main/test_stu.py 。公开pure_MTDVocaLiST.pth，strict加载test_stu的student_thin_200.SyncTransformer(d_model=200)，13280961参数，FP32 eval。

本次直接AST执行作者Dataset、calc_pdist、eval_model原封不动的函数/类体。sys.setprofile仅在eval_model返回读取六个samplewise_acc列表，不改内部公式；calc_pdist仅加保存矩阵的wrapper，实际模型前向与计算由原函数执行。对所有80条新做网络前向，不复用旧分数。首3自然的原512 JPEG可复用审计物，且与原音轨逐点核对。

预处理统一原512分辨率逐帧S3FD，去除实验68先缩256检测的适配；按Wav2Lip预处理裁整脸写默认OpenCV JPEG，再作者Dataset的torchvision RGB0–255、Resize96、下半48×96。作者16k Mel80/FFT800/hop200、Slaney、dB缩放与int(80*f/25)索引。原生min_length−5窗口，作者无效候选−1000掩码、边界平均方式全部保留。S3FD batch1，关cudnn.benchmark避免算法搜索workspace；Torch2.5.1+cu121兼容运行时，不宣称原论文包版本bit-exact。

K5每个5帧窗口在±15帧共31候选中检索，预测offset在−1/0/+1帧记正确，全视频取正确比例。K7–K15对相邻窗口候选分数作作者指定平均，等效上下文延长；允许偏移仍±1帧，没有随K放宽。分数[0,1]是检索正确比例，不是分类概率或Sync-C。

## 全组60对结果

均差T−N，越高越好。差以百分点报告；冲突=与既有Sync-C的配对差严格异号。N每源在3个TTS配对中重复，故N均值与20源等权相同。精确平局另列，不当作一致或冲突。

| 官方指标 | N均分% | T均分% | T−N百分点 | T胜/60 | N胜/60 | 平局/60 | 冲突/60 | 非平局中冲突 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| K5（主） | 3.771425 | 3.205783 | -0.565642 | 21=35.0% | 39=65.0% | 0 | 33=55.0% | 33/60=55.0% |
| K7 | 4.086478 | 2.514971 | -1.571506 | 15=25.0% | 42=70.0% | 3 | 45=75.0% | 45/57=78.9% |
| K9 | 4.008297 | 2.095818 | -1.912478 | 12=20.0% | 48=80.0% | 0 | 42=70.0% | 42/60=70.0% |
| K11 | 4.634242 | 2.235378 | -2.398864 | 6=10.0% | 51=85.0% | 3 | 39=65.0% | 39/57=68.4% |
| K13 | 5.043904 | 1.965505 | -3.078399 | 10=16.7% | 50=83.3% | 0 | 44=73.3% | 44/60=73.3% |
| K15 | 5.345543 | 1.596139 | -3.749404 | 6=10.0% | 54=90.0% | 0 | 48=80.0% | 48/60=80.0% |

主K5冲突33对：30对Sync-C偏T/MTD偏N，反向3对。完整冲突方向与各样本结果在summary.json与paired_scores.csv。

## 来源统计与严格参考子集

20来源是独立统计单位，60生成配对不是独立来源。source-cluster bootstrap10000次seed20261028。K5差95%CI为[-1.320332,+0.326934]个百分点，跨0；只能说本组更常给自然高分，平均差仍不确定。辅指标CI：K7[-2.731662,-0.413208]、K9[-2.826714,-0.996252]、K11[-3.458703,-1.291544]、K13[-4.269931,-1.811668]、K15[-5.034235,-2.426896]个百分点。这些是相关的多上下文端点，不是五个独立裁判。

strict12同句参考36对，K5 T胜15/36=41.7%，自然21/36，平均T−N−0.716549个百分点，CI[-1.419698,-0.048511]，冲突24/36=66.7%。它是预先保留的子集，不替代全20源主结果。

## 验证和解释限制

80条均验证原音轨PCM逐点一致，作者六种归约与独立截断移动均值计算完全一致。二次独立CPU复核从80候选矩阵重算480个指标，再重算360个指标×配对的均差、胜负、平局和冲突，与summary全部一致，independent_result_audit.json PASS。首3来源新矩阵与上一轮完整512审计矩阵最大差<2e−4。两项实际测试PASS：上下文K不放宽时移容忍；真实矩阵的作者eval_model与独立计算六指标完全一致。

按官方方法，K5 T胜35%，并非旧自定义mean logit的0%；二者不能混称同一指标。另一方面，绝对K正确率仅几个百分点，所有80媒体K5范围0至8.9744%。实验69已显示自然片段的模型主峰多在−3帧，而当前K指标要求预测在零附近±1帧，所以存在显著低分/地板问题。40/40已知平移恢复通过，不意味着零时移K分数在当前TFG视频上已被校准为感知同步质量。

这些结果描述官方指标的偏好方向，不能直接证明自然口型更好或SyncNet有偏置。Sync-C是时移距离曲线的相对突出程度，K指标是检索到零附近的比例，冲突也包含评价目标不同。未做逐条校准以改善K值，后续若评估校准后的端点需另立协议。

## 复现与产物

Ubuntu /home/wjj/tts-audio/tts-exp/runs/mtd_official_cloud_rescore_20261028_attempt01；Windows E:/Documents/tts-exp/runs/mtd_official_cloud_rescore_20261028_attempt01。小JSON/CSV与80矩阵均已同步本机；原JPEG预处理与源视频保留Ubuntu。

```bash
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/mtd_official_cloud_rescore_20261028.py score
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/mtd_official_cloud_rescore_20261028.py analyze
```

protocol.json保留评分前冻结的源码SHA、指标、支持和分母。source_manifest.json及source_snapshot/保存驱动脚本、审计helper、官方test_stu与测试文件。score可断点恢复，完成后不重复GPU；不要重跑freeze生成不同源码hash的协议。代码未提交。

## Relations

- follows [[69-mtd-official-parity-and-shift-control-audit]]
- replaces_score_adapter [[68-cloud-mtd-synchformer-native-preference]]
- follows [[MTDVocaLiST官方K指标重评云TTS原生视频 2026-10-28]]
- uses [[MTDVocaLiST部署]]