---
title: PC-Talk Sync-C分差分布与正差样本
type: note
permalink: tts-exp/docs/results/pc-talk-sync-c-distribution
status: concluded
date: '2026-10-31'
analysis: descriptive existing scores only
---

# PC-Talk Sync-C分差分布与正差样本

✅ 已完成。2026-10-31，报告83的既有本机结果做只读描述性分析，没有服务器连接或模型/ASR/TTS前向。定义ΔC=来源内三条TTS平均C−该来源自然C；20来源是统计单位，60逻辑TTS含重复，只有21种唯一TTS媒体，不能当60独立重复。

|来源|参考|自然C|TTS均C|ΔC|
|---|---|---:|---:|---:|
|BAC009S0005W0121|扩展|6.091|8.289|+2.198|
|BAC009S0006W0121|扩展|8.390|9.988|+1.598|
|BAC009S0025W0122|同句|6.181|7.646|+1.465|
|BAC009S0012W0121|扩展|6.712|8.033|+1.321|
|BAC009S0016W0121|同句|5.926|7.067|+1.141|

以上五个来源每人的三条TTS都是相同音视频，代表C1与均值相同；是按正差事后排序的展示例子，不能当总体/人类质量评价。播放为保留原生AAC视频，评分入口为原PCM，继承83的SyncNet内部MP3/延迟边界。

全20：均值+0.490，中位+0.673，样本SD0.959，Q1/Q3=-0.028/+1.107，范围[-1.724,+2.198]；14正、6负、0平。10%/90%分位-0.506/+1.478。分位采用NumPy线性插值，SD为ddof1。

|Δ区间|来源数|比例|
|---|---:|---:|
|(-inf,-1)|2|10%|
|[-1,0)|4|20%|
|[0,.5)|3|15%|
|[.5,1)|5|25%|
|[1,2)|5|25%|
|[2,inf)|1|5%|

主要12的均值+.457、中位+.673、SD.786，9正3负；扩展8均值+.539、中位+.510、SD1.232，5正3负。这里只描述原结果，不做新检验；报告83全20Sync-C Holm9 p=.15475、主要12 p=.38185仍未过，正差样本选择不改变该结论。

产物目录：`./results/pctalk/chinese20_rms24_nine_20261031_attempt01/syncnet_distribution/`。top5.html左右视频对比、distribution.png/PDF、distribution.json、source_differences.csv（20行）、tts_draw_differences.csv（60逻辑行）。

验证：输入分数JSON及旧video_index SHA冻结，20来源Δ与83逐值匹配；stdlib均值/中位数/样本SD独立复算匹配，25HTML媒体链接/10所选视频SHA校验通过，图已目视。测试tests/test_pctalk_syncnet_distribution_20261031.py两项PASS，覆盖来源内平均/重复draw与分箱边界/样本SD。源码scripts/analysis/pctalk_syncnet_distribution_20261031.py。原报告/封存包不改，新增产物为包外补充。

## Relations
- follows_up [[83-pctalk-chinese20-nine-scorers]]
- relates_to [[SYNC-C-SCORES]]
