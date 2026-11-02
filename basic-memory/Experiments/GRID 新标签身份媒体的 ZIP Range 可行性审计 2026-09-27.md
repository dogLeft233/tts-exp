---
title: GRID 新标签身份媒体的 ZIP Range 可行性审计 2026-09-27
type: experiment
permalink: tts-exp/experiments/grid-新标签身份媒体的-zip-range-可行性审计-2026-09-27
status: concluded
date: '2026-09-27'
run_id: grid_range_feasibility_20260927
---

# GRID 新标签身份媒体的 ZIP Range 可行性审计

## Observations
- [status] concluded；仅获取可行性。8个官方ZIP支持严格Range，32候选目录预算；只1个固定cal成员在RAM核验，未取成套媒体、未新模型/评分/GPU、未解封旧评估媒体。
- [source] [官方GRID记录](https://zenodo.org/records/3625687)，API许可证CC BY 4.0；视频标签s1–s34除s21。只是数据标签身份，无生物身份/预训练独立保证。
- [history] CodeGraph先查询后限定搜索代码/配置、项目BM和两个import manifest；历史发现初版2和multiset250中50个GRID均s1，未发现候选其他标签既有TFG使用记录。负搜索不证明绝对未见；本地audio ZIP本已含全部34标签。
- [range] s2–s9均206与精确Content-Range/Content-Length PASS；例s2 suffix bytes=-65557→394578565-394644121/394644122，CD 394490680-394644099共153420B。客户端若200/范围不符在body前关闭，逐响应/总字节封顶。
- [container] 8个ZIP各1000真实.mpg，非JPG帧目录。官方文字的.jpg不能替代实测；本地历史s1_processed的MP4为既有转码路径，本次不读其媒体。
- [clock] 固定s2/bbaf1n.mpg compressed398467B→407552B，CRC与SHA PASS。MPEG-1 360×288，实数75帧25fps/PTS0…2.96s覆盖3s；内嵌MP2 44100Hz stereo，114帧/每声道131328samples=2.977959s；A/V start_pts均0、time_base1/90000。原始PTS/计数已留证，无人工预览。只是单成员事实，不能外推全包或证明物理零延迟；后续常量延迟须cal冻结。
- [audio] local audio_25k.zip MD5 4b3ac37b1a258f55d1eebe657de491a9与官方相同，真实34000 WAV各标签1000。32候选同名支持，但header时长1.49004–2.36004s，代表同名WAV仅1.49004s；不假定与3s容器同步，不据此声称已测裁静音机制，优先容器内嵌轨。
- [initial-candidates] cal s2/s3/s4，eval s5–s9；各字典序前4个同名媒体共32条，只查目录。压缩合计13448238B（12.825MiB），解压13754368B（13.117MiB）；尚未获取该集合。
- [expanded-proposal] 8cal=s2–s9；16eval=s10–s20+s22–s26，排除s1/s21。官方包与本地标签WAV均有；额外16包未查CD。candidate24_metadata_only.json给本地WAV字典序首stem候选，8条remote目录证实，16条仅待核验同名候选，不假定长度或codec。后续科学方案另行前瞻冻结。
- [estimated-budget] 24标签各1clip压缩约8.36–11.39MiB/解压8.58–11.58MiB；各2clip约16.72–22.77MiB/17.16–23.16MiB。基于已测32目录条目外推，非保证界/未获取；不含后续转码/特征/生成视频。
- [validation] 21个HTTP receipts（20 Range）及32本地WAV header、8官方ZIP size独立一致性复核PASS；网络实际2561830B，持久media=0，run/BM合计远低4MiB，保留5GiB门。不是独立重下所有CD或同步真值验证。
- [artifacts] runs/grid_range_feasibility_20260927/report.md；官方metadata、HTTP receipts、目录/候选/预算、单cal header与PTS、local库存、独立复核、artifact_hashes/final。旧报告与Research hub未修改。

## Relations

- relates_to [[multidataset-samples]]
- relates_to [[跨数据集 TFG 测评（5×50 multiset）]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 理论端授权小额获取可行审计，planned；固定标签/目录选择与4MiB持久门，无新效应评分 | September 27, 2026 | parent feasibility request / user authorized |

| 严格Range/官方目录/同cal时钟审计与独立metadata一致性PASS，concluded；24标签仅分级候选，未获取集合，保留物理同步/新身份边界 | September 27, 2026 | parent authorized feasibility and same-cal timeline audit |
