---
title: Codex checkpoint - 2026-10-20T03-19-11Z - May postnorm chart delivery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t03-19-11-z-may-postnorm-chart-delivery
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T03:19:11.618820+00:00'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-20T03-19-11Z - May postnorm chart delivery

## Summary
为用户制作了报告56的中文结果图、矢量PDF和完整12条件CSV，重点区分低尺度训练的模型差异与提高驱动尺度的额外变化。

## Story
原问题是归一化是否改变训练与驱动的效果；最新用户在断电后结果完整交付的基础上问“结果怎样，能制作个图表吗”。本轮从Basic Memory正式报告56和封存F产物读取12格结果，用Matplotlib生成独立导出图，没有恢复GPU或进行新科学运算。上排使用同一Sync-C纵轴展示标准训练/标准驱动、低训练/标准驱动、低训练/高驱动；下排展示最终低/高相对标准的原时钟D0和嘴部像素MSE变化。该分解避免把两个训练模型间的差异误认为提高驱动的收益。

## Working State
最终派生图包为F:/Datasets/tts-exp/may_postnorm_scale_figures_20261020_v3，包含may-postnorm-results-cn.png、同名.pdf、all-12-conditions.csv、plotted-data.json、plot-source.py、figure-receipt.json、visual-and-data-qa.json。科学根F:/Datasets/tts-exp/may_postnorm_scale_20261020保持封存，权威final_delivery_receipt passed，381成员manifestSHA26e3bdf41c7119a3d01accdd8ff69675b520f3347272ee1d092fc95b5ea06265；本轮没有在该根新增或覆写文件。新图不是新实验结果。

主要结果：wav2vec2 Sync-C为1.626→1.718→1.719，高驱动相对同一低尺度模型标准驱动额外仅约+0.001；HuBERT为1.695→1.697→1.697，高驱动额外+8.583e-6。低/高相对标准D0，HuBERT+0.126恶化，wav2vec2−0.051改善；嘴部MSE分别−1.09%/−1.66%。只说明May单身份、单训练seed、每模型12000成功更新的工程小试；不称显著、不外推总体因果或人类感知。尺度是原生完整split z-score之后乘0.5/1/2，约RMS0.5/1/2，不是旧RMS0.1和原录音相对−6dB。

报告56与原实验Task已在上轮正式发布/关闭，本轮没有改Task/地图/结论。断电晚于全部科学作业完成，SHM已消失但F完整，不需要重启。旧Patrick coarse7保存暂停、fine14/Chris/Cathy与24full继续暂停，旧自动化已删除。所有新模型断点保留。

## Changed Files
.codex/scratch/plot_may_postnorm_user_summary_v1.py是机器局部、未提交的导出脚本，最终源精确镜像在v3/plot-source.py。v1/v2派生候选保留：v1缺上标字体且标题重叠，v2仍有标注靠近初始点，最终v3使用ASCII科学记数并分离标题/标注。没有改17冻结科学源，不stage/clean/commit/push其它dirty。此次checkpoint及读回也只写.codex/scratch和Basic Memory，不写封存科学根。

## Verification
读取正式报告、finalreceipt；绘图前按manifest fresh核analysis/cells、effects、interpretation_summary SHA及manifest自身。最终PNG实际view_image查看，中文、指标方向、数值、三位小数、指数、轴、脚注均可读，无重叠或缺字。五导出文件SHA重新读回精确，CSV12行C/D0/MSE与原cells全精度float逐格相等。v3/visual-and-data-qa.json passed。PDF从同一Figure导出，未独立渲染PDF；fontTools有MERG表未subset的提示，没有最终缺glyph警告。0新GPU/训练/推理/评分，未重复长测试或全tensor加载。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)
- [报告56](../../docs/experiments/56-may-postnorm-scale.md)
- F:/Datasets/tts-exp/may_postnorm_scale_20261020/final_delivery_receipt.json
- F:/Datasets/tts-exp/may_postnorm_scale_figures_20261020_v3/figure-receipt.json

## Observations
- [result] 已完成并实际看图核对中文总结图，完整12条件CSV逐格全精度一致，独立派生包不修改封存科学结果。
- [decision] 用相同纵轴和训练/推理两段差异展示，保留单身份单seed与SyncNet代理指标边界，不把提高驱动的微小变化概括为普遍优势。
- [next_step] 向用户交付最终v3图和PDF/CSV，随后按其新研究方向讨论是否需要独立种子重复；不自动启动训练。

只读GitHub证据：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}；当前HEAD `2b71639c3b555293375d248350999377168d515d`未确认远端可解析，不构造commit链接。

## Relations
- continues [[Codex checkpoint - 2026-10-20T03-05-00Z - May postnorm delivered after power loss]]
- relates_to [[56-may-postnorm-scale]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1068条；大量其他任务改动，不批量stage/clean/commit。
