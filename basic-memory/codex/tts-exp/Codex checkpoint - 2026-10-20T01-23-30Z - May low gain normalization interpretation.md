---
title: Codex checkpoint - 2026-10-20T01-23-30Z - May low gain normalization interpretation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t01-23-30-z-may-low-gain-normalization-interpretation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T01:23:30.700914+00:00'
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

# Codex checkpoint - 2026-10-20T01-23-30Z - May low gain normalization interpretation

## Summary
已从报告55与实际72格产物核对May低音量训练关闭归一化的配对结果：嘴部像素MSE略低，同步指标方向混合，不能判整体质量更好。

## Story
用户最初要区分音量、编码器和归一化对生成的作用，随后要求在May上做同encoder权重、训练与推理音量和输入norm的完整交叉。实验已完整交付；本轮最新问题是“训练时用低音量数据，不用归一化，会比用归一化的模型效果好吗”。本轮只读复核，不启动新训练或评分。先从Basic Memory Startup Router、研究地图和实验指令定位，项目全文搜索并读取canonical报告55，再直接读取F盘cells.json与final_delivery_receipt.json。

为回答训练端问题，固定两组都用−6dB训练、推理原音量0dB且推理norm关闭，只比较训练norm开关。HuBERT off/on的C为1.6129927635/1.6520695686，原时钟D0为13.0249495506/13.0843939781，嘴部MSE为0.004725785868/0.004886138441。wav2vec2 off/on的C为1.7710232735/1.6760978699，D0为12.9934377670/12.9830579758，MSE为0.004577990837/0.004670433092。低音量且无训练norm的嘴部MSE在两encoder均较低；HuBERT C较低、D0较低，wav2vec2 C较高、D0较高，评价指标没有一致赢家。这里训练norm开启但推理norm关闭是刻意固定推理预处理的条件对照，不能与常规on/on部署流程混淆。

若比较各自训练推理norm一致的流程，−6dB训练、0dB推理下，on/on相对off/off的嘴部MSE上升HuBERT3.4179%、wav2vec2 1.6701%（off为分母），C差各+0.04033/−0.09130，D0差各+0.05968/−0.04658。这是共同切换两端的流程差异，不能称纯训练端效应。两种读法都支持嘴部像素误差小幅更低，但同步没有跨encoder一致收益；不把像素MSE或Sync-C直接当人类感知质量。

## Working State
权威科学包F:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019已封存，本轮不写入、不运行会改mutable快照的旧poll脚本。final_delivery_receipt passed：8新模型各12000成功FP32更新0skip、4旧0dB模型精确导入，共72格144矩阵，1437成员/6物理根。manifest SHA c065b49686f82440b65361f1bd60485542ba6fa12f8390863f645e313c1e83bc。旧Patrick已按用户要求暂停且不恢复，原24full未完成。当前无本实验活动科学jobs，新结果未声称云备份。

报告55是tts-exp/docs/experiments/55-may-gain-normalization-factorial，物理basic-memory/docs/experiments/55-may-gain-normalization-factorial.md；实际cells、normalization_effects、effect_summary均在F根analysis/。新媒体跨F/E映射、云冷存储locator及主机历史故障恢复均见前一同chat交付checkpoint及final_delivery_receipt，不在本轮重跑或修改。当前norm是native波形z-score，内部LN/GN与encoder权重固定；不是旧Patrick RMS0.1。May单身份、单训练seed、每模型12000步/两相关clip，未证明充分收敛、感知改善或多seed稳定性，不能据此确定最佳训练音量。关闭norm对推理gain仍可测敏感，开启norm使响应接近数值精度；这与质量收益分开。

## Changed Files
本轮没有科学源码、测试、报告或封存数据变更。仅机器局部.codex/scratch/read_may_low_gain_comparison_v1.py、may_low_gain_report_readback_v1.json与本checkpoint脚本/读回记录用于只读核查和handoff；这些未提交。前序实验源码和全部验证入口在已交付checkpoint。大量其它dirty文件保持，不stage/commit/clean/push。

## Verification
本轮实际Basic Memory项目text检索、canonical报告全文读回、F final_delivery_receipt和8个低音量训练/0dB推理条件JSON读取、固定inference norm条件配对与MSE百分比计算已执行。未重跑GPU、训练、评分、CPU全tensor加载或测试；已有科学QA通过结论来自此前已封存receipt，不冒称此次重复全验。Git branch/HEAD/status与GitHubcanonical仓库/当前branch PR/commit解析做只读核查。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)
- [报告55](../../docs/experiments/55-may-gain-normalization-factorial.md)
- F:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019/analysis/cells.json 与 final_delivery_receipt.json

## Observations
- [result] 固定−6dB训练、0dB推理及推理norm关闭，训练norm关闭的嘴部MSE在两encoder略低，同步C和原时钟D0没有一致优势。
- [decision] 区分训练norm条件效应与训练推理共同开关流程效应，区分嘴部像素误差、SyncNet指标与感知；单May/seed/短预算不选最佳音量。
- [decision] 不恢复Patrick或扩展新训练，保持已封存实验不变。
- [next_step] 向用户说明低音量无归一化目前只有嘴部像素误差的小幅优势，同步与总体质量尚无一致改善证据；等待用户对后续验证的新方向。

配对精确读回与计算：[{"encoder": "hubert_base", "fixed_inference_normalize": false, "off": {"encoder": "hubert_base", "train_normalize": false, "test_normalize": false, "gain_db": 0, "sync_c": 1.612992763519287, "sync_d": 11.929676532745361, "distance_d0": 13.024949550628662, "background_median": 13.542669296264648, "mouth_bbox_MSE": 0.004725785867543891, "av_offsets": "-2,-3", "full_sync_c": 1.486238956451416, "full_distance_d0": 13.318861484527588, "train_gain_db": -6, "origin": "new_gain_training"}, "train_norm_on": {"encoder": "hubert_base", "train_normalize": true, "test_normalize": false, "gain_db": 0, "sync_c": 1.6520695686340332, "sync_d": 11.904225826263428, "distance_d0": 13.084393978118896, "background_median": 13.556295394897461, "mouth_bbox_MSE": 0.004886138440808281, "av_offsets": "-2,-3", "full_sync_c": 1.5922198295593262, "full_distance_d0": 13.370773315429688, "train_gain_db": -6, "origin": "new_gain_training"}, "off_mse_change_pct_vs_on": -3.2817853036080757}, {"encoder": "wav2vec2_base", "fixed_inference_normalize": false, "off": {"encoder": "wav2vec2_base", "train_normalize": false, "test_normalize": false, "gain_db": 0, "sync_c": 1.7710232734680176, "sync_d": 11.673946857452393, "distance_d0": 12.993437767028809, "background_median": 13.44497013092041, "mouth_bbox_MSE": 0.004577990837395192, "av_offsets": "-2,-3", "full_sync_c": 1.6914176940917969, "full_distance_d0": 13.29006814956665, "train_gain_db": -6, "origin": "new_gain_training"}, "train_norm_on": {"encoder": "wav2vec2_base", "train_normalize": true, "test_normalize": false, "gain_db": 0, "sync_c": 1.6760978698730469, "sync_d": 11.806534767150879, "distance_d0": 12.983057975769043, "background_median": 13.482632637023926, "mouth_bbox_MSE": 0.004670433091698215, "av_offsets": "-2,-3", "full_sync_c": 1.608971118927002, "full_distance_d0": 13.298330783843994, "train_gain_db": -6, "origin": "new_gain_training"}, "off_mse_change_pct_vs_on": -1.9793079675488934}]
GitHub只读核查：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}
当前HEAD若GitHub不能解析则标为本地未推送，不构造commit链接。

## Relations
- continues [[Codex checkpoint - 2026-10-19T17-52-38Z - May gain factorial verified delivery]]
- relates_to [[55-may-gain-normalization-factorial]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1066条；大量其他任务改动，不批量stage/clean/commit。
