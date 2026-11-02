---
title: Codex checkpoint - 2026-10-19T17-52-38Z - May gain factorial verified delivery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t17-52-38-z-may-gain-factorial-verified-delivery
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T17:52:38.616112+00:00'
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

# Codex checkpoint - 2026-10-19T17-52-38Z - May gain factorial verified delivery

## Summary
May同encoder训练音量×推理音量×native输入归一化72格实验已完成、发布报告55并跨F/E全成员封存；推理gain响应大幅减小，但质量指标没有一致改善，Patrick仍暂停。

## Story
用户先让Patrick当前轮保存后中断，已在三gain coarse7完成。随后要求May小规模测试主流编码器在训练与推理中的影响，并指出整体更换encoder不能排除归一化之外的因素。报告54先固定HuBERT-base或wav2vec2-base各自权重切native输入z-score，训练0gain并交叉推理条件。最新用户要求训练与推理都改音量，因此本次完整执行2encoder×2trainnorm×3traingain×2infernorm×3drivergain=72格，每格两个固定10秒clip。内部LN/GN与encoder权重不改，评分始终原PCM16，以区分生成响应和评分器对音量的响应。

八新±6dB模型各12000成功FP32更新、0skip，共96000formal；四训练0dB模型从报告54按758成员全SHA原件导入，无重训/重渲染/重评分。共12模型、144矩阵，其中新96矩阵/24000帧，旧48矩阵/12000帧有原独立QA。共同完整随机generator initial文件SHAf9cd6cfadbc72812507e293e660b27e74a61f0556373162c487ffca6f5148e7b；seed42、4096rays/maxsteps16、原AdamW和60k LR前缀、固定EMA12 final，不按test选音量、checkpoint或预算。真实入口16success/0skip不计formal、不warmstart。

## Working State
权威持久入口F:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019/final_delivery_receipt.json。完成于2026-10-19T17:49:56Z（客户端Asia/Shanghai为October20）。1437成员、6物理根、5749007608逻辑bytes逐个两次freshSHA；manifest SHA c065b49686f82440b65361f1bd60485542ba6fa12f8390863f645e313c1e83bc，最终report SHA f20d745f2fa686ea12411cf0ea3a665f4fec676d23fff3106dec15bf46d84cf1。新F/E科学包没有声称云备份。全部新fullcp保留，当前无本试验活动GPU jobs；不要重跑任何launcher。

最终报告tts-exp/docs/experiments/55-may-gain-normalization-factorial；实验May训练音量与推理音量的同权重归一化交叉对照concluded/SUCCEEDED/ENGINEERING_PILOT_ONLY；Task运行May训练推理音量与归一化交叉对照done/current_step3，close_date按客户端October20。final_memory_publication_receipt与final_semantic_publication_receipt读回报告/实验/Task/地图/板/索引和实际结论；三图镜像在basic-memory/docs/experiments/assets/55-may-gain-normalization-factorial/。CONTEXT99行/HANDOFF56行当前段已更新，保留其他任务。此Task完成不代表原24full或旧Patrick完成，不归档chat。

实际结果：固定生成器内跨±6/0推理gain，输入norm开启时最大C极差HuBERT1.9550e−5、wav2vec2 1.8597e−5；关闭时各0.004305/0.030837。嘴部MSE推理gain相对极差关闭最高约0.225%/0.470%，开启近数值精度。稳定性不是质量收益：固定推理0dB的六组匹配流程on/on−off/off点估计中，5/6嘴部MSE上升1.21%–3.42%，1/6下降0.84%，C与原时钟D0方向混合。训练gain没有跨encoder一致单调规律，不能选最佳音量或判普遍有害/有益。

输入norm是各split完整音频的native零均值/单位方差后再固定chunk，不是RMS0.1。归一化后各训练gain特征仅近似同，relativeL2约1e−6，存在浮点与CUDA优化轨迹差异；单May、单训练seed、12000步短预算，无人类感知端点，不能把训练组差异全归于音量。旧AVE60k同两clip C约8–9、新约1.6–1.8，预算/表征/入口同时不同，只描述参照，不能归因norm。旧Patrick/RMS0.1/FP16密度Inf的间接训练因果仍未解决。

恢复资产：主要OUT在F，四HuBERT新媒体及最后wav2vec2_base_off_trainp6媒体物理在E:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019_media/。末组F原逻辑路径为junction，media_placement_handoff_v1和media_placement_final_readback给精确映射；最终manifest按resolve后的六根存放，避免重复计数。读取原protocol路径需保留该junction或按manifest物理根恢复。其它三wav2vec媒体F。源代码execution_sources（冻结GPU）、execution_sources_final（最终消费者/封存）、analysis/report_finalization_source_v1.py及report_interpretation_source_v1.py均纳入最终manifest。

原主机故障完整保留：第六训练job仍运行时SSH客户端先退，原host误复制增长stdout被SHA拒绝，底层传输原因未证明。revision02独立核前五完整state、原第六PID/startticks/trace及未运行后两组后，只观察原第六到自然12000成功rc0，再从未运行的第七/八组继续，0科学重放。active_controller/active_workflow指向controller_revision02_remote_completion_handoff_v1；completed_job_locations.json把第六指向已完成不可变readback，原root jobs第六部分副本及root training_failure/workflow_failure是历史。GPU train/runtime/evaluate和protocol数学未改。最终final_delivery_receipt优先于中间workflow状态和历史failures。

存储不是唯一副本删除。旧报告50的完整云包9分片/archiveSHA32f6b648e2236693e97a7eb1ea0ce483f80ea30a68cb2508b6893fb090278b22及全部643成员曾由独立Linuxfresh读回验证。为并行归档造成的实际空间压力，按既有用户授权分批退休本地18render、36crops缓存及87个step2000..58000中间cp副本；旧三个step60000最终cp、全部矩阵/embeddings/track/JSON/PNG仍本地。旧F根synctalk_may_training_pilot_20261017/的四locator：render_cold_storage_locator_20261019.json、crop_cold_storage_locator_20261020.json、early_checkpoint_cold_storage_locator_20261020.json、remaining_intermediate_checkpoint_cold_storage_locator_20261020.json；每项原bytes/SHA/archive member/9parts完整，可选member恢复到新目录并SHA核验。报告54全部758成员仍物理保持，本次所有cp保持，不动Patrick或共享venv/CloudDrive配置。无损NTFS属性压缩前后逻辑SHA保持；卷free同时受其它任务影响，不能当压缩节省量。

三次可选C清理/迁移（旧Edge编译缓存、公开HF缓存迁移、单一旧设备临时log压缩备份清理）均被自动审批在执行前拒绝，唯一理由blocked by policy；0执行、不用另一个工具绕过。通过F/E布局和已完整验证云包的旧本地副本冷存储完成当前任务，C空间自行波动回升。一次后续旧cp退休准备命令PowerShell转义SyntaxError发生在新脚本创建前，0删除/0GPU，随后固定69项源正常执行；所有intent/receipt/source保留。没有真实训练数值失败。

机器易失远端Ubuntu wjj@192.168.50.2，SSH必须-b192.168.50.1；/dev/shm/may-encoder-gain-norm-factorial-20261019依赖parent SHM的official/data/historical只读链接，不将SHM当持久备份、不随意清parent。原GPU锁仍/dev/shm/synctalk-multispeaker-data-20261017/preparation_GPU.lock。工作流已结束，session/PID不能作为重启依据。

旧Patrick user_pause_v1/paused.json SHA545c965703b38a42058fc5c3a65a4e662722470a306a37ca7ea9f97e1a8ec090仍fresh精确：21/30轮、三gain coarse7各42000attempts，实际success41991/41990/41991，全cp保留；旧fine14/Chris/Cathy/24full继续暂停，旧巡检已删除，不重建。本次新实验没有恢复旧训练。

## Changed Files
源码清单已实际枚举：18个Python模块，含两个wrapper。上一同chat交付checkpoint写成19个是数量笔误；其科学结果、封存SHA与交付状态均不变，本checkpoint给出校正后的完整恢复状态。

未提交scripts/experiments/may_encoder_gain_norm_factorial_20261019/（18个Python模块（其中包括两个wrapper）、训练/特征/独立消费/绘图/报告/封存/主机完成权威）；tests/experiments/test_may_encoder_gain_norm_factorial_20261019.py；CONTEXT.md与HANDOFF.md。实际源码由F最终capsule绑定，机器局部.codex/scratch辅助脚本不是已推送Git提交。大量其他dirty保持，不bulk stage/clean/commit/push。

## Verification
4有意义CPU tests实际passed；主机handoff3CPU拒绝/完成权威检查passed；28缓存条件shape/finite/SHA/独立processor公式、16真实入口更新/fullstate往返通过。正式八模型各12000成功0skip，CPU model/optimizer/gradient/EMA/mean_density全部finite、optimizer/scheduler step与EMA12正确，所有实际frame/source/ray/LR逐行一致。原四0gain fullstates与758成员whole import SHA精确。

全部96新无损FFV1/BGR0+PCM16视频、24000帧时钟误差0。独立CPU重建729120距离，最大误差3.892246841274982e−6；RGB/原PCM/GT-ownedROI/crop/selectedfloat/量化界全通过，新评分audio embeddings与报告54逐位同，合计144矩阵。三张真实PNG实际view_image并记录具体观察；common blur只工程观看不作感知排序。跨F/E1437成员fresh双SHA、父758成员再次全读回、无remote科学jobs、Patrickpause SHA精确；最终报告和六BM笔记及科学结论读回。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)；当前HEAD若GitHub不能resolve则为本地未推送，不构造commit链接。
- [报告55](../../docs/experiments/55-may-gain-normalization-factorial.md)，F final_delivery_receipt与manifest是恢复入口。
- 前一报告tts-exp/docs/experiments/54-may-encoder-normalization-pilot及旧报告50同clip表只作相应范围参照。

## Observations
- [result] 新May音量×native norm完整72格/144矩阵已交付，8新增×12000成功更新、0skip，4旧0gain原件导入，跨F/E1437成员读回passed，无活动科学jobs。
- [result] 输入norm将固定生成器的推理gain响应压到约2e−5；匹配质量指标混合，5/6嘴部MSE略升，不等稳定性带来质量收益。
- [decision] 固定encoder权重识别模块开关，训练与推理gain完整交叉；单训练seed/短预算不判总体因果、最佳音量或感知。
- [decision] 本MayTask done不代表原Patrick/24full done，旧训练与巡检不恢复，不归档chat；新全部cp保持。
- [next_step] 从报告55和final_delivery_receipt审阅本次结果；只有用户另授权才建立多训练seed/充分收敛验证，不自动扩展。

最终封存JSON：{"passed": true, "completed_utc": "2026-10-19T17:49:56.693456+00:00", "roots": 6, "members": 1437, "new_persistent_bytes": 5749007608, "manifest_sha256": "c065b49686f82440b65361f1bd60485542ba6fa12f8390863f645e313c1e83bc", "report_sha256": "f20d745f2fa686ea12411cf0ea3a665f4fec676d23fff3106dec15bf46d84cf1", "visual_review_sha256": "97595bde999ce3af605cd6d6a856e1d03830e3fdbefb1617017f361ebbfae780", "memory_publication_sha256": "643b68ee5524f68c2c50f12549279380de7c1d93de2d86f6de4dbbc34d52b3a8", "imported_parent_manifest_sha256": "5d94a98d200702c69f4150a8fb24f89795dbde3ded9973ee0d2e18969c1bd9c9", "new_training_models": 8, "imported_training_models": 4, "new_updates": 96000, "total_updates_including_imports": 144000, "cells": 72, "total_score_matrices": 144, "new_score_matrices": 96, "old_zero_gain_replayed": false, "no_active_scientific_jobs": true, "old_patrick_still_paused": true, "single_seed_pilot": true, "perceptual_claim": false, "new_science_cloud_backup_claimed": false}
GitHub只读证据：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}

## Relations
- continues [[Codex checkpoint - 2026-10-19T17-51-29Z - May gain factorial delivered]]
- relates_to [[运行May训练推理音量与归一化交叉对照]]
- relates_to [[May训练音量与推理音量的同权重归一化交叉对照]]
- relates_to [[55-may-gain-normalization-factorial]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1066条；大量其他任务改动，不批量stage/clean/commit。
