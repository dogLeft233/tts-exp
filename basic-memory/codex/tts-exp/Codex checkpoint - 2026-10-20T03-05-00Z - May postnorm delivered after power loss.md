---
title: Codex checkpoint - 2026-10-20T03-05-00Z - May postnorm delivered after power
  loss
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t03-05-00-z-may-postnorm-delivered-after-power-loss
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T03:05:00.611936+00:00'
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

# Codex checkpoint - 2026-10-20T03-05-00Z - May postnorm delivered after power loss

## Summary
May后归一化低尺度训练/高尺度驱动12格实验已完整完成并交付报告56；断电发生在全部科学作业完成之后，F持久包完整，无需恢复训练或重评分。

## Story
用户此前要求在同HuBERT/wav2vec2前端z-score之后调输入尺度，测试低于标准化水平训练、高于标准化水平推理。本chat已从同seed42完整random generator initial新增两个scale0.5模型，各12000成功FP32更新/0skip；scale1标准模型直接导入报告54同initial/预算/数据/采样/EMA12。每模型交叉推理scale0.5/1/2，共12条件24clip矩阵，20新增/4原标准导入，原自然PCM16/原时钟/GT ROI/SyncNet FP32 TF32off。不是原录音相对gain、不是RMS0.1，也没有换encoder或内部norm。

最新用户问“刚才断电了，能继续吗”。只读发现Ubuntu已重启（boot ecb6271c-7289-4ba3-8605-e9320174df6a），全部SHM依赖已清空；但九个科学job均在重启前自然rc0，最后score_all于2026-10-20T02:18:25Z（北京时间10:18）完成，Windows已保存所有结果与numerical_delivery_ready候选。故没有恢复GPU/重启controller/回滚checkpoint，而是直接从F持久产物做读回、实际看三图、发布结果、关闭本Task与封存。不能把断电当数值训练失败，更不能重跑已完成模型。

## Working State
权威F:/Datasets/tts-exp/may_postnorm_scale_20261020/final_delivery_receipt.json，passed，完成2026-10-20T03:03:07Z；381成员/1406746409逻辑bytes，全部新完整断点/trace/特征/视频/矩阵/源码和报告保留。manifestSHA26e3bdf41c7119a3d01accdd8ff69675b520f3347272ee1d092fc95b5ea06265，最终reportSHA1ae81e758ad6af66b4686f15ccf3c1f38400c2a8cedd6c2e6dbc1128751627b7。**已封存ROOT，不再运行会写入ROOT的旧poll、run、deploy、analysis或任何launcher；checkpoint读回只能写.codex/scratch，不往封存ROOT新增文件。** 新科学包未声称云备份。

报告tts-exp/docs/experiments/56-may-postnorm-scale，物理basic-memory/docs/experiments/56-may-postnorm-scale.md，三图镜像在assets/56-may-postnorm-scale。实验May归一化后低尺度训练与高尺度驱动对照concluded/SUCCEEDED/ENGINEERING_PILOT_ONLY，Task运行May归一化后输入尺度训练推理对照done/current_step3，报告/实验/Task/地图/TaskBoard/索引六笔记均读回。CONTEXT99行/HANDOFF60行当前段更新，旧任务尾部保持。旧active_controller已单调改为completed_verified_delivery，原candidate/workflow_pending记录是历史，最终receipt优先；powercut_review_v1/before_final_active_controller.json保存旧快照。

结果：标准训练/标准推理→低训练/标准推理→低训练/高推理的C，HuBERT1.695093→1.696572→1.696581；wav2vec2 1.625923→1.718407→1.719414。低/高相对标准，HuBERT ΔC+0.0014877、D0+0.125814（恶化）、嘴部MSE−1.08974%；wav2vec2 ΔC+0.0934916、D0−0.0510759、嘴部MSE−1.65833%。固定低训练模型将推理scale1提高到2，C额外变化仅HuBERT8.583e−6/wav2vec2 0.00100756；标准wav2vec2模型提高驱动时C反而约−0.000164。训练×高驱动C交互为2.861e−6/0.00117159。主要可见差异在两个训练模型之间，不能由单seed推稳定因果、普遍协同收益、最佳尺度或感知改善。

输入尺度是native float32完整split z-score后再乘0.5/1/2，实际训练RMS0.499989868/0.999979736、test高1.999943216；无二次norm/clip/limiter，encoder权重冻结、内部LN/GN保持。输入峰值可超过1，属于内部浮点表示，不是播放LUFS。共同完整initialSHAf9cd6cfadbc72812507e293e660b27e74a61f0556373162c487ffca6f5148e7b、原AdamW/60k LR前缀/4096rays/maxsteps16/FP32/12000更新/EMA12、4150/550/1100监督帧及固定两个10s clip保持。训练组只有一个seed，充分收敛和感知均未证明，报告55归一化之前gain与本次之后尺度不可混作同dose；旧AVE60k的编码器/预算差异也不归因此模块。

恢复指针：protocolSHA0110e9fd52fb4f6fe8c0133e191095b4037a73cd46e8f8496c1672e99513ec08；17冻结Python源execution_sources与sourcepins全SHA，科学源未改。新两个finalcpSHA分别c0894a6e10f70b00cb6f8b9cecfb4c9129cedce807a18090c045a88e8b26917f和f30ddf01278e8725508cf29eb22956049b4b663911ff340fccc2a4337c8b0b3e。标准父报告54整758成员保持，parentmanifestSHA5d94a98d200702c69f4150a8fb24f89795dbde3ded9973ee0d2e18969c1bd9c9；其最终包仍本地原件。新全包在F，未依赖已消失的remote内存盘作为持久证据。

已无本任务活动GPU或host；Ubuntu GPU仅桌面约322MiB。不要恢复整个SHM工具链或重建作业来“接着跑”。将来若新用户方向确实要求新GPU，再按所有已持久source/data/运行环境pin另建恢复命名空间并资格，不覆写封存包/旧控制器。SSH在PowerShell必须分开参数-b 192.168.50.1避免解析异常，不能把venv lexical路径resolve成MFA底层解释器，不能修改共享venv/CloudDrive配置。本轮未做清理、迁移或云上传。

旧Patrick user_pause_v1/paused.json SHA545c965703b38a42058fc5c3a65a4e662722470a306a37ca7ea9f97e1a8ec090仍fresh精确，coarse7三档/旧fine14/Chris/Cathy/原24full继续暂停；旧巡检已删除，没有新自动化。本新Task完成不等原24full/fine17完成，不归档chat，不无授权联系其他chat。

## Changed Files
本轮没有修改17冻结GPU/训练/分析源，也没有新增GPU工作。新增正式报告56及3图镜像，更新原实验/Task/地图/索引/板与CONTEXT/HANDOFF当前头部。封存F内powercut_review_v1、host_finalization_sources、analysis/visual_review、interpretation_summary、report_finalization_readback、report_candidate_before_visual、scientific_manifest与最终manifest/receipt记录实际审核；保留原candidate报告与raw收据，不把文档解释当新科学结果。机器局部.codex/scratch有review/finalize/publish/seal/checkpoint脚本；所有其它dirty未stage/clean/commit/push。

## Verification
本轮真实read-only匹配17执行源码/初始和14缓存SHA，18份原F成员清单共239成员1170648504 bytes重新fresh读回；四finalcp/trace SHA与12000成功/0skip/有限loss、完整frame/source/ray/LR账目核验；父758成员再次全SHA。151900个新距离从已保存视觉/原音频嵌入独立float64重建，最大误差3.7701486945707074e−6。没有新GPU、optimizer或CPU全tensor加载，之前已通过的fullstate finite证明结合本次原字节SHA保留；没有声称此次重复构造模型。

前序实际执行的4CPU tests、14缓存独立公式/输入RMS/特征shape/finite、4真实入口fullstate资格与两新各12000/0skip、5000帧RGB/PCM/ROI/crop/selectedfloat与20矩阵/全部FFV1 PCM时钟误差0的QA均已passed，原4矩阵以父SHA和原QA导入。本次实际view_image三图：热图可读且行内近似、曲线基本重合、固定帧共同模糊无可靠肉眼排序；这是工程观看不是盲评。visual reviewSHA7b39db3bdb8483d84f1a8ac6a9c5bfd7983f78c4914ff5bb1f7b296ecaa8c1bf。

最终381成员/1.406GB fresh双SHA，6BM笔记与3镜像读回，99/60行入口约束和旧Patrick pause精确。科学作业完成早于新boot_time，当前remote任务目录不存在且无科学进程。0重训/重评分/新GPU/删除。前序4CPU tests本轮不重复跑，因源码未变且字节fresh精确。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)；当前HEAD若远端不能resolve为本地未推送，不构造commit链接。
- [报告56](../../docs/experiments/56-may-postnorm-scale.md)
- F:/Datasets/tts-exp/may_postnorm_scale_20261020/final_delivery_receipt.json

## Observations
- [result] 12格/24矩阵低训练高驱动实验已完成并发布56；断电晚于全部科学作业，381成员F封存完整，0科学重放，无需恢复训练。
- [result] wav2vec2低/高相对标准C+0.093、D0−0.051、嘴部MSE−1.66%，但高驱动额外仅约+0.001；HuBERT指标混合，不支持跨encoder一致收益。
- [decision] 保持同encoder内部尺度干预、单seed边界和旧Patrick暂停，不自动扩展训练、重建SHM或重跑已完成作业；新封存包不再写入。
- [next_step] 从报告56和final_delivery_receipt审阅已完成结果，只有用户新方向才设计独立重复或新尺度验证，不再恢复本轮训练。

封存最终receipt：{"passed": true, "completed_utc": "2026-10-20T03:03:07.190168+00:00", "members": 381, "persistent_bytes": 1406746409, "manifest_sha256": "26e3bdf41c7119a3d01accdd8ff69675b520f3347272ee1d092fc95b5ea06265", "report_sha256": "1ae81e758ad6af66b4686f15ccf3c1f38400c2a8cedd6c2e6dbc1128751627b7", "visual_review_sha256": "7b39db3bdb8483d84f1a8ac6a9c5bfd7983f78c4914ff5bb1f7b296ecaa8c1bf", "memory_publication_sha256": "7078d1eb279774bc767e4e98dcb918ddb39f42f8aebbfe5a40bbbd4685c3eadd", "powercut_review_sha256": "3a247148e49b1e8fe6158ebf15440cfc83ae8ab84b0daa0021179dd8aeeaecd9", "parent_manifest_sha256": "5d94a98d200702c69f4150a8fb24f89795dbde3ded9973ee0d2e18969c1bd9c9", "new_training_models": 2, "imported_training_models": 2, "new_successful_updates": 24000, "total_updates_including_imports": 48000, "cells": 12, "total_score_matrices": 24, "new_score_matrices": 20, "new_video_frames": 5000, "imported_video_frames": 1000, "no_active_scientific_jobs": true, "power_loss_after_all_scientific_jobs_completed": true, "new_gpu_calls_after_power_loss": 0, "scientific_replays_after_power_loss": 0, "old_patrick_still_paused": true, "single_seed_pilot": true, "perceptual_claim": false, "new_science_cloud_backup_claimed": false}
GitHub只读核查：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 403}}

## Relations
- continues [[Codex checkpoint - 2026-10-20T01-51-05Z - May post-normalization scale training running]]
- relates_to [[运行May归一化后输入尺度训练推理对照]]
- relates_to [[May归一化后低尺度训练与高尺度驱动对照]]
- relates_to [[56-may-postnorm-scale]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1068条；大量其他任务改动，不批量stage/clean/commit。
