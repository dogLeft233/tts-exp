---
title: Codex checkpoint - 2026-10-19T14-33-42Z - May normalization pilot delivered
  with same-encoder controls
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t14-33-42-z-may-normalization-pilot-delivered-with-same-encoder-controls
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T14:33:42.681819+00:00'
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

# Codex checkpoint - 2026-10-19T14-33-42Z - May normalization pilot delivered with same-encoder controls

## Summary
May同权重归一化训练/推理先导已完成48000实际更新、48视频/48评分矩阵、独立QA与三图实际检查，正式报告54与Task收口，全部科学资产F盘758成员fresh封存，Patrick仍暂停。

## Story
用户先要求Patrick当前轮保存后中断，已在三档coarse7完成；随后要求用May小规模测试主流编码器在训练和推理中是否有害并与旧实验对比。用户指出整体换编码器会混入归一化之外的因素。因此主对照固定HuBERT-base或wav2vec2-base各自预训练权重，只切HF native do_normalize，内部LN/GN不动。训练on/off从共同完整随机seed42 generator initial起跑，再交叉两种推理预处理、±6/0gain；不把跨encoder或旧AVE差异归因归一化。

固定四臂各12000成功FP32更新、4096rays/maxsteps16、原AdamW与60k LR前缀、EMA12 final，不按test选模型/预算。May4150/550/1100原帧划分，test预定test_00/test_03各250帧；24条件48无损FFV1/BGR0+PCM16 clip，评分音频始终原PCM。旧May60000更新AVE同两clip重聚合，只描述参照，不能归因归一化。此次native z-score不是RMS0.1，也不是旧Patrick FP16，不能解释其density Inf间接训练因果。

## Working State
权威持久入口F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/final_delivery_receipt.json passed，2026-10-19T14:32:19.861505+00:00；758成员3636776287bytes，delivery_manifest SHA5d94a98d200702c69f4150a8fb24f89795dbde3ded9973ee0d2e18969c1bd9c9。final receipt优先旧workflow中间状态。无活动本试验训练/评测进程；不要重启任何launcher，不自动扩预算/第二seed。

protocol SHA1061e46bd621136b13dc2465420b72aaabf420bfde474a95584e446a8a0b3dbe；共同完整随机initial文件SHaf9cd6cfadbc72812507e293e660b27e74a61f0556373162c487ffca6f5148e7b。training_independent_review实际CPU fullstate检查4模型model/optimizer/gradient/EMA finite、optimizer step12000、scheduler/LR exact、EMA12，trace四臂采样/ray/LR逐行一致且0skip。所有fullcp保留。training和五个evaluate jobs原始rc0，0科学重放。

独立analysis/qa.json passed：12000帧RGB/PCM/ROI hash与selected float原值/量化界、48矩阵364560距离，float64重算最大误差3.7889890442954766e-6。media_clock_review48视频/12000帧原25fps时钟误差0；historical_audio_feature_review两测试clip SyncNet audio embedding与旧May逐位一致。三PNG均本chat实际view_image；visual_review指出曲线/标签可读，固定帧共同模糊/颈部接缝只是工程观看，不是盲评。H264 comparison preview只供观看，不参与评分。

0dB匹配流程on/on−off/off：HuBERT ΔC+0.029、ΔD0−0.049、嘴部MSE+1.38%；wav2vec2 ΔC−0.027、ΔD0+0.069、嘴部MSE+1.68%。条件内独立训练/推理差异见analysis/contrasts.csv；联合匹配流程差异不能冒称单独训练效应。推理norm on时四生成器±6/0 C极差<0.000015，off约0.001–0.014，说明音量不敏感传播到输出，不代表质量更好。四模型新C约1.6，旧AVE60k同clip约8–9，预算/表征/入口不同不能归因归一化。单身份单seed短预算不排除CUDA训练随机性或预算依赖，不推总体有害/有益与感知结论。

正式报告tts-exp/docs/experiments/54-may-encoder-normalization-pilot；实验tts-exp/experiments/may-同权重编码器归一化训练与推理小规模对照 concluded/SUCCEEDED/ENGINEERING_PILOT_ONLY；Task tts-exp/tasks/运行-may-同权重编码器归一化训练推理对照 done/current_step3。研究地图只更新本May段，TaskBoard去掉该完成行，实验README增54，BM6笔记读回与3图镜像SHA见final_memory_publication_receipt.json。CONTEXT99行/HANDOFF56行追加本次终局同时保留其他任务，before/afterSHA在entry_documents_final_readback.json。旧总目标未完成。

机器易失副本Ubuntu wjj@192.168.50.2，SSH -b 192.168.50.1，/dev/shm/may-encoder-norm-pilot-20261019。SHM不是持久备份，此758科学包仅F持久，未声称cloud备份。共享venv/CloudDrive及旧科学资产不动；新公共weight bodies在本F包保留。C约300MiB/F约1.9GiB，若扩实验先核实际空间。之前只清60过期npm/pip下载cache179846604逻辑bytes以恢复BM SQLite，原receipt保留，不删科学数据。

原Patrick user_pause_v1/paused.json SHA545c965703b38a42058fc5c3a65a4e662722470a306a37ca7ea9f97e1a8ec090仍exact；三gain coarse7各42000attempts，成功41991/41990/41991，21/30，全部fullcp保留。patrick-rms巡检已真实delete，旧fine14/Chris/Cathy及原24full仍暂停，不归档chat。旧RMS真实数值失败和强制恢复probe116全skip/0更新封存，不恢复旧controller。

## Changed Files
未提交脚本在scripts/experiments/may_encoder_norm_pilot_20261019/：protocol/features/runtime/train/controller/evaluate/postprocess_controller/analysis/review_training/review_features/review_media_clock/report/make_preview/publish_final/seal_delivery及两wrapper。实际冻结train SHA1f1017539e312223104abc17047c03db557bb84491a69cae49512d324d8b8f3e，evaluate SHA7856b30d34503090b2255c0c22aaf206e4d8ee910a6067fe6a61a9117f833d1e；执行原源和最终消费者源在F两capsule。tests/experiments/test_may_encoder_norm_pilot_20261019.py与CONTEXT/HANDOFF变更。大量无关dirty未stage/commit/push，不bulk clean。

## Verification
4 CPU tests实际passed；20features的shape/finite/SHA/时钟与独立processor公式20/20；入口4×2success/0skip/fullstate reload，正式48000更新与所有48score/媒体独立QA通过。三图实际view_image，BM最终报告/实验/Task/地图/板/索引全读回，F758成员fresh SHA逐个复核。最终remote ps无本试验科学jobs、Patrick pauseSHA exact。

首qualification shell CRLF/路径import失败发生model构造前0updates，attempt02修复LF/PYTHONPATH和资格重载顺序，原失败保留。最后score科学job rc0之后tar149成员复制全后主机SSH收尾未退出；先独立fresh149成员全部SHA，再只停止该主机SSH，原postprocess/workflow failure保留，score_transfer_close_repair_v1重新封口原5科学rc0与Freadback，0GPU重放，原host源不改。根workflow_receipt中间pending不是实际未交付，最终final_delivery_receipt优先。

可选preview首次PIL import缺失在0preview帧前失败，加入隔离plotdeps后正常250帧H264，仅观看；原失败源/receipt保留。sealer首次GBK读取UTF8 BM receipt失败在capsule构造前，-X utf8同源成功，无科学重算。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)，HEAD为本地未推送时不构造commit链接；只读GitHub证据下附。
- [报告54](../../docs/experiments/54-may-encoder-normalization-pilot.md)与F根final_delivery_receipt是恢复入口；本checkpoint不是原始数据替代。

## Observations
- [result] 固定同权重输入归一化的May训练/推理先导已全部完成并发布报告54，758成员fresh封存，无活动科学jobs。
- [decision] 模块效应采用同encoder、同初始化和采样的开关对照及训练×推理交叉；整体更换encoder不能识别归一化因果。
- [result] 当前小样本结果混合，不能说归一化普遍有害；推理norm on显著压低gain敏感性，但不等于生成或感知获益。
- [decision] 本任务done不代表旧Patrick coarse10/new27或原24full完成；旧任务与巡检不恢复。
- [next_step] 若用户继续科学验证，先读报告54与固定预算边界，另冻结等预算多训练seed对照以估计训练随机性；当前不自动启动扩展。

GitHub只读证据：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}

## Relations
- continues [[Codex checkpoint - 2026-10-19T14-24-21Z - May normalization crossed pilot training and scoring complete]]
- relates_to [[运行Patrick的RMS归一化AVE三档粗训练并评测]]
- relates_to [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- relates_to [[048-synctalk-patrick-same-ave-rms-coarse]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1064条；大量其他任务改动，不批量stage/clean/commit。
