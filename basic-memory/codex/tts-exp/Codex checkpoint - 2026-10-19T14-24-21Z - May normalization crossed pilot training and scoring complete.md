---
title: Codex checkpoint - 2026-10-19T14-24-21Z - May normalization crossed pilot training
  and scoring complete
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t14-24-21-z-may-normalization-crossed-pilot-training-and-scoring-complete
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T14:24:21.999865+00:00'
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

# Codex checkpoint - 2026-10-19T14-24-21Z - May normalization crossed pilot training and scoring complete

## Summary
May同权重归一化对照已完成四组12000次FP32更新、48个无损视频与48个SyncNet矩阵；独立评分QA、实际图检和最终发布仍待完成。

## Story
用户先要求Patrick当前轮保存并暂停，已在三gain coarse7全部断点fresh核验后完成，patrick-rms巡检真实删除。最新目标是用May测试主流编码器在训练/推理中的归一化是否有害，并与旧结果比较。用户指出整个换编码器不能排除其他影响；因此主比较固定同一HuBERT-base或wav2vec2-base权重，只切换native processor do_normalize，训练on/off与推理on/off交叉。旧AVE仅历史描述，不能归因归一化。

共同seed42完整随机generator initial，无旧训练warmstart；两编码器各两臂12000成功FP32更新、4096rays/maxsteps16，原AdamW/LR前缀，固定final EMA12。训练数据May原4150帧，验证550帧固定16行，test预定test_00/test_03各250帧；推理-6/0/+6三gain，共24条件48clip。native z-score不是Patrick RMS0.1，不能用本试验解释旧FP16数值失败。

## Working State
持久OUT=F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019，源scripts/experiments/may_encoder_norm_pilot_20261019/。protocol SHA1061e46bd621136b13dc2465420b72aaabf420bfde474a95584e446a8a0b3dbe。共同initial file SHAf9cd6cfadbc72812507e293e660b27e74a61f0556373162c487ffca6f5148e7b。training_complete与training_independent_review均passed：48000 formal成功、0skip、model/optimizer/EMA/梯度finite，四臂frames/rays/LR逐行相同。所有fullcp保留，固定final不是挑best。

四evaluate render jobs真实rc0，48个FFV1/BGR0+PCM16视频和selected float/PNG F全成员fresh读回。remote score job171107 rc0，48matrix elapsed128.292秒。F active_postprocess仍score，Windows postcontroller262152与one-shot workflow268680（unified56430）尚在读回阶段；workflow_receipt旧training字段不能当当前训练状态。尚未看到analysis/qa.json，不能提前报告数值或称交付完成。最新实际产物始终优先此快照。remote media_clock_review.json已实际passed48video/12000frames/零时钟误差，尚须二进制复制到F并freshSHA。remote=/dev/shm/may-encoder-norm-pilot-20261019是易失副本，Ubuntu wjj@192.168.50.2，SSH -b 192.168.50.1。绝不重启任何已完成GPU stage。

独立analysis.py将核48矩阵float64距离、原PCM、12000crop hashes、selected float/像素量化界，再输出cells/contrasts/historical_same_clips/qa。等待one-shot完成，若失败保留具体failure并只修可证明的host/数据消费者问题，不降低科学门或盲重放GPU。report.py尚未执行，随后生成三图，必须view_image实际看图，保存visual_review，最终报告替换pending段。publish_final.py须先核编号54仍空闲，发布新实验/Taskdone/地图当前May段，保留Patrick暂停和其他项目段；seal_delivery.py在BM/clock/visual全通过后制作完整Fmanifest和final_delivery_receipt。H264 make_preview只观看，不参与score。

旧Patrick pause authority仍user_pause_v1/paused.json SHA545c965703b38a42058fc5c3a65a4e662722470a306a37ca7ea9f97e1a8ec090，三gain各42000attempts，成功41991/41990/41991，21/30；coarse8/10/new27未做，旧fine14/Chris/Cathy与24full仍暂停，不归档chat。旧数值失败/强制恢复probe116保持封存。

## Changed Files
新增未提交scripts/experiments/may_encoder_norm_pilot_20261019/{protocol,features,runtime,train,controller,evaluate,postprocess_controller,analysis,report,review_training,review_features,review_media_clock,publish_final,make_preview,seal_delivery}.py及两wrapper、tests/experiments/test_may_encoder_norm_pilot_20261019.py。scratch部署/发布/oneshot helper和全部执行源在F capsule留存。train SHA1f1017539e312223104abc17047c03db557bb84491a69cae49512d324d8b8f3e，evaluate SHA7856b30d34503090b2255c0c22aaf206e4d8ee910a6067fe6a61a9117f833d1e。大量无关dirty文件，不批量stage/clean/commit。

## Verification
20feature数组shape/finite/SHA/clock、独立processor公式20/20、四臂入口8success/0skip/fullstate往返passed；首次CRLF/PYTHONPATH准备失败0model/0update，原失败保留后LF/依赖repair02完成。四formal fullstates CPU实际加载finite/counters/LR/EMA与48000行采样账通过；四render jobs和score job rc0。4 CPU tests实际passed。原May官方源四SHA精确，原17403支持成员fresh恢复、88 frozen encoder chunk forwards。无原Patrick重放/原科学资产删除/sharedvenv或CloudDrive修改。

C满盘曾阻塞BM SQLite，按人类空间授权仅清60个过期npm/pip下载cache179846604逻辑bytes，固定containment/SHA/删除receipt在OUT/disposable_package_cache_cleanup.json。未删科学资产，BM恢复读写passed；当前C约340MB/F约2GB，继续留意实际空间。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)，GitHub只读证据下附，HEAD未远端解析不造commit链接。
- canonical旧报告50和53；新BM experiment tts-exp/experiments/may同权重编码器归一化训练与推理小规模对照、Task tts-exp/tasks/运行may同权重编码器归一化训练推理对照。当前RENDERING/current_step3不是最终结论。

## Observations
- [result] 四组formal训练48000更新和48无损片段/评分矩阵真实完成；最终独立数值QA与图检/发布尚未完成。
- [decision] 只在同编码器固定权重内切输入归一化，训练与推理开关交叉；跨编码器或旧AVE差异不能归因归一化。
- [decision] Sync-C/距离/原时钟偏移/嘴部误差分别解释，单身份单seed小预算不能推出总体有害或感知改善。
- [next_step] 核当前postprocess/analysis真实完成或failure，完成CPU数值QA、实际view_image、最终BM报告与F交付manifest，再给用户结果；不恢复Patrick。

GitHub只读API证据：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}

## Relations
- continues [[Codex checkpoint - 2026-10-19T12-55-50Z - Patrick paused at coarse7 and May normalization comparison scoped]]
- relates_to [[运行Patrick的RMS归一化AVE三档粗训练并评测]]
- relates_to [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- relates_to [[048-synctalk-patrick-same-ave-rms-coarse]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1064条；大量其他任务改动，不批量stage/clean/commit。
