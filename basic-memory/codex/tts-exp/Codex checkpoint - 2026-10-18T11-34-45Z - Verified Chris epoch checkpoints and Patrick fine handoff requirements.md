---
title: Codex checkpoint - 2026-10-18T11-34-45Z - Verified Chris epoch checkpoints
  and Patrick fine handoff requirements
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t11-34-45-z-verified-chris-epoch-checkpoints-and-patrick-fine-handoff-requirements
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 11:34:45.462555+00:00
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

# Codex checkpoint - 2026-10-18T11-34-45Z - Verified Chris epoch checkpoints and Patrick fine handoff requirements

## Summary
Chris和Patrick的六份完整epoch断点已逐SHA核验；安全暂停后转Patrick精调技术上可实现，但当前试跑入口不允许fine，调度交接尚未实现或执行。

## Story
原目标是四身份两seed三训练档的24完整模型及训练/驱动/评分27格。Brown原FP16失败后，用户授权保持FP16、另立更新前guard和上游溢出核查，并先试Patrick/Chris/Cathy。Patrick seed42三档coarse10及27格描述性预览已交付，实际更新数59985/59985/59984不等，CONTROL_FAILED保持，非fine17。Chris随后恢复完整16500监督并开始训练。

用户最近询问SyncNet音频是否归一化，已依据本地coarse_preview_v2.py和官方SyncNetInstance.py/python_speech_features源码说明：原PCM16直接MFCC，±6评分为float64原PCM单位增益，无逐段RMS/峰值/响度归一化；默认MFCC保留log-energy，非音量不变。没有为这次解释新增数值实验。训练和测试音量相反的可能机制仍属假说；单身份单seed粗阶段且更新数不等，不能据分数确定因果或感知改善。

最新用户问当前说话人是否有断点、能否暂停并启动原说话人精调。此次只读核查确认Chris三档ep1和Patrick三档ep10完整pth均存在且fresh SHA与epoch_receipt完全一致；每份原verification passed。train_other_identity_epoch_v1.py保存full checkpoint，恢复验证model/optimizer/lr_scheduler/scaler/EMA、epoch/global_step，并从receipt恢复model counters/stats及lip mode。每轮6000尝试后保存，不是每一步的无损暂停；不能杀当前epoch后声称可从任意步续训。当前V3无通用pause检查，启动拒绝已有receipt及退休identity，不能盲重启V3。

精调在本trial入口第14行显式被phase==coarse限制；遗留fine分支另要求coarse qualification passed，Patrick该门因更新数不等不能通过。正确后续若用户决定切换，应另立版本，先验证完整epoch边界handoff/逐receipt恢复不重放，再建立保持FP16和原更新数控制失败的Patrick三档描述性fine入口，恢复已退休监督及容量资格，保留原coarse及27格。此次用户为能力询问，没有执行暂停、启动fine、杀GPU或改训练源。

## Working State
ROOT=/dev/shm/synctalk-multispeaker-data-20261017；FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；OUT=FROOT/trials/fp16_other_identities_trial_v1；SSH wjj@192.168.50.2。机器易变状态：最新本地receipt摘录 {"pid": 181520, "status": "training", "current": {"identity": "ChrisVanHollen", "seed": 42, "epoch": 2, "gain_db": 0, "relative": "training/ChrisVanHollen/seed42/gain_+0/coarse/ep0002/attempt01"}, "remote_job": {"state": "running", "pid": 814187, "command_sha256": "145f18488995a8ce5578176c9c90e95650767f80df309bb423f294a9c1375616", "passed": false, "started_unix": 1792322511.9638941, "child_pid": 814194}, "failures": []}。

此次实际remote ps814187/814194仍运行Chris0dB ep2，从ep1完整state恢复；一次实时trace global11481、成功11481、0skip、loss有限，只是前缀。Chris其余两档ep1各6000成功，Cathy未开。真实receipts/ps优先于此快照。heartbeat synctalk-27保持ACTIVE，正常健康运行不通知，原全目标不done或归档。

权威交付OUT/early_score_preview_v2/delivery_v1/report.md、scores_27_cells.csv、mean_distance_curves.csv、patrick_27_grid.png、independent_review.json。all27_preview SHA0c727888b1fbeb09e573ab26bc7129d453aa03b45de345b460d66973ac6fac1a，delivery receipt SHA9533b56022e7436774bdcb4d8e80c3d80144d477e03919644b569c1b09ee7f92。10:39:55UTC V3 preview_delivery_release已写，不重复写。预览176764正常退出，不重启。

Patrick监督现仅72controls，train/val/test非control缓存按F全验与lease退休，未来精调必须显式full恢复；Chris当前完整16500监督。216份remote评分npy缓存已退，F原数组与归档保留。旧lossless电影在cloud locators，显式restore后才可加载。空间以真实df为准，未执行新清理。Brown原archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72及所有旧false/原版本保持，105372/133984/610686不重启。guard不是eye/sigma上游增长的完全根因修复。

## Changed Files
本轮只新增本地未提交checkpoint辅助脚本.codex/scratch/checkpoint_resume_finetune_capability_20261018.py及此不可变BM handoff；未改controller、epoch入口、实验合同、数据或模型。工作树1046项dirty含大量既有工作，未批量stage/clean。

## Verification
CodeGraph读取current V3完整源、train_other_identity_epoch_v1.py、train_epoch.py、execution_state.py以及persist逻辑；fresh SHA核验六份F full checkpoint与各epoch_receipt一致。Chris ep1每档20995755B，Patrick ep10每档24078348B。未在本轮torch.load重建六个trainer，完整state恢复语义由源码及既有资格支持；未新跑GPU或暂停测试。实际SSH ps和trace读回支持运行状态。错误schema单数permalink未命中，随后发现并读取正确tts-exp/schemas/coding-session，没有将缺note误报成实验故障。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，本轮GitHub只读确认canonical身份；当前分支只读查询无open PR。
- 本地HEAD `2b71639c3b555293375d248350999377168d515d`，未确认远端commit可解析，不构造commit链接。
- 官方音频入口 https://github.com/joonson/syncnet_python/blob/master/SyncNetInstance.py 与 https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py 是上一解释实际读取的源码。

## Observations
- [result] 六份完整epoch checkpoint fresh SHA匹配；当前训练支持完整epoch边界恢复，未提供任意步无损暂停。
- [decision] 当前仅核查能力，尚未暂停Chris或启动Patrick fine；保持FP16、原失败和原实验版本。
- [blocker] 通用暂停/接管未实现，trial coarse-only限制和Patrick CONTROL_FAILED使既有正式fine入口不能直接复用；监督恢复及空间资格也需先完成。
- [next_step] 向用户明确可在完整epoch边界安全切换及所需新版本资格；若用户决定切换，先实现并验证不重放的调度交接，再运行Patrick三档FP16描述性精调。

## Relations
- continues [[Codex checkpoint - 2026-10-18T10-48-35Z - Patrick 27-grid delivered and Chris FP16 training started]]
- tracks [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- relates_to [[SyncTalk Patrick coarse10 seed42 27格描述性预览]]
- relates_to [[045-synctalk-other-identities-fp16-diagnostic-training]]