---
title: Codex checkpoint - 2026-10-18T09-29-48Z - First identity preview scheduled
  with preserved live training handoff
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t09-29-48-z-first-identity-preview-scheduled-with-preserved-live-training-handoff
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 09:29:48.658491+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-18T09-29-48Z - First identity preview scheduled with preserved live training handoff

## Summary
首身份粗训练预览已获授权；新控制器接管同一末epoch，确保先评分再切换身份，而不重放已执行更新。

## Story
原任务是四身份两seed三训练gain的24完整模型与27格；Brown原FP16非有限失败保留。用户授权更新前保护与Patrick/Chris/Cathy独立FP16粗训练，又要求一个人完成后先看分数。ADR046因此固定首个三档coarse10身份、seed42、epoch10 EMA、原六段test，先driver0/scorer0再全部27格。粗阶段预览不能称fine17完整训练；实际成功更新不等仍CONTROL_FAILED，分数仅描述实际模型，不能据此调训练/选checkpoint/补预算。

原控制器在每个epoch后立即启动下一job，GPU锁为非阻塞，评分抢锁可能损坏后续训练调度。本轮新增独立V3控制器，从29个已持久epoch/job重建只读历史，再接管同一末epoch。仅停止Windows164828；新Windows181520仍观察remote776954/flock776964/训练776965，实际ps确认持续运行。待三档ep10全持久化后，V3在退休Patrick监督与启动Chris之前写preview_ready并等待经过真实交付QA的release。原7源与contract冻结不动。该接管是特定历史恢复，不能盲重跑。

## Working State
OUT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1，RUN=/dev/shm/synctalk-multispeaker-data-20261017/trials/fp16_other_identities_trial_v1；SSH wjj@192.168.50.2。真实receipt/ps优先于本快照。Patrick0/-6 ep10完成，+6 ep10仍运行；Chris/Cathy未启动，formal24完成0。当前SHM约29.6MB，F约1.51GB，容量必须实测并资格；勿重启旧observer或盲删。末checkpoint约24MB，当前空间极紧。

controller_revision03/handoff.json与原controller receipts、29组资格源为权威接管证据。V3源码SHA d0ad60214e0c4d36193505c7aeb6b1b507b9838cb1d2001a1eb89a0f98f6b712；contract SHA787d917d259eeca89311016737893348578335e3623f217858797926311893bd。三档完整ep10后controller_revision03/preview_ready.json标识评分安全边界。只有真实评分/独立QA/全持久字节完成才能写preview_delivery_release.json：identity PatrickLeahy、passed/independent_score_qa_passed/all_persistent_bytes_verified均true、receipt_path与receipt_sha256实值。不得提前释放或杀活跃GPU。

预览request SHA941e1b9dade0285a9c90c8edab5852704e3910d2c337e9acd5a6e0afd7801b54；实现尚未资格，0新视频0分数。原render_model强制fine17，不能改原门；需新coarse入口严格从full checkpoint提取EMA、核验模型buffer/counters，保留FP16 test_step/blend、FFV1/PCM逐像素与音轨复核、FP32SyncNet TF32off、guard20主/full次、独立距离聚合QA。原train仅fine17导出final_ema，coarse必须独立实现和验证。现test监督仍在，不能先全退休再假定可加载。synctalk-27 ACTIVE，尚待更新其旧PID提示为V3。

Brown失败archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72保持，旧105372/133984/610686不重启。FP16保护不是上游权重增长根因修复；本轮fine/seed2026未授权自动开启。

## Changed Files
- scripts/experiments/synctalk_multispeaker_full_20261017/other_identities_controller_preview_v3.py，F controller_revision03同名冻结源。
- tests/test_synctalk_preview_controller_v3.py：existing job observe、closed epoch reuse、lost reject、complete CONTROL_FAILED可描述预览/partial拒绝。
- Task《试跑 Patrick Chris Cathy 的 FP16 粗训练》当前头部与F task_readback.json同步。
这些源码仍未提交，仓库1041项dirty包含既有工作，未批量stage/clean；SHM、PID是易失状态，F封存为持久执行证据。

## Verification
3项路由单元测试实际通过。29epoch及29job的proof、manifest、archiveSHA、working receipt/traceSHA鲜核；本轮未再次流式重验每个tar成员，原成员逐SHA证明保留。新controller已到同一Patrick +6 ep10而非停在历史epoch2；ps确认同一训练进程连续。尚未端到端资格EMA/渲染/评分，未产生新科学结果。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，当前branch只读查询无open PR。
- HEAD `2b71639c3b555293375d248350999377168d515d`未验证远端commit可解析，保留本地标识。

## Observations
- [decision] 首身份固定coarse10 EMA描述预览优先于其余身份，原CONTROL_FAILED和全部失败保留。
- [result] 独立V3接管同一末epoch并设身份结束评分等待点，未重放optimizer更新。
- [blocker] 预览实现、EMA资格、独立像素/评分QA与容量资格未完成；不能只轮询等待或把请求合同当分数。
- [next_step] 在Patrick真实三档ep10完整持久化的等待边界，资格独立coarse EMA渲染与评分入口及容量，再执行原六clip27格并独立核验交付。

## Relations
- continues [[Codex checkpoint - 2026-10-18T03-04-19Z - Other identity FP16 trial running with controller handoff preserved]]
- tracks [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- relates_to [[046-synctalk-first-identity-early-score-preview]]