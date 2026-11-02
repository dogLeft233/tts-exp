---
title: Codex checkpoint - 2026-10-18T10-08-50Z - Patrick coarse first score table
  independently verified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t10-08-50-z-patrick-coarse-first-score-table-independently-verified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 10:08:50.472784+00:00
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

# Codex checkpoint - 2026-10-18T10-08-50Z - Patrick coarse first score table independently verified

## Summary
Patrick三档FP16粗训练完成后，首张固定驱动/评价0dB表已通过独立复核，可先向用户交付；完整27格仍运行。

## Story
原目标为四身份两seed三训练档24完整模型与27格。Brown原FP16非有限失败保留后，用户明确要求保持FP16、另立更新前guard并先试Patrick/Chris/Cathy，又要求首个身份完成后先看分数。ADR046将首身份固定coarse10 EMA、seed42、原六testclips的描述预览提前，原fine17门和CONTROL_FAILED保留。V3训练controller已在Patrick三档coarse10完整结束后等待安全评分边界，未中断GPU或重放更新。

独立early_score_preview_v2资格已完成并实际执行。v1 CPU映射用requires_grad筛选，与实际torch_ema保存全部parameters不符，资格失败保留且无真实渲染；v2改为全部参数顺序，手工named overlay与官方copy_to完整state SHA一致，restore exact、60 EMA updates/counters、9固定帧各两次FP16/FFV1PCM通过，3CPU+2host测试通过。每正式视频1500帧无损RGB/PCM读回，评分FP32SyncNet TF32off，source ROI guard20主/full次。每视频18距离矩阵独立NumPyfloat64核验，云整档读回后才窄退媒体cache。

Patrick首张固定driver0/scorer0表已真实通过独立复核：seed42、coarse10固定EMA、原六段test等权、guard20主端点；训练−6/0/+6的Sync-C分别8.213/8.068/7.770，Sync-D分别6.790/6.840/7.077。仅描述实际三个模型，不能作训练gain受控效应或感知改善结论，原CONTROL_FAILED及59985/59985/59984成功更新不变。

## Working State
FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；OUT=FROOT/trials/fp16_other_identities_trial_v1；ROOT=/dev/shm/synctalk-multispeaker-data-20261017，remote RUN=ROOT/trials/fp16_other_identities_trial_v1，SSH wjj@192.168.50.2。真实receipts/ps优先。Windows训练181520在waiting_first_identity_preview，预览176764继续串行9视频；本轮最后读到5/9已score/F/cloud完，第6 train−6/driver−6运行。Chris/Cathy尚未开。PID、SHM和仓库未提交文件是机器本地状态；F sources、receipts、原始分数矩阵及cloud locators是持久证据。

权威预览OUT/early_score_preview_v2：fixed_driver0_scorer0_preview.json、first_table_independent_review.json、scores/train_±d_driver_+0/local与artifacts.tar.gz、renders/train_±d/driver_+0/cloud_storage_locator.json。媒体原lossless电影的F重复tar已按全验规则退休，需从cloud显式恢复，不能假定仍在；local H264/PNG/head/wav与score矩阵保留。Patrick trainval非controls13452已在F完整archive16562成员与remote16500监督fresh SHA后独占lease退休，全部3000test及48trainvalcontrols保留，未来trainval加载必须restore；不重复cleanup。

预览源coarse_preview_v2.py SHA85d04f5a567dea8174eaec061c9e68dcc2a6d0206d5715610deba13b9c32e515；hostcontroller SHA db1caa3a7fe938da088230c9c03e2a19782defab66f349b68a872587da5b2c06，host_source_manifest固定7源。原trial contract SHA787d917d259eeca89311016737893348578335e3623f217858797926311893bd不变。全部27格完成后，必须独立最终review、持久交付，才写OUT/controller_revision03/preview_delivery_release.json（identity、passed、independent_score_qa_passed、all_persistent_bytes_verified、receipt_path与receipt_sha256）；首表不能释放训练。failed/lost必须读实际job/receipt另立恢复，不盲重启或重放completed。

Brown原failure archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72、原version与旧false保持，105372/133984/610686不重启。guard不是上游权重增长根因完全修复，fine/seed2026尚未自动启动，原24模型目标未完成。synctalk-27保持ACTIVE，首表在本轮回复交付，后续不要重复通知同表，直到完整27格/真实失败/需输入才通知。

## Changed Files
本轮新增.codex/scratch/audit_first_preview_table.py及record_first_preview_table_checkpoint.py，只对存储产物做CPU复核并写入记忆。F first_table_independent_review.json新建；Task当前头部/过时initial段标签、ADR046和研究地图更新且SDK读回。未改冻结训练/渲染/评分源，未进行新optimizer更新。仓库1044项dirty包含既有工作，未批量stage/clean，完整gitstatus保存F first_table_git_status.txt。

## Verification
F early_score_preview_v2/fixed_driver0_scorer0_preview.json SHA4e8d953100af1ae5b6179ef2d27d45fafdc4ca6bf0c0f7e997e25c936a6b7918；first_table_independent_review.json记录本轮fresh三份score archive及所有local成员SHA、54存储矩阵的float64端点重算/六clip等权复核，最大端点误差2.0012563579641096e-06<3e-5。三视频既有独立云整包读回receipt与archive SHA一致；本轮未重跑GPU、未重读云payload或恢复lossless电影，不冒称再次像素推理验证。
本轮所有54矩阵shape245×31/finite，guard20取20:225及full245，手工float64列均值/median/min/offset端点与原值在3e-5内；六clip均值与summary精确相等。三score tar全部75成员fresh SHA与local一致。没有再次运行ML资格测试；之前资格/距离QA由已持久receipt证实。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，本轮只读GitHub查询当前branch无open PR。
- HEAD `2b71639c3b555293375d248350999377168d515d`未验证GitHub commit可解析，保留本地标识。

## Observations
- [result] 首张−6/0/+6训练档guard20 Sync-C=8.213/8.068/7.770，Sync-D=6.790/6.840/7.077，六clip等权、单身份单seed粗阶段描述预览。
- [decision] 原实际更新不等CONTROL_FAILED和全部失败版本保留；不根据预览调参、挑checkpoint或补训练步。
- [blocker] 完整27格与最终交付QA未完成，Chris/Cathy须在安全评分交付后继续；原24完整模型仍未完成。
- [next_step] 完成Patrick剩余驱动条件的27格，独立最终review并持久交付后按V3 schema释放训练继续Chris/Cathy。

## Relations
- continues [[Codex checkpoint - 2026-10-18T09-29-48Z - First identity preview scheduled with preserved live training handoff]]
- tracks [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- relates_to [[046-synctalk-first-identity-early-score-preview]]