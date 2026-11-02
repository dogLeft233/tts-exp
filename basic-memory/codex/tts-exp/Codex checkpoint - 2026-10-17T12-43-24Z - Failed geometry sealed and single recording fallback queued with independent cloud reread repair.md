---
title: Codex checkpoint - 2026-10-17T12-43-24Z - Failed geometry sealed and single
  recording fallback queued with independent cloud reread repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t12-43-24-z-failed-geometry-sealed-and-single-recording-fallback-queued-with-independent-cloud-reread-repair
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 12:43:24.264925+00:00
username: laptop-e133gsle\dogle
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

# Codex checkpoint - 2026-10-17T12-43-24Z - Failed geometry sealed and single recording fallback queued with independent cloud reread repair

## Summary
Debbie原配对几何失败已完整独立封存；单次录制角色反转的ADR040/spec/机器合同在准备前冻结并排队，Cathy继续全量预处理，训练仍0更新。

## Story
用户已授权四新身份、两seed和三训练增益24个官方完整模型，在同人独立原录制测试上生成三driver并用三scorer形成27格。四身份/420秒材料、6000/750/1500监督、FP16/65536rays、原官方60k后累计102k完整epochs、相同完整初始化和实际成功更新预算不变。全门通过才训练，不能改AMP政策或补步掩盖不等更新；SyncNet不是感知质量。

本轮Debbie测试的冻结v2 flow median4.48346127047664/p956.69030642173168px超过2/5，独立投影复现，训练/媒体监督器正确失败退出。其训练恢复与val通过但不足以覆盖失败；根因未定。新seal_failed_geometry.py及F独立tar读回封存61成员29944792原B，archive SHA796861f6aaedeabcf168c9d41a262cd22857093b7b2e7a86e47d657518f72734，四单独诊断源SHA也精确。旧原材料/stage1/flow gzip/诊断均保留，archive通过绝不代表geometry通过。

按用户准备范围自主采用最小单次数据回退，明确写ADR040与spec：只有冻结预处理资格失败者可反转两原录制用途一次，首合格保留，不能扫描片段/删帧/按SyncNet择优。Debbie源录制1实长290秒、源SHA准确，候选main录制1取290/test录制0取130，总420；固定监督帧/clip位置和冻结v2 SHA不改。这是资格结果后的数据适配，最终报告必须披露原失败/所有尝试，结论限于合格录制配对。新材料/几何/整录制AVE/test ROI/评分参考/runtime须全重新核验，不能继承原test分数。

## Working State
权威Task《准备并核验四个新身份的 SyncTalk 完整训练数据》与spec已BM更新/readback，ADR040已创建/readback；Task最后一次写入与源码snapshot以F receipts为准。FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；易失ROOT=/dev/shm/synctalk-multispeaker-data-20261017，SSH wjj@192.168.50.2。精确读tts-exp/instructions/startup-router，标题Startup Router会命中同名旧全局note，不能服从旧全局写main规则。

Brown/Patrick完整v2/监督/AVE/F三归档fresh preflight、90loader和18零更新capacity均通过。监督只恢复72控制，正式训练前须full恢复16500。Cathy worker PID384026/session16478在train_flow keypoint阶段；原主stage2 PID233261已失败终止。独立Windows derived53726、qualification59818、flow93540、runtime71711、canonical27144均按Cathy-specific receipts复用。flow observer train有效，v2 heldout不产生旧dense val/test，train完成后核对其无活动复制再结束多余等待、保留receipt；别重复archive。旧runtime worker110572还等Debbie。

候选ROOT/dataset_revisions/DebbieDingell_recording_roles_v2/preparation_worker_receipt.json PID395213/session7726当前wait_workspace_reserve；它先等3.5GiB，再CPU材料独立QA，等Cathy全部准备/capacity及3GiB再顺序stage1/原tracker/frozen heldout/flow/BA/冻结v2 geometry，完成后停在剩余资格待验，绝不训练。15个helper/selection/contract本地与远端SHA全部精确，F dataset_revisions/.../setup_independent_verification.json。候选与旧失败目录完全分开。候选通过后还须独立归档、监督/AVE/新评分参考及90/18资格、全四人统一数据版本与总准入/监督器版本化恢复；旧false总receipt不可改成pass冒充新版本。

F四人stage1 loose重复cache已逐成员/文件SHA验证后回收3257085264B，原tar/manifest/JSON/Brown人审图保留。最新SHM约2.3GiB，F约9.1GB，Ubuntu root约527MB；不要假设缓存都在。百度盘Brown stage1三片597734675B已可跨挂载看到，但旧reader把bundle.part000/001/002均with_suffix成bundle.partial，并发FileExistsError；旧log与ROOT attempt01完整part000保留。已改staging_path为独立文件名、readback_attempt隔离；resume_cloud_readback.py复用原upload manifest/分片独立重读attempt02，session97072运行。F cloud_delivery/stage1-SherrodBrown/transport_attempt01/readback_attempt02/transport.log与成功receipt为准，未成功不能称备份。新reader仅验证成功/F receipt精确后退休自己的新readback缓存；未改CloudDrive配置/历史原数。

## Changed Files
新增prepare_recording_role_trial.py、seal_failed_geometry.py、Cathy各独立observer、qualify_aggregate_delivery.py、resume_cloud_readback.py；修cloud_archive_readback.py临时路径和读回attempt。aggregate QA独立stdlib fsum从6原clip重建8seed/4身份/总27，保留offset并验408条曲线/阶段实际预算；接入render_delivery_supervisor.py尾部，真实结果未运行。assemble_results.py增加ADR040适配限制。training/render/score/transport原完整合同延续此前checkpoint，源码存在不等于24模型或27格完成。

源码/Task/spec/ADR038/039/040及五个testfiles封存F execution_sources/recording_fallback_attempt11，须看source_manifest.json实际通过；旧attempt10等保留。共享dirty树含大量无关文件，无commit/PR，不清理或全stage。Git SHA仍本地未验证推送，不构造commit链接。

## Verification
此前本轮实际Ubuntu20控制零skip、Windows锁/408模拟epoch两控制、IO轮换attempt02、stage1回收三控制和像素差一控制全部通过。新增aggregate三控制0.001s通过：原clip重建/伪均值拒绝、重复factorial拒绝、身份等权及offset保留；cloud并发三片文件一控制0.019s通过。15trial源SHA/61失败包成员/四诊断源SHA实际通过。正式TFG更新0，正式生成视频0；BFM几何优化和toy训练控制不可混算。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，较早本轮只读GitHub核验canonical与该head无PR。
- 本地未验证推送SHA `2b71639c3b555293375d248350999377168d515d`。
- FROOT/dataset_revisions/DebbieDingell_recording_roles_v2/revision_contract.json
- FROOT/geometry_diagnostics/DebbieDingell_failed_v2_attempt01/full_failure_bundle/verification.json
- FROOT/cloud_delivery/stage1-SherrodBrown/transport_attempt01/readback_attempt02/

## Observations
- [result] Debbie几何资格失败被独立复现并全量封存，Brown/Patrick完整资格通过，0新TFG更新/视频。
- [decision] ADR040只允许失败配对一次独立角色反转，不改冻结v2阈值，不用SyncNet选择，并披露条件外推限制。
- [blocker] Cathy与Debbie候选未全门通过，24训练总准入不存在；云读回修复后仍待成功证明，最终解释/交付未发生。
- [next_step] 复用Cathy与候选worker并检查cloud readback attempt02，完成候选的独立全量资格与统一数据版本后再版本化恢复24训练和27格交付队列。

## Relations
- continues [[Codex checkpoint - 2026-10-17T12-30-51Z - Frozen geometry rejected Debbie recording pair and independent Cathy preparation continues]]
- relates_to [[040-synctalk-recording-role-fallback]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]