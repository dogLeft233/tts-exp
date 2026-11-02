---
title: Codex checkpoint - 2026-10-17T12-30-51Z - Frozen geometry rejected Debbie recording
  pair and independent Cathy preparation continues
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t12-30-51-z-frozen-geometry-rejected-debbie-recording-pair-and-independent-cathy-preparation-continues
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 12:30:51.766512+00:00
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

# Codex checkpoint - 2026-10-17T12-30-51Z - Frozen geometry rejected Debbie recording pair and independent Cathy preparation continues

## Summary
Debbie的冻结v2跨录制几何门真实失败，监督器正确阻止24模型开训；Brown/Patrick资格通过、Cathy独立准备继续，实验TFG更新和正式生成视频仍为0。

## Story
用户已授权四新身份各6–8分钟自然数据、同人不同原录制test，全部门通过后执行4身份×seed42/2026×训练−6/0/+6共24模型，随后三driver×三scorer完整27格。沿固定官方commit43079fb3与AVE，从共同随机初始化运行FP16/65536rays、60000粗阶段后累计102000尝试完整epochs，保留原loss/optimizer/scaler/EMA并验实际成功更新相等；AMP skips不能补步或改政策掩盖。评分0原PCM16、±6 float64原PCM数值单位、source-owned ROI与guard20主/full次不变。

本轮完成训练协调器锁定/分离远端epoch/断线重新接入和安全持久轮换，排队门后自动24训练及后续media delivery链，并加入每epoch官方核心源码SHA验证。准备到Debbie时，测试flow误差median4.48346127047664/p956.69030642173168px超过冻结2/5门；val为0.86748/1.83116通过，训练早晚camera controls通过。独立NumPy投影复现原误差，Torch投影最大差4.745e−5px，原flow重复精确、共享点冻结。姿态覆盖/三角化条件差异只是线索，尚未证明原因，不能解释成音量科学阴性。

主stage2及两个监督器均以失败停止，training_started=false。保留失败原数且不放宽ADR039。另起仅Cathy的同方法准备worker及F derived observer，绕开主worker提前退出。拟议检查Debbie另一原录制是否足够长，若合格可在无TFG结果前版本化反转main/test角色并重做全部材料资格；此方案尚未决定、未写新spec、未执行。不能静默替换失败结果或按SyncNet择素材。

## Working State
持久FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017。易失ROOT=/dev/shm/synctalk-multispeaker-data-20261017，SSH wjj@192.168.50.2。直接读Task/spec/ADR038/039与真实receipts优先；Task目前仍含旧监督器活跃说法，必须修订。四人原材料/stage1/72 GT评分参考此前已独立通过。Brown和Patrick全量v2/AVE/监督、F三归档fresh成员SHA、90 loader和18 capacity通过；监督只保留72控制，正式训练前full恢复。

Debbie失败ROOT stage2_v2_worker_receipt.json及stage2_v2_DebbieDingell_geometry.log保留；prepared/DebbieDingell/geometry_v2包含原失败全数。F geometry_diagnostics/DebbieDingell_failed_v2_attempt01已有diagnosis.json、residual_location.png和原失败log/receipt，scp完成但完整失败包/该四文件SHA独立持久核验待补。F full_experiment_supervisor_failures/20261017T121627.852250Z.txt和render_delivery_supervisor失败receipt保留，不重复启动。旧PID119780在training_started=false时安全停以加源码pin，替代PID113204已因几何门失败退出。

实际12:29UTC检查：Cathy独立worker PID384026在train_tracker，face_tracker PID384028；ROOT stage2_v2_Cathy_independent_worker_receipt.json为权威，Windows archive_Cathy_independently session53726等完成。旧runtime worker PID110572/session22551停在等待Debbie，不能假设它会处理Cathy；Cathy独立qualification/flow/runtime持久观察链仍待补。ROOT SHM约3.6GB空，F约9.4GB空，root约527MB，C约10.8GB；监测实际空间。

F raw stage1重复cache窄范围回收worker session32821/PID5420继续；Brown已经独立archive全成员和loose SHA验证后退休41240文件752344340B，保留完整tar/manifest、全部JSON及五条人审jpg/lms。其余身份以真实receipt为准。百度盘Brown stage1 597734675B三片已写B:，Ubuntu独立readback PID349302还等待可见性，F cloud_delivery/stage1-SherrodBrown/transport_attempt01/staging_receipt.json明确cloud_independently_verified=false；不能称云备份成功。没有动CloudDrive配置或未验证原数。

## Changed Files
未跟踪scripts/experiments/synctalk_multispeaker_full_20261017中的execution_journal.py、remote_epoch_job.py、training_coordinator.py、persist_training_epoch.py、train_epoch.py、qualify_all_data.py及full_experiment_supervisor.py实现训练锁定/恢复/准入。render_model.py按官方final test粗/default patch1 flags加载fine最终EMA；score_identity_seed.py/validate_scores.py/measure_video_changes.py与remote_result_job.py、seal_media_delivery.py、persist_media_delivery.py、retire_result_cache.py、retire_test_supervision.py、render_delivery_supervisor.py、assemble_results.py实现媒体/27格链但尚无真实模型执行。global aggregation independent QA/最终报告尚需完成。

stage2_v2_Cathy_independent_worker.py、archive_Cathy_independently.py和diagnose_failed_geometry.py是失败后独立准备/只读诊断，不改核心v2。reclaim_stage1_raw_F.py、cloud_archive.py/cloud_archive_readback.py是窄范围存储工具。最新F execution_sources/full_delivery_queue_attempt10共91文件SHA/AST通过，时间在独立Cathy/诊断代码之前，需新snapshot。共享dirty树大量无关改动保留，无commit/PR。

## Verification
实际Ubuntu20 tests在execution_unit_tests_attempt06.log通过8.463s零skip；Windows真实锁及408模拟epoch恢复两项通过5.977s。中断70后恢复最终408唯一epoch/16阶段块，第三次无新增；属于synthetic controls，无TFG更新。F controller_qualification/io_state_rotation_attempt02.json跨平台partial复制保留、幂等接受、远端旧state unlink后续跑、metadata变更拒绝通过。另Windows stage1 cache回收三项控制通过0.082s，pixel signed-difference一项控制通过。已观察的BFM几何优化不是TFG训练。正式24训练、正式渲染/27格、最终端到端交付未执行。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，本轮较早只读GitHub查canonical且当前head无PR。
- Git SHA `2b71639c3b555293375d248350999377168d515d` 是本地未验证推送，不构造commit链接。
- FROOT/geometry_diagnostics/DebbieDingell_failed_v2_attempt01/diagnosis.json
- FROOT/controller_qualification/io_state_rotation_attempt02.json
- FROOT/full_experiment_supervisor_receipt.json

## Observations
- [result] Debbie测试几何flow median4.48346127047664、p956.69030642173168px失败；独立投影复现，训练被正确阻断，0新TFG更新/视频。
- [decision] v1失败和v2失败都保留，冻结v2及2/5px门不变，不依据SyncNet选录制；素材角色反转目前仅候选方案。
- [blocker] Debbie当前录制配对不合格，Cathy余下资格及独立观察链未完成，全四人训练准入尚不存在。
- [next_step] 保留并独立持久核验Debbie完整失败包，同时检查原录制长度及版本化角色反转可行性，复用正在运行的Cathy worker继续全量准备。

## Relations
- continues [[Codex checkpoint - 2026-10-17T11-43-48Z - Full epoch controls and four identity scoring references independently qualified]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[039-synctalk-shared-training-flow-geometry]]