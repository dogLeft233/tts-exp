---
title: Codex checkpoint - 2026-10-17T14-36-26Z - Metadata ordered Chris replacement
  frozen and independent qualification resumed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t14-36-26-z-metadata-ordered-chris-replacement-frozen-and-independent-qualification-resumed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17T14-36-26Z
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

# Codex checkpoint - 2026-10-17T14-36-26Z - Metadata ordered Chris replacement frozen and independent qualification resumed

## Summary
用户明确授权固定官方元数据顺序换人后，已冻结ADR042、更新spec并选定首个可获取候选Chris420秒；Cathy完整几何通过、剩余资格恢复执行，正式TFG更新仍为0。

## Story
目标是四个新说话人、每人6–8分钟自然视频、同人不同原录制test，在所有数据门通过后自动做4身份×seed42/2026×训练−6/0/+6共24官方完整训练，随后三driver×三scorer27格，分离训练、驱动和评分器对音量的响应。保留原FP16/65536rays/maxsteps16、60000粗到累计102000完整epoch尝试、原loss/AdamW/default GradScaler/EMA，不补步或改scaler掩盖实际成功更新不等。评分0原PCM16、±6为float64原PCM数值单位，source-owned ROI、guard20主/full次。analysis_contract_v1不变，n4是条件性探索而非人类感知改善或充分数据证明。

此前Debbie原配对及唯一角色反转均未过固定几何门，反转val median2.0101095402098133>2px不能以略超放行。本轮用户明确允许按官方metadata固定顺序替换身份，失败完整保留。ADR042和policy_v1在获取候选前冻结WDA→WRA→RD原文件次序、身份/录制首次出现顺序、首连续标注段和首_000 chunk，不拼不连续段、不按分数/下载速度排序。BenCardin缺镜像原录制0，CarolynMaloney缺录制2；固定全ZIP目录记录二者SOURCE_UNAVAILABLE、0数值预处理而非科学阴性。第三Chris两原录制完整可用，选定300+120=420秒；原视频实际328/293秒、25fps，CRC和SHA、F到Ubuntu复制通过。样本十帧为清晰单人自然讲话；原YouTube独立ID/官方Senate Democrats标题及发布日2019/06/24、2018/01/19已直接存证，发布日不等于拍摄日。相似衣服/同木背景如实保留，全文PCM/RGB去重复与全部独立资格仍待实际通过。

Cathy唯一新test2录制已通过完整冻结v2几何门。其后续post02在任何监督生成前因bare interpreter遗漏既定runtime_exec依赖路径而import face_alignment失败；旧失败、已通过material/stage1完整F档保留。新resume03仅恢复原隔离运行环境并逐阶段复用完整证据，不改数值方法或重放已完成几何。Chris以独立namespace从零预处理，worker等待GPU/足够SHM；新resume04、train-only flow和canonical observers已排队。统一活动训练/渲染/评分/QA/汇总入口已改为Brown/Patrick/Chris/Cathy，禁止用旧Debbie参考。

## Working State
权威Task `Tasks/准备并核验四个新身份的 SyncTalk 完整训练数据.md` 已更新；spec `Experiments/SyncTalk 四新身份官方完整训练电平实验 Spec.md` 和ADR042已BM更新。ROOT=/dev/shm/synctalk-multispeaker-data-20261017，FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；SSH wjj@192.168.50.2 BatchMode/key，V10016GB。运行PID、SHM和本地dirty代码都是fragile状态，先读真实ps/receipts，勿重复启动。

持久选择policy SHA180603ad4e9d2a5d92edfb6e74a68fb824d1a10c9fe7746e966f1f7440391efd，固定HDTF commit bbde889bbab954c9a9ebcabe9962885c29f2165e，镜像6f9ed465f761b8378c690be4f86f090634281584。F identity_replacement_policy_v1内15候选完整审计、pinned_archive_catalog SHA5fdcd80d087214ec385c3817d2965ee7a448ed77f29c71d1496c32adae9f268f、source_availability_attempt01及active_candidate_attempt02.json为准；selected_candidate.json历史Ben不能覆盖。Chris源0 SHA18aa6f842e8b2e42474115a0ea43c2a072599f9b782977d35e996945668642a7，源1 SHA30b00b54bdfbe73aaa81b474b6bf0cdc30e72b97bf9771ad94308e6d966c36b0。HTTP大range中断attempt02留存，attempt03仅4MiB分段重试后完整CRC/SHA通过。

Cathy revision CathyMcMorrisRodgers_recording2_v2：295+110=405秒，primary geometry实际passed；Windows resume03 PID21792，post_geometry_worker_attempt03_receipt.json。14:36UTC actual ps在ave_cache.py PID506988，监督阶段已结束并进入AVE；后续完整资格尚未完成。material15files/raw879260459 archiveSHA2524f3aaba50e73d7b656d940cac6e584d1e895bb48fe4f305481fe50e63bcdd，stage141251files/raw744223516 archiveSHAa177905fc1fb2040f0b70f9dda2005b46491c7885f696948abe56d5f5df4ac76。canonical resumeobserver PID130972读取post03并仅在全部资格和fresh原tar/两视频SHA/无人读后退休缓存。不要改live六local modules：resume_revision_qualification、persist_cohort_prepared、persist_cohort_runtime_qualification、restore_supervision、qualify_all_data、execution_journal；不要启动旧post02/canonical01。

Chris revision ChrisVanHollen_metadata_order_v1：contract SHAade2b500c2c254800ebba115a33b62996a83f84db7315d8dd92ef37fc4c59541，selection SHA86050b50da540ca1d32a438ce55f81e1d649c0882525a0f0276ce7c84bb62b4c。primary PID486668 detached，preparation_worker_receipt phase waiting_gpu_and_workspace，要求SHM>3.5GiB且非阻塞GPU锁，不持锁等待空间。Windowsresume04 PID28404/post_geometry_worker_attempt04_receipt waiting_geometry，train-only flowobserver session13921，canonical resumeobserver PID15524。早期post02/03因decoder ID allowlist和运行环境修正在wait0checks安全停止，源cap/原decoder/失败日志保留；当前decoder SHAa1a7a20ea70e92dbdbf71b11ac1d35edb25df2b113b959d8801075bac02b81a4，仅变ID guard。两candidate各自frozen namespace不随BASE源修改。

cohort_binding.VERSION=metadata_identity_replacement_v3，IDs Brown/Patrick/Chris/Cathy，POST_RECEIPTS=Cathy03/Chris04。Brown/Patrick已全资格但只72监督控制本地，full训练前必须restore。新fresh统一资格实现全来源/整录制AVE/PCM/F archives/全部不变输入pins、实际BASE入口90loader+18完整65536ray、初始化tensor和原pixels/rays逐行精确、72新score audio/GT矩阵。旧false总receipt不冒充新pass。versioned_experiment_supervisor尚未启动、cohort_selection/data_qualification未发布，0正式TFG更新/视频。

空间仅退休本任务已经完整核验的失败反转stage1九精确目录：74250文件逐SHA及无人读通过后逻辑1523292366B/唯一inode862913027B释放；原失败48108-member archiveSHAda2695fcabf7e6408530759b8be24427a3e3bd5f309d2e79834d64db133e4f45保留，原PCM/JSON/几何不动。新flow/bundle cloud attempt01被C整包cacheguard拒绝，0源删；attempt02 session15783/PID114368使用cloud_archive_bounded逐256MiB写入、每片UbuntuCloudFS完整SHA、每片C backpressure，最后整包有序fullSHA后才退休F重复副本。14:36UTC flow part003 reader运行，仅部分片已过，不称整包云备份。F flow1.34GB及bundle仍保留。旧四v1 canonical1.762GB和Brown597MB云完整stream proof已通过事实不变，不动CloudDrive配置。C约1.1GB、F约5.37GB、SHM1.8GiB/Ubuntu root558MB，真实df优先。正式render_delivery_supervisor仍旧整包Cguard后端，须新bounded整包实际通过后再接入，避免后续大媒体失败。

最新完整源码F execution_sources/metadata_replacement_pre_admission_attempt15共211files SHA/AST通过，但随后activeID/resume/bounded变更尚需最终attempt16快照。21个全局BASE依赖/测试副本已本地远端SHA精确。synctalk-27旧prompt仍待同步042/Chris/Cathy03/Chris04/v3；同thread ACTIVE10min，不创建重复automation。所有真实结果独立QA/解释/持久交付后才停用，保留chat。

## Changed Files
select_replacement_identity.py/download_replacement_identity.py、ADR042/spec/Task，prepare_replacement_identity.py、resume_revision_qualification.py、retire_qualified_revision_canonical_resume.py、persist_cohort_prepared.py/persist_cohort_runtime_qualification.py、cloud_archive_bounded.py/cold_backup_rejected_revision_bounded.py。更新cohort_binding及全部活动训练/渲染/评分/QA/汇总identity列表和fresh全局准入。旧原helper、失败bytes与sourcecaps保留，无数值阈值/loss/scaler变更。dirty树1037项，未commit/PR，别全stage或清理其他任务。

## Verification
实际metadata顺序/不拼断续/同原ID拒绝3控制passed；Windowsbinding5 0.294秒；Ubuntu原20完整执行控制attempt09 8.947秒0skip、新binding5 attempt03 0.036秒0skip；factorial3与equal-weight aggregation3 passed。F controller_qualification/cohort_execution_controls_attempt03.json固定25core源/log，这些toy执行控制不是实际模型或27格结果。Chris双源CRC/SHA及两平台精确、Cathy全几何与两个完整F archive、失败stage1退役全SHA实际通过。剩余数据资格、bounded整包、统一准入和24真实模型未完成。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，本轮只读get_repo确认canonical，当前head openPR查询空。
- 本地SHA `2b71639c3b555293375d248350999377168d515d` 尚未确认推送，不构造commit链接。
- FROOT/identity_replacement_policy_v1/active_candidate_attempt02.json
- FROOT/controller_qualification/cohort_execution_controls_attempt03.json
- FROOT/dataset_revisions/CathyMcMorrisRodgers_recording2_v2/post_geometry_worker_attempt03_receipt.json
- FROOT/dataset_revisions/ChrisVanHollen_metadata_order_v1/post_geometry_worker_attempt04_receipt.json
- FROOT/dataset_revisions/DebbieDingell_recording_roles_v2/cold_backup_rejected_revision_attempt02.json

## Observations
- [decision] 按官方metadata预冻结原次序换人，保留前两来源不可用和所有几何失败；首完全合格保留，不以评分选人或放宽门。
- [result] Chris首个可获取候选420秒源CRC/SHA通过；Cathy新录制完整几何通过，剩余资格真实运行；0正式TFG更新/视频。
- [result] 全部活动入口四人版本一致，25core控制0skip通过；仍没有统一数据准入。
- [blocker] Cathy剩余资格、Chris完整预处理和fresh全四人准入未完成；空间及新有界云整包证明仍限制准备调度。
- [next_step] 复用Cathy03/Chris04与真实primary/flow/canonical jobs，完成有界整包读回并接入媒体后端、最终源封存和automation同步，启动版本化监督器等待fresh四人全门再自动执行24模型与27格。

## Relations
- continues [[Codex checkpoint - 2026-10-17T13-54-22Z - Reversed Debbie pair rejected and versioned four identity admission implemented]]
- tracks [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- implements [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[042-synctalk-metadata-ordered-identity-replacement]]