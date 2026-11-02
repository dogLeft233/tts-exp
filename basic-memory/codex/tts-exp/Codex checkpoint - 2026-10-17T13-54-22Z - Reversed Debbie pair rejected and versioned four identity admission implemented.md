---
title: Codex checkpoint - 2026-10-17T13-54-22Z - Reversed Debbie pair rejected and
  versioned four identity admission implemented
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t13-54-22-z-reversed-debbie-pair-rejected-and-versioned-four-identity-admission-implemented
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17T13-54-22Z
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

# Codex checkpoint - 2026-10-17T13-54-22Z - Reversed Debbie pair rejected and versioned four identity admission implemented

## Summary
Debbie唯一角色反转也未通过固定几何门并已完整持久封存；Cathy唯一新录制候选正执行stage1，统一版本绑定/独立准入控制已实现并过25控制，但当前没有四个合格身份，仍零正式TFG更新。

## Story
用户的目标仍是四个新身份、每人6–8分钟自然素材、同人不同原录制test，全部资格通过后4身份×seed42/2026×训练−6/0/+6共24个官方完整模型，再做三driver×三scorer的27格及四身份等权汇总。官方AVE冻结预训练，TFG共同随机完整初始化，FP16/65536rays/maxsteps16，60000粗阶段至累计102000完整epoch尝试；保持原loss/AdamW/default GradScaler/EMA/fineflip，记录AMP skips并比较实际成功更新，不补步/改scaler掩盖不等预算。评分0原PCM16、±6float64原PCM数值单位，source-owned ROI、guard20主/full次。不能把SyncNet响应直接称为人类同步质量，n4仅条件性探索。

前一checkpoint已记录两原pair失败、Cathy405秒唯一新test录制和Debbie420秒单次反转。本次推进得到Debbie反转的真实资格失败：两个训练恢复控制均通过，但完整val750的flow median2.0101095402098133px超过固定2.000，p953.938487529220711px通过5.000仍整体失败。test1500只有landmark前置通过，test shared-reference flow尚未跑，不能冒称通过。geometry_resume_worker_receipt.json和post_geometry_worker_attempt02_receipt.json均false并停止，无TFG更新。官方固定HDTF WDA/WRA/RD metadata只有Debbie0/1两个原录制；没有第三原录制可沿用ADR041。不能再反转、放宽仅超0.010的门、挑片段或把同录制另一chunk当独立test。必须先冻结替代第四身份的非评分选择规则/候选及新spec/ADR，再获取核验；原四人合同当前不可训练，不缩为三人。

同时实现显式数据版本绑定，消除训练用新录制、render/score误读旧原PCM或ROI的路径风险。选择registry固定四身份、双平台版本目录及完整输入SHA；训练每epoch和render/score/独立矩阵/像素QA/封存/退休共用同一解析器。full/test监督从该版本原F archive恢复至该版本原receiver，finally恢复模块globals。旧失败目录不移动或alias覆盖，旧false总stage/监督器不改成pass。fresh准入会重新完整读取各身份三个F archive、核验原始来源下载、全部实际不变输入、原PCM与整录制上下文/新test参考，然后经BASE真实入口对每身份重跑loader90和65536ray资源18，对旧pixel/ray/full-input逐行精确、完整初始化tensor SHA精确。另在只写新view的路径重新做72audio forward/原GT距离与端点独立QA，旧reference不覆盖。全部通过后才写data_qualification.json。源码实现不是实际准入结果，当前版本仍指向已失败Debbie候选，versioned supervisor刻意未启动。

## Working State
权威Task为 `Tasks/准备并核验四个新身份的 SyncTalk 完整训练数据.md`（本轮两次BM更新并直接读回，顶端Current execution gate优先于旧历史段落）；spec为 `Experiments/SyncTalk 四新身份官方完整训练电平实验 Spec.md`，ADR038–041保持原限制。ROOT=`/dev/shm/synctalk-multispeaker-data-20261017`，FROOT=`F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017`。Ubuntu `wjj@192.168.50.2` BatchMode/key，V10016GB，runtime_exec.sh隔离工具链，不能改共享venv。新控制、源码和metadata均持久F；SHM仅运行缓存。

Brown/Patrick原全资格和三个archive preflight仍有效，监督仅72控制，full训练前必须恢复。Cathy候选ROOT/dataset_revisions/CathyMcMorrisRodgers_recording2_v2：原main295秒、新独立test2 110秒，总405秒；remote PID470446/session52906，13:53UTC真实phase=stage1，子进程477524 preprocess_images_landmarks.py --reuse-image-root .../reuse_stage1，原parent flock435526；素材独立QA已实际通过。原main6750输入JPEG/FAN/parser/BFM/BA/flow须SHA精确才能复用，全部新test和冻结v2资格后续仍待跑。Windows post_geometry_worker_attempt02_receipt.json PID128476/session85058继续等待后续，不能重复启动。6个已加载本地helper（complete controller/persist_prepared/persist_runtime/restore_supervision/qualify_all_data等）本轮未编辑，候选独立远端冻结副本未编辑。准备新身份时用独立版本helper，或等Cathy完整结束后再更新统一身份列表和source capsule，不能让live capsule漂移。

Debbie反转失败完整F bundle实际通过：FROOT/dataset_revisions/DebbieDingell_recording_roles_v2/failed_preparation_preservation_attempt01，48108成员2142144023原B、33000 prepared→preprocessed同字节alias全部验收，archive SHA da2695fcabf7e6408530759b8be24427a3e3bd5f309d2e79834d64db133e4f45，manifest SHA0d907102a88c104e2498e59c572944c22e259b50ded137cd5ea1c768661ed028。只在完整tar逐成员SHA后，又逐字节/无人读检查两canonical，再退休此失败namespace main/test.mkv共881988635B；canonical_cache_retirement.json通过，原字节在F tar完整保留。prepared/preprocessed/PCM/JSON/失败数组未删，原失败v1和云locator未动。persist_revision_failure.py session78798/PID122900已完成，不重复封存。其训练flow Fgzip1340711203B已独立解压/原SHA通过（raw1516380128B），gzip SHA db2554c9e91d2d53e84d07f4577ff5098fc3e27e0aa998bb568a2a05a29292ed，dense原缓存先前已退休。旧四个failed-v1 canonical云冷存和Brown三片597734675B stream零payload独立读回仍有效，勿混称新bundle云备份。新bundle只有F持久。最近SHM2.6GiB/F约7.3GB，实际df优先；不动CloudDrive配置/未验证原数。

最新F execution_sources/cohort_binding_failed_reversal_attempt14完整165文件复制SHA/全部Python AST通过，包括代码、Task/spec/ADR038–041、11 testfiles和两候选immutable contracts/sourcecapsules；attempt13保留。十四执行依赖已逐SHA同步Ubuntu BASE及测试副本；候选 own namespace仍原冻结bytes。Windows helper版本化supervisor尚未启动；当前root/data_qualification.json未写任何新pass。旧global训练/媒体监督失败保留。

synctalk-27已按实际故障/代码实现更新为ACTIVE10min同thread；下一轮先解决第四身份选择缺口、复用Cathy，不重复尝试已失败Debbie。保持只关键完成/实际故障/需输入通知，全部真实结果独立QA/解释/持久交付后才停用并保留chat。当前metadata初步发现BenCardin0/1有独立官方URLs及annotation325/284秒，尚未选择、下载或资格；不能把待核验例子当已确定cohort。已下载其他备选Chuck/Jan/Veronica main均不足278秒。最终选择需先冻结与披露规则，而不是连续按评分找最好的人。

## Changed Files
新增 `cohort_binding.py`、`restore_cohort_supervision.py`、`qualify_cohort_version.py`、`cohort_requalification.py`、`cohort_reference_requalification.py`、`versioned_experiment_supervisor.py`、`seal_revision_failure.py`、`persist_revision_failure.py`。更新runtime_common、train_epoch、training_coordinator、render_model、score_identity_seed、validate_scores、measure_video_changes、retire_test_supervision、seal_media_delivery和render_delivery_supervisor的真实输入路径绑定。更新完整训练控制fixture为真实registry下408模拟epoch重连；新增test_synctalk_cohort_binding_20261017.py。没有改变官方数值loss/optimizer/scaler/EMA或几何阈值。共享dirty树大量其他任务，本任务改动未commit/PR；不要全stage或清理其他文件。

## Verification
Ubuntu更新后的20完整执行控制attempt08实际通过10.104秒、0skip，含原FP16/AMP/full-state/EMA与408模拟epochs；attempt07是runtime_exec切换cwd后相对unittest发现目录失败，日志完整保留并修为绝对路径，未伪称模型失败。Windows新binding5控制通过0.205秒，Ubuntu同5 attempt02通过0.056秒、0skip：旧数据没有selection不能准入、错误测试namespace/重复身份拒绝、选择SHA改变拒绝、同长原PCM篡改/路径逃逸拒绝、restore确用所选archive/receiver并在异常finally恢复globals且旧失败文件保留。F controller_qualification/cohort_execution_controls_attempt02.json固定日志、测试源码和cohort_binding源码SHA。更早20/aggregate/namespace/stream/failed-cache控制在前checkpoint保留。真实新role failure48108成员SHA和窄范围缓存退休已实际通过。Cathy只素材QA完成/当前stage1，剩余几何及runtime未通过；没有24模型/27格真实结果。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，本轮只读API确认canonical；当前head PR查询为空。
- 本地、尚未验证推送SHA `2b71639c3b555293375d248350999377168d515d`，不构造commit链接。
- FROOT/execution_sources/cohort_binding_failed_reversal_attempt14/source_manifest.json
- FROOT/controller_qualification/cohort_execution_controls_attempt02.json
- FROOT/dataset_revisions/DebbieDingell_recording_roles_v2/failed_preparation_preservation_attempt01/verification.json
- FROOT/dataset_revisions/DebbieDingell_recording_roles_v2/failed_preparation_preservation_attempt01/canonical_cache_retirement.json
- ROOT/dataset_revisions/CathyMcMorrisRodgers_recording2_v2/preparation_worker_receipt.json

## Observations
- [result] Debbie原pair和唯一反转均未过固定v2几何门；反转val median2.0101095402098133px，不能以略超门放行。0本轮正式TFG更新/视频，无四人准入。
- [result] 反转失败48108成员/2142144023原B全量持久SHA通过后，只退休881988635B本任务canonical缓存；原数完整保留。
- [result] 四人显式版本绑定及fresh准入控制已实现，20+5真实控制0skip通过；代码实现不代表全数据或模型实验通过。
- [decision] 保持4身份/6–8分钟/原FP16/65536rays/实际预算/评分合同；不改门、不复用旧test、不删失败或降为3人。
- [blocker] 已知Debbie两个原录制均无合格方向且无第三官方原录制，现有版本不能启用；替代身份规则/spec/数据尚待冻结核验。Cathy后续资格也未完成。
- [next_step] 先冻结并封存第四身份的非评分替代规则和新版spec，按确定性官方metadata候选顺序准备一个新身份，同时复用Cathy stage1与后续worker；完整更新所有ID/版本源pin并fresh全四人准入后才启动已授权24模型。

## Relations
- continues [[Codex checkpoint - 2026-10-17T13-37-43Z - Independent recording candidates queued and cloud storage verified before cohort admission]]
- tracks [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- implements [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[040-synctalk-recording-role-fallback]]
- relates_to [[041-synctalk-eligible-recording-fallback]]