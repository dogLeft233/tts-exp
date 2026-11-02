---
title: Codex checkpoint - 2026-10-17T17-07-16Z - Four identities admitted paired training
  running and byte exact storage qualified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t17-07-16-z-four-identities-admitted-paired-training-running-and-byte-exact-storage-qualified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 17:07:16.378965+00:00
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

# Codex checkpoint - 2026-10-17T17-07-16Z - Four identities admitted paired training running and byte exact storage qualified

## Summary
Brown/Patrick/Chris/Cathy已全部通过独立准备和fresh统一准入，24模型正式训练正在运行；首组三档各6000次成功更新/0AMP skips及18000行配对trace独立复核通过，字节精确缓存去重和两份历史失败stage1云冷交付已实际完成资格。

## Story
用户希望分离音量在SyncTalk训练、视频驱动和SyncNet评价三个环节的作用。May缩减预算先导及固定视频27格之后，用户授权四个新自然身份、每人6–8分钟、同人不同原录制测试，全部数据核验后自动执行官方完整训练和三轴27格。Debbie原配对及唯一角色反转、Cathy原配对在固定几何门失败，用户明确要求保留失败、不放宽门，按官方metadata原顺序替换身份。ADR042冻结队列，缺源Ben/Carolyn标SOURCE_UNAVAILABLE，首可获得Chris完成全部资格；Cathy独立test2也完成全部资格。

此次工作跨过了最后的数据准入门。四人来源/原PCM/整录制AVE上下文、F完整归档、BASE实际加载/资源、完整共同初始化与像素/rays、72fresh评分参考全部通过，原失败没有被覆盖。已有总监督器自动转入真实训练，首个三档epoch已经完成配对控制。流程启动和单epoch控制不代表完整phase预算、24模型或27格完成，更不能写成科学效果。

实际首epoch显示每个冻结selection达33MB，408epochs不能按初始模型大小估算容量。ADR043因此选择只对独立持久且完成、无人读取的缓存做完整字节硬链接共享和透明NTFS压缩，并把两份原失败stage1档有界上传百度盘、Ubuntu整包读回SHA后退休F重复副本。训练数学、数据pins、初始化、FP16/scaler与成功更新预算均未改。失败的权限/链接/reader探针保留，没有重跑参数更新。

## Working State
权威Task `Tasks/准备并核验四个新身份的 SyncTalk 完整训练数据.md` active/current_step4，step3完成；接近50KB，只改当前头部/observations或拆父任务关联卡，禁止按epoch追加progress日志。Spec `Experiments/SyncTalk 四新身份官方完整训练电平实验 Spec.md` 为IMPLEMENTED/统一准入passed/训练已启动。研究地图已同步。ADR038–043和analysis_contract_v1保持。

ROOT=/dev/shm/synctalk-multispeaker-data-20261017，FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017，Ubuntu wjj@192.168.50.2/V10016GB。SHM、PID、dirty脚本和scratch是机器局部fragile状态；F各完整档、验证和源快照为持久证据。先真实ps/receipts，不能以旧checkpoint或heartbeat的历史计数为当前状态。Git工作树1037项dirty，包含大量其他工作；无本轮commit/PR，不全stage/clean。

cohort_binding.VERSION=metadata_identity_replacement_v3，顺序SherrodBrown/PatrickLeahy/ChrisVanHollen/CathyMcMorrisRodgers。Chris namespace ChrisVanHollen_metadata_order_v1/post04，Cathy CathyMcMorrisRodgers_recording2_v2/post03，两个post/canonical workers全passed退出，不重启。正式统一data_qualification SHA fc437965bbf097a771f2d952508b688d981c52b65a99e266dadaa68290040a84，selection SHA cc27679ed33817158f138d33864d9f6ccea16279fb85482f70ae5a78235792c7。实际360loader/72资源/72fresh GT矩阵、共同完整初始化和原pixels/rays全通过，最大矩阵误差4.240466012106481e-06<原3e-05。独立准入review scope为receipt/source/remote字节和原数值门复核，不冒称又重复计算数组。

versioned_experiment_supervisor PID105372保持原进程，state official_24_model_training/training_started=true；同进程coordinator_restarts=0。最新receipt current_identity=SherrodBrown/seed42/coarse/ep2/gain6。本轮17:03后真实ps见train_epoch.py PID588168，Brown/seed42/coarse ep2/+6运行，非toy。首组三档coarse ep1每档6000成功更新/0skip，blocks/SherrodBrown/seed42/coarse_ep0001.json passed。独立18000行真实trace逐步frame/audioID/pixel/rays/rect/LR、完整state_before_epoch和sampling SHA一致，实际65536rays。phase-end60000/102000成功预算仍须验收。Brown监督16500文件全SHA、16428恢复+72复用完成；其他身份由原coordinator加载前full恢复，勿重复restore；每身份两seed后原handoff全SHA再退休监督。

全部24模型固定官方commit43079fb3f349675f6cb33031bb07262f7566cdd5；每身份两seed42/2026三训练档共同完整随机NeRF/条件state，固定预训练AVE，无May warmstart。原FP16/65536rays/maxsteps16/loss/AdamW/defaultGradScaler/EMA/fine交替lip-patch不改。6000帧粗10epochs=60000attempts，fine至17epoch累计102000attempts，408epochs/2.448M attempts；不是额外100000或102000epochs。保留每步AMP skips，phase末实际成功optimizer更新严格相等；失等CONTROL_FAILED，不补步/改scaler政策或precision掩盖。GradScaler正常增长允许，不能误当政策编辑。冷初始化6lip probe非finite梯度/0update原记录保留，不冒充实际fine结果。

24完整模型后自动三driver三scorer，27格/身份seed及四身份等权汇总，曲线、原距离矩阵、独立像素/距离/aggregation QA、科学报告与持久交付。评分0原PCM16，±6float64原PCM数值单位，source-owned ROI guard20主/full次、FP32SyncNet TF32off，FFV1/PCM正式/H264预览。analysis_contract_v1 SHA9ede60017cb5bfe5cbdfc66c47b98115d4e8081e00d64905440c2e64cf75adc2冻结：driver0验证平台只描述，不早停/按test选条件/加预算，n4条件探索不推人类感知、随机身份总体或数据充分。尚无本轮正式生成视频或生成SyncNet结果。

ADR043缓存observer PID133984/state watching_formal_epoch_persistence，当前已资格5个完成epochs，errors=[]。源`.codex/scratch/epoch_metadata_cache_observer_20261017.py`与F controller_qualification/epoch_metadata_cache_observer_attempt01/source冻结，必须复用不编辑活helper。仅原F逐成员持久验证、job PID退出且非当前epoch才处理；Linux独占F_SETLEASE/F_WRLCK实际reader拒绝控制通过，完整selection SHA后原子换相同字节hardlink。Windows独占READ/shareDELETE再SHA，持久恢复intent、unlink/close/os.link、finally恢复缺路径；仅已完成缓存存在非原子查找窗口。关闭trace/stdout透明NTFS压缩，原verify_persisted优化前后均过，archive/manifest不变。按唯一inode/nlink/实际分配量计，首canonical占空间不重复算释放。helper失败先核intent/原路径/canonical SHA并显式恢复，不能盲重启数值更新。

失败探针全部保留：WinError1314文件symlink、B目录junction Local volumes required、首次procFD不可读在0修改拒绝、attempt02 Linux成功后Windowsatomicreplace WinError5。独占shareDELETE实测通过后attempt03只恢复F缓存绑定，原验证前后过。没有将epochs移到C/B链接、没有删pins/trace/finalstate、没有改CloudDrive配置。

cold_rejected_stage1_archives_attempt01已completed/passed 2/2，PID125192不重启。原Debbie stage1 755426549B SHA39fdb63ab24ee4c2107a852b8ba5267aabb70672402e91933ec7aeffd55421ef，whole receipt f818a72c942e3653747254011651174c0ee25a36b2d24f198a8d19cc848463ee；原失败Cathy stage1 605838830B SHA79fe11935ccc437d6b2cec8ca0d8f645a4c523fd1045fb81d2ff6924a730f7bc，whole receipt03ed7781f9d132c3ed598ac7c1e8c1e99603379d6e3df0618a0545621385b11b。每256MiB Ubuntu独立读回+C backpressure+整包有序SHA、Freceipt字节精确和F最后SHA/独占无reader后，两份F重复tar已经退休；preprocessed_packages/<ID>_stage1_cold_storage_locator.json及worker/<ID>_retirement.json为准。原manifest/verification/PCM/几何/false保留，显式cloud restore后才能访问原archive路径。不要把旧source26中1/2快照当当前2/2。旧反转flow/bundle和四v1 canonical/Brown stage1整包cloud证明继续有效，不重复上传。

17:03后实测Ffree2397892608B/Cfree9861406720B/SHMfree1095110656B/root614109184B，真实df优先，未验证容量变化不归因清理。若继续需要空间，先独立确认已完成且无消费者的本任务material/stage1/flow准备档并用同bounded整包方法；保留当前准入必须derived/qualification/runtime和原下载MP4。旧BASE Cathy失败像素缓存是候选，但尚未执行退休：须cloud whole/原failure SHA/所有当前文件与inode无reader/当前cohort隔离及实际唯一链接释放证明，不能触碰合格Cathy revision。共享venv、CloudDrive配置、未验证原数不动。失败旧像素/档加载前必须显式恢复。

最新完整源码封存F execution_sources/metadata_replacement_first_paired_epoch_storage_attempt26：225主源码/文档SHA/AST，storage_supplement_manifest53份ADR043、helpers、失败与正资格、首组三档独立review和原block、云proof。16实际启动及17–25所有历史snapshots保留。checkpoint本次再核验总监督器loaded_source_capsule全部源码SHA、统一准入/selection SHA、两旧failed总receipt SHA精确。旧renderer/full false不改；新renderer状态render_delivery_supervisor_metadata_identity_replacement_v3_receipt.json，已接bounded cloud。不要编辑loaded sources或pins。

synctalk-27 ACTIVE同chat10min，保持现有自动化，无需再确认。未变无行动信息保持安静，仅关键完成/实际故障/需输入通知；全部24真实完整模型/27格/独立QA/报告/持久交付后才停用并保留chat。

## Changed Files
新增独立准入/首组三档trace review与storage probe/cache observer/cold helper于`.codex/scratch/`；更新Task current_step4、spec、研究地图并新增ADR043，F source24–26封存。实际数学源不改；helpers冻结副本在各F controller_qualification目录。当前dirty/未跟踪代码是机器局部状态，继续以前先对活sourcecaps核SHA，不能全量清理或stage。

## Verification
原20执行控制attempt09+binding5 attempt03/metadata3/factorial3/aggregation3都实际通过0skip，toy不是模型结果。四人独立准备、fresh统一实际360loader/72资源/72参考矩阵和完整初始化全通过；独立receipt review明确其范围。首组三档18000真实trace独立验证passed。Linux reader/lease、Windows NTFS hardlink/透明压缩/shareDELETE与原持久验证前后实测通过；失败探针保留。两原失败stage1整包独立CloudFS读回及最后F SHA/no-reader后退休passed。当前真实ps/df、live receipts、loaded core/selection/admission SHA和旧false精确重新检查；尚未完成任何完整模型/phase最终预算、视频或27格。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)：本轮只读GitHub get_repo确认，当前branch openPR查询空。
- 当前本地SHA `2b71639c3b555293375d248350999377168d515d` 未确认推送，不构造commit链接。
- FROOT/controller_qualification/unified_admission_independent_receipt_review_attempt01/review.json
- FROOT/controller_qualification/first_three_gain_epoch_independent_review_attempt01/review.json
- FROOT/controller_qualification/epoch_metadata_cache_observer_attempt01/receipt.json
- FROOT/controller_qualification/cold_rejected_stage1_archives_attempt01/receipt.json
- FROOT/training_coordinator_receipt.json，真实epoch jobs/receipts优先于快照。

## Observations
- [result] 四身份fresh统一准入已passed，正式24模型训练在运行；首个三档epoch各6000成功更新/0AMP skips、实际65536rays、完整初始化和逐步采样一致。
- [decision] 保持冻结数据/原FP16/loss/optimizer/scaler/EMA/65536rays和60000/102000attempts；实际成功预算不等保留CONTROL_FAILED，不补步掩盖。
- [decision] ADR043仅完成/独立持久/无人读缓存做同字节hardlink与透明压缩，历史失败stage1仅整包云SHA后退休重复档，所有失败和恢复locator保留。
- [blocker] 完整phase成功预算、24最终模型、视频/27格、独立结果QA和持久报告尚未完成；当前无实际控制失败，容量需持续按真实分配量监督。
- [next_step] 复用105372训练协调器与133984缓存observer，核真实epoch及phase-end配对预算并在已验证持久约束内维持空间，完成24模型后继续自动27格和独立QA交付。

## Relations
- continues [[Codex checkpoint - 2026-10-17T15-53-43Z - Chris full geometry qualified and automatic four identity experiment armed]]
- tracks [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- implements [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[043-synctalk-byte-exact-epoch-cache]]
- relates_to [[042-synctalk-metadata-ordered-identity-replacement]]