---
title: Codex checkpoint - 2026-10-17T15-53-43Z - Chris full geometry qualified and
  automatic four identity experiment armed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t15-53-43-z-chris-full-geometry-qualified-and-automatic-four-identity-experiment-armed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17T15-53-43Z
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

# Codex checkpoint - 2026-10-17T15-53-43Z - Chris full geometry qualified and automatic four identity experiment armed

## Summary
Brown/Patrick/Cathy已全独立资格，Chris已通过全部冻结几何门并进入官方监督/AVE等余下资格；版本化监督器已真实运行等待fresh四人准入，正式TFG更新和视频仍为0。

## Story
用户要检验音量变化在训练、驱动及评分三个环节分别产生什么影响。原May缩减预算先导之后，用户授权四个新自然说话人、每人6–8分钟、同人不同原录制test，四人全门通过后自动执行官方完整训练及27格。Debbie原配对和唯一角色反转均未过固定几何门，用户明确要求保留失败、不放宽门，按官方metadata固定原顺序换身份。ADR042/policy在下载前冻结WDA→WRA→RD文件次序、身份/录制首次出现、首连续标注段及首_000；Ben/Carolyn缺固定镜像源标SOURCE_UNAVAILABLE，第三Chris首可获取300+120秒保留，不按SyncNet/TFG选择。

这次恢复完成了Cathy新test2的全部独立资格和Chris全量素材、stage1、原BFM/heldout tracker、train flow/BA及完整v2几何。Chris全750/1500 heldout与两训练恢复控制都在原门内，几何pass不冒充完整数据准入。其post04继续原官方监督、六整录制AVE、新source-owned ROI/三评分音频、独立参考矩阵和加载/资源资格。全局入口已经统一Brown/Patrick/Chris/Cathy，versioned_experiment_supervisor真实启动，自动等候两candidate全资格、fresh全四人复核再训练；不需再次一般性确认。

空间采取可恢复的字节级策略：先F完整归档/逐成员SHA，再在无人读取时退休本任务缓存；失败反转flow/bundle还经百度盘分片及Ubuntu CloudFS整包读回SHA后才退休F重复副本。失败、原PCM及几何数值完整保留。这避免把缺文件、旧false receipt或局部通过误当作新实验结果。

## Working State
权威Task `Tasks/准备并核验四个新身份的 SyncTalk 完整训练数据.md` status=active/current_step3，最新头部与Versioned automatic execution handoff的实际段落优先，早期历史段落保留且不是当前状态。Spec `Experiments/SyncTalk 四新身份官方完整训练电平实验 Spec.md`、ADR038–042及analysis_contract_v1固定。ROOT=/dev/shm/synctalk-multispeaker-data-20261017，FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017，Ubuntu wjj@192.168.50.2，V10016GB。F持久档和source snapshots是交付证据；SHM、PIDs、dirty代码与scratch是fragile状态，恢复先真实ps/receipts。

Cathy revision CathyMcMorrisRodgers_recording2_v2，295+110=405秒，post_geometry_worker_attempt03_receipt passed=true/state revision_independently_qualified_pending_cohort_version_admission。官方监督/六AVE/新ROI及三评分参考/18GT独立矩阵、所有F档、loader90/完整65536ray资源18/preflight实际通过。canonical observer02已全资格、fresh archive和两视频SHA/no-reader后退休canonical，原字节F档。旧原pair几何失败、post02 bare interpreter缺face_alignment启动失败都保留；后者在任何监督前，不能当数值阴性或再换Cathy。Brown/Patrick已全资格但本地监督只有72控制，正式训练前必须从各自验证档full恢复。

Chris revision ChrisVanHollen_metadata_order_v1，contract SHAade2b500c2c254800ebba115a33b62996a83f84db7315d8dd92ef37fc4c59541，selection SHA86050b50da540ca1d32a438ce55f81e1d649c0882525a0f0276ce7c84bb62b4c。完整素材7500/3000帧原RGB/PTS/PCM/三gain逐值独立通过，exact_shared_RGB=0，非静音4秒probe全target lag最大abs相关.21941381950111874<.90未flag；官方发布日不冒称拍摄日或无上游污染。Stage1全8250帧、0failed/review_required=false，两个微小secondary检测保留。Primary486668已passed/geometry_qualified_pending_independent_persistence_and_remaining_gates；原BFM、val750/test1500固定identity/focal tracker、train NeuralNRT flow及500+8000 BA都完成，均不是TFG训练。

Chris完整geometry receipt SHA d747c317c14a58d501ca9067f1c10b474647be76b0e1a5efa00e52ec16acedb3已F逐字节保存。train6000 median/p95=.6805738726437918/1.4044356811654604px，val750=.6388439941457801/1.1557820658687024，test1500=1.0194198952621327/2.0015721947009335，原门median≤2/p95≤5。两恢复rotation p95 .04837010313986232/.06738756868230167°，translation .1870767626292723/.21312132085819438mm，原2°/5mm；condition50.635252598760616<1e6。early_complete_geometry_independent_gate_review_attempt01使用原受保护qualify_geometry与冻结点/重复flow/NumPy投影门重读验证，没有重拟合或改门。train-only flowobserver109152已completed/退出，F gzip900210625B SHA6029f7c5aeb374fbfc687208a4acb2a41390eb6b4f779139974ad507e46fad12、完整解压1039780128B SHA02ef7391f676256817d3ea32802ac4b153ab4c0c1d661f068b673830bb003516 fresh复核后退休原dense。不要重启或等待不存在的val/test dense flow。

Chris Windowspost04 PID28404当前state=supervision_and_AVE；material/stage1 F全member档已完成，实际supervision.py PID560325在计算眼部系数，GT/torso提取已结束。15:52UTC真实ps仍supervision，无AVE/TFG；canonical observer15524仍waiting，仅post04全资格及fresh原archive/两视频字节/no-reader后可退休。后续六AVE、新test ROI/三评分参考/18GT矩阵、derived/qualification F档及监督退休/control恢复、dataset/loader90/65536ray资源18/runtime档/独立preflight必须实际全过。15:52UTC C约9.40GB、F4.135GB、SHM1.1GiB/root548MB，真实df优先；监督2GiB开始门已实际通过，当前图像提取已完成，不能放宽guard。Audit jobs通过BASE/runtime_exec.sh，SyncNet只用own TORCH_HOME/PYTHONPATH，不混3.10deps或改共享venv。

总监督器PID105372真实运行，versioned_experiment_supervisor_receipt state=waiting_unique_candidate_qualification/training_started=false，launcher_attempt01已核验。cohort_binding.VERSION=metadata_identity_replacement_v3，IDs Brown/Patrick/Chris/Cathy，POST_RECEIPTS Cathy03/Chris04。loaded_source_capsule本checkpoint前逐SHA再核验精确；不得编辑已加载全局源、两candidate冻结namespace和六live local模块resume_revision_qualification/persist_cohort_prepared/persist_cohort_runtime_qualification/restore_supervision/qualify_all_data/execution_journal。必须复用105372而非重复启动。fresh四人准入会重验所有来源/原PCM/整录制AVE上下文/F所有档/不变pins、BASE实际每人loader90+资源18、完整初始化和原pixels/rays精确、72fresh audio/GT矩阵。无统一data_qualification/cohort_selection，旧false总receipts精确保留，不可继承旧Debbie参考。

通过后24模型=4身份×seed42/2026×训练三档；原FP16/65536rays/maxsteps16/原loss/AdamW/defaultGradScaler/EMA，6000帧粗10完整epochs=60000attempts，fine至17epoch累计102000attempts。实际成功optimizer更新预算必须成对一致，AMP skips记录，不补步/改policy掩盖不等；不等CONTROL_FAILED而非科学阴性。随后三driver×三scorer完整27格、每身份/seed和四身份等权汇总、曲线/原距离矩阵/独立像素距离aggregation QA/解释/持久交付。0评分原PCM16，±6float64原PCM数值单位，source-owned ROI guard20主/full次，FP32SyncNet TF32off。analysis_contract_v1固定平台只描述、不早停/选test/加预算，n4条件探索不推人类感知改善或数据充分。render_delivery_supervisor已接cloud_archive_bounded，新版本receipt与旧false分开。

失败Debbie原pair4.483/6.690px和唯一反转val2.0101095402098133>2px、Cathy旧pair2.372>2px全部保留。原Debbie stage1 F tar SHA39fdb63ab24ee4c2107a852b8ba5267aabb70672402e91933ec7aeffd55421ef完整41251成员再次SHA后，原preprocessed41250和prepared ori/parsing33000路径无人读/逐SHA退休，唯一payload937367552B，不重复算硬链接。原PCM/几何/旧false SHA未变；首次读旧partial geometry顶层passed字段KeyError在0删除时停止、failure/source保留，attempt02只修字段读取不改门。反转版本此前canonical881988635B/stage1唯一862913027B/六derived driverWAV161280480B也只有完整持久SHA/no-reader后退休。加载被拒绝/已退休像素必须显式恢复。

失败反转cold_backup_rejected_revision_attempt02已completed/passed，不再重启。flow1340711203B/5片 wholeSHA db2554c9e91d2d53e84d07f4577ff5098fc3e27e0aa998bb568a2a05a29292ed；bundle1781143937B/7片wholeSHA da2695fcabf7e6408530759b8be24427a3e3bd5f309d2e79834d64db133e4f45，原48108成员完整云档。逐256MiB每片Ubuntu读回/C backpressure再整包有序fullSHA，bundle210.839秒/payload暂存0。whole receipts及locators持久后F重复gzip/tar已退休，不能假定仍存在；Fthin15102文件409460992B全对原manifest独立SHA/size再次通过，原PCM/全部几何及false保留。independent_completed_cold_delivery_qualification_attempt01与源码固定证明，bundle receipt SHA dc4b326b21b6d613776a2113028d1e84f58d9d6aa798c84d48d123d47baf1f7f。旧四v1 canonical1.762GB/Brown597MB全cloudstream证明仍有效，不动CloudDrive配置。

F execution_sources/metadata_replacement_automatic_execution_attempt16是实际启动源222文件；运行attempt17、素材空间attempt18、stage1/cloud attempt19、flow attempt20都保留。最新metadata_replacement_geometry_passed_attempt21为225主源码/文档SHA/AST全通过及geometry_supplement_manifest五份几何/QA证据；TaskBM已读回。synctalk-27已同步上述最新事实，ACTIVE同thread10min，未变保持安静；全部实际结果独立QA、报告和持久交付后才停用，保留chat。若Chris任何真实资格失败，先保留完整失败/源码并安全封存核对0更新，再按预冻结队列推进，不编辑正在运行sourcecaps。

## Changed Files
本段工作新增独立几何/flow/cloud QA和窄范围失败缓存退休helpers于`.codex/scratch/`；更新Task并封存F sources16–21，启动版本化监督器、接入bounded媒体后端和统一cohort IDs。数学源/阈值/loss/scaler不变。当前git dirty 1037项，未commit/PR，大量既有其他任务变更，不能全stage/clean。活源以loadedcapsule为准，不因后续文档更新重启已运行数值阶段。

## Verification
实测Ubuntu原20执行控制attempt09 8.947s/0skip、binding5 attempt03 .036s/0skip，Windowsbinding5 .294s，metadata3/factorial3/aggregation3通过；F controller_qualification/cohort_execution_controls_attempt03固定source/log。toy控制不是实际模型结果。Cathy全部独立资格、Chris完整素材/stage1/geometry/F trainflow、失败缓存逐成员/当前文件/inode无人读和cloud整包/15102thin再次验证已实际通过；当前checkpoint重新验证geometrySHA、Cathy/cloud pass及总监督器全部loaded源码SHA。余下Chris资格、fresh四人准入、24训练、视频/27格均未完成。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)：本轮只读get_repo确认canonical，当前head openPR查询空。
- 本地SHA `2b71639c3b555293375d248350999377168d515d` 未确认推送，不构造commit链接。
- FROOT/versioned_experiment_supervisor_receipt.json
- FROOT/dataset_revisions/ChrisVanHollen_metadata_order_v1/post_geometry_worker_attempt04_receipt.json
- FROOT/dataset_revisions/ChrisVanHollen_metadata_order_v1/qualification_state/early_complete_geometry_independent_gate_review_attempt01.json
- FROOT/execution_sources/metadata_replacement_geometry_passed_attempt21/

## Observations
- [result] 三身份全资格，Chris全部冻结几何通过，post04后续监督运行；正式TFG更新/视频0，无统一准入。
- [decision] 固定metadata顺序首合格保留，失败完整封存，不放宽门、不按SyncNet选条件或改AMP政策掩盖预算差异。
- [result] 自动监督器已启动且loaded源码SHA精确，等待fresh四人全门；有界分片云交付与整包读回已真实完成。
- [blocker] Chris余下完整资格及fresh四人实际加载/初始化/评分参考复核尚未通过，不能启动正式训练。
- [next_step] 复用Chris post04和canonical observer及总监督器105372，监测真实supervision→AVE→参考/持久/loader90/resource18/preflight，四人fresh准入通过后监督器自动执行24模型和27格。

## Relations
- continues [[Codex checkpoint - 2026-10-17T14-36-26Z - Metadata ordered Chris replacement frozen and independent qualification resumed]]
- tracks [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- implements [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[042-synctalk-metadata-ordered-identity-replacement]]