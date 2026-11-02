---
title: 043-synctalk-byte-exact-epoch-cache
type: decision
permalink: tts-exp/docs/adr/043-synctalk-byte-exact-epoch-cache
status: open
decided: 2026-10-17 16:47:38.826926+00:00
project: tts-exp
tags:
- synctalk
- storage
- exact-bytes
---

# 043-synctalk-byte-exact-epoch-cache

## Decision
在用户已有空间处理授权内，使用“完成且独立持久校验后的精确字节缓存优化”：完整冻结cohort_selection的逐SHA硬链接共享、关闭的trace/stdout透明NTFS压缩，以及已失败准备版本历史stage1整包的有界云冷备份。所有原逻辑路径/字节、模型初始化、数据pins、FP16/GradScaler/optimizer/EMA和训练预算保持；不编辑正在执行的冻结副本或已加载controller源。记录状态open，用于持续核验这项已实施工程选择，不额外引入用户批准门。

## Context and rationale
四人数据统一准入已通过并实际开始24模型训练。首个完成epoch的F输出有33039178B冻结selection、3634293B逐步trace、8370630B完整artifacts gzip及20995755B全checkpoint；消耗旧state轮转后仍有大量重复元数据，408epochs不可按11.7MB初始模型state估容量。F当时约2.5GB空闲，SHM约1.2GiB，保持源与预算的存储处理已成为具体执行需求。

硬链接完整JSON不裁剪任何file pin，普通读取/原SHA完全相同。NTFS压缩由文件系统保留逻辑字节，原归档/manifest不改。对已完成、原F逐成员持久验证且任务PID退出的epoch才操作，运行中的epoch明确排除。首个原始未修改persist_training_epoch.verify_persisted在缓存优化前后均通过，首个0dB完整archive SHA ef23a9e18b7139b261457eb234afab19c173fac74b1be55ed5da2dd1d34173b1保持。0/−6两个完成epoch已由observer核验，不能把缓存资格写成科学结果或phase成功预算。

## Qualified implementation and recovery
Linux采用已实测的独占F_SETLEASE/F_WRLCK证明目标没有其他打开者；实际reader控制在打开时拒绝lease、关闭后通过。完成job父/子PID须退出，原目标/完整canonical逐SHA一致，再原子换入相同字节hardlink。第一个canonical的创建会占同样payload，不能把首个链接逻辑33MB全部宣称释放；后续按唯一inode与nlink计数。

F上先独占共享删除handle确认无其他reader，持有handle再次读完整SHA；原F archive已验证且canonical完整。缓存重绑定前写持久恢复intent，在handle关闭后以os.link恢复相同字节路径；finally恢复缺失路径，非原子查找窗口只针对完成epoch的缓存，不能用于活文件。随后原verify_persisted再次必须通过。NTFS压缩后的trace/stdout逐SHA与原值一致；原归档和manifest留存。canonical原SHA、训练实际input、sourcecaps均不得改变。

当前两个新辅助进程：epoch_metadata_cache_observer_attempt01 PID133984只观察已完成F epochs并执行上述优化；cold_rejected_stage1_archives_attempt01 PID125192仅队列原失败Debbie/Cathy stage1。后者采用原已资格cloud_archive_bounded：每256MiB独立Ubuntu CloudFS读回、C缓存backpressure、最后整包有序fullSHA和F receipt字节核验，完成前不删除F源。只处理不在当前准入archive_proofs的指定stage1旧档，原manifest/verification、PCM/失败几何/false receipts保留，增加locator，原路径使用前必须显式restore。决策采集时上传在运行，尚无两个完整交付；后续实际交付与缓存退休以worker/locator/整包receipt为准，不能用排队标记代替云证明。

## Alternatives and preserved failures
Windows普通文件symlink探针因WinError1314不可用，F到B目录junction探针因需要本地卷不可用，均未搬移科学数据，未改CloudDrive配置或申请权限。Linux首次进程枚举因4个FD不可读在0缓存修改时拒绝，改用更强的kernel lease资格，不忽略拒绝读者。第二次已成功Linux exact-byte dedup，Windows持有独占handle的atomic replace得到WinError5，原F文件与canonical/temp均保留；独立删除共享能力探针通过后，attempt03仅恢复F缓存绑定并通过原验证。全部失败source/proof保留，未重跑任何参数更新。

不能以删pins/丢trace、缩减65536rays或epochs、放宽几何/AMP预算门解决容量。当前未迁移epoch档到C/B junction，也未删除实际训练归档或最终checkpoint。后续若仍需空间，应先按真实文件/分配量与独立持久证据规划，不能假定已失败的链接能力可用。

## Consequences and affected work
守护进程须复用现有PID、源封存，不编辑已加载源。任何kernel/Windows reader guard失败都保留原cache并延期/停止辅助worker，不能绕过门。粗10/fine17完整state和所有trace/曲线的逻辑字节保留；实际optimizer成功预算仍仅由原phase-end配对资格验收。辅助进程失败不会把训练/科学结果改pass；停止旧数值job或blind replay仍禁止。

证据在FROOT/controller_qualification/completed_epoch_selection_dedup_attempt01–03、tmpfs_file_lease_attempt01、storage_native_links_attempt01、storage_cloud_directory_junction_attempt01及两个observer目录。真实辅助源码在.codex/scratch/epoch_metadata_cache_observer_20261017.py、cold_rejected_stage1_archives_20261017.py；F目录有冻结副本。现有BASE训练/渲染/评分源码不变。

## Observations
- [decision] 对独立持久、已完成且无reader的缓存做完整字节硬链接共享和透明压缩；原失败stage1仅整包独立云SHA完成后退休F重复副本。
- [rationale] 真实epoch重复metadata会超过现有空间；精确字节存储优化保持所有科学输入、预算和验证函数。
- [alternative] 不删pins/trace、不改训练预算，不采用已实测不可用的Windows file symlink或B目录junction。
- [consequence] 每次资格须保留完整SHA/no-reader/原验证器复验与可恢复intent；云副本未全验不得删除源。

## Relations
- affects [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- affects [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[038-synctalk-training-owned-geometry-and-lazy-loading]]