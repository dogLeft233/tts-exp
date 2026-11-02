---
title: 050-ubuntu-runs-cloud-retirement-after-independent-restore
type: decision
permalink: tts-exp/docs/adr/050-ubuntu-runs-cloud-retirement-after-independent-restore
status: accepted
project: tts-exp
decided: '2026-10-19'
decision: Ubuntu 旧 runs 完成分片上传后释放；前两根采用独立全内容恢复核验，剩余五根于2026-10-20遵用户明确的流量限制豁免下载核验，保留fresh本地源和云元数据检查。
rationale: 用户要求把暂时不用的 runs 转到统一百度网盘，并实际腾出至少 60 GiB；Windows C 盘缓存余量不足，Ubuntu 也不能先生成大包。
alternative:
- 直接目录移动或仅复制成功就删除；无法证明完整可恢复。
- 在 Ubuntu 本地先生成完整大 tar；根盘只有约 5.7 GiB 空闲。
- 经 Windows 写大包上传；C 盘当时只有约 12 MiB 空闲。
consequence:
- 采用 64 MiB 有序 gzip/PAX tar 云分片和完整文件 SHA manifest。
- 独立 Windows 从网盘读回所有分片及每个文件 body，并在 F 盘有限探针实际恢复与复读 SHA。
- 释放前重新完整哈希源、检查精确候选进程引用，保留原路径的恢复指针和小型报告。
---

# Ubuntu 旧 runs 的可恢复迁移

## 2026-10-20 用户明确修改核验策略

用户先要求暂停核验，随后明确“直接清理ubuntu空间吧，我网络的流量不是很足了”。这条最新授权覆盖本文后方旧的完整下载恢复准入要求，适用于已完成上传的剩余五根旧目录。此前已完成的两根独立恢复凭据保持不变；其余五根不得伪写完整恢复通过。

新的 `retire_uploaded_only_v1.py` 在 Ubuntu 本地复核 producer 清单/源 after 一致性、fresh 全文件 SHA 和身份、保留树 symlink 与候选活动引用，云端只查询分片名称和大小，不读取压缩包内容。通过后仍按确切 manifest unlink 文件和 rmdir 空目录，留下小报告及 CLOUD_ARCHIVE.json。单独的 `retirement_without_download_receipt.json` 明确 `independent_restore_verified=false`、`verification_waived_by_user=true` 和下载 payload 字节数 0。上传时的分片/全流 SHA 保持，恢复前可另行核验。代价是五根尚没有独立全内容恢复保证；这是用户节约流量的明确选择。

5 项隔离测试已通过：无 cloud body 读取的正常退休，源变化/活动引用/云片缺失/名单外根均拒绝。最终7/7清理完成：2026-10-20T02:01:11Z空闲71,184,130,048字节，相对起点5,570,371,584字节净增65,613,758,464字节（61.107574GiB），达到60GiB目标。两根完整核验、五根用户豁免的状态保持分开；7根stub集合、报告SHA和locator最终本地审计通过。权威终局在控制根 retirement_without_download_20261020_attempt01/completion_receipt.json、final_local_audit.json，网盘README/catalog已更新。清理worker及旧下载worker均退出，不自动再下载。下面内容是原方案及历次实现，保留作来源记录。

这是用户“将 runs 旧目录转到百度网盘、尽量腾出60G以上”的执行策略。数据目标统一为 `B:/tts-exp/ubuntu-runs-archive/20261019/batch02/`，Linux 对应 `/home/wjj/百度网盘/tts-exp/ubuntu-runs-archive/20261019/batch02/`。保留原始数据集、环境、10 月资产与其他会话的 May 训练。

源根为 `/home/wjj/tts-audio/tts-exp/runs/`，精确七根名单在 Task [[ubuntu-runs-cloud-retirement-60g-20261019]] 与本地协调器。盘点约 61.456 GiB；最终必须同时看实际退休回执和 df 空闲变化，不以计划数报完成。7931 个保留树链接扫描中，另一个候选被 912 条链接引用而排除；带 1520 外链的旧 LRS3 队列亦排除。

首次 attempt01 在任何 payload/删除之前，因为不区分候选引用而阻断其他会话的 Python 训练。原失败保持；新的 attempt02 允许不关联的任务，但命令/cwd/root/fd 命中源候选或科学进程 fd 无法读取时仍停止。这个普通用户观察不宣称 privileged all-UID 完整观察；源在归档后与退休前的全文件 SHA 和身份比较另行约束快照变化。

每批完整保留普通文件、目录、链接、mode/uid/gid/mtime ns 与逐文件 SHA。生产端根盘低于 1 GiB 时等待上传缓存释放，不修改 CloudDrive 配置、数据库或进程。独立 Windows 读回每片与全流 SHA，解析全部 PAX 成员，一一匹配集合/元数据；每个普通文件 body 都真实写入 F 盘唯一受控探针，重新打开全量 SHA 后才移除该探针。它是全内容有限磁盘恢复探针，不是已在 Windows 重建可立即运行的 Linux 绝对路径树。

科学输入和输出不被重新筛样或重跑。只有独立恢复 receipt 完整合格、fresh 源树逐内容/身份复核和精确关联检查通过，才按原 manifest 逐项 unlink、rmdir 已空目录；不泛化为递归删除权限。原位置重建 CLOUD_ARCHIVE.json 与选定小型原报告，便于今后发现并恢复。失败停留在可核验状态，未通过的原件保留，不自动覆盖或重试已有目录。

## Observations

- [decision] 用户的空间回收通过可恢复迁移完成，源释放依赖独立全内容恢复和 fresh 源核对。
- [constraint] 当前训练不因旧目录归档而暂停；原始数据集、环境和 10 月资产保留。
- [verification] 三个隔离 helper 测试与 8 MiB 真实跨机器传输测试通过；实际批次状态以各自 receipts 为准，本文不宣称已释放 60 GiB。

## Relations

- extends [[049-unified-baidu-project-root]]
- relates_to [[研究服务器缓存清理与可恢复归档]]
- affects [[ubuntu-runs-cloud-retirement-60g-20261019]]
## 有界恢复探针的存储补充：独立输入仍来自B，Z仅作临时落盘

Windows E/F空闲同时不足以安全容纳后续1.262GB单文件加余量，不能通过跳过物理恢复、只验分片或降低空间门来推进。采用Windows独立Baidu挂载读取原归档流，再把每个解出的body写到Ubuntu专用SMB scratch，Windows重新打开全量SHA，校验后立即移除。这里Z是同一Ubuntu磁盘，**不是独立备份**；独立性指归档输入仍从Windows B挂载取得，绝不把Z上的源文件/上传缓存当云副本。恢复receipt明确记录restore_probe_storage、restore_probe_is_backup=false及这种存储边界。整包/每片/全部成员与POSIX元数据/每文件SHA/producer source-after相等/fresh source退休复核全部保持。

实际8MiB B→Windows→Z scratch→Windows重开SHA→Ubuntu直接SHA检查通过，SHA1b95a82cb6dddd4ccfae75914d63f5b11ae34d418d1e32847926de0bfd4fc4c6，临时body已删除。新verify_live_smb_v1.py仅改变receipt描述与存储字段，所有admission之前的验证代码逐字符一致；完整body恢复fixture与坏whole SHA不能产生admission两项通过。不会在原两批之中换验证器；待第二批已有closed retirement receipt后才启用。新caller v6在F盘存小型控制元数据，Z专用scratch每批用数字ID，避免restore path误触原run引用扫描。当前第一、二批仍由原v4及F探针执行，原源码和proof不改。
## Windows磁盘耗尽后的新完整验证attempt

F降至约183MB、E约95MB后，原F验证attempt04在约25GiB已读处安全暂停；保留其内存状态与原partial日志。实际启动coordinator_v7（client_attempt07）：从原第二批完整归档开头执行一轮完整Windows独立B输入→Ubuntu专用SMB body探针→Windows重开全量SHA。并未把旧partial当closed proof或跳过任何成员；未来第一批已closed proof按原SHA绑定复用。新metadata亦在Ubuntu专用scratch，每run用数字ID，避开原run名称引用检查。

源归档、原producer、原manifest/whole SHA与fresh源退休规则保持。SMB探针与metadata不是独立备份；独立云副本仍在百度网盘，输入由Windows独立挂载读回。原attempt04在新完整第二批退休前可择机恢复，或在被新proof替代后终止；全过程只允许一个caller发布/退休同一run。具体活动路由与暂停拥有关系以Task和active_route.json/preserved_v4_attempt01/state.json为准。
## Qualified physical restore transport: persistent SSH pipe

Windows独立B读取和解析全部tar/PAX成员、逐片/全流SHA不变。仅将有限探针的每文件SMB元数据往返替换为固定源码SHA的SSH helper：Windows按64位长度帧发送真实云文件body；helper只使用自建数字scratch，检查256MiB余量，xb写入/fsync/关闭/重开全SHA/精确unlink，返index、字节数、写入SHA和重开SHA，Windows同时对body计数与SHA并核对全部响应。每批关闭时比对总文件/字节计数，helper确认空探针目录后删除；坏body、坏全流或协议失败不产admission。helper不接收run源路径，也不读取原件/上传缓存。scratch仍不是备份。

1024个真实B云输入分段（含空文件）写入与重开SHA通过3.52s，旧SMB为23.44s；新增坏body/whole、超长/截断/低空间拒绝测试和不相交分区聚合测试通过。当前冻结实现verify_live_pipe_v1.py和restore_pipe_server_v1.py/client_v1.py；版本与SHA、原失败、实际当前caller在Task和active_route.json，源码发布batch02/_verification_sources/framed_restore_pipe_v1。此方式仍为全内容有限物理恢复验证，不宣称已完整重建可执行Linux目录树。目标状态仍未完成。