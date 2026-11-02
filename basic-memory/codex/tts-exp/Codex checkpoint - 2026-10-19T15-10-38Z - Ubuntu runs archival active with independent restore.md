---
title: Codex checkpoint - 2026-10-19T15-10-38Z - Ubuntu runs archival active with
  independent restore
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t15-10-38-z-ubuntu-runs-archival-active-with-independent-restore
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
username: LAPTOP-E133GSLE/dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1543e-4ce2-7401-bd94-9dfddaf8204c
trigger: compact
model: gpt-6.1-sol
started: 2026-10-19T15-10-38Z
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-19T15-10-38Z - Ubuntu runs archival active with independent restore

## Summary

Ubuntu旧runs的可恢复云归档正在执行，首批已真实释放4.383754730224609 GiB；至少60 GiB的目标仍未完成，不能以后台上传或盘点体积代替验收。

## Story

用户最初希望找清历史实验数据并整理百度网盘，随后明确要把Ubuntu的暂时不用的旧runs转到网盘，尽量腾出60G以上。此前已把12个项目云目录统一到B:/tts-exp，保留原子目录结构，Windows和Ubuntu验证命名空间与小锚点SHA；路径映射在B:/tts-exp/_index/path_mapping.json，仓库AGENTS.md和ADR049记录新根，不创建旧根兼容链接。

当前采用逐批完整归档、独立Windows云端回读并实际恢复每个文件、再fresh核对源树后释放。Ubuntu直接产出64MiB gzip/PAX tar分片；独立消费者将每个文件体写入F盘有限探针并重新打开做SHA，全部分片/压缩流/成员/POSIX元数据绑定最终producer manifest。只有完整通过后，retire按精确manifest unlink源文件、rmdir空目录，并留下CLOUD_ARCHIVE.json及小报告。活跃的May训练、10月实验、数据集、模型与环境保留。此方法需要云端上下行时间，但不依赖仅有分数或“上传成功”标记来删除实验原件。

## Working State

Durable authority: Task ubuntu-runs-cloud-retirement-60g-20261019与ADR050；实际证据以per-run source/transport manifests、independent_restore_receipt和retirement_receipt为准。云payload在B:/tts-exp/ubuntu-runs-archive/20261019/batch02/<run>/attempt01，Linux对应/home/wjj/百度网盘/tts-exp/ubuntu-runs-archive/20261019/batch02。批次README/catalog只是可刷新导航快照，目前仅首批完成。

Machine-local/fragile authority: Windows .codex/scratch/ubuntu_runs_archive_20261019/client_attempt04/coordinator_status.json；活跃coordinator_v4.py PID265156，verify_live.py逐文件回读。Ubuntu控制根/home/wjj/tts-audio/tts-exp/.codex/scratch/ubuntu_runs_archive_20261019_attempt02，producer/retirer为冻结archive_remote_v3.py。不要覆盖正在运行的源码，proof含实际源码SHA。Ubuntu SSH wjj@192.168.50.2以BatchMode公钥连接，无密码文件。

7个选中根盘点约61.456 GiB：lrs3_phone_rules_full_20260921_proxy；phone_gain_static_tfg_mfa_repair_20260923_sync_c_v3；mfa_linear_intraphone_time_20260925；aishell1_qwen_mfa_linear_n100_20260820；mfa_linear_video_retiming_v1；local_tts_spectral_repair_20260925；lrs3_mfa_scorer_effect_20260923。7931个保留树symlink扫描对7根无incoming引用。另两根分别有1520外链、912incoming被排除，详情candidate_inventory.json/incoming_links.json。

首批已全量恢复1446成员/1423文件、4703724231源字节、3711480841压缩字节/56片；archive SHA2961edfffc53339152bb60d2dbc2595c28bac005564fd78e043741b435716a80，source manifest SHA6f6b41097e234f571e4ef7d45a8b14c4a98ed822a24e605d174ced0516780152。实际退休释放4.383754730224609 GiB，原路径约56KiB小报告与云指针。原transport误写Windows batch01，保留原件并由transport_resolution_addendum修正到真实batch02；不要改旧receipt或重复归档。

第二批phone_gain当前producer PID154968；2026-10-19T15:09:43Z状态UPLOADING，源37156534063字节/324469成员，已产23238541312压缩字节/346片。独立消费者此前已实际恢复约14.6GiB，仍未获整批退休资格。WAIT_NEXT_UPLOADED_PART表示正常等待完整已上传片，不是失败；该状态省略restored_file_bytes字段，不能当0或重置。不要重启已活跃的uploader或coord。

Ubuntu cache guard PID161076只核本次allowlisted archive生产者的UID/命令/启动ticks：根盘小于3GiB暂停producer，超过4GiB恢复；CloudDrive继续上传。最后df可用约4.27GB、上传缓存约6.44GB，暂停154968属正常节流。完成后创建CONTROL/cache_guard.stop或SIGTERM该guard使finally恢复它暂停的有效producer；不可SIGKILL留下stopped进程。不能改CloudDrive配置/DB/缓存或停止它。Windows最后C约278MB、E2.46GB、F2.16GB；F每次仅一个恢复文件（当前最大约130MB），避免Windows上传写C缓存。Z是同一Ubuntu SMB，不能算独立备份。

## Changed Files

本请求已修改AGENTS.md的统一云根说明，创建/维护Basic Memory ADR049、ADR050和上述Task。归档器、消费者、coordinators、tests、清单及回执位于scratch（机器局部、未提交），云端_verification_sources已发布冻结源码与SHA manifest，live_overlap_v1子目录含当前consumer/coordinator和测试。全仓有大量其他会话dirty/untracked代码，不属于本次清理，不清理或提交它们。当前tracked dirty还包括CONTEXT.md、audio_enhancement、portable_source_axis相关文件及测试；未创建commit/PR。

## Verification

本次实际已运行：3项remote归档/边界/无proof禁止退休测试；2项UTF8路由/越界测试；1项guard仅管理本任务的scope测试；2项live分片最终绑定/坏hash拒绝测试；2项live端到端真实body恢复/坏whole SHA禁止admission测试；8MiB Ubuntu→云→独立Windows SHA探针；首批4.38GiB完整真实恢复和fresh source退休。没有跑整仓测试或新科学推理/训练。

保留失败：attempt01过度科学进程互斥导致0payload/0删除；首caller误batch namespace等待；v2 Windows默认编码失败；v3修UTF8并完成首批后，在第二批等待期安全停止，由v4复用既有producer及首批proof实现上传与独立回读重叠。旧失败、原源码和原transport均保留，没有放宽恢复或源核对门。

GitHub只读确认仓库身份；当前分支无匹配PR，commit API对本机SHA返回422未找到，所以该SHA仅标本机、不构造远端commit链接。gh CLI不可用，改用GitHub connector完成查询。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Local Git snapshot: `2b71639c3b555293375d248350999377168d515d`（GitHub不能解析）。
- Cloud navigation: B:/tts-exp/ubuntu-runs-archive/20261019/batch02/README.md。
- Actual cursor: .codex/scratch/ubuntu_runs_archive_20261019/client_attempt04/coordinator_status.json。

## Observations

- [result] 已确认回收4.383754730224609 GiB；至少60 GiB未达成，任务保持active。
- [decision] 数据释放必须在独立完整内容恢复、最终manifest绑定及fresh源复核之后；current/10月与其他会话科学任务保留。
- [blocker] 无当前失败；主要等待云端实际上传和独立回读带宽，cache guard暂停不代表故障。
- [next_step] 继续监看client_attempt04至completion_receipt passed且实际回收与最终df净增均至少60GiB；若不足则加经同样检查的新旧run。完成后停止guard、刷新云catalog/发布闭合恢复日志、fresh df/locator检查，并将Task结案。

## Relations

- tracks [[ubuntu-runs-cloud-retirement-60g-20261019]]
- implements [[050-ubuntu-runs-cloud-retirement-after-independent-restore]]