---
title: 035-privileged-process-attestation-for-shutdown-archive
type: decision
permalink: tts-exp/docs/adr/035-privileged-process-attestation-for-shutdown-archive
status: open
project: tts-exp
decided: '2026-10-10T20:32:05Z'
tags:
- archive
- reproducibility
- shutdown
---

# 035-privileged-process-attestation-for-shutdown-archive

## Context

用户授权当前20轮自然结束、安全归档后正常关闭192.168.50.2，明天自行重启。20轮/121events、原数组及tell全数学和完整sealed journal已实际通过ROOT资格。SHM包含唯一未持久化科学产物、实际QUEUE和runtime，完整备份验收前仍需保留服务器。worker11单次真实preflight在owner进程读取门失败，没有输出其当时详细观察；原11及完整失败捕获保留。

后续diagnostic01两独立时点各85个UID1000进程、0refs/0races/141 EACCES incomplete，三个PID/ticks1809/1166、1826/1168、2139/1210一致。ROOT完整捕获资格QC `.codex/scratch/root_writer_diagnostic01_actual_qc_attempt01.json` self4e9f1c0bc96c5e21f5aaa9f6c4c7722832f212137919b20e96b82b48ba5b9e51。诊断证明后来这两个时点的普通用户观察不完整，不是11早先失败细项的直接捕获。旧initial root ALL_UID06已实际完整观测497进程，但不能用旧时点冒充当前归档前资格。

## Choice and rationale

按ADR028保留owner wjj归档、仅固定只读/proc scanner使用root。fresh worker12采用明确的两阶段准入：先捕获完整但UNADMITTED的inventory，ROOT独立验证实际来源/closure/每文件hash/stat/数量/容量和所有20数学等证明，再取得fresh privileged scanner06归档前时点的实际全UID资格。ROOT create-only seal绑定真实inventory response raw/self、source manifest raw/self、完整scan/source/request/invocation/实际rc与精确scope，才释放单次archive。

inventory-only保留原owner scan诊断，任何观察到的references/races仍硬阻断；EACCES保持原FAIL_CLOSED证据，不做PID白名单、不转换成PASS。资格来自fresh root ALL_UID06对同一完整scope的实际可读观察，要求UID0、原scanner06完整body/raw pin、hidepid0、全可见PID/UID roster、每进程两个stable channel snapshot、0payload refs/0incomplete/0races；transport ancestry仍在扫描表，不泛化跳过。scope的fixed/linked-home/external目标必须与实际inventory闭包精确一致，不能仅覆盖19个根字符串而漏掉新增target。

fresh root prestream scan结束至实际archive进程启动最多300秒；超时闭合失败，保存失败和全部原件，下一次只能ROOT重新资格后fresh显式释放，不能自动重试。记录进程真实启动和内部重新inventory的起止时间。wjj stream必须重新完整inventory且精确匹配封存manifest/源stat与hash，仍执行每member dirfd/no-follow身份、全body SHA、前后stat、end-of-tree和external依赖复核、whole archive SHA、完整member集合/类型/模式/link/time/bytes/hash及fresh所有body restore重新打开、全部执行source pins。旧initial只作谱系证据，fresh prestream是独立META/receipt，不能原地改旧proof或把不完整owner scan当PASS。

归档内容和全部restore资格完成时仍标PENDING_FINAL_PRIVILEGED_ATTESTATION。使用保留的original final04标签，在完整restore后/关机前对实际同scope运行fresh ALL_UID06，完整五原始文件读回与ROOT资格通过后，ROOT另写create-only最终admission。后取得的final scan不能伪装成已关闭tar里的内部记录。只有最终admission和全部原安全停机门通过后才正常shutdown，保存sudo接受/系统关机消息与后续外部SSH22不可达；actualoff后await_user_restart并暂停既有tfg。

这是两个明确时点的完整观测，不声称持续无writer。流中科学快照稳定性由精确reinventory、逐body hash和metadata前后、全member/restore证明约束。未增加不必要的before-inventory或post-stream重复扫描，原协议已有的归档前及restore后资格均保持。密码只私有PTY无回显stdin，不写文件/argv/BM/log，不改sudoers、ptrace、安全配置或用户daemon。

## Alternatives and consequences

忽略141 EACCES或按PID/进程名特判会丢失完整观测；反复重跑相同普通用户预检无法解决权限来源；旧initial缓存无当前资格。全worker作为root扩大文件写权限并改变owner，违背ADR028。额外四五个重叠扫描没有连续覆盖保证，增加执行和schema复杂度；采用原协议两个fresh时点并显式记录时间与内容稳定边界。root SSH key、sudoers、capabilities或ptrace配置修改均没有必要。

这要求fresh source successor更新所有inventory/preflight/readback/META/restore validator的schema和状态，原11、旧initial及所有失败不变。修复由原GPT-6 Luna/max实现，ROOT及原ASR先审源码后释放一次实际动作；不执行实现测试或GPU。完整全body备份仍含19固定根与所有转接目标，不减少科学媒体、runtime、原PCM/frame maps或source payload。CloudDrive本地读回和cloud sync单列，Windows材料化与Linux原绝对路径可重建资料单列。科学cohort/policy01/ADR031/原PCM时钟/全部效果门不变。

## Evidence and affected work

- `.codex/scratch/root_archive11_preflight_failure_actual_qc_attempt01.json` self18daaa9651dc8a36e37ade166b9866a47b7c7eabe84c869ec976aec341eff66b：唯一11实际preflight失败，无manifest/archive。
- `.codex/scratch/fit40_session09_writer_scan_diagnostic_attempt01/capture_attempt01/`：实际后来两次FAIL_CLOSED capture；receipt self5156b566d10ffc50364009ce61ee773ba22d53da3b4b79abadbe321b9a5104da。
- `.codex/scratch/fit40_session09_poweroff_archive_attempt07/all_uid_process_writer_scan_attempt06.py`24154B/rawf81742747c23af339135e615642d5ea40f88eb8003cee0d539e9df13a67e3896保持不变；新prestream launcher/worker12 source尚在实现，没有新预检或归档资格。
- [Repository](https://github.com/dogLeft233/tts-exp)，branch codex/cross-tfg-audio-head-20261003，HEAD5688784dd5c6c1272074f0c4b6c5ff79f117c907 local/unpushed。

## Observations

- [decision] wjj两阶段inventory/stream，fresh root全UID prestream资格绑定实际inventory与全部source/self/rc/scope；300秒内实际启动，完整restore后fresh final及另写ROOT admission。
- [rationale] 后来两次普通用户诊断均有141 EACCES，保留FAIL_CLOSED；实际privileged完整观测解决可读性，逐body/stat/reinventory保证归档快照稳定。
- [alternative] 不忽略EACCES、不复用旧initial、不杀daemon/改安全配置、不用root执行归档；不增加无持续保证的重复扫描阶段。
- [consequence] fresh12所有caller/META/validator区分UNADMITTED/content-pass-pending-final/final-admitted，原11和所有原件保持；任何权限、引用、race、scope、来源或restore门失败均保留服务器。

## Relations

- relates_to [[028-owner-cleanup-with-read-only-privileged-proc-scan]]
- relates_to [[研究服务器安全停机与完整可恢复归档]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]