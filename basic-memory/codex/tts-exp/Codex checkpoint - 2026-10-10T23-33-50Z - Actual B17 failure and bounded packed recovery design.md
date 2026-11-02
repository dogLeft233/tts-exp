---
title: Codex checkpoint - 2026-10-10T23-33-50Z - Actual B17 failure and bounded packed
  recovery design
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-10-t23-33-50-z-actual-b17-failure-and-bounded-packed-recovery-design
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-10T23:33:50Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-10T23-33-50Z - Actual B17 failure and bounded packed recovery design

## Summary

20轮训练、全数组数学和journal封存均已完成；B17完整清单与fresh进程准入通过，但实际归档只产生880050176字节的截断TAR，服务器仍开启，现转向有界分块和独立完整内容恢复包装。

## Story

长期目标是解释TTS对TFG的局部声学增强机制，并实现跨TFG、未见说话人超过固定RMS的自然音频增强头。增强PCM只驱动视频，完整原自然PCM及原时钟用于评分和交付。ROOT维护理论、冻结协议、Basic Memory和独立实际产物QA，代码和实验沿用原GPT-6 Luna/max代理。

用户最新要求在安全节点关闭局域网192.168.50.2，明天自行重启；今晚不运行四中心、Dev、Confirm或新的GPU任务。关机将丢失SHM执行资产，因此先保存完整可恢复闭包。B17 inventory05与ROOT QA08通过后，新prestream06通过全UID扫描并由ROOTseal04封存。archive05实际SSH在扫描结束222.355秒后启动，来源和逻辑B容量门通过，但SSH/worker/wrapper自然rc2退出，零远端stderr。TAR只写出约0.88GB，没有fresh restore或最终内容回执。C盘在写入时降为0，退出后恢复约852MB；这是缓存压力的相关证据，底层错误原因尚未捕获。

ROOT已完整读取并散列截断TAR、核验实际进程和回执，将失败与科学结论分开。最窄历史C重复包删除被自动审批策略拒绝且未启动进程，不能换途径重试。Dev与Signal源码审阅建议保留immutable inventory17/QA08与producer17身份，新增独立executor18，使用64MiB create-only TAR块以及独立完整bodypack和精确POSIX索引，避免单个增长云文件和约60万个云端小文件。该方案此时仅完成源码推理，尚无ADR036、source18或launch。

## Working State

服务器wjj@192.168.50.2:22仍开启，已知coordinator/model/archive子进程都退出；GPU最近实际只读0%、5MiB、无compute app。没有活动私有PTY或归档job。密码/API只走runtime或私有PTY无回显stdin，不写文件/argv/BM/log。不得重启GPU实验、重复math或seal。

JOB=/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/job_control；session09 ID fit40_spsa_bfa027430723_6800c380298d。原QUEUE cwd/source_root=/dev/shm/fit40_queue_remote_package_attempt02/project；dispatch=/dev/shm/fit40_joint_spsa_execution_clean_rerun_attempt03。package12实际package11 runner08、原绝对路径与source pins不变。HOME仅control/storage。

训练权威：`.codex/scratch/root_coordinator26_terminal20_eventchain_actual_qc_attempt02/root_actual_qc.json`（121链事件、20UPDATED、40pointterminals、120actualchildrc0，无pending/writer）；`.codex/scratch/root_fit40_full20_math_actual_qc_attempt01.json` self7660be339b3ada267901fe9d97bf54a6f71afa29d782d5b5c17f1c4ff53c98c3（1200unique NPZ、1920logicaluses、240baselinepins、20tell误差0）；`JOB/fit40_durable_spsa_controller_journal_attempt01.json`16698298B/raw2dd1c1ac08e439496c1f7753ab4bb8ccdd8727d2fc6a53aebeffd145b2167231/selfb731f233eea49ca533f4f21114eff09e342036b7287f63d172b112fe2f8d3b4d已单次封存。coordinator原OS rc未捕获，不改成actual0。

B17清单：596376entries、545819regular、46860dirs、3697links、37345424926B，19roots、65package12pins、linked/external0。ROOT QA08 `.codex/scratch/root_session09_inventory_actual_qualification_attempt08.json` self272e3d68c1d9f2e0d70c963d7ef497ed32e4d82f20f08d63d65da3aeb98ce5ee。manifest selfe7c9c2b8b0a7cbf8fc6cfb983ec4c4adc71351220de89012d5ed732c7a0f64c1。B17response634017302B/rawfa2f471af9de38b08ce62c675e6e104e6076796437c07e9aabdf13650c69f0be；receipt889553B/raw380668ac959ef14a687edd39e7774dc3f99230d0cf79b318c3ba7c1fd2119e36。清单producer17 `.codex/scratch/fit40_session09_poweroff_archive_attempt07/session09_poweroff_archive_attempt17.py`263923B/raw60eeb2704ace537f6ab39d9d20b2c87feddba1503c598cbfcebf6f6cfc8c2c1a，所有历史身份保持。

Actual archive：`.codex/scratch/fit40_session09_poweroff_archive_attempt17/archive_capture_attempt05/`；worker258256、SSH256308、wrapper241128。SSH实际23:10:44.335379Z→23:16:01.396292Z/2，stderr0；本地worker/wrapper2，capture_errors[]，无kill/retry。process_receipt selfe5a11fefa0a93e91586fcadf816c2bf6d9c6d26229fd3514f89334909ca3baa1。B:/fit40_session09_poweroff_archive_attempt17/session09_recovery_attempt17.tar 880050176B/SHA695467666eb7ca98e36da192043a4e1127acec967c1929a09f9e4fdf3d1ea079，仅12完整member；META/source_inventory_response.json声明634017302B，仅157468160B，缺476549142B。

fresh06实际成功但已消费/过期，不能复用admission；scanner06、reserved final04仍在packet07，final04尚未运行。final helper03 source-only70848B/raw65047a9966fcab9812a58d7b5d3a4902d0136de5ce44e3896eaf84bb1886ec30绑定旧transport，新增transport需重新绑定。ADR035的exact19roots、全UID稳定双通道、0refs/incomplete/races、实际scan结束到SSH Popen≤300秒、完整源重清单、whole/member/sourcepins、最终独立准入及正常sudo关机门保持。

本机脆弱状态：B是CloudDrive百度盘虚拟挂载，约3.753TB逻辑空间；C约852MB，E约790MB，F约2.886GB，无D；Z属于同一服务器，不算独立副本。不得改CloudDrive配置/DB/进程，云同步状态仍NOT_CHECKED。现有dirty/untracked/ignored BM、runs、scratch、所有失败产物保留，无commit/push/PR。

下一方案需同时解决大响应内存与写缓存：现worker17仍同时持有约634MB响应与解析对象，并在请求中压缩/Base64聚合。新executor须采用有界请求传输、并行stdin/stdout/stderr、≤1MiB缓冲；每块独立创建、关闭并完整重读散列，写前C可用至少512MiB+下一块大小，固定等待时限后失败保留。独立bodypack实际复制所有TAR regular bodies，包括META、硬链接别名和零长度项，索引绑定manifest ordinal、原绝对POSIX路径、元数据/PAX/link、offset/length/SHA。重开后逐body与全部65pins核验；不能把archive指针称为restore，不能称已经物理重建POSIX树或证明云端持久化。

科学路线仍冻结initial720→20jointSPSA→四full120中心→final720→portable→sealedDev/Confirm。初始四候选NOT_ADMITTED、step060只是initializer，尚不能宣称跨TFG超过RMS。Dev先修旧固定引用，封实际candidate/runtime/base360后跑540新行，不复用旧360、不提前读Confirm；全部9门失败继续显式11谱形/局部功率方法。

## Changed Files

- `HANDOFF.md`整理为61行当前快照；旧历史内容保留`.codex/scratch/root_handoff_before_archive17_snapshot_cleanup_attempt01.md`。
- `.codex/scratch/root_qualify_archive17_actual_failure_attempt02.py`与`.codex/scratch/root_archive17_actual_failure_qualification_attempt02.json`新增独立实际失败QA；attempt01的B/UNC拼写错误失败也保留。
- active Task已修复partial edit和C0 BM读失败，最新全文/context/HANDOFF readback为`.codex/scratch/root_sync_archive17_failed_recovery_attempt03/actual_readback_receipt.json`实际PASS。
- 新storage设计此时只有原Luna代理源码审阅消息，无新增executor实现或运行。

## Verification

ROOT实际失败QA02为PASS_ACTUAL_FAILURE_CAPTURE_NOT_BACKUP，self2e7c0ffde68855f30a0eef2a83ad545b2cb6614bc62535825ca6f0db56c35114：完整重读880050176B TAR并散列，核验local/B process回执、source pins、fresh age与实际rc、无restore/最终receipt。failed archive不等于科学阴性，不是关机资格。

fresh06实际privatePTY ECHOoff、sudo-v/launcher/scanner/SSH均0，23:06:37.802665Z→23:07:02.016661Z，echo restored并关闭；readback03、ROOTseal04实际0。未运行保留final04。Dev/Signal仅做源码与设计审阅，未运行实现测试。Git、canonical repository、当前分支PR与同会话checkpoint分页身份按此次wrapper重新读取；完整checkpoint正文和exact host metadata须创建后直接读回。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`为local/unpushed，无PR，不构造未push commit链接。
- `HANDOFF.md`及active Task是继续工作的入口；此checkpoint为不可变手递快照。
- `tts-exp/docs/adr/035-privileged-process-attestation-for-shutdown-archive`：完整关闭门。
- `.codex/scratch/root_archive17_actual_failure_qualification_attempt02.json`：实际失败及partial全body hash。

## Observations

- [result] 20轮/全tell数学/journal已完成；B17完整inventory与ROOT QA08通过，fresh06+seal04通过，archive17/05实际rc2且只有880050176B截断TAR。
- [result] ROOT QA02完整散列partial并核验真实进程终局，训练原件保持，服务器仍开启且没有新GPU任务。
- [decision] 保留inventory17/producer17、旧失败及科学协议；下一包装方案需独立executor来源、fresh07、完整内容恢复和最终ROOT准入，不复用旧admission。
- [blocker] C缓存容量不足相关于实际中断；确切原因没有archive-time日志。C重复历史包删除被automatic policy拒绝，禁止换途径重试。
- [blocker] 新64MiB TAR/bodypack方案尚未冻结ADR或实现；没有完整备份、fresh body恢复、final全UID证据或关机完成。
- [next_step] 冻结有界分块与独立bodypack的 typed ADR，再让原Luna实现新executor并完成独立源码准入、fresh07和实际完整备份，以既定最终门安全关机。

## Relations

- continues [[Codex checkpoint - 2026-10-10T23-04-24Z - Verified B17 inventory and staged fresh scan closure]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[035-privileged-process-attestation-for-shutdown-archive]]