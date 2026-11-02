---
title: Codex checkpoint - 2026-10-10T22-34-46Z - Actual sealed inventory and failed
  archive request repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-10-t22-34-46-z-actual-sealed-inventory-and-failed-archive-request-repair
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-10T22:34:46Z'
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

# Codex checkpoint - 2026-10-10T22-34-46Z - Actual sealed inventory and failed archive request repair

## Summary

20轮训练和全数组数学已通过、journal已封存；fresh B16完整清单与ROOT QA07通过，prestream05实际通过并seal，但archive04在本地构造请求时以KeyError终止，尚无完整备份或关机。

## Story

长期目标是解释TTS对TFG的局部声学增强机制，做跨模型及未见说话人超过固定RMS的自然音频增强头。增强PCM只用于视频生成，完整原自然PCM与原时钟用于评分及交付。用户最新授权在安全节点关闭192.168.50.2，明天自行重启。因此今晚不启动中心、Dev、Confirm或新GPU训练，只封存现有20轮、完整备份可恢复闭包及正常关机。

关机会丢失SHM中的原执行环境和科学产物，完整body备份是必要前提。ADR035以完整UNADMITTED inventory、ROOT来源/闭包/容量验收、fresh privileged全UID进程检查、ROOT exact admission和300秒内实际archive SSH启动保护归档；全部body恢复后还有reserved final04与ROOT独立最终准入。B盘是CloudDrive百度盘虚拟挂载，逻辑空闲与云同步不同。

本轮把worker13的多份634MB响应展开改成有界压缩传输、流式manifest原body比对和释放TarFile成员缓存，worker16源经ROOT及ASR验收。B16 inventory04与ROOT QA07实际成功。prestream05在无回显私有PTY中实际完成并完整读回，ROOT seal03实际创建精确admission。随后ROOT单次释放archive04：本地worker31456读取B盘634MB响应耗时，最终在请求构造读取context['source_pins']时自然rc2退出；这与过期时间窗是两个不同问题。没有SSH archive子进程、TAR或restore证据，也没有kill或重试。worker17仅候审草稿，修稳定分块本地读取及已核验source_pins的数据流，尚不可运行。

## Working State

服务器仍开启，wjj@192.168.50.2:22。JOB=/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/job_control；session09 ID fit40_spsa_bfa027430723_6800c380298d。原QUEUE cwd/source_root=/dev/shm/fit40_queue_remote_package_attempt02/project；dispatch=/dev/shm/fit40_joint_spsa_execution_clean_rerun_attempt03。package12实际package11 runner08及原绝对路径不变。密码/API仅runtime或私有PTY无回显stdin，永不文件/argv/BM/log。

训练权威：`.codex/scratch/root_coordinator26_terminal20_eventchain_actual_qc_attempt02/root_actual_qc.json`通过121链事件/20UPDATED/40point terminals/120实际childrc0及无pending/lock/模型children。`.codex/scratch/root_fit40_full20_math_actual_qc_attempt01.json` self7660be339b3ada267901fe9d97bf54a6f71afa29d782d5b5c17f1c4ff53c98c3通过1200unique NPZ/1920logical uses/20tell数学，误差0。JOB/fit40_durable_spsa_controller_journal_attempt01.json已单次seal，raw2dd1c1ac08e439496c1f7753ab4bb8ccdd8727d2fc6a53aebeffd145b2167231/selfb731f233eea49ca533f4f21114eff09e342036b7287f63d172b112fe2f8d3b4d。不要重跑20训练或seal。

worker16权威源：`.codex/scratch/fit40_session09_poweroff_archive_attempt07/session09_poweroff_archive_attempt16.py`260696B/raw0e40a1e328a1422e2c2bca07421f2d39fb5ffbff2a44cfcf708b4f868e0ebb93。`.codex/scratch/fit40_session09_poweroff_archive_attempt16/inventory_capture_attempt04/`实际worker241376/rc0于22:17:24.222435Z结束，SSH257740/rc0。B16 response634017302B/rawdf7a885eab29002753f9e0b482fabbe49bb8d2f89556931ed60ac66395828f39；receipt889553B/raw8d3d301abb41fef80617adf5f7b75d14e376f016dbe426b0a9f5dac8a7d683ae。ROOT `.codex/scratch/root_session09_inventory_actual_qualification_attempt07.json`6692B/rawf3aa114e0077ba0a8179e8743cbba99d582f73e55d8a717670c2ab8c8db4509f/self65054b6f56d2be875e0bcba7fb312bb7563b441d69fd2b470b87ea4725aaaf8d实际rc0：596376entries/545819regular/46860dirs/3697symlinks/37345424926regularbytes/19roots/65package12pins，manifest selfe7c9c2b8b0a7cbf8fc6cfb983ec4c4adc71351220de89012d5ed732c7a0f64c1。UNADMITTED owner incomplete仍FAIL_CLOSED，不等于root进程准入。

prestream05已消费：privatePTY10554实际22:20:13.969005Z至22:20:32.547260Z、ECHOdisabled/sudo-v0/launcher0/scanner0/SSH0并关闭。`.codex/scratch/root_session09_private_prestream_capture_attempt01.json`434B/raw9c00eed0e54fd40adec7acae6be7985f9b0f307529af15192c3d70f99ee0abca/self9a49610fb10e25abf1c111909e70e2fe27b9dab775d6045dc7f886b68ab84d34。packet/readback_prestream_capture_attempt05/prestream/含五原始body及readback receipt；500可见进程/19exact roots/0refs-incomplete-races/两稳定channel/actualrc0。ROOTseal03实际rc0创建B16/prestream_admission_attempt16.json8712B/rawcdcb3b60491e25cae1c37079eaaaea7efa671676323b878d92f46c5731145679/selfeca27900f2f430e23d460fdfbf40ac9ed495a16bd59b7380352c0ab3d9d28694。时间窗已失效，绝不可重用。

archive04单次实际capture `.codex/scratch/fit40_session09_poweroff_archive_attempt16/archive_capture_attempt04/`：PID31456/OScreation22:24:01.494711Z，22:27:15.357495Z自然actual child和wrapper rc2，capture_errors=[]，stderr50B/raw7228cf0e1b2d54c373cef178996ff4b130298a641a00d960eb69845efdcd504d，内容SESSION09_ARCHIVE_ERROR: KeyError: 'source_pins'。process_receipt self65d4031320b1ad5a71b73a54cc4df0ec907d0f5bd1049a1b349191cf70fb07cd。ROOT exec23890已rc2关闭；PID31456和parent260608已缺席；没有停止进程。postrun明确SSH receipt/stderr、TAR、fresh restore、worker最终receipt缺席。B16仅清单/source/admission六叶，原件保留，不重跑16。

源码原因：worker16 `_read_prestream_admission`返回已核验source_pins但context.update未绑定；builder3332–3334读取context['source_pins']，fresh3431亦比较context.get。worker17候审未执行草稿将精确修这条数据流，以及B盘Path.read_bytes改为最多1MiB分块及fd/path identity门。Seal03同样大文件读需fresh seal04修复。Signal准备fresh prestream06/readback03/privatecontrol03和17 bindings；ASR逐consumer独立审阅bytearray兼容及source_pins来源，ROOT未准入任何新执行。未执行的草稿可最终绑定，不因草稿文案变动无限升级；实际已执行/接受来源与失败永不改写。

长期科学路线与所有门不变：明天用户重启且要求继续后四full120中心→final720→portable→sealed Dev/Confirm。Dev先修旧固定引用、封实际candidate/runtime/base360后540新行；严格9门失败继续显式11方法。没有科学新胜出结论。dirty/untracked/ignored BM/runs/scratch、B虚拟挂载及SHM均machine-local脆弱状态。

## Changed Files

worker14/15/16内存修复源、B16 inventory与ROOT QA07、privateprestream05/readback/seal03实际产物、archive04实际失败capture；worker17及fresh工具仅候审准备。Task/HANDOFF上一段仍误写inventory运行中，下一步需同步上述真实终局；本checkpoint pointer会先更新。没有commit/push/PR/清理或GPU运行。

## Verification

ROOT及ASR静态源码/delta审阅，AST解析；真实B16 inventory及独立ROOT完整路径/65sourcepins/raw+self/容量QA通过；实际prestream全UID扫描和完整body readback/seal通过。archive04是实际失败，不称备份成功；未运行任何实现测试。checkpoint重新核验Git/canonical repository/PR[]/commit visibility、同chat分页检索及直接身份，要求created和完整body/host metadata读回。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`为local/unpushed，branch PR[]。
- `tts-exp/docs/adr/035-privileged-process-attestation-for-shutdown-archive`、`HANDOFF.md`及上述actual captures/QC。

## Observations

- [result] 训练20、全数组数学及journal已封；B16完整37.345GB清单及ROOT QA07通过。
- [result] prestream05实际通过并seal；archive04实际KeyError rc2、无SSH/TAR/restore/off，失败原件保留。
- [decision] 不重复16或复用已过期prestream05；新候审source17精确修数据流/分块读取，经独立审阅和fresh完整准入后才可启动。
- [decision] 今晚不启动新GPU；完整备份/恢复与final ROOT准入后正常关机，再await_user_restart并暂停tfg。
- [blocker] 本地响应读取耗时且context缺source_pins；17及fresh06工具闭包尚未审阅准入，服务器仍开启。
- [next_step] 完成worker17和fresh prestream06/readback03/seal04完整数据流与分块读取的独立源码验收，随后继续完整备份和安全关机。

## Relations

- continues [[Codex checkpoint - 2026-10-10T21-48-36Z - Actual inventory and bounded memory archive repair]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[035-privileged-process-attestation-for-shutdown-archive]]