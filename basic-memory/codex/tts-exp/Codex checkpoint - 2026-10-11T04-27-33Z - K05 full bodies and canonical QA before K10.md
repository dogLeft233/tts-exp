---
title: Codex checkpoint - 2026-10-11T04-27-33Z - K05 full bodies and canonical QA
  before K10
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-11-t04-27-33-z-k05-full-bodies-and-canonical-qa-before-k10
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-11T04:27:33Z'
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

# Codex checkpoint - 2026-10-11T04-27-33Z - K05 full bodies and canonical QA before K10

## Summary

k05完整原始评分产物已经回读并由ROOT逐body验证；120行canonical CPU重算正修复一处元数据guard，下一步是k10中心。

## Story

目标仍是跨TFG、未见说话人超过固定RMS的自然音频增强头。增强PCM仅驱动视频，原自然PCM和原时钟用于评分与交付。用户已取消关机，并明确继续四中心评估；备份18/fresh07/final04和停机保持停放，服务器及tfg heartbeat保持运行。ROOT维护理论、冻结协议、BM和独立实际QA，代码/实验沿用原GPT-6 Luna/max代理。

20轮联合训练、全原数组/tell数学和journal封存已完成且不得重跑。k05三模型full120于03:13:53Z自然完成，实际SSH、execute和三个model process均为int0。ROOT核验终局、writer退出和journal不变。为了确保评分真正来自原音和完整数组，进一步回读631项科学payload及40原WAV引用，不能以helper派生metadata代替正文。

首次raw读取在GNU stat类型guard停止：实际只读诊断确认中文locale返回“普通文件”。新reader07只将stat固定LC_ALL=C，保留nlink1、文件类型、前后大小/SHA、scope和容量门。新launcher05实际单次运行04:09:55Z至04:13:12Z、RC0，收齐620去重body/59,997,197B；618次SSH均实际0/stderr0，另2项精确local reuse。ROOT独立重开全部620正文和6818项capture body检查，631payload及40WAV全部精确匹配。

120行semantic consumer QA03于04:17:10Z实际RC2，输出为空且未创建目标目录、未启动远端CPU；它错误要求NPZ pin_name等于row_id，而生产者所有120项都明确为null，map的pin_name才是row_id。实际唯一ledger path/SHA/fileID、terminal.raw_npz_by_row及七数组身份链均存在。Dev QA04仅将该guard改为要求字段存在且值为None，其余联接未改；ROOT已核验完整窄diff和103028B/raw84bab13ebe6ffed44426ce31bf7a43d62ce70c6510f4b20b58328b001b8059cb，授权原Dev一次实际QA04。此快照时结果尚未返回，不能声称canonical120通过。

## Working State

局域网wjj@192.168.50.2:22，原QUEUE cwd/source_root=/dev/shm/fit40_queue_remote_package_attempt02/project；session09、package12实际package11 runner08、原lexical Python固定。k10未prepare或execute。四中心dispatch=/dev/shm/fit40_joint_spsa_full_centers_attempt01；禁止launch旧session、未知AutoDL DirectD0或重跑已完成k05/20训练。

已封journal：JOB/fit40_durable_spsa_controller_journal_attempt01.json，16698298B/raw2dd1c1ac08e439496c1f7753ab4bb8ccdd8727d2fc6a53aebeffd145b2167231；完整math QC self7660be339b3ada267901fe9d97bf54a6f71afa29d782d5b5c17f1c4ff53c98c3。JOB=/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/job_control。

本机脆弱raw root：C:/Users/dogle/.codex/scratch/fit40_center_k05_payload_readback_attempt07/raw。manifest1785062B/raw5a0cef586008d846a280524210fa8830107bbadfc36da6f01c03955c89d69147/self4fe28825c0b491d64d0dbf219d8ea55a86ce84717659305ece9cce408710e953。旧raw03 partial、capture04 locale失败、QA03失败全部保留，不覆盖或删除。

QA04输出目标仍为C:/Users/dogle/.codex/scratch/fit40_center_k05_local_body_canonical_qa_attempt01，仅在缺席时使用，必须有新invocation04捕获。冻结CPU rowreader01为9645B/raw049503ea3fd704400bad5312eb99d508f82e3c6c5e7edd51e0cea1782ae5afd6，在原QUEUE/noCUDA/TorchCPU单线程重算FULL及GUARD20。ROOT后续仍须独立实际读取120NPZ/840数组、frame-map、40WAV/120PCM窗直接ledger联接和120CPU输出。

k10只读preflight source01 9851B/raw1da372321c12c7be36c5d21c4e14ad0721c43a768297098bf101a4b254ac4a5c与proc/GPU source03 9384B/rawabe9ad8fe174590666788f67a3f64ccbd11d99cfd6f3ffe5d81a7c0ee8d6f88b已完整ROOT源码审阅，尚未执行。prepare wrapper/harness仅本地source复制，不能当执行release。完整k05 canonical QA后，fresh只读source/input/target/process/capacity核验→另prepare release→完整packet实际QA→execute。ROOT新root_k10_packet_actual_qc_attempt01.py已创建未运行，按真实tell10和family完整重建120行/plan/index，不能代替实际prepare/readback/source/proc门。

之后k15/k20→final720→portable→sealed540新Dev/Confirm。Fit准入后先修Dev固定旧引用并封actual candidate/runtime/base360，不复用旧360或提前读Confirm；cohort/policy01/ADR031/统计门不变。完整9参数门失败继续独立11谱形/局部功率。静态EQ与时变频带组成可改变局部包络，整体RMS不消除该变化；这仍是机制解释，四中心尚未完成选择，无新胜出结论。

## Changed Files

- `.codex/scratch/fit40_center_k05_payload_readback_attempt07.py`及launcher05：固定stat locale后完成原始正文读取；源和全部实际capture保留。
- `.codex/scratch/fit40_center_k05_terminal_qa_attempt04/local_body_canonical_qa_attempt04.py`：仅修NPZ explicit-null guard，单次实际执行已授权，结果待回报。
- `.codex/scratch/root_k05_full_raw_body_actual_qc_attempt01.py/json`、`root_k05_raw05_outer_actual_qc_attempt01.py/json`：ROOT完整body和执行capture实际核验。
- `.codex/scratch/root_k10_packet_actual_qc_attempt01.py`：完整packet独立重建工具，未执行。
- Task context和HANDOFF的03:43 raw段落仍旧，须按实际QA04及k10状态同步；本checkpoint先保存新确证状态。

## Verification

ROOT k05终局QA selffd2055c13398492efe6e4d2ee1e5bbeb766a5bc44da203ab31907e7a2a43b34e；helper完整receipt/631payload QA selff494fb1d6215c3294e412c8f638a8002befae56108bf912b8f4fb7c1c5cf6fd1；40原WAV/120PCM16窗正文QA self39a1749ac58eb54e7caf39b186df687c8c4723179a54899b0d7c8d8d4ef009df已通过。

实际完整raw body QA self6b1fb6f26c0b18dda70615a8b85b511f9765f7091673005c8cc0b22c6e5e3578，outer actual0 capture/source QA selfbd43e1f6fec5d215a4bf0d726db7d3d2196dab56f8d79b0475b39657de4da582。raw launcher05 process receipt4256B/rawef074346f7dc5ceddaa1cd0e257504ae0b5ebe7f76b746f02d4985630bc321ba/self6c53cde917afcae91698361bcba650a4bcedc62771d4ce441a0901b61e1d6813。

QA03真实失败capture `.codex/scratch/fit40_center_k05_local_body_canonical_qa_attempt01_invocation_capture/`，PID273124，04:17:10.8745289Z至04:17:13.6851056Z，actual2，stderr194B/raw71606b428b793e685ade8e52af860510bc206fc28a95f48782455a0e9f20c123；不是科学结果失败。Dev只读结构复核报告120NPZ explicit-null、120map联接/时钟及840数组hash无不匹配；ROOT独立semantic QA仍待完成。不运行实现测试或新的GPU实验。

Git codex/cross-tfg-audio-head-20261003 / `5688784dd5c6c1272074f0c4b6c5ff79f117c907`，local/unpushed。本次重新核验git status、canonical GitHub仓库和当前branch空PR；未commit/push/创建PR。BM与scratch/runs、SHM和挂载盘仍为本机脆弱状态。密码/API不写入文件/argv/BM/log，不修改CloudDrive/sudoers，不绕过被拒绝清理。

## References

- [GitHub repository](https://github.com/dogLeft233/tts-exp)
- `HANDOFF.md`、上述actual capture与ROOT QC为恢复入口；local SHA不构造GitHub commit链接。

## Observations

- [result] k05三模型120行actual0完成，全部620原始body和618实际SSH传输通过ROOT完整字节/SHA核验。
- [decision] 用户取消关机并继续四中心，已完成20训练/math/journal和k05不重跑；canonical120实际核验后继续k10。
- [decision] NPZ pin_name遵守生产者explicit-null契约，row身份继续由独立path/SHA/fileID/terminal/数组链绑定，不能弱化联接。
- [blocker] QA03元数据guard停止，QA04单次执行已授权而结果待回报；ROOT完整semantic/CPU重算门尚未通过。
- [next_step] 完成并独立核验QA04 canonical120实际产物，然后fresh核验k10环境、prepare完整packet并单次执行第二中心。

## Relations

- continues [[Codex checkpoint - 2026-10-11T03-59-18Z - K05 raw body guard diagnosis before next center]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]