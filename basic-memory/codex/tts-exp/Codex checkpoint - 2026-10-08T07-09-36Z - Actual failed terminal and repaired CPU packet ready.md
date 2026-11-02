---
title: Codex checkpoint - 2026-10-08T07-09-36Z - Actual failed terminal and repaired
  CPU packet ready
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t07-09-36-z-actual-failed-terminal-and-repaired-cpu-packet-ready
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
trigger: compact
project: tts-exp
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
username: dogle
git_sha: 7746b01f34b5850f11e322a0abc6934095f95f64
status: open
capture: deliberate
repository: dogLeft233/tts-exp
agent: codex
started: '2026-10-08T07:09:36.9944159+00:00'
branch: codex/cross-tfg-audio-head-20261003
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T07-09-36Z - Actual failed terminal and repaired CPU packet ready

## Summary
旧 Dev360 已完整结束并保留 120 PASS/240 接口失败；修复源码已提交，实际 template08 已核验，新 CPU worker10/index03 待启动。

## Story
目标从解释中文 TTS 的同步增强机制延伸为可迁移的自然音频增强头：增强 PCM 仅驱动视频，最终无损换回完整自然 F32 PCM 和原时间轴。用户要求持续实验、失败后继续迭代，并由 GPT-6 Luna/max subagent 编码与执行实验，ROOT 维护理论、冻结协议和独立复核。

当前冻结 Fit40/20 speakers 的 Adam120 四频带谱形与局部门控候选。下一科学阶段仍是 Dev20/10 speakers ×3 seeds ×3 TFG 的 360 identity/RMS 对照及 180 candidate；三模型方向、等权 +.050 over RMS、D0/Canchor、内容、视觉与 GUARD20 均需满足，Confirm80 和 LatentSync 保持封存。Fit 的谱形贡献较大，正确门控对固定原时钟 D0 有额外改善；尚无当前 Dev 的新科学结果。

上一运行 attempt09 已得到实际完整 360 行：Wav2Lip 两臂各 60 PASS，MuseTalk/Ditto 两臂各 60 FAIL，run_status=COMPLETE_WITH_FAILURES、manifest=INCOMPLETE。失效发生在 native renderer 前，frontend08 未在环境覆盖之后恢复每行 entry/control context；不能把它当作模型或候选阴性。child 完成后 controls17 的严格 receipt 收集抛异常，execution_record 保留 STALE RUNNING/null rc；ROOT 与 Fit 独立核验了全部实际失败账本和 launcher owned PID exited，不伪造退出码。

frontend09 修复环境恢复位置和 per-row work/control 路径，Wav2Lip 分支保持字节级行为。十个修复源码与 source_snapshot07/ROOT QC 已在本地提交 7746b01。实际 27 请求全部成功，得到 template08 与 summary08，ROOT 和 Signal 对实际 raw/self/hash、21 源码、2 native target、旧资格链及 backend descriptor 一致性核验通过。pending seed03 从实际 template08 填充既有 Dev 输入/资源估计，并指向新 controls360_repair_attempt10 namespace。worker10 修正 summary 中 target_root_expected_verified 的 two-row schema；index03 补充实际创建 namespace 的 phase1 q00 ancestry，所有 12 请求已独立审阅和 Bash 语法核验，尚未 dispatch。

上一不可变 FU checkpoint 中 q00“双重转义”的疑问已撤回：ROOT 误读嵌套 JSON 工具显示，原始请求正确且保持未改。Task106/Experiment105 已记录纠正及实际 terminal/template 状态。

## Working State
durable：HEAD 7746b01f34b5850f11e322a0abc6934095f95f64，branch codex/cross-tfg-audio-head-20261003，本地未推送；冻结方法/协议未改。Task 当前 phase=dev_controls360_repair_template08_actual_ready，实验 implemented，transfer pending。

machine-local/fragile：当前服务器 root@connect.westd.seetacloud.com:18625，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea；根 /root/tts-research-system-20261001。密码/API 只在私有 runtime，不入文件/BM。私有 PTY57588/bridge07:12249 当前可用，重启需重新建立。旧 360 媒体仍在 /dev/shm，snapshot05 可用 SHM 25,404,350,464 B、GPU free 32,230 MiB；启动前应重新测量。CONTEXT.md 再次修改、HANDOFF.md ignored 已更新；五个其他历史修改保留不覆盖。

CPU packet：.codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/request_index_attempt03.json（11,091 B，ac2c423bebfbb8bc4e0f2997a8395df3b4bc1d08e361ef57c71cdef16fbc50a9）；worker10 33,635 B/68b98f6fe30e0794e745c28f75a29f1cbe6ee61526c28bfa5dc5f2391c9d7599。CPU 流程生成 fresh sealed seed03、22-file SHA probe、binding09、plan09/360 rows、preflight09，随后八个实际下载需要 ROOT 独立 QC 才能制备 detached launcher07。

旧失败 compact indexer04 的 namespace/leaf 修复已 ROOT 完整 diff 审阅：90,851 B/3163401487e7c788f2c3efdf664a128f4bbec5cfa4893e059a76ee1280d827f5。copier03 全源已审阅；Fit 正准备 fresh 索引包。compact 只保留规定 WAV/NPZ/小文件与部分视觉输入，不能代替完整媒体恢复；不得据此删除 SHM，不阻塞新 GPU 实验。

## Changed Files
- scripts/experiments/cross_tfg_audio_head_20261003/frontend_adapter_runner_attempt09.py 及对应 collector11、builder17、controls18、candidate adapter06/runner13 等十个修复消费者在 HEAD7746。
- openspec/changes/cross-tfg-audio-head/reproducibility/ 的修复 snapshot 与 ROOT QC 随本地提交保留；具体路径以 git show 7746b01 为准。
- .codex/scratch/root_qc_dev_terminal_and_legacy_snapshot_attempt01.json 和 dev360_terminal_snapshot_attempt01/downloads/ 是旧终局实际依据；scratch ignored、需保持本机。
- .codex/scratch/root_qc_actual_template08_attempt01.json 是 template08 实际核验；cpu packet 尚无 response，不重放已有 phase1 请求。

## Verification
已实际完成：旧七个终局下载 raw SHA、manifest selfhash、360 ordered unique rows/六个 60-cell、120 Wav2Lip native entry/pixel/clock 证据；source snapshot06 的原链与 plan08 closure；修复 snapshot07 的 61 local pins 与独立全图审阅；27 phase1 responses、template08/summary08 selfhash 与 ancestry；worker10 producer schema、index03/all12 request pins 及 Bash -n。未运行 pytest；未实际下载/核验旧 score NPZ bytes。CPU09/GPU attempt10 尚未启动，180 candidate/内容/视觉/Confirm 尚未执行。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified historical pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current HEAD `7746b01f34b5850f11e322a0abc6934095f95f64` is local/unpublished; current branch PR query returned no match.

## Observations
- [result] 旧 Dev360 full failure denominator 已实际完整，120 Wav2Lip PASS/240 MuseTalk-Ditto 接口 FAIL；不作科学对比。
- [decision] 保持冻结候选和 Dev/Confirm 分离；新输出 namespace create-only，旧失败源码、请求、媒体和 stale execution_record 保留。
- [decision] 增强保持 STFT phase/原时钟，global RMS 固定；remove_per_frame frequency common mode 不是逐帧能量守恒。
- [blocker] 启动 GPU前还需实际 CPU09 八产物与独立 QC；不存在需要用户授权的新阻塞。
- [next_step] ROOT 私有 dispatch 已审阅 worker10/index03，下载并独立核验实际 binding09/plan09/preflight09，然后启动 fresh controls360 attempt10。

## Relations
- continues [[Codex checkpoint - 2026-10-08T06-37-13Z - Reviewed renderer repair graph and fresh CPU preparation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]