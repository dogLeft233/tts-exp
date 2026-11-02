---
title: Codex checkpoint - 2026-10-08T16-19-29Z - Full failure evidence and CPU repair
  preflight
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t16-19-29-z-full-failure-evidence-and-cpu-repair-preflight
repo_root: E:/Documents/tts-exp
started: '2026-10-08T16:19:29.6832744+00:00'
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
git_sha: 29488e1ec07b7f98add3e117e09fe43cf739f636
repository: dogLeft233/tts-exp
project: tts-exp
cwd: E:/Documents/tts-exp
capture: deliberate
status: open
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
username: dogle
hostname: LAPTOP-E133GSLE
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T16-19-29Z - Full failure evidence and CPU repair preflight

## Summary
完整同机对照与候选工程失败已核验并保存；局部声学增强头继续推进，下一步执行修复后的 CPU 音频预检。

## Story
用户要求把 TTS 对 TFG 的局部声学作用做成适配多模型的自然音频增强头：增强的 mono float32 PCM 驱动视频生成，交付视频换回完整自然音频和原始时钟。ROOT 维护理论、冻结协议和证据；代码与实验由 GPT-6 Luna/max subagents 执行。不能在排除项或失败尝试处结束。

冻结 Fit40/20sp Adam120 与 Dev20/10sp、三模型三 seed 设计保留。360 条 identity/RMS 对照全部成功；RMS 的 equal-model ΔCbest=+0.152045096，simultaneous99% CI=[0.002388913,0.301701279]，ΔD0=+0.052387171，区间跨零。这个分解支持下一方法直接改善零偏移距离并约束目标方向，避免只提高搜索所得 margin。完整 RMS 结论已入项目 Experiments。

旧 candidate02 的 180 条全部因 PyTorch interop 重复设置而失败，发生在原生视频生成前；它是工程失败，无候选科学结果。六个完整 terminal 文件的 SFTP/raw SHA、180 条身份和阶段状态经过 ROOT 与独立 QC。新的 CLI03 仅把线程设置改为条件初始化，保留配置一致性检查。CPU03 worker 已修正真实 producer 字段与 W2L frame binding，metadata load_plan 已通过。ROOT 随后发现 q00 先创建 scripts 子目录再要求 project 为空，派发将必失败；Dev 正封装新 q00/index02。--repo-root 的持久化 provenance 路径亦需按实际证据复核。CPU worker 源码与旧请求不可覆盖。

下一方法 DirectD0/GateDC 已有源码与明确目标：相对 RMS 的 D0 改善和 embedding cosine 改善，加 Cbest worsening guard、波形/TV 正则。严格资格要求非零有限梯度；9 个头参数，Fit-only 三轮120步。GateDC 将有效的正/负门固定在后平滑均值。所有这些仍是 SOURCE_ONLY，尚未 GPU 训练。

## Working State
Durable: 本地未推送 Git `29488e1ec07b7f98add3e117e09fe43cf739f636` 含54个文件，保存完整 failed180、RMS 数值复核、线程修复与 DirectD0 源契约。旧 controls360 proof 保持不变；360完整 terminal 已双读备份到 F/B。主 Task 仍 active（helper120/section119），Experiment118 implemented，最近 phase=dev_candidate180_common_runtime_repair；它们应在实际 CPU 结果后更新。

Machine-local/fragile: 新服务器 root@connect.westd.seetacloud.com:18625，私有 PTY57588/bridge07。密码/API 仅内存 SecureString 和子进程环境，不入文件/BM。PROJECTROOT=/root/tts-research-system-20261001；namespace 为其 cross_tfg_audio_head_20261003 子目录，SHM=/dev/shm/cross_tfg_audio_head_20261003。CPU03/index01 绝未派发，CPU03 修订仅改请求；新候选180须新 binding/plan/dispatcher/output namespace，不重跑360对照。Confirm80/40sp 和 LatentSync 保持封存。GUARD20 原9/180不适格保留全分母。

## Changed Files
- `scripts/audio_enhancement/enhance_v3_attempt03.py`：幂等线程初始化，已提交。
- DirectD0 head、objective04、strict wrapper09、runner02：已提交，未实际训练。
- `openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/candidate02_failure_rms_and_directd0_source_attempt01/`：45份精确证据副本与 manifest，已提交。
- `.codex/scratch/candidate_v3_cpu_preflight_attempt03/`：忽略的待运行预检包，worker SHA=1de5879c3ff81412c494ecf600402eb28f829dc799711a153e59b192fcd1cbac；index01 source-only，不派发。

## Verification
实际完成：完整360控制QC、180失败完整读回/QC、RMS28 family重算及720个float32分数恒等式、CLI差异及 source pin 复核、DirectD0 全源审阅与非零梯度资格约定。CPU03仅 metadata load_plan：180rows/20inputs/20W2Lbindings；未执行增强音频或 GPU。新 commit52项 raw index/worktree字节一致性已验证；HEAD raw byte核验尚待一次。未运行 pytest；ASR/API调用仍0。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)；当前分支无匹配 PR，GitHub HTML 本次直接200核验。当前 SHA 为本地未推送。
- `root_failed_candidate180_full_qc_attempt01.json` 与 `dev360_candidate180_failure_readback_attempt04/failure_readback_qc_attempt04.json`（scratch）定位完整失败证据。
- `fit40_d0_direction_gate_dc_source_contract_attempt02/`（scratch）定位新方法源码契约。

## Observations
- [result] candidate02 180/180 FAILED before render/mux/score；controls360 360/360 PASS，保留无重放。
- [decision] 增强器必须保留自然音频时钟与样本数，最终轨道恢复完整自然 PCM；以三模型相对 RMS +0.050 和全部冻结 guard 验收。
- [decision] 136位 Fit/Dev/Confirm/历史暴露排除后 metadata roster 余264位；S0102–S0111仅待查顺序，未选新 cohort。只读目录定位请求已封装，不读 Confirm audio/score。
- [blocker] CPU03 q00 目录空检查顺序错误已定位；修订请求和持久化 provenance root 复核尚待完成，所有 GPU/音频实际阶段仍待。
- [next_step] 审阅 fresh CPU03 index02，单次私有派发，严格以 PASS receipt/full PCM clock 检查为门，再创建 fresh candidate180 运行。

## Relations
- continues [[Codex checkpoint - 2026-10-08T15-41-42Z - Complete controls and common candidate runtime repair]]