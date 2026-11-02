---
title: Codex checkpoint - 2026-10-08T16-43-24Z - Actual CPU parity and fresh candidate
  preparation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t16-43-24-z-actual-cpu-parity-and-fresh-candidate-preparation
branch: codex/cross-tfg-audio-head-20261003
username: dogle
trigger: compact
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
project: tts-exp
started: '2026-10-08T16:43:24.9727103+00:00'
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
agent: codex
git_sha: 82a77586da5bffc059173ce68d289c46f03735f8
hostname: LAPTOP-E133GSLE
status: open
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T16-43-24Z - Actual CPU parity and fresh candidate preparation

## Summary
线程修复已实际通过全部20开发音频和历史参考的严格CPU预检，真实证据已提交并双盘备份，新三模型候选运行正在封装。

## Story
用户目标是形成可复现、适配多TFG的自然音频增强头：增强mono F32用于生成，成片恢复完整自然PCM与原时钟，持续迭代到可用方法。ROOT负责机制推理/协议/BM/Git/私有SSH，代码与实验由GPT-6 Luna/max agents执行。

已完成controls attempt10完整360/360 PASS；RMS对identity equal-model Cbest+0.152、28族99%[+0.002,+0.302]，主要B+0.155而D0+0.052区间跨零。旧candidate02全180因重复interop初始化在render/mux/score前失败；六完整terminal及全180身份和分母经ROOT/独立QC，不作科学阴性。

新CLI03保持原权重/声学算法，仅幂等线程初始化。CPU03/index02已实际单次派发：同一strict CPU进程完成1历史参考+20 Dev，全部clock/N/RMS/input PCM、20 W2L EXPECTED帧及严格runtime通过ROOT/独立复核；无CUDA、native、scorer、Confirm或选择。历史参考完整WAV和PCM逐位相同，maxdiff/RMSE=0。ROOT修复的q00空目录检查顺序和repo-root路径已经由真实运行确认。新candidate180还未启动，Dev正在创建repair supplemental binding/新plan/guard15/output namespace，controls360保留explicit readonly parent。

并行DirectD0/GateDC源合同02已提交及全文审阅：直接RMS-relative D0与cosine方向reward，Cbest guard、波形/TV正则，保留9参数/32ms/8ms/3dB/RMS/phase/clock。全40 zero equivalence和有限非零严格梯度先于V3 Adam120；尚无GPU训练。Signal准备reviewable资格→固定120包。

## Working State
Durable: 本地未推送HEAD `82a77586da5bffc059173ce68d289c46f03735f8`，当前branch保持codex/cross-tfg-audio-head-20261003。新CPU proof68项HEAD raw/worktree逐字节PASS；此前29488失败/RMS/DirectD0 proof52项亦PASS。两组117文件2,864,150字节在F/B各两次整读PASS，cloud_sync仍未确认。

BM Task120/Experiment119（helper121）active/implemented，phase=cpu03_pass_candidate180_fresh_packet_preparation，实际写入/readback15155/15165字符；git字段29488为写入时事实。HANDOFF已压缩38行，CONTEXT94行。未执行pytest/push/PR。

Machine-local: root@connect.westd.seetacloud.com:18625，private PTY57588/bridge07易失，密码/API仅内存。PROJECTROOT=/root/tts-research-system-20261001；NS=其cross_tfg_audio_head_20261003子目录；SHM=/dev/shm/cross_tfg_audio_head_20261003。持久provenance在PROJECTROOT/runs，独立CPU code-root在SHM/project。CUDA物理GPU-d09622d0-3535-e662-bb45-3df2b049fcea。

Future cohort尚未选择：排除Fit/Dev/Confirm/历史暴露136人，400人roster余264；S0102–S0111仅候选顺序。官方HF100档案全已暴露。实际只读探针04达到首根10k上限、05完成浅层119/176元数据检查，尚无新source；/root/autodl-pub/data实际MISSING。ASR agent正在仅列B的17个generic dataset-audio gzip TAR成员名，总压缩4,052,051,914字节；不抽取/打开audio成员。官方OpenSLR15GB压缩容器可作为后续来源，尚未下载。原GUARD20 9/180不适格保留完整20/10sp分母；未来队列按metadata前10新eligible speaker各2条≥4sec。Confirm80/LatentSync封存，ASR/API/repeatGPU均0。

## Changed Files
- `reproducibility/cpu03_actual_strict_preflight_attempt01/`（openspec change内）：完整CPU worker/源pin/18请求响应/actual receipt/ROOT与独立QC，已提交。
- `CONTEXT.md` 已提交CPU实际PASS；`HANDOFF.md` 是当前ignored snapshot。
- `.codex/scratch/root_candidate_cpu03_actual_receipt_qc_attempt01.{py,json}` 和 `signal11_candidate_cpu_preflight_independent_qc_attempt01.json` 是原始复核。

## Verification
实际CPU receipt45226B/raw8c44b1c80a08e48465f25672ef6b990002653ff7d250ede7369be13dab3a3b09/self66add979541447ec5f8a525576196fd34d9d25306a1c0126fa7957988e3d0486。ROOT QC5950B/aa677f516c14e119e9a219390b4fe8effbfad6b031088a5ae3ace55db331e7ee；独立QC3899B/6f7c7efcbe3ec82b1cd809871c3ed1b797314cfb33786a8fa514c9a239d85ad4。21新输出payload未全部下载本机；明确此处producer实际检查与ROOT回执核验范围。新180/GPU/联合540尚待。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)；本次直接GitHub200/分支PR查询200且无匹配，SHA仅本地未推送。
- `.codex/scratch/candidate_v3_cpu_preflight_attempt03/request_index_attempt02.json`（actual18 responses俱在）与`downloads_attempt02_revision02/cpu_preflight_receipt_attempt03.json`。
- `.codex/scratch/candidate02_cpu03_F_B_full_proof_backup_qc_attempt01.json`：117-file双盘整读证据。

## Observations
- [result] 同一严格CPU进程的20+1实际PASS，参考输出逐位相同；旧180工程失败和完成360对照保持原样。
- [decision] 新candidate180使用独立output/plan/repair binding，原runtime09与控制结果immutable；实际execution记录区分原binding raw SHA与补充repair/overlay SHA。
- [decision] 保留三模型各正、等模型额外+0.050 over RMS及D0/Canchor/content/visual/GUARD20/失败分母验收。
- [blocker] fresh candidate180 packet待完成审阅；新未暴露speaker source待定位，DirectD0严格GPU资格/训练尚未执行。
- [next_step] 审阅并单次派发fresh候选180实际metadata与detached launch，随后完成原音换轨/独立评分和联合540全部端点。

## Relations
- continues [[Codex checkpoint - 2026-10-08T16-19-29Z - Full failure evidence and CPU repair preflight]]