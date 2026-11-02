---
title: Codex checkpoint - 2026-10-08T18-48-26Z - Verified independent staging and
  local timing controls
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t18-48-26-z-verified-independent-staging-and-local-timing-controls
hostname: LAPTOP-E133GSLE
trigger: compact
project: tts-exp
model: gpt-6.1-sol
branch: codex/cross-tfg-audio-head-20261003
started: '2026-10-08T18:48:26.6696785+00:00'
git_sha: 4887c362dc69a1d43e625a259eb0bd2a8dddec84
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
cwd: E:/Documents/tts-exp
username: dogle
repository: dogLeft233/tts-exp
agent: codex
capture: deliberate
repo_root: E:/Documents/tts-exp
status: open
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T18-48-26Z - Verified independent staging and local timing controls

## Summary
旧 Dev 的冻结候选已完成 107/180 条且全部流程 PASS，未来独立 Dev20 的远端完整暂存回执已全文核验，DirectD0 新训练与投影后机制对照保持未运行。

## Story
用户要求从本地 BM、Git 和代码恢复 TTS 增强 TFG 的原因，并产出适配多个 TFG 的自然音频增强头，持续迭代到可用方法。增强 F32 音频驱动原生视频，交付恢复完整自然 PCM 及原时钟。ROOT 负责理论、协议、BM/Git、私有远程连接和证据；GPT-6 Luna/max 子代理实现代码和实验。

同机 identity/RMS 360 条完整对照已完成，等模型 Cbest 的 RMS 收益为 +0.152。旧候选因接口控制失败在音频生成前全体失败；保存失败分母后，新的严格 CPU 资格验证与历史 PCM 完全一致，冻结 step120 候选唯一一次启动。实际只读快照 2026-10-08T18:47:07Z 为 107 PASS、0 FAILED、0 INCOMPLETE，Wav2Lip36/MuseTalk36/Ditto35，父 PID164225/start_ticks333464927 未变。流程通过不等于科学晋级，仍需完整终局、原音评分、内容、视觉及边缘支持。

未来 Dev20 在效果暴露前固定为 S0102–S0111 每人两段长自然音频，来自固定第三方 AISHELL-1 mirror commit。20 WAV 和规范 F32 的全量本地核验已完成；远端 producer 将 48 项 create-only 写入并回读。ROOT 下载了完整实际回执，逐项检查全部20行/48输出/51来源 pin、自哈希、ZIP、N 与时钟；独立子代理核验同样 PASS。ROOT 未重新下载全部远端音频正文。暂存回执 raw0896932659649d3328d9986329c9bddfa4fd73f3b7b3c6f4268dba44441ff379，ROOT核验 root_future_dev20_actual_stage_receipt_qc_attempt01.json。

新 DirectD0 在 Fit40 上直接优化原音轨零延迟距离和单位特征方向，固定 Adam120，并对同一权重记录 GateDC。全部23科学源码和几何输入已实际暂存并核验，launcher03/status03 修复实际 cwd 和回执比较且已上传，GPU 训练未启动。研究推理 CLI 与最终便携交付胶囊应分开：前者严格绑定训练运行时；后者仅在实际权重经 Fit-only 数值资格验证后允许明确的 CPU/其他 GPU 运行时。

ADR020 已实际写入并读回：当前全局径向投影会把 gate 时序与频谱幅值耦合，因此用投影后的逐频点时间均值和 ±64 帧循环移位拆解，固定 theta、原投影、STFT/ISTFT、N、时钟与 RMS。GateDC 仍有有效帧 mask，不称为全时常数；构造谱与实现 PCM 重分析相位分别衡量，披露所有接缝和完整支持分母。240 个 Fit-only 波形对照和独立评分仍未执行。

未来 Dev20 协议06固定 identity/RMS/新 DirectD0 三臂、三模型、三seed共540行，全部重新生成对照，旧 Dev 不替代其分母。GUARD20 必须在实际评分轨上三模型共同查询集至少81帧，裁两端20后至少41帧。源代码审计发现 MuseTalk 的144帧来自当时输入长度而非上限；新20通过必要输入与名义生成帧检查，但实际共同查询集仍需产物核验。D0/Canchor/content/visual 和联合统计族的数值准入界限仍须在未来效果前绑定；Confirm80/40sp 与 LatentSync 继续封闭。

## Working State
持久锚点为本地未推送 HEAD4887c362dc69a1d43e625a259eb0bd2a8dddec84，分支 codex/cross-tfg-audio-head-20261003。原候选冻结 Fit40/20sp、Adam120；旧Dev20/10sp×3seed×3models，360 controls+180candidate。既定主收益要求三模型均正且等模型比RMS至少+.050，其他准入仍保留。

机器本地脆弱状态：私有PTY57588、bridge07、observer10 PID117048和 /dev/shm 运行。服务器 root@connect.westd.seetacloud.com:18625；PROJECTROOT=/root/tts-research-system-20261001，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码/API仅私有内存，不入笔记或文件。候选不得重放；成功上传/下载不重放。

## Changed Files
CONTEXT.md 当前94行且新改动未提交；ignored HANDOFF.md 已更新，均包含实际暂存与运行状态。5项无关 tracked dirty 保留，完整状态见 checkpoint_GO_git_status.txt。Task123/Experiment122 已实际更新并读回。readiness04 只修正 PROJECTROOT/scripts 路径并使用新 namespace，尚未执行；其 q03 仍裸 python3，先绑定已资格确认的绝对解释器。ASR客户端07只修正候选 rows 文件名，源代码静态审阅完成但未运行，未来终局 pin 仍null。D0 CLI01/postcontrols03/胶囊方案 source-only；完整 ROOT 源码审阅尚待补齐。

## Verification
ROOT 已完整核验未来20实际 producer receipt 的所有行和输出，独立复核 PASS；此前本地全量 WAV/F32 检查及 D0 23项实际暂存 PASS。候选实际 metadata、唯一启动、启动回执、连续只读快照已核验；最新107行仍RUNNING。当前Git证据87路径及F/B89项完整双读核验在上一轮完成。没有pytest、新ASR调用、DirectD0训练、未来Dev效果、push或PR。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)；GitHub当前分支 PR 查询无结果。HEAD `4887c362dc69a1d43e625a259eb0bd2a8dddec84` 本地未推送。
- `.codex/scratch/dev360_candidate180_status_observer_attempt10/polls/20261008T184705Z_poll_c6a6aca9/observation.json`。
- `.codex/scratch/dev20_future_cohort_source_stage_readback_attempt02/downloads_attempt02/source_stage_qc_attempt01.json`。
- `.codex/scratch/fit40_d0_gatedc_training_packet_attempt02/launcher_status_receipt_fix_attempt03/dispatch_packet_attempt01/request_index_launch_only_attempt09.json`，仅q12且未运行。
- `.codex/scratch/dev360_candidate180_terminal_contract_attempt05/candidate180_terminal_transfer_contract_attempt07.json`；实际终局7文件 pins 待产出。
- `.codex/scratch/dev20_future_cohort_protocol_attempt06/` 和 `tts-exp/docs/adr/020-post-projection-spectral-timing-controls`。

## Observations
- [result] 候选107/180流程全部PASS且仍运行；未来20完整暂存回执ROOT和独立核验PASS。
- [decision] 投影后频谱/时序对照遵守ADR020；新Dev重新渲染完整540行，数值门槛和支持政策先冻结再暴露效果。
- [decision] 研究推理运行时与最终便携胶囊分开资格确认，输入全N/时钟和恢复原音轨继续保持。
- [blocker] 候选未终局，DirectD0必须等待完整终局与GPU空闲；新的数值准入门槛、实际共同查询支持、便携交付仍有必要工作。
- [next_step] 先修正 readiness 的绝对解释器绑定并审阅新 packet，归档本轮实际暂存证据；继续现有候选至终局，核验完整540配对数据后唯一启动DirectD0。

## Relations
- continues [[Codex checkpoint - 2026-10-08T18-10-36Z - Live native candidate and verified staging]]