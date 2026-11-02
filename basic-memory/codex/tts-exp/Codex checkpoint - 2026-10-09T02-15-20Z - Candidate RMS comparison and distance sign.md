---
title: Codex checkpoint - 2026-10-09T02-15-20Z - Candidate RMS comparison and distance
  sign
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t02-15-20-z-candidate-rms-comparison-and-distance-sign
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T02:15:20Z'
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

# Codex checkpoint - 2026-10-09T02-15-20Z - Candidate RMS comparison and distance sign

## Summary

独立复算确认step120头相对RMS的平均FULL Cbest仅多+0.011149，Ditto下降；此前D0方向解释反了，正确的原时钟改善是−0.033662，当前头尚未达到跨TFG准入。

## Story

原问题是解释TTS为何能提高TFG的SyncNet分数；用户把目标扩大为可适配多TFG的自然音频增强头，增强波形只负责视频生成，再换回精确原自然PCM和时钟。最新问题是模型是否已经超过RMS。ROOT维护理论、协议和Basic Memory；实验与代码继续交给GPT-6 Luna/max，不在工程失败或一条排除结论后停止。

本轮从实际180候选和360对照ledgers进行配对核对，再由独立子代理复算支持键、时钟和指标。180组三臂支持身份全部相同，所以此前平均Cbest数值保留；但必须修正D0距离符号，并把9组短片不适格的GUARD20完整端点标记为INCOMPLETE。当前问题是跨模型迁移：Wav2Lip有分数收益，MuseTalk近零、Ditto退化，平均原时钟距离变差。Cbest=B−Dmin，因此其微弱增益同时包含B增加和最佳距离恶化，不能据此收敛到合格方法。

DirectD0已在修正40个源WAV根目录后唯一实际启动，仍只有STARTED_DETACHED证据。本轮尝试读取启动后状态再次被本机自动审批策略拒绝，未发送命令。保留所有失败和冻结政策，继续准备540原矩阵/媒体联合读回，并审查实际Fit损失是否覆盖足够模型；未来独立Dev20和Confirm的分界不变。

## Working State

可恢复的研究状态在BM Task《开发跨 TFG 的自然音频增强头》和Experiment《跨 TFG 自然音频增强头的训练与原音换轨验证》；本轮两条Observations已替换并直接读回，状态分别active/implemented。D0正向改善固定reference−candidate，Cbest/Canchor固定candidate−reference；冻结policy/analyzer已正确实现，无需改门槛。

machine-local证据目录`.codex/scratch/independent_candidate180_support_recalc_attempt01/`包含分析源码、180组明细和16,898B receipt（SHA `a0e14087bf777aa03112df482ce798556560dd392b6eecfd2f6d597e191f23e2`）。FULL Cbest逐模型W2L+0.068454、Muse+0.000900、Ditto−0.035908；正向D0改善逐模型+0.030824、−0.028086、−0.103724。FULL B+0.058594/Dmin+0.047445。GUARD20共171适格/9不适格，全部失败来自BAC009S0022W0123的三模型×三seed；不以适格子集替代180分母。完整NPZ body与媒体没有在本地复算，本轮不是实际joiner/readback执行结果。

远端最新服务器✅ `root@connect.westd.seetacloud.com:18625`，project `/root/tts-research-system-20261001`。DirectD0 actual control为`cross_tfg_audio_head_20261003/job_control/fit40_d0_gatedc_training_attempt03`，supervisor PID573383/start_ticks334637264；status04两请求均无response，勿重放launch。私有PTY57588、loopback bridge127.0.0.1:12249/PID29868和scratch是fragile runtime；密码和API不入文件、不从进程环境或内存读取。已审查CPU包的实际目录是`.codex/scratch/dev540_joint540_cpu_consumer_attempt02/`，旧checkpoint中的dev540_joint_media_readback_attempt02路径误写，不存在。新readback有MediaPipe视觉特征推理，但没有TFG生成、SyncNet重评分或训练。

未来独立Dev20的natural重复/RMS ASR基线60/60成功，20片段归一化转写全相同，候选20尚未调用。Confirm80/40新说话人与LatentSync仍封闭。Git branch `codex/cross-tfg-audio-head-20261003`，HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`是本地未推送；当前远端无该分支ref，GitHub PR查询返回[]且不能解析该SHA。五处原有tracked modifications和大量untracked scratch保留。HANDOFF是本地快照，不替代BM。

## Changed Files

- BM Task与Experiment的Observations：更正raw D0符号、增加独立支持核对及GUARD20不适格分母，旧完整正文和已执行实验保留。
- `HANDOFF.md`：更新当前比较和执行边界；旧字节在`.codex/scratch/root_candidate_rms_memory_correction_checkpoint_attempt01/HANDOFF_before.md`保留。
- 两个Luna/max子代理完成ledger支持/sign和dispatcher/source审查；独立分析目录及`.codex/scratch/fit40_d0_gatedc_bridge_status_route_audit_attempt01.json`为新的本地审查产物。另一个代理继续准备540离线导出与聚合consumer。

## Verification

ROOT核对独立分析源码27,682B、paired details624,474B和receipt16,898B的完整SHA，均匹配。子代理实际读取全部540 ledger行，所有配对、query clock、mask/lag/support hashes和clock字段一致（0 mismatch）。两个独立源审查确认core06定义D0/Dmin为distance、Cbest/Canchor为confidence，joiner输出raw候选−参考差，未来analyzer在两处显式把D0反向后才验收。无需修正冻结代码。

status04本地11个source/request pins又核对PASS，两请求仍无response。私有PTY非空write_stdin被自动审批拒绝：`approval required by policy, but AskForApproval is set to Never`；命令未执行，无远程请求，不能据启动时间推断训练完成。实际loopback监听和空PTY轮询成功只说明本地runtime存在。原540包已核对43pins但未派发。本轮未运行pytest、TFG、SyncNet或新云API；无新commit、push或PR。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)，通过本轮只读GitHub API验证canonical URL；本地未推送SHA仅保留为代码，不构造GitHub commit URL。
- `openspec/changes/develop-cross-tfg-audio-head-20261003/future-dev20-admission-policy-attempt01.json`与`analyze_future_dev20_admission_attempt01.py`为正确的D0方向与冻结门槛。
- `.codex/scratch/independent_candidate180_support_recalc_attempt01/independent_candidate180_support_recalc_attempt01.json`和paired明细为当前比较的实际证据。
- `.codex/scratch/fit40_d0_gatedc_status04_readback_attempt01/request_index_attempt01.json`为已封存待执行的只读状态包。
- `.codex/scratch/dev540_joint540_cpu_consumer_attempt02/`为正确的原矩阵/媒体联合读回包目录。

## Observations

- [result] step120头FULL Cbest对RMS平均+0.011149，但Ditto−0.035908；原时钟D0正向改善平均−0.033662，当前头未满足跨TFG准入。
- [result] 180组三臂支持身份和时钟在ledgers中全部一致；原矩阵和媒体完整独立读回仍待执行。
- [result] GUARD20完整端点不适格9/180，保持全部计划分母；此前+0.007161不是合格完整端点。
- [decision] 修正叙述中的距离符号，保持本来正确的冻结政策/代码/阈值和确认集封闭；不按结果换片、删失败或改支持。
- [blocker] host自动审批拒绝私有PTY非空只读状态请求，未获得DirectD0启动后状态，联合540包也未派发。
- [next_step] 用获准的ROOT私有路径读取封存status04获得实际训练状态及终局pins，再完成540联合读回并推进冻结未来Dev20验证。

## Relations

- continues [[Codex checkpoint - 2026-10-09T01-42-51Z - Verified D0 restart and ASR consistency]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[021-future-dev20-natural-audio-admission-policy]]