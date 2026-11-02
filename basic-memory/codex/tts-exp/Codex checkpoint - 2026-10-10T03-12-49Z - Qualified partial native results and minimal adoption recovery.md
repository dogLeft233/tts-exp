---
title: Codex checkpoint - 2026-10-10T03-12-49Z - Qualified partial native results
  and minimal adoption recovery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-10-t03-12-49-z-qualified-partial-native-results-and-minimal-adoption-recovery
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-10T03:12:49Z'
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

# Codex checkpoint - 2026-10-10T03-12-49Z - Qualified partial native results and minimal adoption recovery

## Summary

ROOT已独立核验首个联合SPSA plus点的8条真实Wav2Lip官方结果及56个NPZ数组，并将完整2152文件归档读回；dispatcher15最小继承修复已获ROOT与Signal源码审核通过，正在封存新的coordinator/session恢复链。

## Story

用户要求持续探明TTS提升TFG的局部声学机制，并产出跨模型、未见说话人可用且超过固定RMS的自然音频增强头。增强PCM只驱动生成，完整原自然PCM和原时钟用于评分及交付。ROOT负责理论、协议、BM及独立核验；原GPT-6 Luna/max子代理已在额度重置后恢复，继续负责代码与实际实验，没有等待模型替换批准。

冻结科学路线为actual initial720→20 joint SPSA→tell5/10/15/20四个full120中心→final720选择→portable→封闭Dev+Confirm。初始四候选均NOT_ADMITTED，step060只是minviolation初值；对固定RMS等模型Cbest增益+.035378766，Ditto负且D0/anchor约束未全部满足。当前单plus的8条W2L结果不是联合端点、没有tell或收益判定。恢复沿用相同packet、ask、θ与PCG64随机状态，仅补齐缺少的模型，已完成块只计一次。

## Working State

**权威实际部分结果。** Session05/attempt07于2026-10-10T02:35:22.080158Z开始，W2L块8/8 COMPLETE、official PASS、child/process RC0后，02:38:19.520892Z因原producer不提供可选NPZ bytes字段而被dispatcher11汇总器误判退出。旧coordinator720156、pair child720199、W2L720290均已退出；outer receipt UNKNOWN/rc2。session只有SESSION_INITIALIZED/PAIR_ASKED/POINT_DISPATCH_INTENT三个事件，iteration0、0tell；Muse/Ditto/minus未运行。这是PARTIAL_NATIVE_OUTPUT，不能当成零科学产物，也不能重放旧UNKNOWN。

ROOT实际资格证据 `.codex/scratch/root_session05_partial_native_actual_qc_attempt02/root_actual_qc.json`，self9f36290560334280d71571c50409a08327d89e8c79ae0446faca99a20677e9ce，PASS_ROOT_ACTUAL_PARTIAL_NATIVE_W2L8_SOURCE_AND_ARRAYS。其输入为 `.codex/scratch/root_session05_partial_native_actual_readback_attempt01/actual_inventory.json`，2152文件/64437169B，raw94c049b32b50500e5358508c498d1259111279fa6b4c3cf6f65ba2b44395813e。ROOT核验全部56数组dtype/shape/hash/finite/[N,31]/31lags/原时钟mask，独立2-array官方NPZ矩阵与7-array产物完全相同；原自然评分窗口及完整delivery PCM、generator shared WAV、8视频hash、projection/shared self链、原ask及session事件链均PASS。verifier01仅因ROOT误用receipt_sha256代替实际self_sha256而失败，02修读取字段通过，旧证据保留。

W2L ledger504617B/cc8e0f9af098073949301ee559edd5565b49daf9fffbcfff051492aebc41b4ea，manifest49645B/74d9e9acefb899ee39867fa87c9445b7e1a93f898489223ff4785dcc1f2e2c58，status12267B/0293bd7365dce8ca9558cfccc27154331887865000ff8c40278a2ebe5c1155fb。实际plus point为 `/dev/shm/fit40_joint_spsa_execution_attempt07/pairs/pair_01/dispatch_root/points/e869a1bb8f667ee867ecfa7f8cc5a5c4cca7d1a218c7b8c3438df52bbc9cd9c8_plus`。pair e5fa43c586f7665835da25c62d2e0acb3260f9662a2412baf913e2a688d73fc3，canonical ask79c34b67d28578d68d6a663f4ce9dc1ab46f6568af93f0c3d11d7a828；original packet为JOB/fit40_spsa_twenty_pair_coordinator_attempt07/pairs/pair_01/packet，必须复用原packet路径以保留binding。

**完整备份。** `B:/tts-exp-archives/partial_native_spsa_w2l8_20261010_attempt01/partial_native_actual_tree.tar.gz`，55879044B/raw4419abade67e9e3265663d5419052bbfafc29de0b6449452a7021c473a0c88d7。ROOT逐一核对全2152个文件含视频/帧的真实body/hash/size，archive_actual_readback_receipt self3af87042105d4e89d953635bff16a6d63e22f8251aae533f476e7ff925e4063e。服务器原件保留；这是本地网盘挂载备份，未验证云同步。此前218文件机制证据linked chain另行保留。

**最小恢复源码。** dispatcher12只修可选bytes合同，原NPZ path/SHA/七数组检查保持。dispatcher13曾有_adoption_applies读取错误顶层identity而默默重新渲染的问题，已拒用并保留。dispatcher14修完整14字段nested proposal_identity；dispatcher15加真实child/process RC0断言。最新 `fit40_spsa_point_dispatcher_attempt15.py` 136808B/b77700e89361288ba99389967fd57ba357baa927e22a81a0b3f40095db897785，ROOT/Signal源码审核通过。它用生产验证函数检查旧8块、三模型24条sidecars/shared WAV及七个block文件pin，只继承旧W2L，跳过W2L渲染及重复shared apply，继续Muse/Ditto，最终24行分母恰好含旧8一次。

bootstrap14(28519B/ca6846…3f2408)转交self-pinned adoption receipt；center dispatcher10继承15。新coordinator16/lineage/adoption receipt仍由Dev封存，尚未launch。具体可达约束：冻结DurableSPSASession04.resume在旧POINT_DISPATCH_INTENT遇fresh dispatcher ABSENT会直接UNKNOWN。因此必须不同fresh session root，不复制旧intent；用相同初始selector/package/state确定性ask，并比较旧event0/1/2与新event0/first ask前后PCG64/完整proposal。外层lineage绑定旧事件/dispatch/UNKNOWN process receipt；不用在每个adoption receipt重复所有旧历史。ROOT已授权Dev完成真实只读preflight后直接启动一次，不再等待重复批准。Signal正在并行审核新closure。

**交付/Dev20。** portable05 58811B/29bc744d250924221373a9244a735f3f7c3b2285d1b0d97357e6eda7a965f987与finalizer02 source-only接受，尚无真实admitted center/portable应用。ADR031 selected_spsa_center接口保持20新说话人clips/10人、三模型、3seeds、完整540行及原所有阈值，Confirm封闭。

新的builder04(74982B/36a84af3da6c92c54dc4a8456d50fc804cbbaa6d828dfc10250bb05073554b7c)/analyzer03(32444B/5435efe54cd15ea4ded5477b9b76b567daa2b88d5b1c21c86e21e6d6e1a58a25)已由ROOT与Signal源码审核PASS，`.codex/scratch/root_dev20_selected_sources_review_attempt01/root_source_review.json` self072f9f4b34e5eeb4281ddb64af4e30cf83342ae56d93d6b503736f6a92112714。共享_verify_exact_plan_rows在build/freeze/读取effects前重构精确360 identity/RMS+180 selected rows、27jobs/顺序/方法/原音cohort/portable应用，填补仅hash/count/roster不能保证candidate输入refs的缺口。没有运行应用、Dev效果或tests。

ASR新发现实际执行scope不适配：package11真正W2L `_render_and_score`已有precomputed_fit_snapshot_waveform生成输入路径，随后独立mux/scoring完整自然PCM；但三后端stock run都锁定240行/六方法，旧SPSA adapter锁8/40条和helper07 schema。Dev20需要20源×3seeds=每模型60候选/总180，不可伪造40源或用增强替换natural_samples。ASR已获ROOT授权写独立最小Dev20 adapter：复用实际部署row-level native hooks/scorer，独立验证builder04冻结计划及portable05 application lineage，exact6-row invocation从每模型60候选中执行，base identity/RMS计划用于容量资格；不改核心/旧adapter/冻结cohort/seeds/thresholds。source-only，真实Fit admitted前不做应用/模型/API/Dev/Confirm。

**机制与理论。** 已有描述性结论 `近静态谱形通过局部谱成分改变三类TFG的时变条件特征`、96实际frontends与886个40ms窗ROOT核验保留。相同全局RMS下，近静态谱形借局部频带构成产生local gain SD均值.4551433875dB，q05/q95 −.391333214/+.949170490dB；DC几乎复现full前端变化方向W/M/D cos .995256/.969548/.991782。这是provisional Fit8机制诊断，不是新的Sync收益或全TTS因果比例。已有11参数local-power head是CPU-qualified既有实现，不能记成新产出或已获TFG收益。

ROOT对已评分initial60冻结objective做了closed-form endpoint-space导数诊断，`.codex/scratch/root_initial60_frozen_objective_weights_attempt01/closed_form_diagnostic.json` self5131766eb16fdc7ce728960c4ee0a7571581e4707498454728452f11ec6619ed；对RMS softmax W/M/D权重.172537794/.179350659/.648111547，对identity .008053307/.889051199/.102895494。它解释当前联合loss主要补Ditto RMS及Muse原音端点，不是波形梯度、预测收益或首8batch梯度，不改冻结优化器。±.25是每次更新increment限幅，不是θ参数box。

唯一资格机器 wjj@192.168.50.2:22/V100 GPU-d12b3309-c299-da83-064d-a3cd619bd560，public-key BatchMode/StrictHostKeyChecking=yes。JOB=/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/job_control；QUEUE=/dev/shm/fit40_queue_remote_package_attempt02/project；W2Lsource=/dev/shm/fit40_w2l240_stage_attempt05/project。lexical Python=/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python，不能resolve为MFA。PowerShell→SSH文本会CRLF污染，使用binary Python subprocess/base64包装；凭据不入文件。旧AutoDL DirectD0状态未知，不重复launch。

Git codex/cross-tfg-audio-head-20261003 / `5688784dd5c6c1272074f0c4b6c5ff79f117c907` (local/unpushed)，已有dirty/untracked/ignored source/BM/runs/scratch保留，ROOT无新增tests/commit/push/PR。远端SHM、未跟踪代码与忽略scratch属于机器本地脆弱状态。Task/HANDOFF已同步partial8及Dev20源码PASS/B2152备份；Task metadata的Signal-awaiting旧字段和CONTEXT中的dispatcher12叙述待下一实际恢复节点同步。

## Changed Files

- `scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_dispatcher_attempt15.py`及bootstrap14/center10恢复closure草稿；13错误版本保留不执行。
- 同目录 `build_future_dev20_selected_spsa_plan_attempt04.py` / `analyze_future_dev20_selected_spsa_admission_attempt03.py` 已审source-only；新Dev20执行adapter编写中。
- CONTEXT.md（96行）、HANDOFF.md、BM active Task/implemented Experiment；ROOT actual8/56array QC、完整archive/readback、initial60导数诊断及本checkpoint。

## Verification

ROOT实际读取完整远端inventory及全部非media文件，独立解析数组/RIFF/事件/实际raw/self链；8条官方结果与56数组全PASS。完整2152文件backup逐body读回PASS。ROOT AST/diff审核dispatcher12→15，Signal独立检查身份匹配、RC0、24行无重复与resume可达性。Dev20 builder04/analyzer03 ROOT+Signal source-only审核PASS。没有新增科学归约/选择/tell、ROOT tests、API调用或Dev/Confirm效果。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；当前SHA `5688784dd5c6c1272074f0c4b6c5ff79f117c907` 为local/unpushed，branch PR实际[]。
- `.codex/scratch/root_session05_partial_native_actual_qc_attempt02/root_actual_qc.json`
- `B:/tts-exp-archives/partial_native_spsa_w2l8_20261010_attempt01/archive_actual_readback_receipt.json`
- `.codex/scratch/root_dev20_selected_sources_review_attempt01/root_source_review.json`
- `.codex/scratch/root_initial60_frozen_objective_weights_attempt01/closed_form_diagnostic.json`

## Observations

- [result] 首plus的8条W2L真实官方分数、56数组及原自然PCM/clock链ROOT实际核验通过，并完整备份2152文件；三模型双符号尚未齐，0tell。
- [decision] 最小partial-native继承在fresh session比较原ask/PCG64后只计旧W2L一次，旧UNKNOWN不重放；Dev真实只读preflight后可直接launch一次，无重复ROOT许可门。
- [decision] Dev20保持完整540及封闭端点，以独立adapter适配实际20源×3seeds，复用generator-only row scorer，不修改旧核心或伪造Fit40。
- [blocker] 新coordinator/session lineage与adoption preflight尚未完成，首pair缺Muse/Ditto与minus；没有真实admitted center或未见说话人收益。
- [next_step] 核对新partial-adoption closure及实际preflight，接续相同ask的Muse/Ditto和minus取得首个有效联合tell，并持续冻结20步/四中心/最终选择路线。

## Relations

- continues [[Codex checkpoint - 2026-10-10T02-45-11Z - Qualified package11 and partial native Wav2Lip recovery]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[031-selected-spsa-center-dev20-provenance]]
- relates_to [[近静态谱形通过局部谱成分改变三类TFG的时变条件特征]]