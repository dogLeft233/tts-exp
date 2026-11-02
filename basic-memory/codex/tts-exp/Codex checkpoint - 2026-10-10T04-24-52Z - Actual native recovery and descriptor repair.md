---
title: Codex checkpoint - 2026-10-10T04-24-52Z - Actual native recovery and descriptor
  repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-10-t04-24-52-z-actual-native-recovery-and-descriptor-repair
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-10T04:24:52Z'
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

# Codex checkpoint - 2026-10-10T04-24-52Z - Actual native recovery and descriptor repair

## Summary

首个联合SPSA恢复真实运行得到Ditto 8/8 PASS；Muse生成后因运行descriptor错传而失败，汇总又触发继承路径冲突，0tell。保留全部产物，改用最小Muse修复和相同初始参数的全新完整点重跑。

## Story

用户要求持续探明TTS增强TFG的局部声学原因，并交付超过固定RMS、适配多模型与未见说话人的自然音频增强头。增强PCM只驱动生成，评分和交付始终换回完整原自然PCM与原时钟。ROOT负责理论、协议、BM和独立复核，原GPT-6 Luna/max子代理负责代码与实验；额度重置已恢复原代理，没有替换审批待办。

冻结主线为actual initial720→20 joint SPSA→tell5/10/15/20四个full120中心→final720选择→portable→封闭Dev+Confirm。初始四候选均NOT_ADMITTED，step060仅为minviolation初值；其等模型Cbest对RMS +.035378766且Ditto及部分D0/anchor不合格。工程失败不构成科学阴性，也不能用部分W2L/Ditto行宣称增强成功。

## Working State

**最新真实运行。** coordinator19（87655B/ad9f5bcd9a4bb9f3096810074114cb93429a2724d3eea931e7e0474480507ea5）只启动一次，session06 ID fit40_spsa_recovery06_package11，dispatch /dev/shm/fit40_joint_spsa_execution_recovery_attempt03。实际PID731901/startticks9031891、pair731975/9032867已由Dev确认全部退出。ROOT在04:14:40Z实际读回时仍活跃，完整15字段fresh ask和PCG64状态与旧ask精确相同，event0/1/2 self及前链PASS；旧W2L8被继承且未重跑。随后Dev实际terminal读回：Ditto plus 8/8 official PASS、child/process rc0；Muse plus 0PASS/1FAILED/7INCOMPLETE，一行实际生成了部分视频但无有效score。剩余minus没有启动。

Muse原因由ASR在actual package11完整66源码中定位：musetalk_spsa_stagea_adapter_attempt12在overlay._validate_runtime_evidence调用传入 self.generator（source_model_protocols中的科学协议map），而overlay要求 model['generator']的运行descriptor并立即索引adapter_path。后续renderer_path/python/adapter_sha256/musetalk_adapter_sha256/renderer_sha256/overdub_helper_sha256/model_source_root七字段也全在descriptor、全不在protocol。最小Muse13只改该传参，保留self.generator作为科学协议；core实际render此前已用同一个model descriptor。没有调整科学参数或加入缺省假字段。

汇总额外失败：session报 point file reuses role/path with different bytes or local path，coordinator报 point-entry files do not exactly match event-referenced terminal receipts。session仍只有INITIALIZED/PAIR_ASKED/POINT_DISPATCH_INTENT，0tell；point entry文件存在但未获耐久链接受。Signal源码确认正常非COMPLETE plus会阻止minus并NO_UPDATE/halt，但本次更早在保存point入口拒绝，所以不能写成已NO_UPDATE。Dev/Signal追溯该继承路径冲突。

**恢复决定。** 新package需要Muse修复，ROOT选择简单且严格的恢复：保留所有旧W2L8、Ditto8、Muse部分产物与session05/06事件，将这些失败物理尝试明确排除出新canonical reduction；从同一qualified initial selector receipt建立fresh session/controller、相同θ/RNG/首ask，完整重跑plus24，然后继续正常三模型双符号路线。不用跨package继承框架，不修改旧pin/receipt，不重开旧UNKNOWN/intent。唯一科学上下文计划仍2160，额外失败尝试的物理渲染另计。该决定将另写typed ADR；fresh包与preflight尚未完成，未启动新一轮。

**独立资格证据。** ROOT fresh06 initial读回 `.codex/scratch/root_session06_initial_actual_qc_attempt02/` self0a77ebb36aefa56ba06805a6827cee752cf32ca32aaa342bfb5bad4f900dea9f：manifest与qualified05只变sessionid/self，event0同理；完整12source闭包实际PASS。ROOT `.codex/scratch/root_partial_native_adoption_actual_qc_attempt02/` selfde8ff9e5327b08296fa0f2737bd67c9903012664ddbd0e9912d456a5f2898b95：24完整实际文件及22referenced body、旧8/7block pins/3sidecars/原packet/RC0/前事件全PASS。adoption06 raw2ec3b30985596ecd6029b59bed05b13d3dafc328caa65ab6ac737427d476e3d4/self8130f3441420d4131d26af04c1713b4d1005362029040960d78355e881a0b5a3；lineage06 raw8809c53d64b1d74454044c926c31aa6ed51cc37c5e545971530dc951ecb82ebd/selfeea343ad8b14dc5d88cc12ed89cbc422a1b4df59c442d8e31792394166f42752。full AskProposal为15字段含speakers，dispatcher身份投影14字段不含speakers；builder05错误15-field adoption实际被ROOT拒绝，builder06/coord19才正确。失败版本保留。

ROOT实际live evidence `.codex/scratch/root_coordinator19_actual_live_readback_attempt01/`包含capture、实际源码、事件、状态及dispatch inventory。Dev最新terminal文件需ROOT下一次完整读回复核，当前终局数字明确来自Dev实际报告。

旧W2L8的ROOT56array/原自然PCM/clock/源链资格、完整B盘2152文件backup继续有效。QC `.codex/scratch/root_session05_partial_native_actual_qc_attempt02/root_actual_qc.json` self9f36290560334280d71571c50409a08327d89e8c79ae0446faca99a20677e9ce。backup B:/tts-exp-archives/partial_native_spsa_w2l8_20261010_attempt01/partial_native_actual_tree.tar.gz 55879044B/raw4419abade67e9e3265663d5419052bbfafc29de0b6449452a7021c473a0c88d7，原服务器保留，未验证云同步。

**交付与科学。** portable05/finalizer02及Dev20 builder04/analyzer03已source-only审核接受，无真实admitted center/portable应用。新Dev20 native adapter01 90625B/a0208abf9c1de8a421920803e8e0fbac64391c6495531510c897e3dd8811c3f3已完成源码，复用实际row级generator-only scorer，candidate每模型60行、exact6-row调用。ROOT待独立审核且追查新S0102–S0111 cohort的360 identity/RMS是否已有实际生成路线，不能只完成candidate180而缺完整540。所有Dev/Confirm效果继续封闭。

描述性机制结论《近静态谱形通过局部谱成分改变三类TFG的时变条件特征》保留：96实际frontends与886个40ms窗ROOT复核；相同全局RMS下近静态谱形借局部频带构成产生local gain SD均值.4551433875dB，full/DC变化方向cos W/M/D .995256/.969548/.991782。这是provisional Fit8描述性证据，不是新Sync收益或全TTS因果比例。已有11参数local-power head只完成CPU资格，不能写为新TFG实证方法。

实际资格机器wjj@192.168.50.2:22，V100 GPU-d12b3309-c299-da83-064d-a3cd619bd560，公钥BatchMode/StrictHostKeyChecking=yes。JOB=/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/job_control；QUEUE=/dev/shm/fit40_queue_remote_package_attempt02/project。lexical Python=/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python，合法软链接不可resolve后执行为MFA。PowerShell文本SSH会CRLF污染，使用binary Python/base64。03:14Z rootfs约4.32GiB free低于5GiB，SHM约17.89GiB free，保留旧证据并以真实容量门控制新输出。旧AutoDL DirectD0仍UNKNOWN不重放，凭据不入文件。

Git codex/cross-tfg-audio-head-20261003 / 5688784dd5c6c1272074f0c4b6c5ff79f117c907 (local/unpushed)，PR实际[]。tracked dirty及未跟踪source、忽略BM/runs/scratch保留；ROOT没有新增tests/commit/push。Task/Experiment/HANDOFF/96行CONTEXT最近同步session06initialPASS，但仍写coordinator17pending，需跟随最新实际终局同步。远端SHM及未跟踪本地代码属脆弱机器状态。

## Changed Files

- 恢复closure dispatcher15/bootstrap14/center10/coordinator19及adoption06/lineage06；source/pin检查通过，但actual恢复运行被上述两项工程故障终止。
- 新Dev20 selected adapter01完成source；ROOT完整540执行路线审核待续。
- ROOT fresh06initial、24文件adoption及真实live readback脚本/证据，BM当前checkpoint；Muse13尚准备中。

## Verification

ROOT实际完整文件、raw/self链、事件/15字段ask/PCG64对照PASS；未把预检当模型结果。ASR actual66源调用链审计精确定位Muse descriptor错位并核对后续八字段。Signal源码确认失败plus不能UPDATED。Dev实际运行Muse与Ditto并读回terminal；ROOT终局产物独立复核待下一步。没有本轮测试、新的联合归约/参数更新、Dev/Confirm效果或API调用。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；5688784dd5c6c1272074f0c4b6c5ff79f117c907为local/unpushed，无PR。
- `.codex/scratch/root_partial_native_adoption_actual_qc_attempt02/`
- `.codex/scratch/root_coordinator19_actual_live_readback_attempt01/`
- `.codex/scratch/root_package11_actual_sources_qc_attempt01/remote_full_readback.json`：actual66文件source body。
- `scripts/experiments/cross_tfg_audio_head_20261003/dev20_selected_spsa_stagea_native_adapter_attempt01.py`

## Observations

- [result] 同一首ask恢复真实继承W2L8并新增Ditto8官方PASS；Muse一条生成后校验失败、其余7未尝试，汇总未过耐久链，0tell。
- [decision] 新Muse descriptor修复需要fresh package，放弃跨package继承；旧失败尝试保留并排除新canonical reduction，从相同initialθ/RNG完整重跑plus24，物理额外尝试另计。
- [decision] 保持2160唯一科学上下文、20步与四中心、原PCM/clock、RMS/3dB约束和所有Dev/Confirm阈值不变。
- [blocker] Muse13/new immutable package及继承路径冲突分析尚待完成；没有三模型双符号完整首pair或真实admitted center。
- [next_step] 完整读回本次terminal产物，接受最小Muse13和fresh无继承包的实际preflight后，以相同θ/RNG重跑完整首点并继续20步/四中心主线。

## Relations

- continues [[Codex checkpoint - 2026-10-10T03-12-49Z - Qualified partial native results and minimal adoption recovery]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[031-selected-spsa-center-dev20-provenance]]
- relates_to [[近静态谱形通过局部谱成分改变三类TFG的时变条件特征]]