---
title: Codex checkpoint - 2026-10-09T14-32-17Z - Fit414 native input proof and durable
  joint optimization
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t14-32-17-z-fit414-native-input-proof-and-durable-joint-optimization
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T14:32:17Z'
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

# Codex checkpoint - 2026-10-09T14-32-17Z - Fit414 native input proof and durable joint optimization

## Summary

三模型初始Fit已独立核验414/720，原生Wav2Lip输入与共享波形全量验证通过，联合优化的耐久journal和最终中心选择正在完成。

## Story

用户希望持续解释TTS提升TFG的局部声学原因，并做能迁移到多个TFG和未见说话人、超过固定RMS的自然音频增强头。增强PCM只驱动生成；交付及评分恢复完整原自然PCM和原时钟。ROOT维护机制、协议、BM和独立审核；代码和实验交给GPT-6 Luna/max。最新新增机器为wjj@192.168.50.2:22，native Windows公钥SSH可用，密码/API永不入文件或argv。旧AutoDL DirectD0终局未知，不重复launch。

平均谱形与音素内部轨迹的受者相容性已有机制证据；当前九参数头为四物理频带的三组参照对比×static/positive-flux/negative-flux，FFT512/hop128、common-mode消除、全局3dB约束、固定RMS，无peak limiter。旧W2L单模型训练step120在三模型Cbest对RMS平均+0.011149，W+0.068454/M+0.000900/D−0.035908，未达跨模型门槛。ADR022联合路线为完整720初始真实评分→20步SPSA960评分→四中心480评分，Dev/Confirm及LatentSync保持封闭。每次工程失败保留原记录，修复使用fresh attempt，不能把控制失败写成科学阴性。

## Working State

最新ROOT实际SSH队列证据在`.codex/scratch/root_actual_live_queue_readback_attempt05/root_actual_qc.json`，20180B/raw6f70af967e459549b0b89112ee993f590d760a48fa31bbd911eb4a04681c7f88/selfb07c01b75ee7fb5fd7f438de6b7fbc04e6ca95ec72c5c3945ff3113f66fd363f：414/720，W210/M108/D96；71事件、35新终端块210新行均PASS/实际childrc0，当前claim Ditto offset96。核验raw/self/previous事件链、所有terminalprocess/manifest/status/results pins、rowIDs和PIDticks，不称全新媒体解码或效果选择。原单队列coordinator260683/ticks3113948、entry260692/ticks3113952实际存活，source `/dev/shm/fit40_queue_remote_package_attempt02/project`，state `/dev/shm/fit40_queue_runtime_attempt01/state`。三offset240且无active/paused/failure才完成；不要重启或重复launch。

Family03冻结plan路径`/dev/shm/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/fit40_spsa_remote_family_composition_attempt03/fit40_three_model_family_plan_attempt03.json`，2584596B/raw9f7914033f9e7e89a60ae2618f7d429860d11de902a8400855fe57e37eb216a2/selfb071434696b2c094180473677d484934d7e3d04c808b7915000c8cfc6c6abb0c；720行、六方法、W2L/Muse/Ditto固定顺序和源码闭包保持冻结。

W2L05实际native资格完成：fresh builder03共享Py3.10/Torch2.5.1 CPU1/1生成48WAV/144projection；native Py3.8/Torch2.1.2 CPU1/1完整constructor/pre-render48通过，两个实际childrc0，zero render/score/controller。ROOT实际SSH流hash384文件/2,346,213,291B、完整解码88WAV（40自然+48增强），并复算144投影/sourceclock/theta/RMS通过。权威ROOT QC `.codex/scratch/root_native_w2l05_shared48_readback_attempt03/root_actual_qc.json`38365B/raw4e0add43867957ec09b1b4351c7d7d5fb11ddd26281e95b7ec6bd27506f369bd/selfe808e207bfff61862f61bdcd99296bd1bcf4eaceae2f84bb47a7526d0b771818；同目录`root_sourceclock_theta_qc_attempt02.json`11742B/rawf26634e7888699908ff29447fc1158af85b406ad1ae98032ba44c69d71260d57/self6a3ec309a8132d319d4470d03a1d8bfdda6903eaceac1556bc606f6c6be98285。builder02错字段失败exit1、ROOT只读01/02文件大小guard失败均保留，fresh streamhash03通过，不改原产物。

已撤回虚假W2L blocker：冻结runner core96537B/6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e的_render_and_score在_transform前有precomputed_fit_snapshot_waveform分支。无需更改冻结transform；只有旧CLI240行/块六方法限制，新薄point runner直接调用adapter.render_subset。selector06模块不能在Py3.8 import，科学选择与journal置于共享Py3.10，nativeW2L只做JSON传输和原生render/score。

ADR027已typed open创建并直接读回：`tts-exp/docs/adr/027-durable-spsa-journal-and-final-center-selection`。pending ask pair必须先于apply/render持久化；保存完整theta/iteration/history/PCG64大整数、input/artifact/runtime/source/packagepins，实际childrc与全量NPZ后才tell，20entry hashchain和确定性replay。complete ADMITTED或complete NOT_ADMITTED的冻结least-violation fallback均可做初值；missing/CONTROL_FAILED不行。最终720选择集是reference240+actual tell5/10/15/20四中心480，与初始720分开，portable theta与选择资格证据分开。

Dev已封存point runner02 source61702B/b02d5870ef4a9e25700b7903e5feedbbfa08133d7fb1976edcd09980dbd55e4a、tests23442B/0805fd66168a9b2b1937307d727890d34bd9d5e1aa33696aca28659c19901d56，agent七项CPU合成测试和Py3.8 grammar通过，ROOT待source/tests复核。真实8/40 per-model行，无240假padding，parent-captured childreceipt含numericrc/PIDticks/package raw+self。官方decoded support mask可mixed/all-false，transport保留全数组，由selector判common support，不能将合法unsupported窗口冒充执行失败。Dev继续parent launcher/dispatcher/Ditto09/package集成；Signal负责initial720 selector receipt producer与durable session/journal，不存在可跳过的新journal源码。

Signal最终中心consumer02 source88340B/09d431d608cb7dd49dd28b728d0e1e99c43070996ae3869190deaeb2d637228f、tests13739B/bc5303012c5db3fc3f5c921f22d971c8d887ed36ee213218c74acbd98e54a5d6，agent六项CPU测试通过、ROOT还需读剩余source与独立测试。ROOT发现consumer01把三个完整protocol强制相等会误拒绝Muse专属generator，实际family03只该字段不同；fresh02严格匹配各原source protocol和composed.execution.source_model_protocols，只剔除generator比较共同评分协议，再独立绑定generators_by_model。历史Ditto protocol.generator保留原样，不篡改source。新consumer将实际20ask/tell journal、960proposalNPZ、480centerNPZ和初始reference240完整回读归约，无实际科学init/优化/最终选择。

归档06恢复成功：F与B同名tar各355450880B/SHAa1005131389943d080c17279b0768a56d22c30c9ab37d171a4eb0a364f44bdc2，ROOT独立全包/11473members/334638160解压B/实际七source树通过。ROOT证据`.codex/scratch/root_archive06_recovery_independent_readback_attempt01/root_actual_qc.json`raw5bf5546d98539c60c155a5038125dbf91f5b8e35fb63436cf74e58ba72388225。F:/tts-exp-archives/fit40_queue_completed_blocks_attempt06/及B同路径长期恢复入口。failed06不重流；fresh archive07修复source_root并ROOT实际8tests PASS，仅支持旧七leaf，不是未来通用chunkworker。B挂载回读通过，cloud sync未知。

没有清理媒体。ROOT发现inventory10全部252direct candidates缺source_path、worker07部分unlink后谎报cleanup_performed=false、每leaf执行前缺重新控制/writer复核；ASR新inventory11/worker08/policy11已封存，agent实际测试7/11/8通过，ROOT还未核验或运行verify-only。inventory11共有9936候选245861096B（252direct+9684JPEG），全部精确映射archive source_path/bytes/SHA。完整保留集1537文件88777064B，84rawNPZ+1453metadata/source；84仅NPZ计数。fresh worker08每leaf第一次unlink前重读实际terminal/control/writer、逐文件dirfd/hash/inode核对，perfile ATTEMPT/FILE_UNLINKED journal与部分删除准确计数。第二文件失败夹具实际deleted1/attempted2/cleanup_performedtrue。ROOT待source/test审查→fresh actualverify-only→精确单次授权→unlink后独立1537keeper全SHA。ASR正并行做未来≤500MiB genericarchive source+tests，F+B全成员回读后方可清理，B作为长期已验证归档、F可按独立relocation/evictionreceipt退缓存，fixedcapacity gates不改。

Muse09本地_validate_rows48应用/六语义负控制ROOT通过；不是完整原生constructor。mask07单manual point的43NPZ数组独立数学复算通过，gainCV约0.003–0.016，均值EQ占主导并有小动态调制；static_only移除非零gateDC并改变全局非线性projection，非纯时序消融，第四参照带实际gain变化。没有新增TFG效果结论。

Git branch codex/cross-tfg-audio-head-20261003，HEAD 5688784dd5c6c1272074f0c4b6c5ff79f117c907（local/unpushed），既有tracked dirty、新untracked代码与ignored BM/runs/scratch保持，未新commit/push/PR。Task/HANDOFF最近mutable为378，须立即同步414和上述新资格。活动process/SHM/本地回执/未提交代码属fragile；目标仍active/unlimited，持续到方法完成与封闭泛化验证。

## Changed Files

- `scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_execution_attempt02.py`及对应unittest：8/40真实point接口、完整family/receipt/NPZ闭包，agent测试待ROOT。
- 同目录`fit_only_final_center_stagea_ingest_attempt02.py`及对应tests：各模型generator与共同score协议分离，pending完整回读/replay/final选择。
- `.codex/scratch/fit40_queue_media_inventory_attempt11/`、`fit40_queue_archive_stream_attempt08/`：source_path/部分清理telemetry/每leaf live复核fresh修复，未删文件。
- BM ADR027、Task/Experiment/HANDOFF、上述ROOTactual readback receipts是恢复入口。

## Verification

ROOT实际queue414事件/terminal/PID链通过；原生shared builder及W2L constructor两个实际rc0，384文件全SHA、88WAV PCM、144sourceclock/theta/scoreoriginalPCM独立通过；archive07 ROOT8tests、F+B全members/源树通过。Muse09与mask07资格沿此前实际ROOT证据保存。point02/consumer02/cleanup08 agent测试通过，ROOT尚未独立运行；没有任何initialselector科学结果、SPSA update、finalcenter评分、新Dev/Confirm或媒体unlink。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；fresh branch PR查询为空，HEAD为local/unpushed，不构造无效commit URL。
- `HANDOFF.md`、Task《开发跨 TFG 的自然音频增强头》、Experiment《跨 TFG 自然音频增强头的训练与原音换轨验证》及上述ROOT实际回执。
- `tts-exp/docs/adr/027-durable-spsa-journal-and-final-center-selection`定义持久优化与最终选择边界。

## Observations

- [result] ROOT实际414/720，W210/M108/D96；35新终端块210PASS、numericchildrc0，原队列持续运行。
- [result] 原生W2L05/shared48输入资格完成，ROOT384文件/88PCM/144projection全量通过，零render/score/controller。
- [decision] ADR027固定pending ask先于render、完整PCG64/20entry replay、初始与最终720分离，complete fallback可初始化而missing不能。
- [decision] point transport保留官方support mask；模型generator逐模型精确绑定、score协议共同校验，冻结historicalsource不改。
- [blocker] 完整720尚未结束；point/package/journal/consumer02及精确media cleanup仍需实际资格，未有新的泛化收益。
- [next_step] 完成point02/consumer02/cleanup08的ROOT独立复核和fresh控制验证，同时维持单队列，完整720后实际选择初值并启动耐久联合SPSA。

## Relations

- continues [[Codex checkpoint - 2026-10-09T13-48-41Z - Fit342 archive recovery and native point execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[026-shared-waveform-and-model-projection-receipts]]
- relates_to [[027-durable-spsa-journal-and-final-center-selection]]