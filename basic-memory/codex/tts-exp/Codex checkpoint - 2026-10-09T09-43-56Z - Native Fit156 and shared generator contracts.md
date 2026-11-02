---
title: Codex checkpoint - 2026-10-09T09-43-56Z - Native Fit156 and shared generator
  contracts
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t09-43-56-z-native-fit156-and-shared-generator-contracts
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T09:43:56Z'
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

# Codex checkpoint - 2026-10-09T09-43-56Z - Native Fit156 and shared generator contracts

## Summary

正式三模型Fit已完成156/720条有效评分；ROOT复核Ditto首组六的1445个实际文件、完整原音换轨及矩阵通过，并发现共享执行hook混淆了两个不同的generator契约。

## Story

用户要求恢复TTS增强TFG的局部声学机制，随后要求产出适配多个TFG、在新说话人超过固定RMS的自然音频增强头。增强PCM只驱动视频，交付与SyncNet评分使用完整原自然PCM及源时钟。ROOT维护理论、实验协议、BM与结论；代码和实验交GPT-6 Luna/max。失败后持续推进，音量排除不是终点。最新用户确认局域网192.168.50.2的SSH用户名wjj；实际Windows公钥可登录。实验、历史网盘数据、下载和云接口已授权，秘密永不写文件/argv/日志。

旧W2L训练step120在独立三模型FULL相对RMS等模型ΔCbest+0.011149，W2L+0.068454、Muse+0.000900、Ditto−0.035908，D0正向改善−0.033662，未达跨模型目标。ADR022选择在同40条Fit自然音频/20人上先取得三模型×六方法720个唯一评分，然后以真实三模型原音分数联合优化9参数。当前是正式Fit执行、实际产物验证和共享SPSA执行层闭合，没有新的完整效应、科学初始化或优化更新。

## Working State

权威Task《开发跨 TFG 的自然音频增强头》active，Experiment《跨 TFG 自然音频增强头的训练与原音换轨验证》implemented。ADR022/023/024保持：六方法共同FULL支持，说话人等权再模型等权；每模型Cbest对identity/RMS均正，D0/Canchor不恶化，等模型比RMS≥+0.050。缺行/失败阻断初始化；完整但不适格才按预定违例函数初始化。Canonical TorchCPU1thread，NumPy仅诊断。实际decoded五帧/有理PTS与原始矩阵及完整分母保留。独立新Dev20/Confirm80、LatentSync封闭。第一720完成才开始20步SPSA：新960proposal+四中心480，总2160物理评分。

✅执行机wjj@192.168.50.2:22；使用native Windows ssh.exe BatchMode/StrictHostKeyChecking=yes，WSL无该公钥，不切WSLSSH。持久repo /home/wjj/tts-audio/tts-exp；V100-SXM2-16GB UUID GPU-d12b3309-c299-da83-064d-a3cd619bd560；W2L环境 /home/wjj/.venvs/w2l_syncnet_v100_attempt01，Muse/Ditto共享 /home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01。SHM媒体/包、F或B完整归档；每launch实际测GPU/RAM/cgroup/SHM/output与5GiB余量。一次只跑一个六行块，CPU工作并行。旧AutoDL DirectD0终局未知，拒绝私有PTY路径不绕过/不重放launch；旧凭据不入任何文件。

正式完成W2L blocks0–20各6PASS，共126/240；Muse0–2各6PASS，共18/240；Ditto05 blocks0–1各6PASS，共12/240。W2L20 PID206742/ticks2280261已退，manifest raw166f596e92e3be0b2aee04389a37d95975e9d741ee633cdfe108afc410881db9；旧detached直接SSH退出码没有保留，标historical_exit_code_unavailable，不推断0。Muse2 ROOT实际SSH读终端BLOCK_COMPLETE，PID199178/ticks2146901，manifest raw7ee64918a0a290765869f4f9d58508c7e6c35ab5b165e91e45bf4c4621ad38f0。Ditto1由Dev实际read-only确认6PASS及PID210516/ticks2303492退出，GPU0%/5MiB；actual launch receipt8677B/raw89896e20fb7d4728687e07f05da65644211ed271b85bf6d6211e1e3219e2b3b7。下一个公平块Muse3，再W2L21，再Ditto2；已让Signal继续相同批准launcher而不因CPU实现停GPU。

Ditto05修复旧03首组六renderer前KeyError bytes，只推导实际regular WAV stat并核fullfile/PCM/clock；builder补actual model.preprocessing身份，失败不改写。ROOT全文审查同一launcher26097B/0c0ef0c656e91e3602976cdb064bc42cd80e498a67815973aae7195f8ba7b6f1，后续相同固定六行块已授权，无需每块重审。Adapter68998B/ba699de2be6edb4597d5336049798501f7ebf8a4b1118a6337c87f5d09d1c2db，builder57135B/aa9af8a8528d9ec8d8c352b5ffde23828bf55ccc8b3695efd0fea9f888712f6f。Plan895194B/raw82ceb8ebd93f541bee9ae3ac9f93f3a3e657ff4979dcb451d03732cd2ad52475，binding22530B/raw275c8ac529653ab4d066320b65a25dd3c3fb181ca89f09f6928d092013acfc2b，rows660128B/rawf386441f3ed54da8d69f38610f1cb7c8d79b0320a214ce20d1421fda3c232164。Actualstage /dev/shm/fit40_ditto240_stage_attempt05/project，8payload fullbytes/actualCPU/launcher13pins已ROOTPASS，细节见前checkpoint。

ROOT独立实际首Ditto六完整核验 .codex/scratch/root_ditto_fit40_first_block_actual_qc_attempt02/actual_readback_qc.json，raw78e7056fd3d0eb2873e9a32c9809d1852252b8cb6d6ff1d65ba980c11540c91a/self fa12953e400495f6cd402838bdc555f83a6c48527b28a404893d6b6ad54a811d。Fullreadback F:/codex_scratch/v100_ditto_fit40_block_attempt05/full_readback_attempt01/block_000_attempt01；1445 regular files/42,041,282B全部相同远端inventory/raw/self/六plannedrow。六F32交付mux解码完全等原N62639/PCM f4e611…，六PCM16评分mux完全等全原音转换；六generator实际WAV及98ceilnativeframes/有理PTS；sevenNPZ完整hash/dtype/shape/finite/query92×31/lag−15..15；official score/raw_matrix_arrays.npz只有distance/lags两个arrays，与syncnet七array相同；actualtrack100entries中0..97 decoded、98..99 nominaloutside，五帧FULL92/GUARD20 52；严格rendererreceipt/确定性/threads1通过。ROOT01仅本地plan后缀FileNotFound（1445文件核验已通过），fresh02修路径后PASS，失败保留。

真实消费器fit_only_stagea_ingest_attempt03.py53184B/22cd7f983b22babcae0d9c120f4720cb84a218ec4d2b84551d5e1668d23aeb95已ROOT10CPUtests PASS；actualMuse/W2L各首6的helper ROOT独立run()结果与agentreceipt逐byte相同，各6PASS_ROW_TRANSPORT_ONLY，没有family归约/init。Muse全1463文件及原音/PTS/arrays先前ROOT实际PASS。Score.preprocessing绑定model descriptor.preprocessing SHA；PCM转换code独立。Snapshot enum为NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN，canonical record字段snapshot_sha256，pin.self_sha256仅alias。

SPSA composer04 prepare_fit40_spsa_stagea_packets_attempt04.py47581B/fb8f4ddb7c314320bcdee1db886d9ea4669e111f5e2c2a2fff4cb07afa2d8a21，只改03版本标记与consumer01→03导入，ROOT实际完整窄diff+composer函数/seedlineage审查、4/4focused tests PASS。Dev实际三冻结plan metadata composition720行，output .codex/scratch/fit40_spsa_stagea_packets_attempt04/composition_attempt01/fit40_three_model_composed_plan_attempt01.json2584958B/rawb3983aeb0136f227bc3618603d1e66475ba496aac592cb381fa63838a59baa97/self5a35c5506c2ef9411b37438af1fbc60fc3b4b72be5e5c9900083f55d52e91811；receipt3376B/rawa27884b730b27267a2f45057a806e66812013e3555738303753b3e46291c8cdf/receipt_sha256a2c22d361be7552c840a14e6a443701c69fca6dd5328db86c2812a898960b61b。Source manifest5660B/raw1edb6ab948bc4b401a873d0c13566c435dc4ab92d2540d724c3082452937b9ba/selfd737bf667b6f4c1ddf71577493a2bff16955fd1ff857625e5e3f543404ebae24。ROOT已全文读actualdriver/receipt/manifest，独立实际composer replay下一步。Composition只metadata，所有选择/proposal/render/score false。

ROOT刚实际对三份冻结plan比较，models[0].generator与protocol.generator全部不同。前者是各model renderer/runtime descriptor，后者是原protocol/science/frontend/scoring对象（Ditto协议仍有W2L model_id来源标签）；不能要求两个字典等值或静默覆盖。Composer正确分别保留descriptor、generators_by_model=source_model_protocols.generator、source_model_execution/runtimebinding。Muse/Ditto SPSA hook错误等式已通知两agent修freshattempt；Dev确认Ditto02必失败，保留02并新03。普通core.run固定240不能塞24/120；仅复用实际_render_and_score，由hook恢复每模型原协议和runtime并核actualdrive证据。

当前四seed实际architecture是cross_tfg_shared_recipient_transition_head_v3，implementation32233B/77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87，ctor eaf51303012125c2de40dc102fc04a983c130b6d2aac31cd96b6e5648c3e27b8，n_fft512/hop128,targetRMS0.0376838172675746,3dB。record parameter.values为3×3九θ，seed method.config没有θ。不能因metric叫D0静默换d0_gatedc head。source_snapshot_step30/60/90/120与新centerstep5/10/15/20分离。Signal负责公共fit40_snapshot_spsa_application_attempt03.py和共享snapshot/receipt/PCM validators +Musehook；Dev负责Dittohook/packet04；ASR先fairqueue、再narrow dispatcher。公共receipt schema cross_tfg_fit_snapshot_generator_drive_result_v1必须保存exacthead/ctor/seedartifact/step/θ/config/point/parent/input/outputF32/PCM/N/clock/RMS/projection，不加不存在gate_mode等字段。实际seed120θ→existing160application中对应WAV PCM replay尚待Signal完成。Private methodconfig precomputed_fit_snapshot_waveform marker只能伴实际candidate generator_condition_input与application raw/self证据，序列化科学config SHA保持。

ASR公平queue02刚报告8/8CPUtests，source68225B/7cf3b2e7085680e8e0feeaf895cc3f3f80a9beb2c07c4f67373809f39bb2010c，tests17513B/6e1f47afa6bd7e3c2e36de9349fa6ef4b9e52164a58716728df973e0661a514e；ROOT全文review及实际重复tests尚待执行。历史terminal-prefix inspector03按W0–20/Muse0–2/Ditto0–1实际manifest/run_status/exact6PASS/plan/rows/launcher/PID退核验，不按计数导入，旧detach退出码显式unavailable。Queue一次六行roundrobin、exclusive nonce/PIDticks锁/appendonlyclaim+terminal、同模型controlfailure封存暂停其他继续，不retryfailedleaf、不delete、不绕reserve。ROOT已指出并ASR修正：stdout只minimal receipt pointer，读fullreceipt才验row/plan/source；run_status在control/；W2L runtime_binding_path+rawSHA+capacity_stage env必须精确且project_root为实际SHM，output在/blocks，不是/output。FinalROOT源/test/index核验后启用，当前手动相同已审launcher持续。

机制沿既有《冻结增强头的门控时序与频谱贡献》《音素平均谱形与内部轨迹的因果分解》：Cbest主要背景B，正确门控D0附加贡献小；512msroll保留Cbest，gateDC混静态谱形，μ/ρ受者相容。当前9θ静态/正负flux×3bands,32ms/8ms STFT/global3dB/RMS，以W2Lmel/MuseWhisper/DittoHuBERT实际三frontend优化。源前端差异只是推理依据，没有冒充新增因果结果，不重跑已完成全局音量族。

Git branch codex/cross-tfg-audio-head-20261003，HEAD 5688784dd5c6c1272074f0c4b6c5ff79f117c907 local/unpushed；原五tracked dirty保持，新增experimentuntracked，BM/HANDOFF/scratch/runs/env/SHM是machine-local或fragile，无commit/push/PR。MutableTask/Experiment/HANDOFF当前仍W120/Muse12/Ditto6+Muse2live，需下一materialmilestone更新；旧immutable08-59checkpoint不改。代码与实验仍由三名既有GPT-6 Luna/maxagent执行，不新开用户thread。

## Changed Files

- Ditto05adapter/builder/launcher及freshstage已实际执行；首组六ROOT全媒体/矩阵QC02为新增实际证据。
- Composer04只更新实际consumer03依赖，ROOT四tests和窄diff通过，实际720metadata组合保留各模型独立descriptor/protocol。
- 公共v3application/Musehook/Dittohook与queue02/index03仍在封存/复核，当前source枚举须重新取agent最终pins。
- ROOT checkpoint创建新immutable handoff；不编辑旧checkpoint、不把数据运输写成科学收益。

## Verification

ROOT独立实际Ditto1445文件fullSHA/manifest/ledger/六原音PCM与评分PCM/98nativeframes/有理PTS/NPZseven及officialtwo/decodedfive/strictreceipts PASS；前actualMuse1463文件和consumer03十CPUtests与两模型十二真实行运输replay PASS。ROOT本轮composer04四focusedtests实跑PASS和source/fullactualcompositionreceipt读回。未完整720、未科学初始化/更新、未读取sealed新Dev/Confirm。Queue02八tests当前为ASR报告，ROOT尚未独立执行。Commonapplication seed120 actual replay尚未完成。

本checkpoint已重新读bm-checkpoint/bm-writing、StartupRouter→实验指令/CodingSession/Task、projectconfig；freshGit/GitHubcanonical/branchPR[]、samechat coding/codex/legacy分页搜索并直接验证前checkpointexactsessionidentity。CLI实际实现显式write_note(overwrite=False,output_format=json)，只接受created后directreadback exactmetadata；秘密不写入。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；当前SHA local/unpushed，不构造GitHubcommit链接，branch实际PR[]。
- ADR022 tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement；ADR023 decoded-video-window-support-for-native-syncnet；ADR024 live-process-resource-capacity。
- 本checkpoint WorkingState artifact路径与raw/selfpins为实际产物权威入口；server-access tts-exp/docs/reference/server-access保存无凭据机器入口。

## Observations

- [result] 三模型正式Fit有效156/720：W2L126、Muse18、Ditto12；保持完整40条/20人×三模型×六方法分母，未开展科学初始化或SPSA更新。
- [verification] ROOT独立Ditto首组六1445实际文件/完整原PCM换轨/NPZ及nativePTS核验PASS，composer04四CPUtestsPASS。
- [decision] 模型descriptor.generator与protocol.generator是不同冻结契约，分别对原值及binding验证；共同增强头必须沿实际v3九θ实现而非换另一个D0架构。
- [blocker] 完整720尚待公平执行，共享application实际seedreplay/模型hook和queue最终source+实际index审查尚待闭合；旧AutoDL终局未知不阻断V100主线。
- [next_step] 继续Muse3等固定六行公平Fit并启用已复核queue，同时闭合公共v3application/模型hook；完整720后按ADR022三模型联合优化，再完成独立Dev/Confirm与可用接口。

## Relations

- continues [[Codex checkpoint - 2026-10-09T08-59-43Z - Ditto repaired execution and verified real Fit ingestion]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]
- relates_to [[024-live-process-resource-capacity]]