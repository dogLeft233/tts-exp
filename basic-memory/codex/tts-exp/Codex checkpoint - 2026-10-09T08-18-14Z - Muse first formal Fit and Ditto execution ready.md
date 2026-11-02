---
title: Codex checkpoint - 2026-10-09T08-18-14Z - Muse first formal Fit and Ditto execution
  ready
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t08-18-14-z-muse-first-formal-fit-and-ditto-execution-ready
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T08:18:14Z'
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

# Codex checkpoint - 2026-10-09T08-18-14Z - Muse first formal Fit and Ditto execution ready

## Summary

执行agent已取得Wav2Lip正式114/240和MuseTalk首组6/240实际原音评分；Ditto完整240包实际stage及CPU闭包通过，三模型共享Fit路线已接齐。

## Story

用户希望恢复TTS增强TFG的局部声学机制，并做适配多个TFG的自然音频增强头。增强PCM只驱动视频，交付和SyncNet评分换回完整原自然PCM及原时钟；方法要在独立说话人与多种生成器上超过固定RMS。ROOT维护理论、协议、BM与结论，代码及实验由GPT-6 Luna/max执行，失败后继续推进。最新用户确认局域网Ubuntu用户名wjj，现有公钥可连192.168.50.2:22；数据下载、网盘、云API和实验已经授权，凭据不入文件。

旧step120仅在Wav2Lip训练，跨模型共同FULL Cbest对RMS等模型+0.011149，D0正向改善−0.033662，没有达到跨TFG门槛。当前按ADR022先用同40条Fit原音/20人获得三生成器×六方法720唯一评分行，再对共享9参数执行20步SPSA与四个完整中心。三个模型的native重复/完整原音换轨/官方矩阵与decoded支持工程资格均已完成；当前新工作是正式Fit运行、实际结果运输与初始化/更新执行层，工程通过不等同于新方法效果。

## Working State

权威任务《开发跨 TFG 的自然音频增强头》active、实验《跨 TFG 自然音频增强头的训练与原音换轨验证》implemented。ADR022/023/024不变：六方法共同FULL，说话人等权再模型等权；每模型Cbest对两参考正、D0/Canchor不恶化、平均对RMS至少+0.050；缺行/失败阻断选择，完整而无适格才按预定违例函数初始化。五真实decoded帧/有理PTS支持保留raw与完整分母，liveRAM/cgroup/SHM/output容量保持5GiB。新Dev20、Confirm80和LatentSync封闭，尚未实际720选择或SPSA更新。

✅执行机wjj@192.168.50.2:22，repo /home/wjj/tts-audio/tts-exp，V100-SXM2-16GB UUID GPU-d12b3309-c299-da83-064d-a3cd619bd560。W2L/SyncNet alias /home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python；Muse/Ditto共享Torch2.5.1cu121环境 /home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01。仅六行块串行用GPU，各agent在CPU侧准备/归档，ASR维护公平队列。每launch重新测真实容量，媒体SHM、完整字节归档B/F。网盘目录存在不表示云上传完成。旧dirty源码/环境保持，密码/API不入文件。

共同science SHA ae96def4c16fecabc6e49b2aeabea0a76a0308ea159fa6c7688f7908c62c7180，W2L parent实际plan902628B/raw25f49cf6e87800da57fc797210b5d0c4bb6bb3396af04f9f9e9ff1aec0146832，rows678368B/54a69d99b34cf70b02f58b4a4bae032bcf5446fb9777d7a5594f6cc45ffeffbe。40自然+160snapshot预生成源PCM/时钟共同绑定绝对SHM路径。W2L blocks0–18各6/6PASS共114/240，block18 PID178189终局BLOCK_COMPLETE、GPU显式释放。不能因时间推断block19完成。block16错误callerSHA预GPU停止为空dir，fresh16_attempt02才是6PASS，失败保留。

Muse正式packet02 local .codex/scratch/v100_musetalk_fit40_240_attempt01/project_payload_attempt02/，remote /dev/shm/musetalk_fit40_240_attempt02/project，12文件1,981,392B。overlay57751B/589184880821522d4f90609244b6cfc0230616079bb927b1cea7b259d87fb4d3；plan888737B/ae79df59ef5c3f05ba99d7850a1e0158c9fefb07e6b0e0fee8cc557af9f52dc4；rows653408B/2c69885517a36e571406fa7a864097a5eb593eb8cb4006d35dcf6ade2c9d0fb0；binding4438B/ee474826c0cea8b2f039f4b7dac986d7da98fa88fa1f0dcff85620c12a3f18de。ROOT实际全部12远端body/ZIP/source/240science投影核验PASS，QC .codex/scratch/root_musetalk_fit40_packet_qc_attempt03.json selfc015dc9fcd0ceb34b601011c2006ef2b3e91f6ed82e62173c10c36528e763379。只修实际producer计数9 vs布尔True契约并保留01FAILED；02 rows与01逐byte相同。ROOT最初packet02QC过严比较模型namespace而失败，无输出；fresh03限定仅namespace/overlaypin工程差异后通过。

Muse真实CPU worker04完成整个_verify_input_bundle/core parser、219科学pins/53,472,480B、40natural/160snapshot、108sourcepins和9assets/3,986,182,419B。receipt .codex/scratch/v100_musetalk_fit40_240_attempt01/cpu_closure_readback_attempt04/cpu_closure_probe_receipt_attempt04.json 25212B/rawa782afdb36fcf9f082c3f97252215fe376cf12dd67130b3c93eecebcac7eb5eb，self365646eac8be6278f94b786cc15b6e719991a50b034739cab321771d0f3b6717，ROOT已实际全文读验。之前01字段契约、02额外refs错误锚、03Path JSON错误均保留。launcher02 PID166739预render因CROSS_TFG_DETERMINISM_RECEIPT_DIR env遗漏而失败；freshlauncher03仅补exact env并使用fresh block_000_attempt02，实际PID170562成功、terminal BLOCK_COMPLETE 6PASS/0failed/0incomplete，repeat NOT_RUN_SUBRUN。成功launch6374B/rawf46b499229e61d34f5f8baaf4e2002234e2c48f6ab51779348027705b0ad8ab9，manifestd90d18a5e5293c3becaa076601bd68236ca01d5cf8a0542ad029397486903dfa。Signal已完整下载1463文件/43,519,913B，实际QC PASS .codex/scratch/v100_musetalk_fit40_240_attempt01/block0_actual_readback_qc_attempt01.json 616217B/raw84e49316e34613c9e87101385167a984254fb41436dd5c4b4bb327c21165359d/self49309c54070ce44d8a7350c95593a0e2b9147cdc6b8612b72db862aa4611462e。该独立agent QC验证97native帧、92query、seven NPZ/raw matrix、五帧selected-track与两nominaltail支持、fullF32/PCM16/媒体；ROOT尚待实际全文及底层body核验，不能声称ROOT已完成。

Ditto完整builder03 scripts/experiments/cross_tfg_audio_head_20261003/build_stagea_ditto_fit40_plan_attempt03.py 56627B/19da5abc5c0ecf472410adb6f38fd727e5014b07cde62903389021013b6c75cf，adapter04 run_stagea_ditto_fit40_block_attempt04.py 66156B/2fac8fcf07f2d7bc55e808365878f34a130e5aae7e505b8d895d735f1b26a7e3。ROOT完整builder/adapter累计delta审查、实际6/6stdlib tests PASS，log .codex/scratch/root_ditto_block04_cpu_tests_attempt01.log raw112306a9d4ad1f10ad6706cb37ea53c8145678f1d9737f0aed4c8ad84beafaa5。Fraction/native等值PTS、真实结构化seed与SyncNet alias/prefix/binaries固定。ROOT实际metadata重放22全文pins+四输出逐byte一致，QC .codex/scratch/root_ditto_fit40_plan_replay_qc_attempt01.json self285bc8085ec9c772283871438dcb2fca48d957099fec3989ce132ff5f2f1c4bd；packZIP全部8source bodies核验PASS，QC root_ditto_fit40_stage_packet_qc_attempt01.json self374011a8a5320b4e11208364a479c23181ab05eaee543aa10d2dc85f9942b072。

Ditto remote /dev/shm/fit40_ditto240_stage_attempt03/project 已actual8/8 CREATED_EXCLUSIVE，stage receipt3084B/raw59913f2237d346846f942eea88623dda34e979d52f4419471bcb0297a27f6522/self954ca92b9b5967185c6f2494ded604f5baad7d4a958d798bc7c04c47fc465cae。plan895096B/130af03c9dcf791e024ae59f7ec21df67ef1e61092b409eda2065d48360175ee、rows660128B/f386441f3ed54da8d69f38610f1cb7c8d79b0320a214ce20d1421fda3c232164、binding22530B/cbf0bc67cf3d300f2c68257e0ef19aa4f5b62341d444c7069309fbbe2e7a31d2。实际8body读回 .codex/scratch/v100_ditto_fit40_stage_payload_attempt02/stage_remote_readback_attempt01/，ROOT尚待独立逐byte核验；Dev已核验。新增持久tiny helper/renderer目标exact pin，其余stage SHM，不改原modelsource。

Ditto真实CPU04 PASS .codex/scratch/v100_ditto_fit40_stage_payload_attempt04/cpu_readback_attempt01/cpu_closure_receipt_attempt04.json 14608B/raw0a51476c4b323dae68c538edf8b00611db7652487c7bbd0f2b7a1aed77bb3191/self4f70017d197cf5a5a07bfee1754c274c5c93cba83dc5ae1f21401cb825a58cd0。ROOT已实际全文读receipt：Python3.10.20、四exact import origins（baselines由pinnedpackage closure预载）、240ordered/40samples/20sp/200audio、per-sample ceilframes、exactparent/binding/renderer/model/scoringpins、无Torch/ORT/GPU。worker04 23840B/90a8ed9d18aa804a0eb8a87d33becceccb78846d3cb1d8c1903890734649a577。ROOT此前完整worker01+累计delta审查，修parentbindingSHM锚、每sample长度、shared_source_* pins来源、包已载模块检查；两次实际CPU03失败（argv receiptguard、预载baselines拒绝）明确保留。Dev随后实际启动首组六方法，PID181942已退出，terminal BLOCK_COMPLETE_WITH_FAILURES六条FAILED、0incomplete；真实容量门控PASS，stdout431B/stderr空。正在只读逐行错误，未retry或第二块，失败不是科学阴性。

真实720消费者01 fit_only_stagea_ingest_attempt01.py 51811B/a3806783d871d015f6d17aa8846c2da88948180e57deb47ff86b11485d4aadef，ROOT完整审查+实际WSL6/6 unittest PASS，log root_stagea_ingest_cpu_tests_attempt02.log rawfba4479891e36f642631c83dc177e6a54846831ca7c4c56264a9b7a2ee113be6。完整720分母/三plan/ledger/status、实际wholeNPZ SHA再解码seven arrays/源PCM/时钟/模型与anchor15协议严格固定。ROOT初次4/6失败是synthetic model循环最后plan路径泄漏，修fixture后PASS。ASR实际W2L首块consumer helper PASS；刚接Muse真实source快照时发现记录selection_status为NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN，而01要求NOT_SELECTED。工程契约差异，不是效应失败。ROOT已要求fresh narrow02读全四实际记录，仅接受确证精确未选枚举、保留所有全文SHA/step/model/720条件并加未知/SELECTED拒绝测试；原01及实际失败receipt保留，不能改源snapshot或plan。02尚在准备，未完整消费720。

SPSA packet03已agent4项CPU PASS，ROOT审看实际snapshot来源为fit_snapshot/precomputed_fit_snapshot_waveform/null checkpoint/config.snapshot_sha256=step record self，不能误叫训练weight。提案center source_step与legacy source_snapshot_step分离；execution.planned_rows 24/120。三真实plan composite720刚因protocol.generator不同而预检查停止，正在fresh修composer仅比较共同protocol字段并保留各模型完整generator/binding/parent pins，不更改前端。ROOT未完成最新composer全审。narrow执行adapter/livecapacity与theta→波形应用receipt仍需实现；20steps×8clips×3×2=960proposal、四full40×3center480，exact缓存两参考来自初始720，共2160物理行；canonical TorchCPU1thread，NO_UPDATE/fourway FULL/9θ/3dB/RMS/时钟与ADR022不变。

机制仍依据已恢复的门控谱形及μ/ρ因果笔记：Cbest主要背景距离B贡献，正确门控时序对D0有较小附加收益，512msroll仍保留Cbest，gateDC混静态谱形。当前9参数head32ms/8ms STFT静态+正负flux×三bands/RMS匹配/global3dB，沿实际多模型原音评分优化；不同源码前端不当作新因果实验结果，不重做已做的全局音量族。

Git 5688784dd5c6c1272074f0c4b6c5ff79f117c907 local/unpushed、branch codex/cross-tfg-audio-head-20261003，五原tracked dirty保持，新实验untracked，BM/HANDOFF/scratch/runs/SHM/runtime属machine-local或fragile。无新commit/push/PR。最新可变Task/Experiment/HANDOFF约记W2L102/240、MuseCPU04READY/launchpending、DittoROOTmetadata/待stage，HANDOFF57行并指前checkpoint；本checkpoint真实最新114+Muse6+DittoCPUready及首组六工程失败，需继续同步可变Task而不改旧immutablecheckpoint。旧AutoDL DirectD0终局未知，拒绝私有PTY不绕过不重放，V100主线持续。

## Changed Files

- Muse packet02、whole CPUworker04/launcher03：只修producer/schema/path/serialization与env，实际首组六方法成功。
- Ditto builder03/adapter04、create-only8memberpack、CPUworker04：真实native长度/PTS/seed与导入闭包接齐，ROOT6tests与metadata重放通过。
- fit_only_stagea_ingest_attempt01.py与tests：ROOT6tests通过，实际snapshot未选状态契约将由fresh02窄修；不修改固定source记录。
- SPSA packet03：真实fit_snapshot与source step身份修正，composer模型协议和后续执行层仍在准备。
- BM Task/Experiment/HANDOFF维护权威工作状态，旧immutablecheckpoint保持。

## Verification

ROOT实际完成Muse全12body/ZIP/science核验、CPU04全文receipt核验；Ditto6/6stdlib测试、22全文pins与四输出逐byte重放、8memberZIP/source全文核验、CPU04全文receipt；720consumer6/6CPU与selector06此前24/24CPU PASS。agents实际运行W2L114行、Muse6行并取得首组全部产物，实际stage/CPU Ditto全240包。Muse首组底层及Ditto8remotebody ROOT独立复核尚未执行。没有新的完整跨模型效果、初始化或SPSA更新。

本次重新加载bm-checkpoint/bm-writing、Router→实验指令/Coding Session/Task、配置；same-chat coding/codex/legacy分页检索及直接身份核验，freshGit/GitHub canonical仓库与branch PR[]。write-note CLI已验证显式overwrite=False/output_format=json；只接受created并直接读回精确host元数据。失败checks均按上列来源保留，不把未运行或synthetictests当科学结果。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)，local SHA不构造commit链接，branch PR实际[]。
- ADR022 tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement；ADR023 tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet；ADR024 tts-exp/docs/adr/024-live-process-resource-capacity。
- 无凭据持久连接 tts-exp/docs/reference/server-access；实际产物/QC/失败attempt路径上列。

## Observations

- [result] 执行agent实际取得W2L114/240、Muse6/240；Ditto240包stage/CPU闭包实际通过，Ditto首组terminal六条FAILED、正在诊断，不计有效科学行。
- [verification] ROOT实际Muse12body、Ditto6tests/plan重放/8ZIPbody和两CPU成功receipt核验，实际720consumer6tests通过；Muse首组与Ditto远端body另待ROOT复核。
- [decision] 保持ADR022/023/024、原音完整时钟、封闭Dev/Confirm与公平六行GPU队列；工程窄修freshattempt，失败记录不改写为PASS。
- [blocker] 消费器实际未选状态枚举与source契约差异须fresh02窄修；完整720与SPSA真实执行层尚未完成，不能提前选择或宣称跨模型收益。
- [next_step] 定位Ditto首组六工程失败并以freshattempt修复重跑，同时完成Muse首组/远端stage的ROOT实际读回核验，与持续三模型Fit及fresh消费者组成完整720后执行ADR022联合优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T07-41-12Z - Formal Fit packets and verified actual score ingestion]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]
- relates_to [[024-live-process-resource-capacity]]