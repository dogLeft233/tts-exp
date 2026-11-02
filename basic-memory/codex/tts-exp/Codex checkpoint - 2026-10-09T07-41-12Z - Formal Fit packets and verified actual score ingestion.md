---
title: Codex checkpoint - 2026-10-09T07-41-12Z - Formal Fit packets and verified actual
  score ingestion
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t07-41-12-z-formal-fit-packets-and-verified-actual-score-ingestion
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T07:41:12Z'
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

# Codex checkpoint - 2026-10-09T07-41-12Z - Formal Fit packets and verified actual score ingestion

## Summary

Wav2Lip正式Fit已由执行agent取得90/240行；Muse240实际远端CPU验证发现字段契约错误后正在修复，真实720矩阵消费器已通过ROOT独立6项CPU测试。

## Story

用户希望找出TTS增强TFG的局部声学机制，并据此做可适配多模型的自然音频增强头。增强PCM只驱动视频，交付和评分必须换回完整原自然PCM及源时钟；方法要在独立说话人与多种生成器上超过固定RMS。ROOT维护机制、协议和本地BM，代码与实验由GPT-6 Luna/max执行，失败后持续推进。用户最新确认局域网Ubuntu用户名wjj，现有公钥可直接连192.168.50.2:22；下载、历史数据、网盘和云API已经授权，凭据不入文件。

旧step120只经过Wav2Lip训练，跨模型FULL Cbest对RMS等模型+0.011149，D0正向改善−0.033662，未达到跨TFG门槛。当前按ADR022在同40条Fit原音/20人上先取得三生成器×六方法的720唯一评分行，随后对9参数执行20步SPSA及四个完整中心；不以部分Fit结果选初值。三种模型的实际native重复、原音换轨、官方矩阵和decoded支持工程资格已完成，本次重点接齐正式Fit执行与严格实际结果消费。

## Working State

权威任务《开发跨 TFG 的自然音频增强头》active、实验《跨 TFG 自然音频增强头的训练与原音换轨验证》implemented。ADR022保持四legacy快照30/60/90/120+identity/rms_fixed的六方法共同FULL、说话人等权后模型等权；每模型Cbest对两参考正、D0/Canchor不恶化、平均对RMS至少+0.050。缺行/失败不能选；完整但不合格才用预定违例排序初始化。ADR023按五真实decoded帧和有理PTS判支持，raw及完整分母保留。ADR024实测RAM/cgroup/SHM/output并保留5GiB。新Dev20、Confirm80、LatentSync仍封闭。

✅执行机 `wjj@192.168.50.2:22`，项目 `/home/wjj/tts-audio/tts-exp`，V100-SXM2-16GB UUID GPU-d12b3309-c299-da83-064d-a3cd619bd560。W2L/SyncNet Python alias `/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python` 的有效symlink与隔离prefix已核验；Muse/Ditto共享Torch2.5.1cu121的环境 `/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01`。最新容量读数rootdisk仅比reserve多27,563,168B，源packet与媒体放SHM六行块，B/F归档；每次launch重新测。旧dirty代码和环境不改。网盘挂载文件存在不表示云端上传完成。

W2L正式packet位于 `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05/`，remote `/dev/shm/fit40_w2l240_stage_attempt05/project`，共230文件/55,302,272B，全部远端SHA stage已实际通过。三模型共同science SHA ae96def4c16fecabc6e49b2aeabea0a76a0308ea159fa6c7688f7908c62c7180。执行agent已报告blocks0–14各6/6PASS/0fail共90/240，block14 PID154344 terminal；block15正在准备，不能凭时间推断完成。GPU按六方法块边界交给真正CPU closure就绪的Muse/Ditto，不能等待包装而闲置。

ROOT前轮实际nativeQC `.codex/scratch/root_native_actual_arrays_readback_qc_attempt02.json` self391fe1a5d2135ac5543841f6320361ec79d03c60afd0c813a39c2f2c79939356：Muse47body和W2L block0共83body的原数组/七数组/完整自然F32/实际ffprobe与PCM16转换逐bytePASS；W2L native95/raw92/共同FULL91，原N62639。Muse两个native144帧、官方139×31矩阵/139支持 exactrepeat。Ditto score-only04复用两个原145帧视频，官方track/crop评分actualPASS；ROOT `.codex/scratch/root_ditto_actual_repeat_readback_qc_attempt01.json` self533d16c8ee10bfed9e58e9d0aac7de546782778ceca6630c754d78044ff0cd71实际核验139×31/seven数组/video elementary exactrepeat、145有理PTS、139支持、fullF32 N92225和PCM16 scoringmux源byteequal。工程资格不作为效应结果。

Muse240 attempt01已完成ROOT完整source/本地packet7项CPU测试、12文件/1,979,307B full-file/ZIP-member/240projection/science/SyncNet协议核验，并在 `/dev/shm/musetalk_fit40_240_attempt01/project` create-only stage；ROOT实际读回全部12body核对PASS。权威本地 `.codex/scratch/v100_musetalk_fit40_240_attempt01/project_payload_attempt01/`，plan raw85154911e14913e422ba9ea100919d7302118b43af6d1340f7c48b3ad935cb9f，overlay55666B/b32a7440dedfa1b6ebf01179f8b4b6d21b6f9376bf5b1f2a2d41909762aea2a6。ROOT packet QC self3a4fd2d8f5f7cb8b28c17bfe582c8696e00da5cdfdeae4b2edfa0010df1efdaa；remote-stage QC selfb5c51156acdb8d25181f31ac83b75954e8a35c604178f09aa2edb4e6a41b39b4。

随后Muse真实remote CPU closure attempt01 FAILED，在filehash/GPU前停止：overlay `_verify_input_bundle` 要求import report的 `all_loaded_musetalk_python_files_verified is True`，实际已固定producer attempt10输出整数9，与九模块mapping一致。失败receipt `.codex/scratch/v100_musetalk_fit40_240_attempt01/cpu_closure_readback_attempt01/cpu_closure_probe_receipt_attempt01.json` raw06af98a21b47d7689ae7cca0f411cf81f0d344ab458a200ade59f832539c7196/self3a30a9182ad61e61ba5f0541d50e280b8efc9c000d51ee2fde35fbb416a2e914。ROOT七CPU测试覆盖source/evidence/PCM/frames，未执行真实machine `_verify_input_bundle`，因此未发现该producer契约差异。Signal已获授权只改counter/schema gate，以正int计数==module mapping长度并核对每模块path/bytes/SHA，创建fresh overlay/plan/binding/payload02与remote namespace；必须用真实路径执行完整 `_verify_input_bundle` 和core import后交GPU。01保留FAILED，当前无Muse正式GPU行。overlay位于payloadROOT，relative evidence锚正确，不把之前已撤回的nested-path疑虑重新加入。

Ditto240 adapter02完整ROOT审查后发现native frame5的reduced PTS1/5被错误要求5/25；Dev fresh03用Fraction/交叉乘法比较等值并补全SyncNet alias+resolved/evaluator/ffmpeg/ffprobe pins。source `scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt03.py` 65912B/SHA53d32e1326893e5d85d3aaf27221dd177718b0c414f238e5319d88e7585d9c80已agent AST parse，但ROOT尚未review03 delta。完整40×20×6 plan/binding builder正在准备，execution adapter不能代替packet。ROOT的pytest尝试因WSL marker无pytest而0项执行，log `root_ditto_block02_cpu_tests_attempt01.log`保留；已要求stdlib unittest counterpart实际测试。原生run02 64,682B receipt已agent完整读回 `.codex/scratch/v100_ditto_native_receipt_readback_attempt01/run_receipt_attempt02.json`，整体FAILED_ENGINEERING_ONLY_REPEAT是后续整图score错误，三code/12weights实际renderer报告用于runtime绑定，成功资格由artifacts+score04+ROOTQC组成，不能重标旧receipt PASS。不再重复engineering identity。

严格真实720消费器已冻结：`scripts/experiments/cross_tfg_audio_head_20261003/fit_only_stagea_ingest_attempt01.py` 51811B/SHAa3806783d871d015f6d17aa8846c2da88948180e57deb47ff86b11485d4aadef；tests22185B/SHA2c6472d0963816c08c030165aeebf11dfc3f54d33d2756eee3fa8ae5282ccce2。ROOT完整source审查并实际WSL unittest6/6PASS，log `.codex/scratch/root_stagea_ingest_cpu_tests_attempt02.log` rawfba4479891e36f642631c83dc177e6a54846831ca7c4c56264a9b7a2ee113be6。实现三模型×40六行块status/manifest/ledger+720完整分母，declaredNPZ→actual restore，先whole-file SHA再np.load(BytesIO,allow_pickle=False)，验证七数组元数据/finite/queryclock/sourcePCM/artifact/binding/单模型descriptor/共同SyncNet protocol及anchor15，模型/score SHA和自然PCM16映射严格匹配。失败/缺行/损坏保留CONTROL_FAILED，成功verified计数只含成功读验。初次ROOT测试4/6失败因synthetic fixture模型循环复用最后模型的plan路径/hash；修复fixture后6PASS，不是模型效应失败。另一次hash guard被agent口述test SHA误抄拒绝，未运行；actualfullhash验证后再执行PASS。selector06 ROOT此前实际24/24PASS/0skip。未实际消费720、选择初值或SPSA更新。

ASR正在准备ADR022实际SPSA执行器：`prepare_fit40_spsa_stagea_packets_attempt01.py` 26664B/SHA80437dbf49854e89e11272e0803c7ef7c0c201699c129fc0309f16891296990d为plan层draft，agent AST parse，ROOT未review。必须只渲染plus/minus各8clips×3models=24proposal行，缓存exact identity/rms rowIDs来自完整720；20步共960proposal，四full40×3中心480。原240 wrappers/core/gate的固定40×6契约需要narrow新adapter和live capacity binding，共用 `_render_and_score`；完整verified720前不可初始化/更新。科学超参数、四路共同FULL和NO_UPDATE规则保持ADR022。

机制依据仍为已全文恢复的《冻结增强头的门控时序与频谱贡献》《音素平均谱形与内部轨迹的因果分解》：head的Cbest主要来自背景距离B，正确门控时序对D0有较小附加收益；512ms错位保留Cbest，gate DC约0.5混入静态谱形。当前9参数head按32ms/8ms STFT、静态+正负flux gate、频带common-mode去除/3dB平滑掩码与全局RMS匹配，在不同前端实际原音评分上继续优化。源码推理不是新因果结果，不重跑已做过的全局音量族。

Git 5688784dd5c6c1272074f0c4b6c5ff79f117c907 local未push，五原tracked dirty保持，新实验源untracked；BM/HANDOFF/scratch/SHM/runtime为machine-local或fragile。没有新commit/push/PR。旧AutoDL DirectD0唯一launch终局仍未知，拒绝的私有PTY路径不绕过、不重放launch；V100持续路线可继续。最新可变Task/Experiment/HANDOFF写入在MuseCPU失败前已完成，仍记78/240和等待closure，需同步90/240及01失败/02修复；本checkpoint保留准确最新状态，旧immutable checkpoint不改。

## Changed Files

- `fit_only_stagea_ingest_attempt01.py`及对应tests：实际NPZ运输、数组和模型/协议身份严格适配，共同完整失败分母，ROOT6项实际CPU PASS。
- Muse240 builder/overlay/payload01及ROOT packet/remote-readback QC：实际12文件stage成功，真实CPU closure失败保留，fresh02正准备。
- Ditto adapter03/完整builder：native PTS等值和实际runtime pins修复，尚待ROOT review/实际CPU与完整packet。
- SPSA packet layerdraft：缓存参考和24row proposals/120row centers；执行层尚待实现。
- BM Task/Experiment/HANDOFF继续跟踪实际状态，全部immutable checkpoint保持。

## Verification

ROOT在此前实际工具调用中完成Muse240 7/7 unittest、完整source/12-file本地packet+ZIP/science/240projection、全部remote12body读回核验；这些不是machine全bundle通过，真实CPU closure01随后失败已经明记。ROOT实际720consumer6/6 CPU tests覆盖合成完整720、缺行、损坏NPZ、错误descriptor/anchor；此前selector06 24/24与Muse/W2L/Ditto实际native数组/PCM/PTS核验PASS。agents实际完成W2L90行，Muse02/Ditto240/SPSA当前仍在准备。无三模型完整720或新泛化效果。

本次重新读取bm-checkpoint/bm-writing、Startup Router/实验指令/Coding Session/Task与配置；same-chat coding/codex/legacy分页检索、previous exact身份、freshGit/canonicalGitHub仓库和branch PR[]读回。CLI write-note契约已验证显式overwrite=False/output_format=json，须actioncreated后直接读回精确host身份。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)，当前local SHA不构造commit链接，branch PR[]。
- ADR022 `tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`、ADR023 `tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet`、ADR024 `tts-exp/docs/adr/024-live-process-resource-capacity`。
- `tts-exp/docs/reference/server-access`保存无凭据连接/runtime；actual arrays/QC/failed closure路径上列为恢复证据。

## Observations

- [result] 三模型native工程资格已实际通过；执行agent完成W2L正式blocks0–14各6/6PASS共90/240，未选择或更新新head。
- [verification] ROOT真实720消费者6/6 CPU PASS，先NPZ全文SHA再解码、模型/协议/原源严格核验，完整失败分母保持。
- [blocker] Muse01 remote stage成功但真实CPU closure计数9/布尔True契约不匹配而失败；fresh02正在只修producer形状。Ditto完整packet和CPU测试仍待接齐。
- [decision] ADR022/023/024科学协议与封闭Dev/Confirm保持，六行边界公平交GPU；工程修复用freshattempt，保留失败证据，不重复native资格。
- [next_step] 让Muse02通过真实完整CPU bundle后执行正式六方法块，并接齐Ditto240 packet，与持续W2L组成完整720真实原音评分后按ADR022联合优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T07-07-36Z - Three native models qualified and shared Fit execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]
- relates_to [[024-live-process-resource-capacity]]