---
title: Codex checkpoint - 2026-10-09T05-03-54Z - Decoded support verified and V100
  model assets
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t05-03-54-z-decoded-support-verified-and-v100-model-assets
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T05:03:54Z'
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

# Codex checkpoint - 2026-10-09T05-03-54Z - Decoded support verified and V100 model assets

## Summary

ROOT已独立打开四条Wav2Lip工程评分的原始矩阵及七数组，确认尾部补帧只有151个真实支持窗口；ADR023已冻结修正，MuseTalk九资产和共享运行依赖已完成，正进入正式240行Wav2Lip及另外两个模型资格。

## Story

用户从TTS增强TFG原因的恢复研究，推进到可适配多个TFG的自然音频增强头：增强波形驱动视频，成片与评分换回完整原自然PCM和源时钟。ROOT维护理论、协议、BM与调度，实验和代码由GPT-6 Luna/max执行，失败后继续迭代。最新确认局域网V100用户名wjj，公钥可用。

旧step120仅在Wav2Lip训练，因此在W2L有收益、MuseTalk接近零、Ditto下降，尚未超过固定RMS的跨模型门槛。现在按ADR022进行Fit40的三模型真实生成/原音官方SyncNet联合选择与9参数优化。工程四行实际完成并重复一致后，ROOT发现旧allTrue mask把官方裁剪流程额外补的一帧也当作生成帧。已将该问题写成ADR023，保留原矩阵，仅依据真实帧来源机械生成有效窗口，不按效果筛窗口。这避免将评分器尾部填充当作增强效果。

## Working State

任务《开发跨 TFG 的自然音频增强头》active，实验《跨 TFG 自然音频增强头的训练与原音换轨验证》implemented，目标持续执行。新Dev20/10人、Confirm80/40人和LatentSync封闭。ADR021准入不改。

权威设计`tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`：4快照30/60/90/120与identity/rms_fixed，共40Fit/20人×3模型×6方法=720唯一物理行；六方法共同FULL支持、完整分母、说话人等权再模型等权。每模型Cbest对两参考正、D0/Canchor不恶化且等模型对RMS至少+0.050。完整但不合格才可按预定违例排序确定优化初值；缺失/失败不删行。随后20步SPSA与k5/10/15/20完整中心，总2160计划行；canonical Torch CPU负责选择和更新。W2L单独240行不会成为三模型选择或效果结论。

ADR023 `tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet`已创建并直接读回；文件`basic-memory/docs/adr/023-decoded-video-window-support-for-native-syncnet.md`4917B/SHAfb473bb267954f04658d84e42dfd2e433bbac2622e991edeaa285e29af4b3456。所有模型统一要求每个SyncNet五帧窗口都映射真实decoded render、非负/inbounds、源帧连续、精确有理PTS。保留官方原矩阵和所有原query；全六方法交集与原40context分母，不能截掉原自然音频或伪造尾帧救活。旧工程155 decoded/157名义crop、152queries的旧mask有152true，其中query151含补帧，实际151适格。该工程四行不是native准入。

✅实际执行机`wjj@192.168.50.2:22`，现有公钥BatchMode/StrictHostKeyChecking=yes；hostwjj-Default-string、V100-SXM2-16GB、UUIDGPU-d12b3309-c299-da83-064d-a3cd619bd560、driver580.178.04。repo`/home/wjj/tts-audio/tts-exp`，原模型dirty源码/环境保留。W2L隔离runtime`/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python`为Python3.8.20/Torch2.1.2cu121；pip/import/help/CUDA tiny probe实际PASS。GAN/S3FD/SyncNet/portrait实际byte/SHA已核验。工程identity和rms_fixed各两次，BAC009S0052W0495/S0052/j15，100992自然采样、RMS0.0376838172675746，4/4工程PASS；同方法的原矩阵、七数组和视频ES身份逐次相同。remoteoutputs`/home/wjj/tts-audio/fit40_w2l_syncnet_qualification_outputs_attempt02`41,767,131B。ROOT没有本地打开这些视频body；媒体证据是实际producer hash和终局元数据。

ROOT实际独立35终局文件以及4个原始2-array NPZ的full bytes/SHA、自哈希、矩阵shape/dtype/finite及重复QC通过：`.codex/scratch/root_verify_native_smoke_terminal_attempt03/readback_receipt.json`rawSHAcf20aef23421bd681bb92580b5274c616cc69e1da4a2e5d31055b3c544b09f1b，self6442064a821741d0da7252ec8232321b65a7b5ceee442e21a116916fbf80b299。另4个七数组已实际下载并ROOT打开，distance与原2-array逐点相同、raw SHA/dtype/shape/query有理时钟/track/lag/mask全部核验，机械计算151 decoded支持：`.codex/scratch/root_verify_native_seven_arrays_and_muse_assets_attempt01/seven_arrays_readback_receipt.json`selfd5964f44d13a1b66afe3e638e831ec88d3baac058458399fce2c091cb7829351。无efficacy聚合。初次ROOT QC因self-hash convention/status猜测在写结果前失败，后续03按真实schema通过，失败sources保留。

160候选WAV和49输入已实际完成、ROOT此前逐文件byte/PCM/RIFF/clock/RMS及远端读回PASS。160应用产物在`.codex/scratch/fit40_snapshot_audio_application_attempt04/application_output/`，45,261,808B，receiptSHA74369d24f09a38e47f830d724bb2bb896f33d47791a7d435bbf4a0d482c60743，ledgerSHA656b25fff7622be257719a53161421656a24c9f829dadd45c599538b13437874，maxRMSerror1.080632340555443e-8、mask2.7626898288726807dB。快照index`.codex/scratch/fit40_training_snapshots_attempt05/legacy_native_attempt11/snapshot_set_index.json`34139B/SHAf6fe8ea2f447194921ae1b505bc4dd8b7dd40fe57363e2cee1b13980a233e338，真实legacy Fit120，不是未知DirectD0。160波形与来源封成176成员ZIP52,545,709B/SHA914eec35802ff40d4a8ba34f84faac0771d431889656d5641ea7293dcb01892d，F/B双份全成员读回PASS；`F:/tts-exp-archive-transfer/fit40_snapshot_audio160_verified_attempt01/`及`B:/experiment-archive/cross_tfg_audio_head_20261003/fit40_snapshot_audio160_verified_attempt01/`，后端云同步未检查，源保留。remote49输入`/home/wjj/tts-audio/fit40_w2l_syncnet_qualification_attempt01`；本地ZIP`.codex/scratch/fit40_v100_snapshot_stagea_attempt01/fit40_v100_input_bundle_attempt01.zip`可读出完整原自然PCM。

✅MuseTalk实际V1.5，mmpose-free官方Whisper/UNet fp16/face parser/固定portrait Haar框。九资产3,986,182,419B都下载到新run`/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/model_assets/musetalk_v15`，worker全文件SHA及Signal独立远端树/byte/SHA PASS。receipt5743B/SHA85954eb398d73df01bf3f91e3e587d568ce87993117eb7770bcf0c2464ea2c89/selfa9d01ba338e8d5409170c8692a5db6ed090a0e326f479787eb80d0cac646c771；ROOT actual receipt/manifest QC self fedc0440d2a7cd6e8a609626aeb24dccf59f1117fae61a268d952a6007e6624c。manifest8255B/rawSHA53d556e02da47af1b52c26aa8bc53102af75686b5bff7e50fd74a41c6cffd9b6；source revisions精确冻结。source closure正在补official faceparsing/resnet和__init__等真实transitive依赖，旧renderer路径/cu124报告需fresh V100/cu121资格，不把旧report改标签。冻结Whisper feature file12886B/SHA579e4d965290295de49cd49cd1a72357b59083d6cdb5f70cd27d7d5cf3de50bf恰为官方4.39.2源码，无缺失旧私有patch阻断。

✅共享runtime`/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01`已实际安装Python3.10.20/Torch2.5.1cu121/vision0.20.1/audio2.5.1；ASR实际V1001MiB fill/sync/read/delete/release PASS。Dev实际安装35-distribution hashlocked union，35/35版本和artifactSHA一致，pipcheck/import-only全部PASS，无model/GPUload：NumPy1.26.4、ORT CPU1.20.1、mediapipe0.10.21、jax/jaxlib0.4.35、protobuf4.25.5、唯一opencv-contrib4.10.0.84、diffusers0.30.2、transformers4.39.2、imageio-ffmpeg/Cython等。install report533797B/SHAf46f86fb19b7ad2f2bbc5c403731ce510fe9a02cd687c662fea3c3960550eede。venv6,024,421,169B，最后实测rootfree7,778,492,416B。原mfa/autoavsr不修改。

🔄Ditto frozen Strict09 PyTorch12assets2,314,719,638B，而非旧Ampere_Plus TRT。官方HFrev e4a2f60328ee7c32af585ac4b3cce299e4c8e254；source官方新checkout commitc3e47eee2e626500017a0556b470d6d4182f85e8已实际核验。ROOT完整读`.codex/scratch/v100_ditto_asset_prepare_attempt02/stage_ditto_v04_hubert_assets_v100_attempt02.py`12408B/SHAa277641d29f448e56270aa0e28b8e378fb28dcb4ab8d7a52fdb85f34f66f1b83与manifest11290B/SHA66650578642e218f0110c933e22255b8c84f6777a4e67c0fea6d70153de73389/self ff356e9a09682985fe47cd30e1e65a13ec14e0bf566182ebbc874603247e86ec，byte ceiling/create-only promotion/full readback/same st_dev/5GiB动态reserve正确。Dev当前已一次启动direct-final transfer，无GPU/import；最后消息final2files进展，尚无terminalreceipt，不能写完成。target`.../model_assets/ditto_v04_hubert`，jobcontrol`.../job_control/ditto_v04_hubert_asset_stage_attempt01`，workerPID39268/localexec77881是fragile active state。完成后fresh路径/Cython build/wrapper与actual V100资格，科学math/weights不变。

存储：Ditto payload+5GiB reserve只比实测空闲少95,063,658B，不能新增cache/duplicatearchives或把160WAV/240media堆SSD。SHM实测free16,769,675,264B、RAM MemAvailable约29.8GB、session/user cgroup memory.max=max，240工程media约2.5GB适合有界SHM+至少5GiBreserve。720/2160需有界工作块、精确归档并完整byte读回后清理已验证临时媒体，canonicalNPZ/ledger/clock/mediapins/failure分母保留。远端百度盘父mount属clouddevice85，subdir mountpointfalse不等于本地盘。E仅约434MB，大下载F。不要为部署凭空放松reserve。

正式W2L source已增量审查：attempt01曾错观察args.outfile而官方实际writer是temp/result.avi，且expected_frame_count未传播，两项在dispatch前发现，未运行。fresh `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt02/wav2lip_child_adapter_attempt04.py`21077B/SHAcb21bb66c66fe8319dfbd33ed72110fa320d9ecf11f6d822cb939c6c03a61b3a捕获真实writer每帧prewrite bytes/dtype/shape/hash、不改生成frame，绑定actualWAV/PCM/N和native melcount。overlay core06 89487B/SHAe2cb6b39e2e3244f7d771708f9a4385d88223c0af0f7c8b6bd79bc0e16a6e2cb绑定160wavepins、prewrite/decodecount和ADR023 mask。ROOT已读增量，修复通过。ASR正做attempt03的实际single-W2L runtime/capacity/240 plan，不用旧三模型runtimecollector硬编码Py3.10/cu124，不等不存在的历史binding。

ROOT又发现selector04强制source_video_sha256为64hex，但W2L真实静态portrait的source_video=None。禁止用faceSHA冒充videoSHA。已指示ASR新selector05增加显式source_media_kind：原video默认仍需真实videoSHA；static_portrait允许None视频但要求实际faceSHA，plan/result/family binding保留kind，并更新对应tests。selector04和旧test不改；科学门槛和候选identity不改。typed修复尚无实际完成receipt。

共同scientific input identity已明确：所有三模型使用同一logical repo-relative path的source40、applied160、receipt/ledger/index、RMS/identity configs/headθ/roster/portrait/static description及ADR022/023 byte pins；model runtime/source/assets pins另置，不混入shared科学identity。candidateartifact=snapshot.snapshot_sha256，不是waveSHA；referenceartifact=config_sha256和fit40_reference_method_binding_v1(method_id/artifact/inputpins/seed)。240只W2L不能伪称三模型readiness；完整720前不选初值。

理论方向来自实际BM读回的历史局部门控消融：旧头的Cbest主要来自平均频谱整形，门控正确时序额外改善D0约0.010而不额外增加Cbest。因此仍同时约束三模型置信度、固定时钟Canchor与D0，在完整Fit真实端点上更新；不把平均频谱移植普遍有效当事实。旧candidate180/control360的FULL等模型Cbest对RMS+0.011149、D0正向改善−0.033662，仍不合格；GUARD20不足9行保留完整失败分母。

旧AutoDL`root@connect.westd.seetacloud.com:18625`的DirectD0启动后终局未知；唯一launch supervisor573383/start_ticks334637264已保存，status04两请求0response，私有PTY非空read自动审批拒绝。没有重放launch、绕过拒绝、读取processenv/memory。V100提供可继续执行的路径。凭据不在notes/files/argv/log中。

Git `codex/cross-tfg-audio-head-20261003`/`5688784dd5c6c1272074f0c4b6c5ff79f117c907`仍local；fresh canonical GitHub/branch PR[]本次实际查询，HEAD不构造未核验commitURL。五原trackeddirty保留，新实验脚本大量untracked，BM/HANDOFF/runs/scratch是ignored machine-local，远端venv/activePID和F/B挂载也不是git持久状态。ROOT本轮同步Task/Experiment/HANDOFF/serverguide至ADR023及Museassets；shared35依赖完成、Dittotransfer和static-kind修复需接着同步。这些增量不创建进度日志。

## Changed Files

- `basic-memory/docs/adr/023-decoded-video-window-support-for-native-syncnet.md`：typed decision已创建/直接读回，sourcepin冻结。
- `.codex/scratch/root_verify_native_smoke_terminal_attempt03/`和`root_verify_native_seven_arrays_and_muse_assets_attempt01/`：ROOT全数据独立QC，engineeringscope/无efficacy选择。
- `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt02/`：child04/prewrite观察和core06 decoded mask；attempt03与selector05由ASR继续准备。
- `.codex/scratch/v100_musetalk_asset_prepare_attempt01/`：九文件实际下载及独立readback证据；Signal正在修新source/shim。
- `.codex/scratch/v100_ditto_asset_prepare_attempt02/`：已reviewed source/manifest，实际单次transfer在运行，terminalreceipt待到。
- `HANDOFF.md`与BM Task/Experiment/server-access：已保存ADR023/工程QC/Muse部署状态，依赖完成增量待同步。

## Verification

ROOT actual 35file raw/self pins、4原distance NPZ及4七数组全内容/shape/dtype/finite/equality/queryclock/track/lag/native mask151核验PASS，sourceZIP中完整自然PCM16直接复核PASS。ROOT读取Muse实际workerreceipt和manifest crosspins PASS；Signal独立远端9资产全SHA/tree PASS。W2L四行和共享CUDA tinyprobe确有执行且PASS；Dev报告35真实依赖安装/pipcheck/导入PASS，noGPU/noTFGload。ROOT完整读Ditto transfer worker02/manifest/sourcepins/diff无阻断；当前资产尚无terminal结果。

此前selector04的ROOT19/19测试和snapshot工具5/5实际PASS保留，本次只读/资格修复尚未跑selector05测试，也未执行正式240/720新科学分数。Startup Router、实验指令、Coding Session/Task schemas重新读完。same-chat coding/codex/legacy搜索分页去重并直接核对前checkpoint exact agent/session/started，CLI源码确认overwrite=False显式转传及output_format=json；新immutablecheckpoint只接受created并直接读回元数据。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；当前SHA`5688784dd5c6c1272074f0c4b6c5ff79f117c907`为local证据，不构造commitURL，branchPR实际[]。
- `tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`和`tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet`。
- `tts-exp/docs/reference/server-access`记录可恢复局域网用户名/host/runtime/资产定位，凭据不入文件。
- `.codex/scratch/root_verify_native_seven_arrays_and_muse_assets_attempt01/seven_arrays_readback_receipt.json`是ROOT实际七数组QC权威回执。

## Observations

- [result] ROOT actual四条官方矩阵及七数组独立QC PASS，旧mask152true里1个query依赖补帧，实际151完整decoded窗口，工程四行不能代替native准入或头效果。
- [result] MuseV1.5九固定资产实际下载及独立远端SHA/tree核验PASS；共享Torchcu121的真实CUDA tinyprobe及35精简依赖安装/import检查PASS。
- [decision] ADR023统一三模型五帧真实decoded/sourceclock/PTS规则，raw和完整分母保留；共同科学input identity与permodel execution pins分开，静态portrait不能伪造videoSHA。
- [blocker] Ditto单次权重传输尚无terminalreceipt，Muse/Ditto fresh V100 source/path/runtime资格、W2L正式240 dispatch及selector05 static-kind修复仍未完成；SSD reserve外约95MB必须走有界SHM。
- [next_step] 完成真实single-W2L binding/plan和static-portrait consumer修复并在V100执行40Fit×六方法240行，同时接齐Muse/Ditto资格，以完整720原音评分按ADR022初始化和联合优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T04-24-23Z - Native Wav2Lip verified and cross-model Fit execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]