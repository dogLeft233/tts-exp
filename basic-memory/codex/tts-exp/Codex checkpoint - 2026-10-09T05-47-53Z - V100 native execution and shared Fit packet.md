---
title: Codex checkpoint - 2026-10-09T05-47-53Z - V100 native execution and shared
  Fit packet
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t05-47-53-z-v100-native-execution-and-shared-fit-packet
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T05:47:53Z'
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

# Codex checkpoint - 2026-10-09T05-47-53Z - V100 native execution and shared Fit packet

## Summary

V100三模型权重与运行依赖已实际就绪，MuseTalk首条自然音频渲染已完成；正式Fit波形包改为共享内存执行，当前在修复评分解释器路径并完成重复资格。

## Story

用户的目标从恢复TTS增强TFG的机制研究，推进到可适配多个TFG的自然音频增强头。增强音频驱动视频，成片与所有评分必须换回完整原自然PCM与源时钟。ROOT负责机制、实验协议与BM；代码和实验由GPT-6 Luna/max执行，失败后继续迭代。最新用户确认局域网V100的SSH用户名wjj；现有公钥可用，实验授权持续有效。

旧step120只经过Wav2Lip训练，跨模型FULL Cbest对RMS均值仅+0.011149，D0正向改善为−0.033662，未达准入。按ADR022，现先在Fit40/20人内生成三模型真实视频和官方原音SyncNet分数，完整选择四个旧快照的联合优化初值，再做9参数20步SPSA。声学机制方向仍是平均频谱整形加局部门控时序，但不同前端响应不一致，须保留Cbest、固定时钟Canchor和D0共同约束。历史局部消融和原始工程矩阵已经复核，不把工程部署成功写成增强收益。

## Working State

持久任务《开发跨 TFG 的自然音频增强头》active，实验《跨 TFG 自然音频增强头的训练与原音换轨验证》implemented，目标未完成。ADR022为权威设计：四快照30/60/90/120加identity/rms_fixed，40Fit×六方法×三模型=720唯一物理行，六方法共同FULL支持，完整分母，说话人等权再模型等权。每模型Cbest对两参考正、D0/Canchor不恶化，等模型对RMS至少+0.050；完整但不合格才允许预定违例排序初始化，缺行不选。20步SPSA及k5/10/15/20完整中心总2160计划行，canonical Torch CPU归约。新Dev20、Confirm80与LatentSync封闭，ADR021不变。

ADR023冻结所有模型真实decoded五帧支持；旧W2L工程4行155真实decoded帧、157名义裁剪帧、152queries，仅151窗口真实支持。ROOT实际打开原distance NPZ、七数组并核验bytes/dtypes/shapes/finite/逐点重复及有理query clock；原矩阵与完整分母保留。权威QC `.codex/scratch/root_verify_native_seven_arrays_and_muse_assets_attempt01/seven_arrays_readback_receipt.json` self d5964f44d13a1b66afe3e638e831ec88d3baac058458399fce2c091cb7829351。旧工程不等于正式native准入。

ADR024 `tts-exp/docs/adr/024-live-process-resource-capacity`已新建并直接读回：每次stage/row检查实际当前进程的RAM/cgroup/SHM/output容量。SSH session/PID/leaf路径只是观察；稳定GPU/model/runtime/mount identity和容量阈值仍绑定。有限cgroup必须满足实际预算，不能因会话路径变化误拒绝或放松reserve。failed UTF8 write attempt01保留，attempt02正确编码后created。

执行机✅ wjj@192.168.50.2:22，hostwjj-Default-string，V100-SXM2-16GB UUID GPU-d12b3309-c299-da83-064d-a3cd619bd560，driver580.178.04，repo `/home/wjj/tts-audio/tts-exp`。公钥BatchMode/StrictHostKeyChecking=yes；密码/API key不入文件/argv/日志。shared runtime `/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01` 为Python3.10.20/Torch2.5.1cu121/NumPy1.26.4/ORTCPU1.20.1/cuDNN90100，实际CUDA tinyprobe、35 hashlocked依赖安装、pipcheck和导入PASS。原环境/原模型dirty source保留。repo根clean不表示模型目录clean。

新run `/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01`：MuseV1.5官方Whisper/UNet fp16/faceparser、固定portrait Haar框、mmpose-free profile，9资产3,986,182,419B全文件下载及worker+Signal远端全SHA PASS。108源/运行文件create-only stage实际PASS，local `.codex/scratch/v100_musetalk_cpu_runtime_prepare_attempt02/remote_readback_attempt01/source_stage_receipt_attempt01.json` raw c636c73018cfd8d0e18a3a85f81a27d5b5b48fa836079ed78e2091e56e7c7d00，self6f8cbc5c50274ee3d984f41738e1fd38398f35493d2fba10c6c0369b84cd181c。actualCPU import-origin报告 `run_attempt10/import_origin_report.json` 11804B/raw45bc2d9e51d55b7ac808cbc2b40f7de168773763bf9541bf0ae01ce6041850dd/self2e926ff9b433d4a32d85ece2bb83849f49679cda1c88a2cd1e4128604342a753 PASS，仅CPU导入，无模型/GPU加载。

Muse首条实际render：row muse_v100_natural1_identity_20261009_01，renderer PID45973/start_ticks908790，Signal报告terminal rc0。remote `/dev/shm/musetalk_v100_native_smoke_attempt01/rows/<row>/process_terminal_receipt.json` 6326B/raw7ac2c25997023bf0e38aea50fbc5cae3dea1d8c4ab2e3d15bdd0a5ea87403a42/self7e42f96299617606234090590b407d628fc2c6d91c06284987de297d98a80bb5。ROOT此时收到真实执行回报，但尚未直接打开媒体/receipt。原F32 remux/官方SyncNet worker在解释器路径guard前失败、0评分：预期venv/bin/python的symlink target不可用，base/bin/python3.8仍可执行。Signal/ASR正在核验所有别名、sys.prefix/package origins和实际新路径，保持隔离overlay而非默换底层环境，保留失败，新worker继续；无评分结果、无完整重复资格。GPU空闲时让Dev的实际渲染与CPU修复并行，严格单GPU进程协调。

Ditto✅ Strict09 PyTorch12资产2,314,719,638B，HFrev e4a2f60328ee7c32af585ac4b3cce299e4c8e254，source commit c3e47eee2e626500017a0556b470d6d4182f85e8，旧Ampere_Plus TRT不使用。actualworker02终局PASS，receipt `.codex/scratch/v100_ditto_asset_prepare_attempt02/readback_attempt01/asset_stage_receipt_attempt01.json` 7737B/raw867dc66c701899ee2163e17030a3254d53f2ae226b4fc23ba4babd9420b42687/self5aa48d80c5dd6041adfff81a05200d2bf972e3fed00c2072ec7d758eb19a7c96。ROOT独立receipt/manifest/worker/source identity和12个asset triples QC `.codex/scratch/root_verify_ditto_asset_receipt_attempt01.json` self98d850bb7b40487b7b61c60f17334c254c33f6be1dcd6e6460c8048dd03c1f7a，ROOT未直接读取2.3GB远端body，worker做全SHA。实际完整StreamSDK CPUimport+隔离Cython build PASS，receipt `.codex/scratch/v100_ditto_cpu_import_probe_attempt01/readback_attempt01/ditto_v100_import_cython_receipt_attempt01.json` raw2b1ba33d34808813f21059b0b743be30f5b55e1c224c1e22408b7c3a3b83df1e/self3c46031cab73234111b8d31c0b3df5f7bea358d57ef755107cbf5a73c2e06f74。非model-forward资格。fresh renderer02 40010B/SHA7b37381777ad8cabf3665a7ab636295bc779b4a1fde5fa15dd485360772a53e4已ROOT关键源码审查，绑定freshruntime/strictbootstrap/actualWAV-F32PCM-N/face/weights/source/25fps frames/PTS；Dev被明确授权等待GPUrelease后直接做同输入两次render/remux/官方评分，不等额外许可。Ditto GPU尚未dispatch。

SSD最新free5,426,327,552B，仅57,618,432B高于5GiB reserve。所有55.3MB正式packet/媒体都走SHM；不得用SSDstage或放松reserve。SHM初始16,769,675,264B，RAM近期30,623,191,040B available，cgroup v2祖先max；每次live重测。ROOT从实际输入ZIP读取40个WAV header：sumN2,828,423、maxN109918、mean4.419411s；240场景按工程单行最大值外推并×2slack+5GiB约10.38GB，小于SHM。此为预算外推，不是硬上界。720/2160用六方法一context有界block，精确归档并全byte读回后清理已验证临时媒体，canonicalNPZ/ledger/clock/media SHA/failure分母保留。F/B已有160波形双份176成员档案全读回PASS；源保留，后端云同步未检查。

W2L正式packet08准备完成、未远端stage/执行：`.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05/`，230files/55,302,272B，runID fit40_w2l240_20261009_attempt05，remoteproject `/dev/shm/fit40_w2l240_stage_attempt05/project`，outputs `/dev/shm/fit40_w2l240_stage_attempt05/blocks`。index56491B/rawa5ccc1124174a3eda0954dcd9ee7def4af69fa4113734875a93cb8938b44afdb/self27568940e12324a0ba2a5276c915a06dc72d46aec5af2199366ee62796d6c8ae；planraw25f49cf6e87800da57fc797210b5d0c4bb6bb3396af04f9f9e9ff1aec0146832；bindingrawa6638095c8ca05ffc6c48fe3d1017e9fd2f3d5da01b3279c01ca97b3e0a67d58，物理与env均control/w2l_runtime_binding.json。共享科学pins hash ae96def4c16fecabc6e49b2aeabea0a76a0308ea159fa6c7688f7908c62c7180不变，permodel runtime/source/assets单独绑定。所有三模型同literal repo-relative科学源、4snapshots、RMS/identity/seed/staticportrait；candidateartifact为snapshot.selfhash，reference为config.selfhash及reference_method_binding，不伪造video SHA。blockcore02 96537B/SHA6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e，保留parent240分母，block-start为6倍数、block-count=6，不把一块写为完整240。ROOT增量/source及08逐文件QC正在完成，旧07QC脚本不能原样运行（payload/index/rootbinding路径不同）。typed static-portrait selector05尚未完成，必须在完整720聚合前修复并跑对应tests；不阻断真实240render。

160candidate waves已全部实际应用并ROOT逐文件byte/PCM/RMS/N核验，authoritative `.codex/scratch/fit40_snapshot_audio_application_attempt04/application_output/` receiptSHA74369d24f09a38e47f830d724bb2bb896f33d47791a7d435bbf4a0d482c60743/ledgerSHA656b25fff7622be257719a53161421656a24c9f829dadd45c599538b13437874。49inputsZIP `.codex/scratch/fit40_v100_snapshot_stagea_attempt01/fit40_v100_input_bundle_attempt01.zip`已ROOT实际读取header与完整自然PCM。无新head成绩、无三模型完成或selection/SPSA更新。

旧AutoDL root@connect.westd.seetacloud.com:18625的唯一DirectD0 launch后终局仍未知，status04两请求0response、私有PTYread自动审批拒绝；不重放launch、不绕过、不读取process env/memory。V100提供可继续执行路线，goal不设blocked。当前Git HEAD 5688784dd5c6c1272074f0c4b6c5ff79f117c907 local未push，五既有trackeddirty保留，新实验代码untracked；BM、HANDOFF、scratch、runs是ignored machine-local证据，远端env/PID/SHM为fragile，canonical GitHub本次实际查询branchPR[]。

## Changed Files

- `basic-memory/docs/adr/024-live-process-resource-capacity.md`：新typed decision created/直接读回，修复临时SSH会话身份与实际容量混用。
- `.codex/scratch/v100_ditto_asset_prepare_attempt02/`、`v100_ditto_cpu_import_probe_attempt01/`：actual权重与完整CPUimport/Cython回执；renderer02已reviewed、未GPU执行。
- `.codex/scratch/v100_musetalk_cpu_runtime_prepare_attempt02/`：108file source stage与actualCPUorigin preflight；SHM Muse首行actualrender rc0，评分路径失败修复在进行。
- `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/`：SHM230file packet、control binding、六行block执行，science identity不改。
- BM Task/Experiment、`HANDOFF.md`、`tts-exp/docs/reference/server-access`已同步权重/source/依赖完成；本checkpoint进一步保存ADR024/packet08/Muse真实render与评分失败。

## Verification

ROOT已实际完成旧4行原矩阵/七数组全byte和decoded151window QC、160波形逐文件PCM/RMS/N QC、Ditto12资产回执/manifest/source/triple crossQC；agent实际执行共享CUDA tinyprobe、35依赖/pipcheck、Muse9资产全SHA+108filestage/CPUorigin、Ditto12资产全SHA/CPUimport+Cython和Muse首个render rc0。区分ROOT实际读取与agent回报。packet08全byte independent QC及正式240尚未执行。selector05 tests未运行；旧selector04 ROOT19/19和snapshot5/5工程tests保留。评分alias失败0score、旧prep失败与其sources保留，科学上无新增效果结论。

本checkpoint先重读Startup Router/实验指令/Coding Session/Task schemas，same-chat coding/codex/legacy搜索分页去重并直接核对前checkpoint agent/session/started；CLI write_note non-overwrite默认经源码确认，output_format=json，要求created后exactmetadata读回。不编辑旧checkpoint。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)，当前commit为local证据，不构造未验证commitURL。
- `tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`、`tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet`与`tts-exp/docs/adr/024-live-process-resource-capacity`。
- `tts-exp/docs/reference/server-access`记录无凭据的V100连接/runtime/模型位置。
- `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/packet_rebase_receipt_attempt01.json`保存230file rebinding及共享science不变的actual准备身份。

## Observations

- [result] Muse首条实际自然音频render PID45973/start_ticks908790 terminal rc0；完整原音评分和重复资格尚未完成，首个评分调用在解释器pathguard前失败且0score。
- [result] Ditto12权重actual下载及ROOT回执crossQC PASS，完整CPUimport/Cython实际PASS；shared35依赖/Muse9资产与108file source stage实际完成。
- [decision] ADR024检查当前进程实际容量，不把临时SSH cgroup路径当稳定身份；保持5GiB reserve和科学比较条件，55MB正式packet及媒体改SHM六方法block。
- [blocker] 评分venv alias目标须修复并重新绑定，Muse/Ditto actual重复资格及W2L240/完整720尚未完成；selector05 staticportrait consumer未完成。
- [next_step] 完成Muse/Ditto真实渲染与原音官方评分重复资格，核验并stage SHM W2L240包后立即执行首个六方法block，持续完成720以启动ADR022联合优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T05-03-54Z - Decoded support verified and V100 model assets]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]
- relates_to [[024-live-process-resource-capacity]]