---
title: Codex checkpoint - 2026-10-09T08-59-43Z - Ditto repaired execution and verified
  real Fit ingestion
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t08-59-43-z-ditto-repaired-execution-and-verified-real-fit-ingestion
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T08:59:43Z'
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

# Codex checkpoint - 2026-10-09T08-59-43Z - Ditto repaired execution and verified real Fit ingestion

## Summary

正式Fit已实际取得Wav2Lip120/240与MuseTalk12/240；Ditto05修复包远端stage/CPU和ROOT字节核验完成，真实评分消费器已通过两模型实际首组六方法运输重放。

## Story

用户要恢复TTS增强TFG的局部声学机制，并产出适配多个TFG、超过固定RMS的自然音频增强头。增强PCM驱动视频，成片与SyncNet评分换回完整自然PCM/源时钟。ROOT负责理论、协议、BM及结论，代码和实验继续由GPT-6 Luna/max执行；失败后继续，不以音量排除或某次控制失败结束。最新用户确认192.168.50.2的SSH用户名wjj，现有公钥实际可用。实验、数据下载、网盘和云接口已授权，秘密不入文件。

旧step120仅经W2L训练，独立FULL等模型对RMS ΔCbest+0.011149（W2L+0.068454、Muse+0.000900、Ditto−0.035908），D0正向改善−0.033662，未跨模型合格。按ADR022先取得同40条Fit自然语音/20人×三模型×六方法720唯一评分，再用真实三模型原音分数优化同一组9参数。当前进展是完整Fit执行与真实数据契约闭合，没有新的完整三模型效应、科学初始化或SPSA更新。

## Working State

权威Task《开发跨 TFG 的自然音频增强头》active，Experiment《跨 TFG 自然音频增强头的训练与原音换轨验证》implemented。ADR022/023/024保持：六方法共同FULL、说话人等权再模型等权，每模型Cbest对identity/RMS均正、D0/Canchor不恶化、等模型对RMS≥+0.050。缺行/失败阻断选择，完整但不适格才预定违例函数初始化；canonical TorchCPU1thread，NumPy仅诊断。实际decoded五帧/有理PTS支持、原始矩阵及完整分母保留；每launch实测GPU/RAM/cgroup/SHM/output与5GiB余量。独立新Dev20/Confirm80及LatentSync仍封闭。

✅执行机wjj@192.168.50.2:22，native Windows ssh.exe BatchMode/StrictHostKeyChecking=yes已实际登录。WSL没有该公钥/agent，曾Permission denied，不改用凭据文件。repo /home/wjj/tts-audio/tts-exp，V100-SXM2-16GB UUID GPU-d12b3309-c299-da83-064d-a3cd619bd560；W2L/SyncNet环境 /home/wjj/.venvs/w2l_syncnet_v100_attempt01，Muse/Ditto共享 /home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01。GPU一次一个六行块，ASR调度公平队列，其他agent并行CPU；媒体/包SHM，完整字节归档B/F，cloud上传未验证。旧dirty和原环境保持，秘密不入文件/argv/log、不读进程env/memory。旧AutoDL DirectD0终局未知，拒绝的私有PTY路径不绕过、不重放launch，V100主线持续。

W2L blocks0–19均terminal6/6PASS，共120/240。最新19 PID189787/ticks2034718，/dev/shm/fit40_w2l240_stage_attempt05/blocks/block_019_attempt01；ASR已确认PID退出及显式释放GPU。Muse blocks0–1各6/6PASS，共12/240；0在block_000_attempt02，1在block_001_attempt01，remote /dev/shm/musetalk_fit40_240_attempt02/output。Muse可复用launcher13822B/acd5955fafe1c8708bb0358c11d02c38e28b6936365244658bad5d5e45c58b49已经ROOT全文审查并实际成功，后续六行块无需重复源审。原计数/CPU锚/PathJSON/receipt env失败均保持。

Ditto旧03正式首组六FAILED，renderer前KeyError:'bytes'，无有效科学行；实际core生成audio_ref本来无bytes。freshadapter05仅由actual regular WAV stat推导bytes并核验fullfile/PCM/clock，错误declared size/hash/PCM拒绝；builder05另补actual scorer要求model descriptor.preprocessing_sha256（2b4e0aaa6d43cca007ddcd6fee750baf96a4e8b9861152768ad2f4890f8ffe64），不能误比PCM16 decoder code。04draft状态ADAPTER04标签未消费，05绑定ADAPTER05及freshnamespace；原失败不改写。

当前adapter05 scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt05.py 68998B/ba699de2be6edb4597d5336049798501f7ebf8a4b1118a6337c87f5d09d1c2db；builder05 57135B/aa9af8a8528d9ec8d8c352b5ffde23828bf55ccc8b3695efd0fea9f888712f6f。plan895194B/82ceb8ebd93f541bee9ae3ac9f93f3a3e657ff4979dcb451d03732cd2ad52475、binding22530B/275c8ac529653ab4d066320b65a25dd3c3fb181ca89f09f6928d092013acfc2b、rows660128B/f386441f3ed54da8d69f38610f1cb7c8d79b0320a214ce20d1421fda3c232164（与03逐byte相同）。本地 .codex/scratch/v100_ditto_fit40_stage_payload_attempt05/，actualremote /dev/shm/fit40_ditto240_stage_attempt05/project，六fresh SHM文件/两既有tiny helper与renderer字节相同。八payload1,856,786B全部remote=ZIP=local由ROOT实际核验。

Ditto actualCPU05 receipt .codex/scratch/v100_ditto_fit40_stage_payload_attempt05/remote_stage_readback_attempt01/cpu_closure_receipt_attempt05.json 14608B/raw11073710e84807db7db1f6ed7c9aedea3510bb600a58581d51e8b144f1828a6d/selfa63f8c0e2c1e9ea61e33148d4f039221be0cdbdd817b1cad2179f9b735f521fd。全240/40samples/20sp/200audio/four exactimports/all sample ceilframes/actual parent/runtime/scorepins通过，无Torch/ORT/GPU调用。stage receipt raw5ed38779320fcea24cf0454324b3d9cb8b99215d719305b02b06cb54b6e3cd3c/selfaa9bd81e09068fd5e2a42c87b4a2f4002285b912b331b00c20223ccb9d0a8254；ROOT日志 root_ditto05_remote_stage_cpu_qc_attempt01.log raw2fe362a9d8ed8e4b49f0ec06cce2cee29890477d538c4351cf80a34649cd0b66。

ROOT刚完成exact freshlauncher全文审查 .codex/scratch/v100_ditto_fit40_block_attempt05/launch_ditto_fit40_block_attempt05.py 26097B/0c0ef0c656e91e3602976cdb064bc42cd80e498a67815973aae7195f8ba7b6f1：十三source/rawpins、actualCPU/self/roster、strict runtime、fresh output/job、O_EXCL intent、oncePopen、strictdeterminismenv、PID/ticks/PGID、receipt闭合。Dev已被明确授权先remote verify_cpu_and_staged_inputs()仅CPU，成功则利用释放slot一次启动block0attempt01。尚未收到actual launch/PID，不能写已运行。外壳错误后先查receipt/process，不重复Popen。GPU/live容量由pinned adapter/core在model前检查。

真实720消费器03 scripts/experiments/cross_tfg_audio_head_20261003/fit_only_stagea_ingest_attempt03.py 53184B/22cd7f983b22babcae0d9c120f4720cb84a218ec4d2b84551d5e1668d23aeb95；tests26569B/43cbf5fe6e83d6abf4d9d496e135ad12b26e84dcd804fa34b720ec3557714c52。准确读取source snapshot enum NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN和canonical snapshot_sha256；pin.self_sha256仅metadata alias，不能编造record字段或modelweight。score.preprocessing绑定descriptor，actualcore无toplevelalias，若未来有alias则须相符；PCM conversion独立核code/ffmpeg/operation/codec/channels/rate。wholeNPZ fullSHA先于seven解码，sourcePCM/queryclock/model/anchor15/完整720failure分母保持。

ROOT实际WSL10/10tests PASS0skip（root_stagea_ingest03_cpu_tests_attempt01.log raw57695ba46d1bfb113f52c662a0f8df271b10537075a916d1fbdfed6dc6bd6fbc）。再完整审阅actual Muse helper04与W2L helper02差异、实际调用其run()并输出freshROOTreceipt；两者byteidentical agent原receipt，各6/6 PASS_ROW_TRANSPORT_ONLY，raw logecc283e8a63bb0ea77678a47e6a9ce1f3e783d90b8540e1171360850a4b556a6。真实N62639/source PCM/portrait/query92×31/sevenarrays/decodedFULL支持核验；Muse FULL92、GUARD20 52，W2L FULL91、GUARD20 51。未做family归约/init/选择。

SPSA packet03 composer已ROOT全文分段审阅：actual三240plan共同science/40自然source/6方法比较，排除仅generator差异并保留各完整source_model_protocols/generators_by_model/runtimebinding/sourceparentpins，720metadata可组成而不提前init；pair24eachside/8clips4sp、center120/foursteps、缓存identity/RMS exactrow不重render。仍引用consumer01需fresh04更新，ASR应发送actual三plan05 metadatacomposition回执与当前pins。CORE实际_render_and_score的methodtransform目前无定制D0九theta，不能把inherit identity generator_condition_input当增强输入。ASR负责common theta→fullF32/applicationreceipt/narrow executor，Signal负责Musehooks，Dev负责Dittohooks；保持permodel exact nativefrontend/runtime/protocol，不能core.run仍硬240路径直接塞24/120。第一完整720后20步SPSA新960proposal+四fullcenter480，总2160物理评分；四路共同FULL/no-update/θ及global3dB/RMS/N/clock约束保持。

为避免模型串行任务等agent推理使GPU闲置，ASR正写CPU控制器和thin fixed allowlist runner：exclusive nonce/lock/PIDticks、三个reviewed launcher/plan/binding、六行roundrobin、完整terminal/manifest/rows才推进。单模型控制失败暂停该模型、失败叶保留，其他继续；不自动retry失败叶、不删除/绕过reserve。启用需要Ditto首次实际成功及ROOT fullsource/tests。当前手动公平slot继续。

机制依据既有《冻结增强头的门控时序与频谱贡献》《音素平均谱形与内部轨迹的因果分解》：Cbest主要背景B，正确门控对D0小幅附加收益；512msroll保留Cbest，gateDC混静态谱形，μ/ρ供受者相容。9θ32ms/8ms STFT静态+posflux+negflux×3bands/global3dB/RMS沿三真实前端优化。源代码前端不同只是理论依据，不冒充新增因果实验；不重跑已完成全局音量族。

Git 5688784dd5c6c1272074f0c4b6c5ff79f117c907 local/unpushed、branch codex/cross-tfg-audio-head-20261003，五原tracked dirty保持，新experimentuntracked，BM/HANDOFF/scratch/runs/SHM/env属machine-local或fragile；无新commit/push/PR。可变Task/Experiment/HANDOFF刚同步至W114/Muse12及05待stage，需更新到本checkpoint120/12及05stage/CPU/ROOT完毕；旧immutable08-18checkpoint不改。

## Changed Files

- Ditto adapter05/builder05/create-only stage/CPUworker05/launcher05，科学rows与frontend不变，只修actual producer契约、model preprocessing字段和新attemptnamespace。
- Consumer03及对应tests：读实际未选状态与独立两类preprocessing身份，ROOT10CPUtests和12真实row运输PASS。
- ROOT实际媒体/array/remote bodies核验脚本与freshreceipts；BM Task/Experiment/HANDOFF已有当前工作快照，后续新执行继续更新。
- SPSA commonexecutor/Musehooks/Dittohooks和公平队列正在agent CPU实现，未冻结/未GPU运行。

## Verification

ROOT实际 Muse首组六1463files/43,519,913B/NPZseven/raw92×31/五decoded支持/12严格runtime receipts PASS，六fullF32mux decode等完整原N62639、六PCM16scoringmux decode等完整source转换、六97nativeframe有理PTS PASS。合并旧Ditto八remotebody/CPU04/四snapshot raw+canonical theta seal checks见 .codex/scratch/root_muse_fit40_block0_and_ditto_stage_qc_attempt01/root_actual_readback_qc.json raw064622e0ba32f155607e1ff57d22e957fb651612e9d0a0ca5acdd41f06edb857/selffa34f0eddce26058b0256964b3faf0cafb22876eb523e8acd3d58cb3ed07216f。

Muse block1 agent actual fullreadback2183files/68,120,901B，QC930548B/rawcecbc1a9a250b1e2bc4b1f58b25fa9a9e6380dd935069757a9f095a31cadc7b0/self1f935738fe19e4b81a442e09f7f1904b0e3d1f3488e1fad95557a637ce3ccc3a；ROOT未独立block1底层复核，不延迟公平GPU。

ROOT Ditto05 full04→05source delta/4stdlibtests/actualcanonicalWAV helper250600B/N62639 PASS；8ZIP bodies/science rows unchanged PASS，log root_ditto05_stage_packet_qc_attempt01.log raw0f888822fcca8d4948dcb37bd03dd7866eccd84f4cc12fb318e0d97768ea165c；八远端body和actualstage/CPUraw/self/fieldsPASS见上述freshQC。Consumer03 actual10CPUtests+W2L/Muse12rowhelper独立执行PASS。工程checks不代表新方法收益。

本次已重读bm-checkpoint/bm-writing、Startup Router→实验指令/CodingSession/Task和projectconfig；freshGit/GitHubcanonical/branchPR[]、samechat三类分页搜索及直接前checkpointidentity核验，创建后读回exactsession_id/agent/trigger/model。CLI实现已核验调用write_note显式overwrite=False/output_format=json，只接受created；不覆盖前checkpoint。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；local SHA不构造GitHubcommit链接，该branch实际PR[]。
- ADR022 tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement；ADR023 tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet；ADR024 tts-exp/docs/adr/024-live-process-resource-capacity。
- 实际authority产物/失败attempt/source/CPU和独立QC均按WorkingState相对路径定位；无凭据server-access tts-exp/docs/reference/server-access。

## Observations

- [result] W2L正式120/240和Muse12/240均有效评分完成；Ditto05八payload actualstage/CPU与ROOT全文核验PASS，正式firstblock尚待actual launch回执，旧03六FAILED不计有效行。
- [verification] ROOT实际Muse首组全PCM/media/PTS/arrays、Ditto05全stage/CPU、consumer03十CPUtests及两模型十二真实行独立运输重放PASS。
- [decision] 按实际producer记录分别核model.preprocessing和PCMdecoder代码，snapshot canonical字段为snapshot_sha256；工程窄修freshattempt、失败保留，ADR022/023/024和sealedDev/Confirm不变。
- [blocker] 三模型720尚未完成，SPSA common变换/应用证据/窄执行hooks和固定公平runner仍需实现及审查；没有actualfamily初始化或优化更新。
- [next_step] 收到Ditto05首组六actual launch及terminal后接齐公平三模型Fit，并完成commonSPSA执行层；完整720真实消费闭合后按ADR022联合优化并进入独立Dev/Confirm。

## Relations

- continues [[Codex checkpoint - 2026-10-09T08-18-14Z - Muse first formal Fit and Ditto execution ready]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]
- relates_to [[024-live-process-resource-capacity]]