---
title: Codex checkpoint - 2026-10-03T14-07-16Z - Frozen head development launch and
  native frontend evidence
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t14-07-16-z-frozen-head-development-launch-and-native-frontend-evidence
started: '2026-10-03T14:07:16.3201581+00:00'
capture: deliberate
repo_root: E:/Documents/tts-exp
git_sha: d512876881fd630d4c8fb89920a031bdff72a144
hostname: LAPTOP-E133GSLE
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
model: gpt-6.1-sol
trigger: compact
status: open
agent: codex
username: dogle
repository: dogLeft233/tts-exp
project: tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T14-07-16Z - Frozen head development launch and native frontend evidence

## Summary

固定零时差新head已冻结并实际启动60次开发生成；原300行特征重放运行中，MuseTalk原音成片已独立核验，完整portable归档已恢复并逐文件复核。

## Story

用户最初要求从BM、Git、本地与百度盘恢复TTS增强TFG路线，随后要求产出适配较多TFG模型的可用自然音频增强头，失败后继续迭代。增强音只驱动生成，成片与主要SyncNet评分恢复原自然音。旧机制报告25已完成；新方法目标ACTIVE，不在启动、依赖通过或单候选失败时停止。ROOT负责理论/协议/BM/Git/运输与独立核验，实验代码由gpt-6-luna/max子代理执行。最新用户授权Qwen ASR按需使用；密码/API只在私有运行时，不入文件或BM。

现有五方法300次Wav2Lip GAN开发结果中，局部压缩相对RMS的equal-speaker FULL Cbest改善.032826487223，并降低D0 .01844037374；原音40ms能量与独立嘴开合相关改善.01367764475。fit24主要提高曲线背景B、D0近零；旧v2的96步相对RMS降低Cbest .01668353081。因此ADR008直接优化原音固定零lag D0与cos0，各.5，并保护视觉范数、波形和mask平滑。新方法96步训练通过独立数学核验，实际开发效果仍待60行与六方法联合360行分析。

## Working State

### Durable scientific evidence

当前本地未推送HEAD d512876881fd630d4c8fb89920a031bdff72a144，分支codex/cross-tfg-audio-head-20261003，7文件选择提交CLI/runtime v3、fullcontext v2、v3 plan builder、CONTEXT、README、transfer-iteration。既有dirty/untracked legacy/tests保留。40个confirm说话人与heldout LatentSync保持未读；dev20片段10新男性说话人各2、三seed；historical fit8片段4人，训练与开发隔离。至少三种不同前端、尽量五模型、heldout/内容/运动/权重CLI/时延/可复现报告26仍待完成。

v3 fit实际归档658731B SHA5ea47f6483242b30796b8ab2d0867d7991e5db54dd873775b64237cd8b6da1a5；ROOT receipt `.codex/scratch/root_v3_fit_terminal_readback_attempt01.json` 是ROOT_FIT_V3_TERMINAL_ARCHIVE_AND_96_STEP_MATH_PASS。全部26成员/96步reward loss/fullcontext支持独立复核，8零初始化严格等于有界RMS，96步梯度/更新finite非零；checkpoint SHA286d085674fe1eb10fb4fff51559bc9923a947327d971a9ab67738a2d1b9cecc。100次五方法Qwen3-ASR与官方文本CER均0，receipt `root_stageb_asr_terminal_and_cer_readback_attempt01.json`；这个内容结论不覆盖新v3。300独立视觉轨迹通过分层重算，开口变化与分数分别解释。

新candidate05五冻结文件实际取回，ROOT receipt `.codex/scratch/root_v3_candidate_frozen_plan_readback_attempt01.json` 是ROOT_ACTUAL_V3_FROZEN_5_FILE_60_ROW_SEMANTIC_PASS。plan raw dbe81fcde4f152d33486fe3eb537d7845199a344f87ce0b2822c39e0a51aa4c4/payload0b06396d677780e0372794cb1e64074f3196c93ecad6bae768e9b264a150213f；60 ledger SHAe12a9fb037d53f8acd605410ab96642d21b40503bd355248d88e953d4cb08f79。spec methodid在method.method_id，不是top-level；manifest无status。policy10新端点（5 contrasts×FULL/GUARD20），joint需重建全部6方法360行28端点，不能复用旧300消费者输出。FULL≥1 query/GUARD20两边trim20且共同≥41，S0022短clip仍保留规划2片段分母、开发不输出CI。

portable archive `.codex/scratch/cross_tfg_audio_head_20261003/portable_archive_attempt02/archive_reassembled_attempt03.tar.gz` 447534356B SHA05260729197c87262b2f97402b196be78776e6cf2c6ef4914dbb6e292f328eb4。ROOT receipt `root_portable_archive_full_restore_readback_attempt01.json` 是ROOT_ALL7532_ARCHIVE_AND_RESTORED_MEMBERS_AND61_CHUNKS_PASS：7531 payload+internalmanifest、全部SHA/size/安全成员/本地实际恢复、61 chunks全核验。内manifest row_status_counts有PASS300/FAILED0/INCOMPLETE0。权威E副本完整；B网盘副本此前hash观察且本次未重hash、读慢。远程源保留。

### Machine-local active runs and native engineering qualification

NS `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`；新v3 OUT `/dev/shm/cross_tfg_audio_head_20261003/spectral_head_fit_v3_dev_attempt05/run`，CTRL `NS/job_control/spectral_head_fit_v3_dev_attempt05_attempt01`。execution_packet_attempt01全部3请求全文/hash审阅并实际成功；13:50:54Z launcher208917 STARTED，初次0rows/output未创建/无错误（启动前hashing）。当前新status request已准备未派发：`.codex/scratch/cross_tfg_audio_head_20261003/spectral_head_fit_v3_candidate_plan_attempt05/execution_packet_attempt02/03_read_candidate_status_attempt02_request.json` SHA2845bb5e6a7e07e52f81c71e9c20949c6935b5926309983b83af1d0f4a66067a。下一次要读取实际进度，不以早期0rows当科学失败。

300原音特征重放packet03全部10请求实际成功；13:45:12Z watcher123551/tick760691505、CUDA child123566/tick760691511 RUNNING，无错误/manifest未写。CODE `NS/syncnet_embedding_replay_attempt04/code`；JOB `/root/autodl-tmp/tts-research-storage-20261001/job_control/cross_tfg_audio_head_20261003/syncnet_embedding_replay_attempt04`；OUT SHM同名/output。observer捕获官方lip/aud原N×1024float32输出再返回原值，31lag raw重放、PCM16/源帧/PTS严格封存，绝不替换producer科学行。replay source SHA1be090c647637b000d41f224403e9c584cd6f910e5e3c66ecbca4bfb3b88f1ec。mechanism_audit跟踪实际终态、另比较8 fit cache 31lag与原生identity的27query居中支持；fit参考平均D0 13.886/Dmin6.403仍明显偏离原生identity约6.95/6.91，当前未定位proxy静态人像/crop或平均支持差异，不据此修改head或重定时。

MuseTalk attempt15真正生成144帧25fps/512x512 silentMP4 SHA2351f4461c2041c01da5429f1cd11f90bc518ad95977d51025a7779e961ba27f；F32原音mux SHA903b2965494cb35cf39c0830ab1313313e934b6f964826a9321ec6ee49e9bf4c，decoded92225samples与原PCM SHA b9a24dd67fd557e012e58339ebe5b1d22655240cf13dd3e31ae7173f3bd0d15a严格相等。native report SHA6ab3cc3248f02411c0caf76fc48179205b6f7ff3d83421d470ff255649dc690f为PASS_ENGINEERING_ONLY；独立只读collector actual READBACK_PASS_ARTIFACTS_ONLY。外层worker仍FAILED：adapter runpy以SystemExit0退出，导致外renderer报告没写，不能倒改FAILED或重渲染现成证据。renderer08仅捕获0/None exit继续验证已ROOT diff审阅；copy-only25成员小archive packet02与commonStageB集成准备中。固定portrait Haar engineeringcrop不冒充官方全视频landmark；此项还不是效应验证。

Ditto实际导入缺Cython；官方env是3.0.11，CPU gcc11.4可用。三个实际adjacentbuild源已只读取回：pyxbld451B SHA0db7d5f7cf2dc67e93b5029b5da3b5d12d890dcda63f2b0d63b748dd909e9234，c1301B SHA0034fadd9742d30a221bbdcddd2984341a17a93111ef5f508b68b50260965f2a，h294B SHA5a62a02e7a5a3aadb3cf1c8683d57d6ae0534434d593d44cb211b353ae23f635。probe04未执行且缺这三pin，保留；local_recovery准备fresh05，exactCython3.0.11wheel1171267B SHA0e25f6425ad4a700d7f77cd468da9161e63658837d1bc34861a9861a4ef6346d。ROOT可执行overlay/cache限250MiB、SHM noexec，不升级base环境；实际全StreamSDK import与nativeportrait待完成。

服务器root@connect.weste.seetacloud.com:12349，ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。私有PTY50274/bridge2133/runtime ENV与SecureStrings易失；密码/API不写入任何归档。ROOT每个request全文与SHA核验后通过`dispatch_requests_root_attempt02.ps1`，实际response顶层ok/code/stdout，不将outer0误作实验PASS。RTX4080SUPER实际32760MiB；GPU/空间有余量。不要resolve venv python symlink，使用literalW2L/SyncNet环境路径。Windows无recursive删除，隐藏后台窗口。

## Verification

上述ROOT归档/数学/实际音视频流/PCM/源pin审阅均已执行；v3新60效果、360消费者、多模型确认尚未完成。没有运行pytest或unit tests。此次重读配置、Router、实验指令与CodingSession/Task schema；71同chat/0legacy搜索与直接BS身份核实、Git/GitHub canonical/noPR/pushedbaseline只读核实；新checkpoint created后直接读回核对。源执行旧HEAD保持真实历史，不逆写当前HEAD。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current local/unpushed HEAD `d512876881fd630d4c8fb89920a031bdff72a144`.
- `openspec/changes/develop-cross-tfg-audio-head-20261003/` contains the shared waveform acceptance and frozen data split.

## Observations

- [result] 新60 head候选冻结ROOT通过并实际启动；300特征重放实际RUNNING，portable完整恢复通过。
- [result] MuseTalk实际144帧silent/originalPCM mux独立PASS；外层FAILED仅报告控制问题，保留两种状态。
- [decision] 原音固定时钟评估、fit/dev/confirm隔离和source byte freeze保持；跨模型native资格、分数效应、独立运动与感知证据分别记录。
- [blocker] 新head开发效果、proxy/native gap定位、Ditto全导入及跨模型效应仍待实际执行；Task13/Experiment12与HANDOFF需更新本次事实。
- [next_step] 派发新v3与replay实际状态，完成60生成和360联合独立核验，同时执行fresh Ditto05导入与Muse共同StageB集成，持续到可用跨模型方法。

## Relations

- continues [[Codex checkpoint - 2026-10-03T13-25-43Z - Verified fixed lag training and original audio content]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[008-fixed-lag-and-rms-reference-head-objective]]