---
title: Codex checkpoint - 2026-09-30T18-24-22Z - Source mechanism continuation and
  physical transfer
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-09-30-t18-24-22-z-source-mechanism-continuation-and-physical-transfer
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 54599e0c543a534a5f87ce17d94f4f64099f2b35
started: '2026-09-30T18:24:22Z'
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

# Codex checkpoint - 2026-09-30T18-24-22Z - Source mechanism continuation and physical transfer

## Summary

新 Qwen 队列的来源方向确认已完成主脸80条，第二张脸正在执行；MFA校准20对全部合格，物理声源/谱形移植和女性外部复核已在首读正式效果前固定研究路线。

## Story

用户最初要从本地代码、Git、BM与百度盘恢复旧电脑未同步的路线，随后明确要求持续自主探究TTS增强TFG的原因，允许重跑、下载数据和使用阿里云key调用Qwen TTS。root维护理论与主线，三个gpt-6-luna/max子代理执行代码、实验和独立复核。目标是找到可移植的生成路径并明确端点与范围，不能以排除清单或仅有计划结束。

历史同PCM的Wav2Lip/Qwen优势已经从Ditto旧codec混杂和缺失的9/27 runs分开。新确认采用官方AISHELL固定revision、cal20/eval80、固定RMS生成/评分四格、校准拟合的512维来源方向及排除实测level secant的干预。所有正式条件和两张脸的产物先封存，再读取效应；独立consumer检验99%主门槛、剂量匹配随机方向、raw/unit几何与时间置乱null，视觉端点另有MFA/MediaPipe。

retry2在第76条遇到TTS仅40个共同SyncNet窗口。保留原始音频、guard20、anchor和2560行计划，retry3把空端点明确记为too_short；它继承75条完整主脸产物和1200行评分，新执行代码单独封存。父run只读，子目录所有JSON/JSONL/log硬链接已解除并验证父文件未变。主脸现80/80完成；短T四格的缺失不阻断拥有完整N支持的来源方向主端点，但T相关80条全分母估计必须标记不完整。

物理线索来自cal-only描述：TTS原始平均响度高7.620dB、浊音占比高0.191、归一化谱包络相邻变化低0.760dB。全轨迹AP下降混合了浊音/静音，不能直接当成浊音噪声更少。新的WORLD双向声源与谱形移植保持recipient音素时钟，分出F0、AP、谱形、局部能量控制，先做CAL waveform QA，再开放EVAL波形和GPU。官方speaker.info显示现cal/eval均为男性，因此另固定20女性speaker×2句的外部复核，复用现d/u/anchor，不重新拟合方向。

## Working State

### Durable repository and protocols

本地HEAD为54599e0c543a534a5f87ce17d94f4f64099f2b35，未push；当前branch无开放PR。此前提交保存producer/Qwen/main spec、独立validator/consumer/visual和两个新spec。主实验协议在openspec/changes/confirm-tts-source-mechanism-20260930/spec.md，物理协议在openspec/changes/confirm-world-source-filter-20261001/spec.md，女性复核在openspec/changes/tts-sex-replication-20261001/spec.md。物理协议经静态审阅，仍待CAL波形控制冻结，不能写成实验结论。

### Machine-local or fragile state

精确SSH入口为root@connect.weste.seetacloud.com:12349，注意weste拼写；已存项目reference/server-access和服务器记忆。密码/API key只由原用户授权和运行内存提供，不入文件/笔记/日志。RTX4080SUPER CUDA可用。研究根/root/autodl-tmp/tts-research-20260930/；Wav2Lip Python在/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python。remote_recovery唯一管理GPU；loopback SSH bridge和token是临时状态，恢复时检查存活或重连，不把token写记忆。

当前正式run为evaluation_20260930_retry3。primary nohup于17:55:50UTC启动，17:58:59UTC退出0，80clips/1280rows。secondary于18:01:20UTC启动，PID318006/worker318048；最近子代理状态50/80clips，总2080/2560rows，尚未汇总效果。全条件封存后优先下protocol/plan/score NPZ/context/visual NPZ/seals，随后MP4，再JPEG/crops；分片不超过8MiB，并逐文件SHA校验。

原protocol SHA256 8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39；2560行计划SHA256 19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab。续接evaluate.py SHA256 e50a31eaa2493b83e87cb4426b742e8f28e73836cbb9374b2319e7b60c940880，recovery_code_seal.json SHA256 55c1512e35bf72563d0e01623f7e14f77aaa776dd94ee565926d9df9bea90417；新执行封存与原protocol分开。continuation_parent.json和mutable-detach manifest保留全部继承证据。

MFA3.4.1实际CPU安装在/root/autodl-tmp/tts-research-20260930/mfa3.4.1-cpu，官方Mandarin模型与词典已下载/逐hash上传，pkuseg资产落地。CAL40 TextGrids齐全，20/20pair qualify，qualification SHA256 556d8577aeba9a36de155397c0201a0ebddfd3c732d6e9d1ae5c8481b0e10832。natural-only CAL包在runs/recovered_20260930/independent_validation/calibration_visual_20260930/mfa_cal/，archive SHA256 e47b387a23fb759e7fd2fbf6e1354139311b297a8be9db484c57956db6b8066a。EVAL160 grids正在CPU对齐，最近12已生成。

冻结旧visual classifier把a˥等IPA声调元音漏算为open vowel：旧规则CAL 0开口元音支持。root在正式效果未读前批准CAL-only visual v2，仅分类时移除U+02E5至U+02E9五个tone bars，保存旧规则与映射表后重冻latency/min-count及控制；physical phone identity仍保留声调与发音区别。旧MFA绑定的远端visual snapshot不可被当前源码覆盖。

本地basic-memory/和runs/均Git忽略，BM持久图在本机，尚无云端备份。Task解释 TTS 的 TFG 与 SyncNet 优势、主实验、物理实验、女性复核均已存在。不要flush含原始凭据prompt的hook inbox。B盘CloudDrive历史缓存仅部分可复核；9/27完整统计仍未恢复，新实验继续推进。

## Changed Files

scripts/experiments/portable_source_axis_20260930/evaluate.py当前dirty，修正短T端点并支持验证继承。local_recovery正在编写未跟踪的source_axis_audio_audit_20260930.py、source_axis_mfa_20260930.py、world_phone_transfer_20261001.py及对应tests。mechanism_audit将修改visual分类并新增CAL-only冻结，需单独版本和commit。临时.codex/scratch/不可连token一起提交。

## Verification

工程官方例子重复的MP4/像素/A/V/矩阵完全一致，+5帧audio控制使固定anchor C从9.502降至0.829，best-lag C仍高；这是工程QC而非科学结论。Qwen100/100对成功，原24k WAV header sentinel通过EOF PCM审计，重封装和canonical16k PCM逐值一致。length-preserved 16→24→16 sham波形与manifest已上传，尚未GPU评分。

full cal20独立validator valid=true，SHA256 f7bc58b72b59a87e3b26c06606b502b2a5bfd05c600c5c49e22d926a6d0c410f；d norm0.3028934609、u norm0.07944404536、cos0.5763336951、global k3。CAL20 MediaPipe视频无解码/时戳/finite失败；已知shift/static/shuffle controls通过。secondary首clip cache/direct-decode head/tail抽检48/80帧逐值一致，含16帧tail batch；artifact faces/secondary_direct_decode_parity.json SHA256 cbb1ef2648a47d05cec43d29cdc790ebebf664c6aee748b8964eb4187c3f5941，不能称全帧检查。

子代理报告consumer/visual相关27 tests通过，WORLD producer当前5/5通过；root未额外运行研究源码测试。GitHub REST只读确认canonical仓库、baseline pushed commit和当前branch PR列表为空。BM同agent/session查询覆盖coding_session/codex_session，legacy查询为空，最新17:00 checkpoint identity已读回核对。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未push HEAD：54599e0c543a534a5f87ce17d94f4f64099f2b35
- runs/recovered_20260930/remote/evaluation_20260930_retry3/
- runs/recovered_20260930/independent_validation/calibration_visual_20260930/mfa_cal/
- openspec/changes/confirm-world-source-filter-20261001/spec.md
- openspec/changes/tts-sex-replication-20261001/spec.md

## Observations

- [result] 新Qwen100对、来源校准20条及主脸80条已完成并核验；双脸正式统计尚未读取。
- [decision] 单个短T empty guard20记too_short，保留固定分母，N来源主门槛支持独立检查；不重抽TTS、不移动guard/anchor。
- [decision] 在正式效果未读前以CAL-only视觉v2修正普通话IPA tone-bar分类，保留旧快照；物理音素映射仍保留声调。
- [decision] 当前男性cohort之后另做20女性speaker×2独立复核，复用冻结d/u/anchor/RMS，不替换或混合现主队列。
- [decision] WORLD双向移植先过own roundtrip和局部能量CAL波形QA，分析条件化AP而非把全轨迹AP低直接解释为低噪声。
- [blocker] 正式双脸尚未全封存，视觉v2 CAL覆盖冻结及WORLD波形QA待完成，EVAL MFA运行中；这些是执行依赖，不能写成科学阴性。
- [next_step] 完成第二张脸并全产物封存，交独立consumer首次读取四格/来源效应和null，同时推进CAL-only视觉v2与WORLD波形QA，随后运行sham、物理移植和女性复核。

## Relations

- continues [[Codex checkpoint - 2026-09-30T17-00-00Z - Sealed fresh Qwen cohort and source mechanism evaluation]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 来源方向与声学机制确认]]
- relates_to [[TTS 机制确认分离生成端与评分端]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[女性说话人上的 TTS 来源方向独立复核]]
- relates_to [[TTS 机制研究服务器 weste 12349]]