---
title: Codex checkpoint - 2026-10-07T18-02-31Z - Verified Adam120 chain and repairing
  fixed terminal rescore
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t18-02-31-z-verified-adam120-chain-and-repairing-fixed-terminal-rescore
project: tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
hostname: LAPTOP-E133GSLE
capture: deliberate
trigger: compact
cwd: E:/Documents/tts-exp
started: '2026-10-07T18:02:31.6966581+00:00'
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
git_sha: cd6039cdeca1a7ee3b3ca2e6efd2de2731cc4ef9
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
username: dogle
agent: codex
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T18-02-31Z - Verified Adam120 chain and repairing fixed terminal rescore

## Summary

九参数局部谱形增强头真实完成冻结 Fit40/Adam120，root 独立核验全部更新链；终端重评分因 GPU UUID 比较在零行退出，下一轮保留权重与协议重跑。

## Story

用户从恢复未同步的 TTS/TFG 机制研究推进到可用的跨模型自然音频增强头，并要求持续执行。增强音频只驱动视频，最终严格换回原始自然 PCM 与时钟。root 维护理论、冻结协议、Basic Memory、私有 SSH 和独立复核，三名 GPT-6 Luna/max 子代理负责训练、后端实现和声学验证。历史静态共享谱形未达到额外 RMS 增益门槛，当前头以九参数调整稳定、上升和下降局部谱形，固定全局 RMS、采样时钟、平滑及 ±3 dB。局部40ms能量只作诊断，当前头并未逐帧保能量，不能据此排除局部能量机制。

attempt11 严格完整反向资格化通过后实际训练120步。root 下载七份终端原始文件并检查全部120行：三遍同序40样本/20说话人、连续 Adam 更新、九个有限非零梯度、每次更新与下次输入逐位相同，最终权重与 checkpoint 精确相同，无 Dev/Confirm、选择或早停。训练过程中平均 ΔC 从首遍 +0.037 到末遍 +0.055，是训练诊断，尚非固定权重或跨模型验收结果。

root 实际派发 fresh no-grad rescore09，核验27份源包、实际绑定、64项预检和原训练环境。新回读证明进程已退出，0/40行在 native_runtime_load 因 checkpoint/geometry/current scorer GPU UUID 比较失败。评分器源码也已做大小写归一化，训练与几何GPU字符串相同；需只读同环境torch UUID实际表示再决定最小修复的新 attempt10；保存旧失败，不改已训练权重或验收门槛。

producer05 全树持久备份实际完成。root 独立读取 tar.gz 全部4,979项及每项字节/哈希，核验整包263,629,735字节，归档通过；原始解码在首行因 MKV audio stream duration 缺失退出。需要从实际 packet/frame/sample clocks 恢复计时验证，保留 raw-QC 失败并只重跑解码。MuseTalk 七份资产已完整下载核验，另两份因旧 Google Drive 来源失败；agent 准备固定 HF revision 的同哈希替代来源，07尚未派发。Wav2Lip/Ditto十份重复试验源实际安装，但预检遇到17个未展开的 @PROJECT_ROOT@，新 runner04 与 preflight03 正在封包。候选权重 materializer03 已安装，实际运行在缺失全局模板前退出且未创建候选输出；模板与完整CLI闭包待补齐。

## Working State

冻结 Fit40/20说话人、Adam120/三遍/seed20261003；迁移后同 GPU Dev20/10说话人×三seed×Wav2Lip/MuseTalk/Ditto，fresh identity/RMS360 加 candidate180。验收仍是三个模型均正、等模型相对 RMS 额外至少 +0.050，同时通过 D0/Canchor、ASR内容、视觉、GUARD20 与完整失败分母。Confirm80/40未见说话人与 LatentSync 密封；尚无新 Dev 或 Confirm 结果。

持久源码 HEAD `cd6039cdeca1a7ee3b3ca2e6efd2de2731cc4ef9` 本地未推送，branch codex/cross-tfg-audio-head-20261003。CONTEXT93行，Task87/Experiment86记录实际训练完成，原始归档 root 检验和最新 rescore09失败需在下一次状态更新补入。五项无关旧修改未触动，大量未追踪源与 .codex/scratch 为本机状态，HANDOFF/runs/basic-memory为忽略或本地产物。

服务器 root@connect.westd.seetacloud.com:18625，指纹 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0，GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码/API key 仅在私有运行时。持久命名空间 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003，易失树 /dev/shm/cross_tfg_audio_head_20261003。私有PTY57588/bridge07:12249为机器本地易失状态，所有SSH由root唯一派发。最新rescore进程已退出，GPU空闲；GPU作业串行，CPU下载与解码可并行。不要向小空间autodl-tmp写大产物。

## Changed Files

- scripts/audio_enhancement/v3_v4_native_runtime_adapter_v7_attempt01.py 与 run_fit40_native_transition_head_v3_attempt11.py / late_bind_fit40_native_transition_head_v3_attempt11.py：已提交的严格训练实现，原CUDA前向和局部CPU反向。
- .codex/scratch/root_training11_terminal_qc_attempt01.py 与 fit120_final_rescore_packet_attempt09_revision04/root_independent_terminal_readback_attempt01/：已实际下载的七份训练终端与独立120步更新链验证。
- .codex/scratch/fit120_final_rescore_packet_attempt09_revision05/：实际派发的27源包及绑定；launch_revision02保存启动和最新失败状态。
- .codex/scratch/fit40_native_rms_geometry_attempt05_archive_integrity_attempt03/：实际全成员归档独立验证；timing_probe_attempt01为待审阅的一行计时探针。
- .codex/scratch/musetalk_asset_restore_attempt07/：保留07revision01，预检先验证manifest的revision02已准备，尚未派发。
- CONTEXT.md 本地提交cd6039c记录真实训练完成；新repeat runner与候选模板修复仍由agent准备。

## Verification

训练 terminal root receipt `root_independent_terminal_qc_attempt01.json` SHA256 52ff06de9be9381126733204be9237194a3ce25174dceaac142775759c3e499c。completion SHA3535908e79bea9854b1d30c6af214b68735d6db2d20e11d3b2111d07a58f6a1c；checkpoint8d5eb607f77963291372b33245f13f9b9578ebf71657acf1b43243746c538eb8；120步 ledger7f6d054993a96ca8db4808e6bd06b5751abd87a0c778ac7653404d379a1bec8d。严格资格化122次原生前向、72次CPU反向重复逐位通过，zero40与历史相同。未运行通用pytest；执行了用户授权实验与独立原始证据检查。

rescore09实际绑定 d756a9936794d02a885dfae66992589ada29edb976c000da129296500a581758，实际preflight f0a2171e7f4147747a6a7979ea2c8db858db9e240cd5026308a184b63c8eaa1b，launch5fe2ee2c6f16a7d1beccbe7d62cef3daaa7d66eae302f0faf55186689ae8ed8e。PID51723/tick325044356已退出；q11实际response SHA27cd70431a625d67ac401a0ab92917ddff08843aaf91ccb33e304f237367a4fa8，FAILED_FINAL120_FIT_ONLY_RESCORE，0行/0更新/无heldout，failure71fb04a01257c2a4c4f25ff8f8b0cd2c17cd1065cf936345a97fb515b37b28e0。

producer05 tar SHA c35092a0478e0cc9a4738f9918596465d00944ab17a8322171faae08f1d92b1a、manifest905f985220212af8d26ecaa9862f8619da67820a6615dea9f3762047ac68b82d，root独立4,979项/287,367,187逻辑字节全通过。actual verifier03 response21d1e6b024fba60ac54e8a7854610b757c1542eee3d14148857b0f92ac9fe7bad，保持RAW_QC_FAILED，不重归档。Muse06旧失败响应059a5b3cee40c264697573ed390d89fac2e43ae29655292764f2b4b3a355ff61确认七资产完整，两份待恢复。Ditto十二权重已全部核验。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)；当前cd6039c是本地未推送。
- 当前branch GitHub只读PR查询无匹配。
- .codex/scratch/root_rescore09_status02_index_attempt01.json 中唯一请求的response为最新实际评分失败锚点；禁止重复派发已存在response。

## Observations

- [result] 冻结Fit40/Adam120实际完成，root已独立核验120步完整更新链；固定权重与跨模型效果仍待执行。
- [result] producer05归档全部成员和整包独立通过，原始解码因MKV stream duration缺失在首行退出。
- [decision] 保持原始PCM/时钟、冻结Dev验收和sealed Confirm；工程修复必须新运行目录并保留失败证据。
- [decision] 当前谱形头固定全局RMS但局部能量可变，后续机制分析需把局部能量路径纳入验证。
- [blocker] rescore09零行GPU UUID比较失败；MuseTalk余两权重、repeat模板展开、候选全局CLI闭包需修复。
- [next_step] 核验UUID实际值并派发最小修复的fresh终端评分，同时完成后端重复资格化与候选物化，然后实际执行三模型同GPU Dev540。

## Relations

- continues [[Codex checkpoint - 2026-10-07T17-29-10Z - Completed Fit40 Adam120 and preparing transfer evaluation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]