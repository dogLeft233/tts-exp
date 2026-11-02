---
title: Codex checkpoint - 2026-10-01T09-13-52Z - Local acoustic protocol and verified
  storage recovery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t09-13-52-z-local-acoustic-protocol-and-verified-storage-recovery
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 3566ac45eead923ae7ec1f765cff654652a256c8
started: 2026-10-01 09:13:52.427260+00:00
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

# Codex checkpoint - 2026-10-01T09-13-52Z - Local acoustic protocol and verified storage recovery

## Summary

已保存完整RMS机制证据并修复服务器空间，局部声学协议在读取EVAL效果前冻结；失败WORLD v5将完整可恢复归档后重跑160 contexts。

## Story

用户要求通过本地BM/Git/百度盘恢复并自主完成TTS增强TFG的原因研究，root维护理论与记忆，三个gpt-6-luna/max代理负责代码和实验。最新重点是局部声学特征，音量/能量已有实验。RMS与mel分区均已有独立完整结果，作为已完成基线；当前用固定受体音素时钟、句级RMS的WORLD声源/谱形/局部能量双向移植，区分谱包络、清浊音覆盖、F0/AP及局部动态。最终报告25仍待完整独立消费者、逐窗音素定位与视觉验证。

RMS07权威报告 `runs/recovered_20260930/independent_validation/rms_consumer_attempt07_pass_readback/independent_rms_factorization_report.json` SHA6371c540e15ff9f1a0ba8a0bda21f7836be5d92305d100e20712c02bdaa9afe6：guard20 raw C direction+.161457[+.094298,+.232449]、radius+.019834[-.005550,+.046192]、direction−radius+.141623[+.076968,+.212289]。方向、联合、方向减半径在16组观测及时间null excess均正。已有female独立复制与视频运动证据，不能直接称为人类感知或音素准确率。CAL已有active∩jointvoiced AP差各频带接近0，TTS有声覆盖增加；详细exactphone AP/SP与动态诊断仍未完成。

## Working State

WORLD v5在158/160 contexts时ENOSPC失败，两末context不完整，5692 score NPZ、1590 videos、12008 scored rows；无最终完整五flag/count侧车。已有19595前向日志只记三flag。效应尚未读，不能将158项拼成合格全run。源05/冻结plan保持原样，选择fresh full160 v6。原v5 run和job-control所有失败/部分产物先归档本地E+B两份tar，每成员hash、源前后相同、E完整restore探针通过后才受控释放远端旧bytes。

RMS09 1680文件/1806611203B已经完整迁移至 `/root/autodl-tmp/tts-research-storage-20261001/rms_latent_factorization_20261001_attempt09/gpu_attempt09_artifacts`，原科学路径为精确symlink。迁移原attempt因ENOSPC在symlink阶段失败，fresh resume完整复核源/目的/retired所有成员后完成，retired树已删除；独立全成员读回PASS。receipt raw SHAe04a5854f0739191ac13cbee801bd1ac407bfeaa474228637d9fdd51a472822e，payload944f1c61662abc9a8136f66f945e3ff66ad4d277b14f450c0214f659d66914ba，treeeb52f7e675712f2b752b9978ffda4581d33e0aa6715ece48dc19997358e0cbc0。为释放symlink创建空间临时移出的WORLD forward log已在原路径恢复5891715B及SHA98b3783cea0467cc3958b8fe44a29f81ccba00bd47dcd89017ccb7f775a608e8。当前root约1.81GB、data约873MB；不能重复迁移或删旧缓存，需v5归档释放约4GB。

v6 stage00–03实际PASS，未launch：`.codex/scratch/world_gpu_attempt07_v6_request_index_v3.json`。原runner SHA275a3169c7c7531a01d5942bd06278843faae70724f0e9ae5b7cf8c90ac7a6ac仍在原attempt05绝对路径，以满足input seals。独立wrapper5 SHA708a33d67a1e734f3247ffb3cccd6605a0a596621fac6adff818bf61cd623afa，每次模型forward核五flag并在内存累加计数/hashchain，周期原子侧车，无百万行日志；USR1可受控停止并保留失败终态。worker02 SHA4c7ea04bdcc28fb5bb3af5285835b06d618c6c3124d777e6849e3896aadd2742要求归档gate和live root≥5.5GB，root≤1GB受控停止。run `/root/tts-research-system-20261001/world_phone_transfer_gpu_20261001_attempt07_v6/run`，control位于data分区。consumer04 SHA9feaf39eb6cad01c9dc76d86535fae4ce95379a8ed775a2e70e0bc93286eb056固定新wrapper路径和完整160×76矩阵，独立重算D0=curve[15]、search=D0−Dmin，raw/embedded plan SHA分开，不做模型前向。

局部冻结addendum `openspec/changes/diagnose-world-local-acoustic-response-20261001/spec_addendum_attempt03.md` SHA0faf10ad357b0ce2a655e2eb2612ae2db9282c889cc4fa4025b40bfcf743ad4c；consumer attempt03 SHA8582e6f86d8cae583561899c903668b90d16cb2098e2827b91be70c629161b70待root完整审阅。CAL-only固定16个exact phone labels，16384 phone×phase×support端点与256 unweighted support-G端点同一16640 max-|t| family，同20k speaker draws。所有arm包括EX必须相同mel starts；short S0099_BAC009S0099W0123 guard20留INCOMPLETE，intended分母40，不39救援。G为固定RAW anchor下RT视频距离减移植视频距离，不是global median C可加分解；局部权重取五lipframes实际mel列并集，−85ms仅MediaPipe视觉。剂量保留phone×phase/support及可重建每窗口的WORLD帧/clock/masks索引。没有EVAL选择或effects read。

CAL detail01 raw tableSHA误与payloadSHA比较，实际preflight失败exit75，无科学执行。02修dualraw/payload但root静态发现build preflight引用validate_inputs局部phone_table未返回的NameError，02未dispatch；local_recovery正在fresh03最小修复，不能改01/02。raw1673092554bfc03dbc385136c3a66c08821499c0ec6c25602d385d1d0d7cf16f/payload586d86fedd159ba1c654586bf6892d87b36b1ed8d88c666217d648d230fb2259分别验证，数学不变。

机内易失状态：authenticated rootPTY50274、SSH daemon93533/port3019，token仅运行时；授权入口root@connect.weste.seetacloud.com:12349，密码/API绝不入文件/BM/argv。archive client `.codex/scratch/world_v5_stream_archive_attempt01/stream_world_v5_archive.py` SHA9193ebf8448888b1011c2d9a99e6a5e1aac7d364575ddfd4963b75a4292f7a01已就绪、尚未执行；需isolated Paramiko runtime及已验证ED25519 host-key pin。runtime TTS_RESEARCH_PASSWORD暂在PTY内，hidden child启动后立即清父ENV。AutoDL没有现成网盘mount，本地B: CloudDrive可归档。MediaPipe环境尚待隔离安装及完整v6视频，不修改W2/WORLD环境。

## Changed Files

本地未推HEAD3566ac45eead923ae7ec1f765cff654652a256c8保存17项已合格科学源和CONTEXT；原dirty tests/consumer/validator及untracked失败attempt仍在，不能stage all。新的局部attempt03与addendum、wrapper5/consumer04/CAL02及归档worker处于未提交状态。BM主Task/storage Decision已读回，后者记载实际迁移结果；WORLD主笔记仍需追加v5失败/v6重跑与16640 family修订。CONTEXT≤100行，ignored HANDOFF待更新。

## Verification

root实际核对v6四个stage response与source/worker/runner/plan/QA hash；RMS迁移独立1680成员读回及原WORLD日志精确恢复PASS；完整RMS/MEL/female reports与PNG已SHA读回。CAL02静态发现未定义变量，尚未运行。局部attempt03代理报告py_compile和协议计数PASS，尚无数据运行；未新增或运行pytest。现查git root/branch/HEAD/status、username/hostname及BM配置，加载Startup Router/实验指令/Coding Session；samechat20项无更多、legacy0，直接读T核agent/session。GitHub只读API核canonical仓库、pushed baseline和无PR。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未推HEAD `3566ac45eead923ae7ec1f765cff654652a256c8`。
- `.codex/scratch/rms_storage_migrate_attempt02/migration_receipt_remote.json` 与 `.codex/scratch/emergency_forward_log_restore_attempt01_request.json.response.json`。

## Observations

- [result] 完整RMS基线方向响应已明确，当前转向音素内部谱形及声源/清浊覆盖的物理移植原因。
- [result] RMS09全成员迁移与WORLD原日志恢复实际PASS；v5全部失败证据保留。
- [decision] v5 ENOSPC且缺最终五flag证据，保持失败run归档，fresh v6全160复跑，不放宽cohort或控制。
- [decision] 读取EVAL前局部family明确扩为16640含whole-support G；空guard20保留不完整及计划分母。
- [blocker] v5完整E+B归档/restore与空间gate、CAL fresh03、v6及独立局部/视觉分析仍未完成，报告25不应写成最终结论。
- [next_step] 完成v5双副本全成员归档与恢复探针后释放精确旧树，满足空间gate并launch v6，同时推进CAL fresh03与局部消费者。

## Relations

- continues [[Codex checkpoint - 2026-10-01T08-24-06Z - RMS mechanism confirmed and local acoustic response underway]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[句级音量在编码器中的方向与范数响应]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[研究服务器缓存清理与可恢复归档]]