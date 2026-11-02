---
title: Codex checkpoint - 2026-10-08T02-04-37Z - Qualified native Wav2Lip repeat and
  restored Dev source frame map
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t02-04-37-z-qualified-native-wav2-lip-repeat-and-restored-dev-source-frame-map
capture: deliberate
project: tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
agent: codex
model: gpt-6.1-sol
started: '2026-10-08T02:04:37.0858269+00:00'
hostname: LAPTOP-E133GSLE
git_sha: 42e124806900b8e4e7951a31d513efeed8f0f573
branch: codex/cross-tfg-audio-head-20261003
trigger: compact
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
username: dogle
status: open
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T02-04-37Z - Qualified native Wav2Lip repeat and restored Dev source frame map

## Summary
迁徙后的服务器已完成 Wav2Lip 严格双次重复、Dev20 原始 PCM 恢复和原生帧数核对，冻结增强头的门控机制证据已持久保存；下一步接通三个模型的正式开发集。

## Story
最初要解释 TTS 为何改善中文 TFG，用户随后要求形成适配多个模型的自然音频增强头。增强音频只驱动视频，最终换轨和评分必须使用完整原始自然 PCM 和源时钟。用户授权持续实验、失败修复、下载及 Qwen 服务，代码与实验由 GPT-6 Luna/max 子代理执行，root 维护理论、协议和独立复核。当前冻结 Fit40/20 说话人训练 120 步候选，Dev20/10 说话人 × 3 seeds × Wav2Lip/MuseTalk/Ditto，360 identity/RMS 控制与 180 候选。Confirm80/40 未见说话人和 LatentSync 保持封存。验收维持三模型正向、等模型均值相对 RMS +.050，并检查 D0、Canchor、内容、视觉、GUARD20 和全失败分母。

实际已在新服务器 connect.westd.seetacloud.com:18625、GPU-d09622d0-3535-e662-bb45-3df2b049fcea 执行 Wav2Lip strict12 两次原生推理。修复了 SFD 设置 cudnn.benchmark 的时机及策略契约 SHA 与规范 payload SHA 混用，并在实际 forward 入口记录策略。两次 141 帧写入前像素序列、生成 MP4、原始评分 NPZ 和距离矩阵全部严格一致。MuseTalk/Ditto 依据各自 strict09 原始复核；strict12 的两者明确 NOT_RUN，不拼造一次全模型运行。strict10 像素不同与 strict11 策略回执失败均保留。

Dev20 缺失的 20 个原始 WAV 已从合格历史归档逐字节恢复，合计 5,465,536 B，不经重编码。原生 mel/尾帧循环的 20 条帧数映射经 root 独立复算，范围 77–159，六条存在预期的重复末窗。该映射描述源前端预期，不冒称观察了尚未生成的 Dev 视频。工程代理正在将 Wav2Lip 接为原始 inference.py 入口，维持 Muse/Ditto 的合格入口、每条 PCM/时钟/帧数绑定和独立来源谱系。

冻结候选 Fit40 机制分析已有结论：相对 RMS，Cbest +.057816875、D0 -.046865845。Cbest = B−Dmin 的原始数组分解显示背景距离 B +.101554346，而 Dmin +.043737471；固定时钟 D0 的改善不能与最佳偏移置信度混写。补充 D0−Dmin 从 .711810207 降至 .621206892（Δ -.090603316，27/40 缩小）。512 ms 门控平移保留几乎全部 Cbest，但正确门控另有约 -.010 的 D0 优势。静态频谱项、时序项和评价器间隔分别报告，未来可研究去 DC 门控/角度损失，当前 Dev 不改候选。

## Working State
服务器主目录 /root/tts-research-system-20261001，实验命名空间 cross_tfg_audio_head_20261003。密码与 API key 只保留在 private PTY 57588 的 SecureString/进程环境，不写文件或 BM。该 PTY 已完成 11 个 strict12 原始回执/NPZ 下载并返回提示符。严格资格 GPU 进程 PID 96199 已退出，GPU 空闲。

Fit40 四臂 41 文件证据已存 durability_snapshots/fit40_gate_diagnostic_attempt02_snapshot_attempt01；新增置信度与 anchor-gap 两回执另存 fit40_gate_diagnostic_attempt02_analysis_annex_attempt02，3,400 B 全部原样下载与只读回查 PASS，未改原 41 文件快照。Task95 active / Experiment94 implemented 已 BM 直接读回。机制笔记“冻结增强头的门控时序与频谱贡献” concluded 已含原始 anchor-gap 结果，但其中 annex 待保存措辞需更新为已实际保存。

本地 HEAD 42e124806900b8e4e7951a31d513efeed8f0f573 未推送，CONTEXT.md 本轮有未提交修改；HANDOFF.md 是忽略文件且需跟进 strict12 完成状态。大量 untracked 实验文件和五个无关 dirty 文件保持原状。.codex/scratch 是机器本地工作区，不能视为服务器持久备份。正式 540 条 Dev GPU 运行尚未开始；当前工作为源接口和当前宿主容量核对。

## Changed Files
root 已检查 strict12 wrapper04、runner13、preflight13、launcher12/status12 及下载请求；这些实际消费源码保持不可变。Dev20 恢复/帧映射与科学分析 receipt 在 .codex/scratch 相应 attempt 目录。根文档 CONTEXT.md 的机制段落为未提交更改，HANDOFF.md 为本地忽略快照。

## Verification
11 个 strict12 原始文件全 raw SHA/bytes PASS。signal 代理已执行 verify_strict12_w2l_native_raw_numpy_attempt04.py（50,816 B，cb625d90c4dcf6538b2863b3194384c7f44d69b2cbdf3336d017ad89636487a5），输出 independent_readback_attempt04.json（32,066 B，0b927619e8bc99e5631c79eeb95232c8a241ca289a0d063e950d69276df89bc2）PASS。root 已读完整旧 verifier 和新差异：qualification raw SHA d1b778… 与 canonical ee942… 分开，adapter refs 比 raw whole-file SHA，内部 canonical self-hash仍独立校验；旧索引与失败 verifier 保留。root 独立再执行尚待进行。

root 已复算 20 源帧数/窗口开始位置/尾窗，源回执 72,121 B ca1ec73328a2c22eee4260d3890a29d87d394dae6a3d97bba61482993d302223。root 实际保存并直接回读两科学分析 annex；copy receipt 1a414e3fb33aa8d3f9cad990985a644461bc57ec8fd55e871de33bef2c3341c3。未执行通用 pytest，未打开 Confirm、未使用 Dev 分数选型。

## References
[仓库](https://github.com/dogLeft233/tts-exp) 与 [已验证远端 commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97) 均直接只读 HTTP 200；当前分支 PR 搜索为零。当前 HEAD 为本地未推送 SHA，不构造远端链接。

## Observations
- [result] Wav2Lip strict12 两次原生运行及原始 141 帧像素/NPZ 严格重复性合格；Muse/Ditto 使用各自 strict09 合格来源。
- [result] 20 条 Dev 原始 WAV 已精确恢复，全部原生帧数映射经独立核对；正式 Dev 尚未生成。
- [result] Fit40 门控机制与 Cbest 背景距离分解有原始数组依据，新增两回执已持久备份。
- [decision] 保持冻结 120 步候选、原音完整换轨、360+180 分母、严格原生入口和 Confirm 封存；工程失败不记录为科学阴性。
- [blocker] Dev 仍需完成新原生 Wav2Lip adapter、三后端来源绑定和当前宿主容量预算。
- [next_step] root 独立运行已审查的 strict12 原始回执/NPZ verifier，再批准新 Dev 入口和容量封装并推进正式 360 控制及 180 候选。

## Relations
- continues [[Codex checkpoint - 2026-10-08T01-33-13Z - Durable gate mechanism and identified Wav2Lip startup policy mismatch]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]