---
title: Codex checkpoint - 2026-10-08T04-54-35Z - Restored Dev inputs and native adapter
  schema gate repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t04-54-35-z-restored-dev-inputs-and-native-adapter-schema-gate-repair
repository: dogLeft233/tts-exp
agent: codex
status: open
model: gpt-6.1-sol
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
project: tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repo_root: E:/Documents/tts-exp
started: '2026-10-08T04:54:35.0039778+00:00'
git_sha: 6126c7d1aeba5498f1f61042a49afe8f1e9ff59f
trigger: compact
capture: deliberate
branch: codex/cross-tfg-audio-head-20261003
username: dogle
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T04-54-35Z - Restored Dev inputs and native adapter schema gate repair

## Summary
已恢复当前服务器的全部22项冻结Dev输入，下一步是修复MuseTalk/Ditto原资格回执的核验接口并完成CPU绑定后启动Dev对照。

## Story
用户最初要求解释中文TTS为何增强TFG，随后将目标扩展为可用、适配多个TFG的自然音频增强头。增强PCM仅驱动视频生成，最终交付严格换回完整原自然F32音频与源时钟。用户授权实验重跑、下载、Qwen ASR/TTS和网盘，要求持续推进；根代理负责研究主线与独立复核，GPT-6 Luna/max子代理实现和运行实验。

冻结的Fit40/20说话人Adam120三遍候选保持不变。开发验证固定为Dev20/10说话人、三个种子、Wav2Lip/MuseTalk/Ditto，先360个identity/RMS对照，再180个候选。接受标准为三个模型均为正、模型等权超过RMS +0.050，并满足D0、Canchor、内容、视觉、GUARD20和完整失败分母。Confirm80/40未见说话人与LatentSync仍封存。Fit机制诊断支持时变门控改善零延迟距离；Sync-C变化还包含频谱/背景距离贡献，不能据此直接宣称视觉更同步。

原19文件执行链已提交为本地未发布6126c7d。实际恢复过程先独立验证并创建三个缺失公共源码、确认六个已有文件完全相同；CPU recovery02随后发现SHM固定源视频缺失。从本地历史恢复目录找到了原175827字节视频，按原SHA创建回原路径，并实际核验全部22项冻结输入。CPU recovery03继续收集时遇到回执格式缺陷：原MuseTalk/Ditto资格产物用外层整文件SHA，collector06与controls13却要求原文件没有的内部receipt_sha256。四个原回执的字节数、整文件SHA、解析JSON与manifest嵌入对象及schema/status/row/model/seed均已独立核对。

工程修复采取最小新版本链collector07、builder13、controls14和candidate guard09：核验真实原文件与外层SHA，保留原资格manifest和科学门槛；四个小原回执若远端缺失，只按精确字节创建。子代理正在准备fresh recovery04的binding05/plan05/preflight05和后续launcher03。当前没有Dev GPU结果；这是运行接口问题，不能计作科学阴性。旧失败源码和回执均保留，避免覆盖历史。

## Working State
持久研究状态以Basic Memory任务“开发跨 TFG 的自然音频增强头”、实验“跨 TFG 自然音频增强头的训练与原音换轨验证”和ADR017为入口。任务snapshot100/实验99实际写入并读回，记录22项输入已恢复；最新schema失败待下次更新。CONTEXT.md为93行，当前修改未提交；HANDOFF.md为机器本地被忽略文件，较旧恢复段落仍需更新。

当前服务器root@connect.westd.seetacloud.com:18625，仓库/root/tts-research-system-20261001，namespace cross_tfg_audio_head_20261003；GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码与API key仅在根代理私有运行环境，不写入笔记。私有PTY57588、bridge07端口12249 PID10892为易失状态，恢复后先核对。SHM容量用六个真实pilot加显式2倍余量，540行场景需求22,972,800,792字节；这是估计场景而非硬上界，执行仍逐行检查GPU≥18GiB、RAM≥16GiB、SHM/输出空间及cgroup。

`.codex/scratch/`下封包、响应、原文件、SSH状态及本地B盘CloudFS归档均为机器本地状态。B盘放置回执不等于网盘服务端同步确认。不要重建整个SHM、重跑已有资格或改写原manifest。科学冻结、完整原音交付、分开的PCM16评分ROI/source-clock规则和所有失败分母仍有效。

## Changed Files
- 已提交：`openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/dev360_samehost_source_snapshot_attempt01.json`和19文件源图，包含精确保留core06 CRLF的`.gitattributes`规则。
- 当前仅有源码的新版本：`samehost_runtime_binding_collector_attempt07.py`、`build_v3_samehost_dev360_plan_attempt13.py`、`run_v3_samehost_dev360_controls_attempt14.py`、`run_v3_step120_candidate_samehost_attempt09.py`，待根代理完整小diff审查与部署。
- 精确恢复证据：`.codex/scratch/dev360_common_source_restore_attempt02/readback_attempt01/install_summary_attempt02_root_readback.json`和`.codex/scratch/dev360_input_restore_probe_attempt02/readback_attempt01/downloads/input_restore_probe_attempt02.json`。
- 实际失败：`.codex/scratch/dev360_phase2_recovery_attempt03/requests_attempt03/02_run_recovery03_cpu_bind_plan_preflight.json.response.json`。
- launcher02已完成本地失败清理/原子回执发布修复并由根代理完整diff核对；尚未实际运行，必须重定向到修复后的runner14与CPU05产物。

## Verification
公共源码installer02实际PASS：3 CREATE_EXCLUSIVE+6 ALREADY_IDENTICAL；原报告3526字节、raw SHA458f7b2cf3e8f12900d062d08b8be1ab216b616ea0bdcc1459bfe2f0b9a92a4a和canonical self SHA均复核。

inputrestore/probe02实际PASS：22/22唯一输入精确匹配冻结seed，20 WAV共5,465,536字节；视频175827字节原SHA c76617141188acfea5f4c122513b982997ec45ebaa35a2b6ed9f67101e9c40de创建成功。原报告12117字节、raw SHA5f162dc906d614b72037592ae4707a3808b7c88d94f3e1489d21ebf8fccb91de、canonical self SHA及22项映射独立复核PASS。

CPU recovery03 q00/q01实际PASS；q02实际FAIL于缺失receipt_sha256格式要求，尚未产生binding04/plan04/preflight04。根代理直接查看原manifest的四个native对象，独立子代理读四个原文件确认外层SHA/bytes及JSON完全相等。没有运行通用测试、没有Dev GPU启动、没有新科学结果。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，本次直接只读HTTP200核验。
- [已推送历史提交c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)，GitHub直接解析成功。
- 当前本地HEAD `6126c7d1aeba5498f1f61042a49afe8f1e9ff59f`尚未发布；分支`codex/cross-tfg-audio-head-20261003`的直接GitHub PR搜索未找到匹配PR。

## Observations
- [result] 全部22项冻结Dev输入已实际恢复并独立整文件核验；原资格结果与候选权重未改。
- [decision] 保留冻结科学协议；用新的连贯源码版本修复真实资格schema核验，外层原文件SHA不能替换为虚构内部自哈希。
- [blocker] collector06/controls13对MuseTalk和Ditto强制了原producer未定义的receipt_sha256，造成实际CPU收集失败；修复尚未部署运行。
- [next_step] 完整审查并部署最小新源图及四个精确原回执，执行fresh CPU recovery04核验binding05/plan05/preflight05后启动360行Dev对照。

## Relations
- continues [[Codex checkpoint - 2026-10-08T04-25-46Z - Sealed Dev seed and exact missing-source recovery after CPU collection failure]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]