---
title: Codex checkpoint - 2026-10-08T02-35-50Z - Independent native QC and measured
  SHM capacity for Dev integration
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t02-35-50-z-independent-native-qc-and-measured-shm-capacity-for-dev-integration
username: dogle
branch: codex/cross-tfg-audio-head-20261003
trigger: compact
cwd: E:/Documents/tts-exp
status: open
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
started: '2026-10-08T02:35:50.2374324+00:00'
model: gpt-6.1-sol
project: tts-exp
git_sha: 96b47a3a3e530ebfc34510abeb4dc23b8799b132
repository: dogLeft233/tts-exp
agent: codex
hostname: LAPTOP-E133GSLE
repo_root: E:/Documents/tts-exp
capture: deliberate
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T02-35-50Z - Independent native QC and measured SHM capacity for Dev integration

## Summary
Wav2Lip 原生严格重复的 root 独立复核已完成，当前宿主的完整历史输出占用已实测，开发集正在接通可容纳 540 条输出的共享内存路径。

## Story
原始问题是解释 TTS 对中文 TFG 的增强，用户随后要求交付适配多种 TFG 的自然音频增强头，并持续推进失败后的修复与迭代。增强波形驱动视频，最终成片换回完整原始自然 PCM 和源时钟；ADR007/016 已声明的 selected-track PCM16 评分切片带绝对源时钟映射，不能把它们误判为最终音轨裁切。root 维护理论、协议与独立复核，代码和实验由 GPT-6 Luna/max 子代理执行。

当前冻结 Fit40/20 speakers、Adam120/3 passes 候选。Dev20/10 speakers ×3 seeds×Wav2Lip/MuseTalk/Ditto，360 identity/RMS 控制和180候选。验收保持三模型正向、等模型均值相对 RMS +.050，连同 D0/Canchor、内容、视觉、GUARD20 与全失败分母。Confirm80/40 未见说话人及 LatentSync 封存。现阶段修复执行接口，不改候选或评价协议。

迁徙后服务器 connect.westd.seetacloud.com:18625 的 Wav2Lip strict12 两次原生 forward、141帧写入前像素序列、全视频与138×31原始评分矩阵严格重复。root 完整审查 verifier04 后另行执行 independent_readback_attempt05.json，实际 PASS，raw SHA 0b927619e8bc99e5631c79eeb95232c8a241ca289a0d063e950d69276df89bc2。MuseTalk/Ditto 使用各自 strict09 合格来源，strict12 两者 NOT_RUN 保留。资格运行 GPU PID96199 已退出。

20 条 Dev 原始 WAV 逐字节从合格归档恢复，共5465536 B；原生 mel窗口/77–159帧映射经 root 独立复算，六条重复末窗是原生前端行为。严格样本原生141帧、SyncNet处理143帧/138queries 是实际边界行为，保留其名义帧和源时钟映射。GUARD20 支持不足记录预定不适格及全十说话人端点空值，不能丢掉失败/改成九说话人结论。

冻结候选机制证据已有结论：相对 RMS，Cbest +.057816875、D0 -.046865845。Cbest=B−Dmin；B +.101554346、Dmin +.043737471。正确门控相对512ms平移约额外改善D0 .010，Cbest差异很小；D0−Dmin 从 .711810207 降至 .621206892。静态频谱贡献、时序贡献和评分器间隔分别记录，当前候选不因这份分析改动。41文件原始证据与3400 B分析 annex 已实际持久保存，机制笔记现为 concluded 并已纠正待备份措辞。

## Working State
当前主目录 /root/tts-research-system-20261001；实验 namespace cross_tfg_audio_head_20261003；GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。private PTY57588/bridge07仍在，密码和API key只走SecureString及运行环境，不入文件或BM。共享内存产物属易失状态，持久快照另存 durability_snapshots；本地 .codex/scratch 是本机审查工件。

完整六个PASS pilot目录的只读容量探测及原始回执下载已实际完成。回执 .codex/scratch/dev360_resource_footprint_probe_attempt01/downloads/resource_footprint_receipt_attempt01.json，57746 B，raw SHA164bff9e05094428af4de0901e4d38e1139754cbfcfa7e693a80251be5479bf7。overlay余7021514752 B，SHM余26444845056 B，cgroup可用57921232896 B，GPU空闲/32230 MiB可用。全部 retained 文件（含JPEG/AVI）复制180条每模型的逻辑基数4624335000 B，加2×余量及源PCM为13878470536 B；按allocated最大值的同类保守场景约14.97 GB。是规划场景，非未来占用实测或硬上界。dev 正将输出绑定同一新SHM根的 controls360/ 与 candidate180/ 分离叶目录，不广泛清理旧证据。

最新未封存Dev源码审查发现跨adapter旧SHA、Linux相对源路径解析、helper09和collector03残留来源字段。dev 正构建一致的新链；signal独立审查 full JSONL payload、完整原音属性、原生decoded帧数、精确indexed crop及实际generator PCM与预期增强波形相等校验。正式360+180尚未启动，草稿接口不能宣称部署完成。

Task96 active / Experiment95 implemented 已直接读回，阶段 strict12_native_raw_qc_complete_dev540_native_integration_pending。本地 HEAD 96b47a3a3e530ebfc34510abeb4dc23b8799b132 未推送，仅提交CONTEXT.md一段；94行入口、21行忽略HANDOFF已更新。五个无关dirty与大量untracked保留。

## Changed Files
CONTEXT.md 已本地提交；HANDOFF.md 为本机忽略快照。原始strict12复核工件位于 .codex/scratch/root_strict12_original_readback_attempt01/；Dev20映射位于 .codex/scratch/dev20_w2l_native_frame_binding_attempt03_readback_attempt01/；待审新Dev链在 scripts/experiments/cross_tfg_audio_head_20261003/，草稿 pins 在 runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/dev_chain_review_snapshot_attempt01.json。

## Verification
root 独立 strict12 原始回执/像素/全NPZ NumPy复核 PASS；20条源PCM/原生窗口映射 PASS；容量worker/config/全部5请求/Bash/pins审查及实际执行 PASS，原始回执下载和whole SHA PASS。没有执行通用pytest，没有打开Confirm，没有用Dev分数选型。新Dev链尚待一致封存与CPU实际绑定。

## References
[仓库](https://github.com/dogLeft233/tts-exp) 和 [已验证远端commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97) 本次直接只读HTTP200；当前分支PR搜索无匹配。当前HEAD是本地未推送SHA。

## Observations
- [result] root独立确认原生Wav2Lip strict12全输出和raw矩阵严格重复，资格链已齐全。
- [result] 完整pilot树已实测，540条约15GB规划场景可由当前SHM承载，overlay不够。
- [decision] 保持冻结120步候选和360+180分母，保留原生帧边界、ADR016评分切片/时钟和GUARD不足的预定语义；最终成片保留完整原音。
- [blocker] 新Dev链仍需修复交叉源码pin和残留来源路径，然后实际CPU绑定和GPU启动。
- [next_step] 封存一致的新Dev源码与SHM容量绑定，root审查后启动360控制及180候选，推进完整配对评价。

## Relations
- continues [[Codex checkpoint - 2026-10-08T02-04-37Z - Qualified native Wav2Lip repeat and restored Dev source frame map]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]