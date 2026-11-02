---
title: Codex checkpoint - 2026-09-30T17-00-00Z - Sealed fresh Qwen cohort and source
  mechanism evaluation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-09-30-t17-00-00-z-sealed-fresh-qwen-cohort-and-source-mechanism-evaluation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: c4a199890bd529928ec34b9be6da86a10a737c97
started: '2026-09-30T17:00:00Z'
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

# Codex checkpoint - 2026-09-30T17-00-00Z - Sealed fresh Qwen cohort and source mechanism evaluation

## Summary

已完成新 AISHELL 100 对 Qwen 自克隆、独立 cal20 来源轴与锚点复核，正式 eval80 四格及来源方向实验已按冻结协议在新服务器运行。

## Story

用户最初要从本地 BM、Git 和代码恢复旧机未同步的研究路线，随后授权持续研究 TTS 增强 TFG 的原因、重跑实验、使用百度盘及网络数据，并明确允许阿里云 Qwen TTS。root 负责理论与实验主线，三个 gpt-6-luna/max subagent 分别负责远程 GPU、输入与声学/MFA、独立验证与视觉统计。目标是给出正向机制干预和明确的适用范围，持续推进物理声学解释，不能仅停在排除项。

历史报告将路线定位到同 PCM 的 Wav2Lip/Qwen 优势、响度生成/评分路径、自然音频替换、音素时钟、来源表示方向及 WORLD 声源/谱包络。旧9/27完整结果没有恢复，A-only WORLD 缓存不能补出视频或评分。为此冻结新的20校准/80评估、50说话人队列，排除旧研究说话人。新数据已全部生成并分片送入服务器；校准只拟合 d/u 和 global k，正式评估仍未做效果汇总。

四格实验分离 RAW/FIXED 生成和评分端；八种表示干预保持自然评分音频与原时钟。独立统计、时间置乱 null、MediaPipe/MFA 视觉测量和 WORLD 物理干预将检查方向特异性、几何解释与可见口型响应。响度校准描述已有具体线索，但不能用表示夹角或校准描述替代效应归因。

## Working State

Git HEAD 仍为已推送的 c4a1998，新分支 codex/tts-mechanism-20260930 无 PR。CONTEXT.md 修改，新增协议、脚本、测试均未提交。basic-memory/ 与 runs/ 被 Git 忽略，是本机状态；远程产物目前在用户服务器，不应视为已有 Git 备份。

服务器入口已存于项目 reference/server-access 和记忆笔记：root@connect.weste.seetacloud.com:12349。密码/API key 只在原用户授权和运行时内存，不入文件/记忆。临时 loopback SSH bridge 尚在运行，进程、token 与本地缓存均脆弱；恢复时优先用持久服务器入口重新连接。远程 GPU 仅由 remote_recovery 管理，勿同时启动另一训练/推理作业。

正式 run：/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/run/evaluation_20260930_retry2。2560预定评分行（两张脸）已在首次效果前封存；当前 primary 先跑，secondary 随后复用音频/潜变量。retry2 已完成至少11/80 primary clips、176/1280行，单clip约25–31秒。attempt1完成首clip视觉产物后因 CUDA np.zeros_like 边界距离 bug 失败，写0分数行；保留原run不改。retry2改为 torch.zeros_like，独立新路径/代码封印，尚无错误或效果选择。

MFA3.4.1 隔离CPU环境/200句语料和 WORLD CPU cal40 分析由 local_recovery 继续。视觉校准只需 cal20 的 baseline 自然视频，不需8臂校准视频；正式eval需8臂共同支持。自然16→24→16 length-preserved 与未裁版本已分片上传，后续稳健性阶段使用。

## Changed Files

- CONTEXT.md：87行入口，标明当前fresh研究、原始自然/英文配对数据和旧响度配置状态。
- openspec/changes/confirm-tts-source-mechanism-20260930/spec.md：冻结 cohort、RMS四格、KKT轴干预/随机剂量匹配、global anchor、40说话人99%CI、缺失规则、cached null、独立视觉及物理机制后续。
- scripts/experiments/portable_source_axis_20260930/：ops、Wav2Lip、scoring、pilot、calibrate、evaluate与独立validator。retry2 evaluate.py SHA256 4a0fce9bcf9f2fff3e798d4ffe519fd32a7670007c2bbb3e2ed0700b44801617。
- scripts/experiments/qwen_tts_pair_prepare_20260930.py、source_axis_audio_audit_20260930.py、source_axis_mfa_20260930.py、tts_mechanism_visual_20260930.py 及对应测试。
- .codex/basic-memory.json 已配置项目local/mixed/coding；.codex/scratch 临时桥接辅助文件不应随配置一起提交。

## Verification

100/100 Qwen成功，0失败。AISHELL revision bbe295d530192a4cd41644b711c9aecd087df653；cal S0052–S0061，eval S0062–S0101，各2句。hf_selected/input_manifest.json 的原始文件 hash 为6b0ce325…，b18f83ad…是不含自hash的canonical fingerprint，两者不能混用。source/frozen manifest 仅路径相对→绝对，其ID/speaker/text/split/ref_sha一致。

20/20 cal完成，10说话人，72.98秒。||d||=.3028934609346225，||u||=.0794440453625454，cos=.5763336950760752；所有说话人median k=3，global k=3，column18，official offset=-3。独立validator重算所有20 NPZ、d/u、PCM/MFCC/掩码与q10 lag，valid=true：runs/recovered_20260930/independent_validation/calibration_artifact_validation.json，SHA256 f7bc58b72b59a87e3b26c06606b502b2a5bfd05c600c5c49e22d926a6d0c410f。

官方例子重复输出MP4、像素和A/V/距离完全一致。已知+5帧延迟：best-lag offset -3→-8，fixed-anchor C 9.502→0.829；best-lag C仍约9.2。两个新中文pilot的PCM16/float身份一致；官方特征/距离maxdiff 3.8147e-6。cal的static-face cache和formal primary head+tail direct decode抽样float/uint8 exact；这些是抽样，不是全帧/双脸等价承诺。

新pilot MediaPipe两条轨迹valid率1、单调25fps、零解码错误；±5帧/static/shuffle QC通过。JSON：runs/recovered_20260930/independent_validation/pilot_mediapipe_qc.json，SHA db64b472889f77bfa153f908cabe4930c13169cce799d0c96162d1ef8d28ab06。尚无音素效应。subagent报告独立validator+visual18项及Qwen/audio9项测试通过；root核对实际JSON与摘要，没有另行运行测试。

原24k Qwen WAV的RIFF/data头均为sentinel，不能用header帧数计时；EOF解码及PCM16重封装保留真实波形。canonical16k有效。audio_integrity_correction_manifest.json SHA6066e653…；100个自然roundtrip中24个多1sample，length-preserved版本只右裁该sample，100长度一致，manifest SHA e3c96ee23b08982c49da73855d2a776e496b4ded1acedf6cd442a70f22cb27c6。

正式retry2 protocol SHA8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39；2560预定row manifest SHA19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab。本地mirror runs/recovered_20260930/remote/evaluation_20260930_retry2/。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- openspec/changes/confirm-tts-source-mechanism-20260930/spec.md
- runs/recovered_20260930/local/qwen_source_axis/hf_selected/tts_outputs/tts_meta.json
- runs/recovered_20260930/local/qwen_source_axis/hf_selected/audio_integrity_20260930/calibration_physical_descriptors.json
- runs/recovered_20260930/remote/calibration_20260930/calibration_summary.json
- runs/recovered_20260930/independent_validation/calibration_artifact_validation.json

## Observations

- [result] cal20 RAW T−N平均响度+7.620dB，19/20更响；固定RMS后静音占比平均低.068、active局部动态范围低.845dB。固定目标.0376838172675746不可重估。此为校准描述，无正式效应归因。
- [decision] FLOAT PCM用float64×32768送入MFCC，不round/int16/clip；正交干预仅排除实测level secant，随机臂必须检查最终float32实际剂量。
- [decision] 80clip/40speaker×2为固定分母；主要支持缺失则eligible=false INCOMPLETE_SUPPORT，同时保留complete-pair估计和每条失败。先封存所有条件产物再汇总效果。
- [decision] cached null使用每clip PCG64(20260930+ordered_eval_index)的100个全A行置乱，跨8臂/双脸同π，固定guard20查询支持；raw/unit及sync_excess均为secondary。
- [decision] 四个99%主门槛满足仅支持定向响应；TTS特异性仍需随机对照，视觉/感知结论分别取证。继续冻结并执行物理声源/谱包络或局部包络干预。
- [blocker] 旧9/27完整统计未恢复；新eval尚在运行，MFA/WORLD依赖与视觉音素支持待完成，不能写成科学阴性。
- [next_step] 保持primary→secondary正式GPU运行，独立统计消费者按已封存schema就绪；并完成cal WORLD/MFA以冻结下一轮物理机制干预。

## Relations

- continues [[Codex checkpoint - 2026-09-30T15-40-00Z - Fresh TTS mechanism experiment and GPU controls]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 来源方向与声学机制确认]]
- relates_to [[TTS 机制确认分离生成端与评分端]]
- relates_to [[TTS 机制研究服务器 weste 12349]]