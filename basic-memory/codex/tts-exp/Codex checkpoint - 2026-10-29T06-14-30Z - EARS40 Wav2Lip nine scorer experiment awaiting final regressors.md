---
title: Codex checkpoint - 2026-10-29T06-14-30Z - EARS40 Wav2Lip nine scorer experiment
  awaiting final regressors
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t06-14-30-z-ears40-wav2-lip-nine-scorer-experiment-awaiting-final-regressors
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T06:14:30Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a18566-b890-7a41-9d5a-4b9e2ca55c9d
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-29T06-14-30Z - EARS40 Wav2Lip nine scorer experiment awaiting final regressors

## Summary
EARS 40 位英语说话人的自然/TTS 配对 Wav2Lip 视频已生成并交付，九评分器中七项完成独立复算，PEAVS 与 SyncReward 仍在顺序执行。

## Story
用户先要求用 BM 回忆 EARS 音频与既有评分工作，随后明确授权对 40 人自然音频和既有云 Qwen3.1 TTS 沿用 Wav2Lip 生成并运行全部九种评分器，检验当前英语条件下是否存在 TTS 优势。冻结 40 个配对说话人、双臂相同 RMS 与原采样时钟、80 视频、每模型 80 条 ±400ms 固定视频移音控制；没有新 TTS API 调用。完整执行证据位于 runs/ears40_w2l_nine_scorers_20261029_attempt01，既有输入 run 为 ears40_cloud_qwen31_20261028_attempt01。

七项已有结果提示评估器分歧：Sync-C 平均 T−N +0.859（TTS 胜34/40），CoGenAV +0.010686（20/40），AVS_u −0.006360（15/40），Synchformer P0 +0.007594（32/40），SparseSync P0 +0.002007（23/40），MTDVocaLiST K5 −0.007258（14/40），原 VocaLiST K5 −0.002784（19/40）。这些是暂存的未完成九重 Holm 校正结果，不能作为最终九项结论。所有40人同一目标文本、单次TTS、固定头像；不能外推普遍英语优势或因果语言交互。

Synchformer 在最长样本遇作者预处理先裁10秒与冻结中心5秒冲突，并发现四个视频整数帧起点被浮点回转截断。修复为按冻结整数帧/样点预取窗口、作者变换不变，保留旧24条和完整 source/hash/tensor parity 凭据。22条复用记录完全相同，必要4次前向重算。SparseSync 在启动前应用同一修复且与作者路径张量/logits一致。ADR052 已记录历史中文一个样本也有旧起点风险，不回写冻结历史报告。

## Working State
持久入口：run/protocol.json、各模型结果/控制/原始数组、execution_repairs/01_classifier_integer_center.json，以及 parent_analysis（最终全九项分析尚未运行）。视频80条独立副本已交付 results/wav2lip/{natural_raw,tts_raw}/<speaker>/ears40_20261029.mp4；manifest 为 results/wav2lip/ears40_20261029_manifest.json。

机器临时状态：Ubuntu wjj@192.168.50.2，远端仓库 /home/wjj/tts-audio/tts-exp。顺序 driver 最近 PID128731，权威状态是 run/status.json；PEAVS 原生特征已54/80，之后固定控制批63/57及SyncReward自动排队。/root/ears_executor 负责监控、真实前向和最终校验传回；不能并发另启 GPU 阶段。远端 /dev/shm 为暂存，最终交付应不依赖它。外部桌面 GPU 进程和其它训练不动。

工作区已有大量其它任务脏文件；本次全部新增脚本未提交，禁止把继承脏文件当成本次改动。完整只读状态保存 .codex/scratch/ears40_parent_20261029/git_status_checkpoint.txt。当前 HEAD 仅本地核验，不声称已推送。GitHub 查询确认当前分支没有 PR。

## Changed Files
本次新增 scripts/experiments/ears40_w2l_nine_scorers_20261029.py、ears40_cogenav_scorer_20261029.py、ears40_extra_scorers_20261029.py、ears40_retrieval_scorers_20261029.py、ears40_regressor_scorers_20261029.py、ears40_results_20261029.py 及各对应 tests。独立分析器不导入 producer，完整命令：
`python scripts/experiments/ears40_results_20261029.py --run runs/ears40_w2l_nine_scorers_20261029_attempt01 --raw-audit --central`
等待全部九项本地齐全再执行。表格构建器 .codex/scratch/ears40_parent_20261029/render_report_tables.py 尚未运行。

## Verification
Root 已核验全部输入 SHA、归一化误差小于1 PCM16 LSB、80视频哈希/帧与PCM凭据、863项原生交付150411329bytes，以及已完成七模型原始数组/标量和控制复算。修复后 root 亲自运行六个针对测试文件，30 passed（0.54s），凭据 tests_postrepair_root.json；如无代码变化不需重复。Sync-C控制分数下降仅15/80但相对offset80/80恢复，符合best-lag分数性质，不是工程失败。SparseSync 79/80 P0>.9995，均值受p105主导；FP64方向不变。仍未完成九项全量主分析、Holm9、最后两项原始审计及最终manifest。

## References
[Repository](https://github.com/dogLeft233/tts-exp)（read-only connector核验）；本地 HEAD `2b71639c3b555293375d248350999377168d515d`。
run 内 parent_analysis 和 .codex/scratch/ears40_parent_20261029/ 为恢复证据；material_passport.md 与 material_passport_run_ledger.yaml 尚待最终结果封口。

## Observations
- [decision] 全九项和所有40对、控制及失败诊断保留；最终仅完整九主终点一起Holm9，主统计为10000配对bootstrap及100000次双侧sign-flip、seed20261029。
- [decision] CENTRAL125 是保留原完整上下文的查询中心敏感性分析，不称作统一5秒基准；SyncNet官方PCM前端也不等同其他模型。
- [result] 英语当前设置的Sync-C优势清晰，但其他评分器有分歧，最终结论等待两项回归器和多重比较。
- [next_step] 继续监控PEAVS→SyncReward并核验最终传回，随后运行root全九项独立分析、完成最终报告/BM证据地图与Material Passport，再向用户完整交付；不得在七项时结束任务。

## Relations
- relates_to [[052-integer-frame-native-window-preselection]]
