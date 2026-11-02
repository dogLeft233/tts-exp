---
title: Codex checkpoint - 2026-10-30T05-58-32Z - EchoMimicV3 Flash Chinese20 complete
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t05-58-32-z-echo-mimic-v3-flash-chinese20-complete
status: completed
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T05:58:32Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1880e-73d5-7863-a38b-20bcdf7d5ebc
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-30T05-58-32Z - EchoMimicV3 Flash Chinese20 complete

## Summary
EchoMimicV3-Flash 已在192个人目录完成中文20人、80逻辑视频、九评分器实验，证据全部下载并独立复算通过，尚需向用户交付最终链接。

## Story
用户原先要求部署FlashHead并比较自然语音和TTS驱动TFG；旧实验76已冻结。用户随后选择原生静态图输入的EchoMimicV3-Flash，明确授权复用同一3.png、中文20人的20自然与60既有TTS、RMS−24dBFS归一化和九评分器。已读取更新的server-192 skill及飞书服务器手册，检查公共权重，在个人数据区建立独立环境、核验官方代码和权重后运行。没有重合成TTS、覆盖FlashHead证据、改共享base或终止他人进程。

权威结论见 [[77-echomimic-v3-chinese20-nine-scorers]]，部署见 [[echomimic-v3-flash]]。主分析12同句来源、次分析8扩展来源及全体20来源分别做来源级配对，TTS先取三条件均值。所有分析组九项Holm校正均不显著。全20 Sync-C自然4.945、TTS4.755、差−0.191，95%配对bootstrap CI[−0.600,0.218]；主12差−0.226。不能写成TTS优势、等效性或感知质量结论。PEAVS全20差−0.494原始p=.00769，但Holm=.0692，仍不显著。

## Working State
实验已完成，无需再启动GPU。远端代码 /home/data2/wangjiajun/EchoMimicV3-Flash；实验代码 /home/data2/wangjiajun/av-evaluation/experiments/echomimic_v3_chinese20_20261030；冻结run /home/data2/wangjiajun/av-evaluation/jobs/echomimic_v3_flash_chinese20_rms24_20261030_attempt01。当前run及归档不可覆盖，新实验必须新attempt。run_experiment_v2.sh含固定路径，不得直接重跑。

本地交付根 results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/：report.md、summary.json、paired_scores.csv、per_video_scores.csv、paired_differences.png/pdf、example_natural.mp4、example_tts.mp4、evidence/、local_readback.json及local_statistics_audit.json。80标准AAC视频另在 results/echomimic_v3_flash/{natural_raw,tts_raw}/<source>/。lossless ALAC评分视频及原WAV均在evidence中。

归档287333571字节，2377成员（2376内容文件），SHA256 ca8cd17cb6974016e69e3470c9cd01fb9927bb2d9d0b8eefca2f51ff3f09380b，manifest SHA256 a43c66ab44dc716ec421cbaff3067dbaf2765826ede04d45d6f9a7c92a2f5b05。本地逐成员哈希与80标准导出PASS。

推理512²/25fps/8步/BF16/CPU offload，单A6000 UUID GPU-de7ff5ff-3510-739f-917c-cb98892de759、实际CPU affinity0–3、Torch22GiB上限；有效生成NVIDIA采样峰值15214MiB，41唯一音频生成43.36分钟。不等同于已在RTX3090实测。最后只读检查本次生成/评分/控制/监控进程均结束，未操作他人GPU进程。

前端实际ChineseW2V配置do_normalize=False，实测输入float32不变；绕过上游−23LUFS保持冻结RMS−24。不得沿用旧preview英文W2V True说法，也不宣称内部LayerNorm不存在。帧数4k+1，保留完整音轨，尾部最大.154125秒无新运动，4N+9T尾部仍有声；报告已披露。不是作者默认SOTA基准。

最初T5旧loader兼容失败回退FP32，诊断PID在首个完整有效视频前停止，但可能已进入网络；initial日志、memory及preflight_interruption_audit.json保留。修复为CPU加载后显式BF16并断言四模块dtype、strict keys全匹配，后续有效结果单独保留。不要将初始22.3GiB诊断峰值混成有效14.9GiB。VAE缓存清理、TeaCache与generator重置、UniPC历史重置已审计。

机器本地SSH辅助 C:/Users/dogle/.codex/tmp/server192_run.py 仅运行时DPAPI解密，固定指纹；不复制或打印凭证。仓库1355行既有dirty/untracked，未清理、未提交。本地git状态快照 C:/Users/dogle/.codex/tmp/echomimic_v3_20261030/git_status_final.txt 是临时证据，勿作为共享归档。

## Changed Files
- 新未跟踪 scripts/experiments/echomimic_v3_chinese20_20261030/：准备、生成、九评分器控制、分析、审计、打包、作图、本地交付及验证等12个Python和README。
- 新未跟踪 tests/test_echomimic_v3_chinese20_20261030.py。
- CONTEXT.md新增实验77入口，保持100行。
- Basic Memory新报告77、部署echomimic-v3-flash；更新SYNC-C-SCORES及DEPLOY，均已读回验证。
- 全局server-192 references/personal-deployment.md记录前端实际配置、BF16 loader、完整checkpoint复用、缓存清理、CPU affinity及显存测量边界，未削弱手册要求。

## Verification
全部80视频尺寸、帧数、非静态运动及lossless PCM逐样点一致PASS，首个N/T接触表已查看，仅工程QA。七评分器80/80；Synchformer/SparseSync各33/80逻辑视频、7来源，21唯一输入UNMEASURABLE，无填补；0失败。5个评分环境pip freeze与旧冻结集合完全一致。AVS_u/MTD/VocaLiST/Synchformer/SparseSync真实±400ms控制达标；CoGenAV±160ms峰位移动达标。SyncNet/PEAVS/SyncReward未在本批重测控制，保留历史PEAVS/SR限制。

25个非空统计组用独立本地NumPy重新计算bootstrap、signflip、Holm及来源均值，最大差0。七项原始张量独立归约PASS；PEAVS/SR真实前向/有限性/覆盖验证，不宣称独立模型复现。SyncReward共同4.75秒支持7来源；PEAVS20×4同特征batch敏感性不改变方向/p。服务器3项测试PASS，12脚本编译PASS，BF16 SDPA独立参考PASS。本地普通Python测试2PASS/1缺Torch依赖失败，未伪称全部本地通过。全局skill UTF8 quick_validate PASS。配对图和N/T接触表已实际查看。

GitHub只读get_repo再次确认dogLeft233/tts-exp；当前head PR查询为空。本地branch codex/cross-tfg-audio-head-20261003，SHA 2b71639c3b555293375d248350999377168d515d，未验证远程SHA可解析，故不构造commit链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Official model](https://github.com/antgroup/echomimic_v3)
- [Server manual](https://gcnadw6o1ien.feishu.cn/wiki/Vq5gwxt51i2DVdkDhobcP4iqnkh?fromScene=spaceOverview)
- basic-memory/docs/experiments/77-echomimic-v3-chinese20-nine-scorers.md
- basic-memory/docs/deployment/echomimic-v3-flash.md

## Observations
- [result] EchoMimicV3-Flash中文20人实验、九评分器、80视频本地交付及独立统计复算完成；全部九项Holm不显著。
- [decision] 冻结报告77及attempt01证据，不覆盖旧FlashHead实验；任何新参数或重跑使用新attempt。
- [decision] 保留RMS−24dBFS输入与lossless评分音轨，明确披露上游LUFS差异及4k+1尾部支持限制。
- [next_step] 向用户交付报告77、配对CSV和N/T示例链接；本实验无需继续GPU任务。

## Relations
- continues [[Codex checkpoint - 2026-10-30T03-05-00Z - EchoMimicV3 Flash Chinese20 experiment authorized]]
- documents [[77-echomimic-v3-chinese20-nine-scorers]]
- relates_to [[echomimic-v3-flash]]
