---
title: Codex checkpoint - 2026-10-03T20-57-42Z - Verified Muse results and repaired
  shared fit continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t20-57-42-z-verified-muse-results-and-repaired-shared-fit-continuation
repository: dogLeft233/tts-exp
hostname: LAPTOP-E133GSLE
capture: deliberate
model: gpt-6.1-sol
agent: codex
project: tts-exp
username: dogle
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
status: open
git_sha: bdf4654db5128b60d177b97aeb2d9362b3a37fa5
repo_root: E:/Documents/tts-exp
trigger: compact
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
started: '2026-10-03T20:57:42.7743619+00:00'
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T20-57-42Z - Verified Muse results and repaired shared fit continuation

## Summary

MuseTalk240开发完成并独立核验，v3低于RMS；共享360fit的原音交付编码失败已保留并安全停止，修复执行与原生裁剪校准继续推进。

## Story

用户要求恢复TTS增强TFG机制研究并持续形成可用的跨模型自然音增强头h(x)。h(x)只驱动生成，原自然音x用于最终播放和主SyncNet评分。报告25已完成，新目标仍ACTIVE：至少三种前端，目标五模型，未见40确认说话人与heldout模型，内容/独立运动保护，权重、配置、CLI、时延、可复现报告26。ROOT维护理论、协议、BM/Git与独立QC；代码实验由gpt-6-luna/max代理执行。服务器/QwenASR和TTS/百度盘/下载/重跑已有授权；秘密只走运行时。

MuseTalk实际240行PASS，20clips/10speakers/3seeds。ROOT独立重建2400指标和原音PCM/视频比特流保护，等seed→clip→speaker聚合：RMS−identity Cbest+.050919779；v3−RMS Cbest−.012742805/Canchor−.043664726/D0−.002545484/B−.046210210，local−RMS Cbest−.011814022。guard20实际228 eligible/12 ineligible=19clips；local−RMS+.020427879，v3−RMS−.008850940。ROOTreceipt01错误把12ineligible计入guard，保留并用02明确过滤纠正；20clip主结果未变。现有v3不能作为跨模型有效方法交付。

ADR009固定12有界频谱候选+identity/RMS/localcompress，3models×8历史fit×1seed=360，筛≤3短名单后运行已暴露dev20三seed，再进入未见40确认和heldout。fit attempt02实际启动269511，前8行均因自然源PCM16交付mux不能通过PCM_F32LE保护失败。ROOT复核exact PID/startticks/NULargv SHA后只SIGTERM该fit树，3秒survivors空，Ditto99490保留；失败日志/计划不覆写。新core attempt03只写相同canonical natural float32 samples用于F32交付，验证hash/count/FFmpeg解码；主评分继续由原natural_path直接构建PCM16。新360计划保持方法/样本/上下文/模型，仅执行core身份变化。需独立等价复核和实际重启，不能停在prepared。

full512八条RMS全部保存数组56seals已核验。ROOT独立重建lag距离/norm-direction Shapley分组，字段最大误差1.776e−15、float32矩阵误差<5e−6；far(|k|>=4)−zero变化+7.086574620，其中direction+7.041925328/norm+.044649291。这是生成后视觉链分解，不是增强头因果收益。逐帧图proxy/native口型有明显差异。ROI01未执行；ROI02实际失败为错误32frame假设。ROOT下载context.json并完整SHA核验46bab1ff...94450，实际141frames/TCHW[141,3,224,224]/trackstart0；旧context_index=offset也错。fresh03应按q0+offset索引/验证全25fps timestamps，校准实际裁剪后执行八条RMS parity，再训练lag-contrast头。

LatentSync官方源码70members/allGitblob已真实上传且本地独立sealPASS。VAE reuse02实际完整SHA/PASS，live Muse sd-vae可复用；probe01 unexpected .cache保留。source preflight01未执行，fresh02源解压/import与HF离线/实际cuda初始化检测prepared待ROOTreview→dispatch。B盘获取receipt标COMPLETE，但代理只见5,072,222,488B latentsync_unet.pt.partial，正式文件缺失，receipt早于mtime；需查writer、稳定seal和完整SHA再最终化。Whisper未做波形标准化，max-relative floor保留uniform gain log offset，不能误称增益不变。

## Working State

Git本地未推送bdf4654db5128b60d177b97aeb2d9362b3a37fa5，branch codex/cross-tfg-audio-head-20261003，无PR。早期dev/Muse冻结d512876881fd630d4c8fb89920a031bdff72a144不改写。CONTEXT93行dirty、HANDOFF31行ignored；BM Task21/Experiment20 wholewrite/readbackPASS。工作区大量旧dirty/untracked，不stage-all。Ditto retry04实际74/240仍运行，尚无全体结论。

Server root@connect.weste.seetacloud.com:12349，hostpin SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。private PTY50274/bridge2133是脆弱机器状态；.codex/scratch/dispatch_requests_root_attempt02.ps1须先全读请求/source/pins。输出cap8MiB、parts<=7MB，实际JSON必须完整。

storage03 actualPASS_METADATA_ONLY_NO_DELETIONS：SHM free6,717,218,816B，W2L/Muse pyframes为空；full512 outputs440.7MB/officialtmp45.6MB保留视频/矩阵/track；旧v2portable archive904.6MB可能已恢复本地，需fresh seal和引用闭包才退休。Ditto active JPEG251.7MB勿动；public NFS只读。未见确认40仍未读取。

## Changed Files

未提交core scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py SHA d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94，entry run_cross_frontend_fit_profiles_attempt02.py SHA dacda9729db79ec96c73c19402866b78fb3d695b2560ca6f285ce4ce0c056660。ROOT已复核delivery窄diff；sharedfamily六源已提交bdf。ROI03/differentiable_native_crop.py代理新增待ROOTsource review。CONTEXT/HANDOFF更新真实结果，未编造报告26完成。

## Verification

- Muse三文件实际本地下载/完整SHA/PASS；runs/cross_tfg_audio_head_20261003/musetalk_stageb_attempt04_readback_attempt01/results.jsonl 6,378,104B SHAa5556d78...5b55。
- root_muse240_ledger_metric_hierarchy_independent_readback_attempt02.json SHA53463763305c9c74a0f290570c511eecc21418154b4ba3a6d02d755792d80b48：2400metrics/20主clips/19eligible guard。
- root_embedding_decomposition8_lag_groups_independent_readback_attempt01.json SHAb95ea22d1ef66ae733a9e759c37ce8d697ad5021d42096820702ab1475c5fe23：8原生向量/lagShapley真实PASS。
- shared_frequency_fit_attempt06/stop_attempt02/00_stop_fit_tree_request.json SHA1a46c73e29ccce08269d3709bf00d03804610fefd5c55ed4a2480a9e7c00ab3d：实际code0/仅fit tree/无survivors。
- shared_frequency_profiles_fit_attempt03/frozen_plan raw1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7，payload64f7e5693a7f830495a4bc0d8749039bc127a6a7f7c8858092221723df4a9880，rows2105cd4182757bbc73a5339c7fa7ed49d1027dcac4159c4b84483ba5404c795b：prepared未执行。
- .codex/scratch/latentsync_deploy_attempt02/latentsync_source_preflight_attempt02_index.json SHA525b5f637b002d0e1176f94bc4cc6f39fc2e5757f9f604798ceb63b07ae0f338：prepared不是importsPASS。
- 未跑pytest/单元测试。实际科学重构/执行QC如上；没有声称最终泛化或人类感知改善。

## References

[Repository](https://github.com/dogLeft233/tts-exp)。当前bdf未推送，仅保留本地SHA。[已核验线上历史commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。本次直接GitHub读查询确认canonical repo和branch无PR。

## Observations

- [result] Muse240 v3−RMS主Cbest−.013、local−.012；eligible19clip local+.020不是20clip主结果或最终确认。
- [result] full512 far−zero direction+7.042/total+7.087，norm-only修正不足，先校准真实视觉链。
- [decision] F32交付修复保持相同natural samples/hash/count；主评分natural_path/PCM16不变；失败attempt保留并新执行冻结。
- [decision] ADR009保持fit/dev/未见confirm边界，原音换轨及内容/运动保护不弱化。
- [blocker] ROI141frames索引待校准；Latent imports未实际执行且B UNet partial需实际最终化；未见确认和heldout待完成。
- [next_step] 独立比较新360plan/bundle/admission请求后立即启动修复fit，并并行执行ROI03和Latent source-preflight02，继续训练及泛化交付。

## Relations

- continues [[Codex checkpoint - 2026-10-03T20-20-52Z - Complete Muse cohort and calibrated head continuation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]