---
title: Codex checkpoint - 2026-10-03T20-20-52Z - Complete Muse cohort and calibrated
  head continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t20-20-52-z-complete-muse-cohort-and-calibrated-head-continuation
username: dogle
branch: codex/cross-tfg-audio-head-20261003
hostname: LAPTOP-E133GSLE
project: tts-exp
repo_root: E:/Documents/tts-exp
agent: codex
repository: dogLeft233/tts-exp
status: open
git_sha: bdf4654db5128b60d177b97aeb2d9362b3a37fa5
cwd: E:/Documents/tts-exp
started: '2026-10-03T20:20:52.2657457+00:00'
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
capture: deliberate
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T20-20-52Z - Complete Muse cohort and calibrated head continuation

## Summary

MuseTalk开发240行已全部完成，ROOT完成全512八条矩阵/embedding与逐帧图的独立读回，Ditto240实际32行通过；共享频谱360fit已预检但尚未启动。

## Story

用户最初要求通过本地BM/代码/Git恢复TTS增强TFG的机制研究，随后明确授权持续形成可用通用自然音增强头h(x)。h(x)只驱动视频，最终播放和主SyncNet评分换回原自然音x。报告25机制已完成，新的交付目标仍ACTIVE：至少三种不同前端有效，目标五模型，未读取40确认说话人与heldout模型，内容与独立运动保护、权重/配置/CLI/时延/可复现报告26。ROOT维护理论、协议、BM/Git与独立科学QC，所有代码/实验交由gpt-6-luna/max代理；服务器、Qwen ASR/TTS、百度盘、下载及重跑授权持续有效，密码/API只走运行时。

Wav2Lip此前360开发显示v3相对RMS改善固定D0−.264和Canchor+.153却Cbest−.105/B−.111；局部压缩Cbest+.033。故继续共享有界频谱family和原生校准学习两条路线，不能以改善D0、一次失败或审计完成替代方法交付。ADR009固定同12候选/三模型/历史8fit一seed→≤3短名单→已暴露dev20三seed→未见40确认及heldout；模型等权兼顾最差模型、Cbest、固定匹配与远滞后区分度，fit/dev不作确认CI。

最关键新证据是同一增强音、同一W2L预测经不同合成/追踪裁剪路径后，SyncNet视觉表示变化很大。ROOT对全512八条compact的56数组逐一验证byte SHA/shape/finite/q0/原音时钟并查看真实图：S0055源帧47/55 proxy嘴明显张开，而native生成/解码/追踪crop较闭合，不能凭图直接宣布纯几何归因。代理报告固定native音嵌入时，native视觉D0下降6.828而far距离上升.259，far−zero差改善7.087主要来自方向7.042、范数.045；此新的代数分解待ROOT独立重算。已要求代理自行封精确96预测→512粘贴→逐帧原生track crop的只读ROI抽取，然后实现可微校准renderer并在历史8fit RMS先建立parity再训练lag-contrast head。保持原时钟，无dev/确认调参。

## Working State

Git codex/cross-tfg-audio-head-20261003，HEAD bdf4654db5128b60d177b97aeb2d9362b3a37fa5 local/unpushed/无PR。六个共享profile/CLI/selector-v3/protocol-v4/builder/runner代码已选择性提交；旧fit/dev/Muse冻结Git d512876881fd630d4c8fb89920a031bdff72a144不变，禁止stage-all。CONTEXT93行dirty、ignoredHANDOFF31行仍Muse222，需更新实际240。BMTask20/Experiment19已whole-write/directread PASS(6563/6573字符)，active/implemented，phase actual_full5128_shared360_preflight_muse222_ditto_retry04_running，需要下一次whole-update真实新阶段。

MuseTalk：TASK NS/stageb_musetalk_attempt04，CTRL NS/job_control/stageb_musetalk_attempt04_attempt02，OUT SHM/stageb_musetalk_attempt04/run。实际19:53:59结束/exit0，240/240 PASS，四方法identity/RMS/local/v3各60，repeatQC1PASS。status11 c8818cca…0109 ROOT读回；结果仍tmpfs，archive insufficient persistent space保留。result_manifest.json1884B d064ff327b23d3d7dfb7be559fd17b62a87c88d7bcc7a19aa04ab2b1d052a4ce、generation_repeat_qc.jsonl2956B8ed516751a1f0504b8260dc0eda11e7ecc7269be81e31161b069a5df7aaa3892、results.jsonl6378104B a5556d7821286669cee4f741de88443b05172584121fa7c2643f5c9b4660ba55待下载完整配对分析。planraw d8414f0c41919b627bbbc24f26b97e989c0a059e82164ebe3f34b2dce34b9116/payloadc968b854…db3f/rows9c720da9…5b33。必须用score.primary_pcm16_window、score.raw_matrix_npz.arrays.distance_by_query_lag及render字段；旧status脚本null是错key，不是文件缺失。11的Ditto绑定错误只影响只读该字段，已新12修正。

Ditto：pkt09 retry04实际19:31:53启动 launcher98735/startticks762771619→runner99490/startticks762772749，TASK NS/stageb_ditto_attempt03_retry04，CTRL NS/job_control/stageb_ditto_attempt03_retry04_attempt01，OUT SHM/stageb_ditto_attempt03_retry04/run。status12 bf45d772…5c1 ROOT真读回32/240全PASS、当前S0022W0122/seed20261005/identity，无退出错误。bundle148457B df33468e…fff、launcher18778B85ab2c45…ad0、planraw170080a0c1608cbf34f8353b1da656f5d9c56de850fdd3f50f4eff4a967e43a0、payload a5c5f4ba…c0bc3、240rowsf2fa4c0b…5635。旧attempt07因Sync无soundfile exit90/0 science保留；新W2L编排解释器有soundfile、Sync评分解释器有PSF且不装soundfile。exact runner argv须从12response获取作为共享GPUgate注册竞争根。

共享360：runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt02/frozen_plan/，raw d0086c40e4a651a2f099a24952f73fbef9b8d6c1b5cd0323fc10e1531ca953e7/payloade0f5e3ea82ae1e171c138bb5c34259d71d1a1262d7bf5f698cca0d298f73fea7/rowsea5efccf0ca36d4ada8cc91ea1a2e98bb0b01b3045871ab5ac8fc91e2f965d36。3models×8fit×15methods/24contexts/seed20261003/dev0confirm0。package166774Bcc51f8a3…1df、24uniquesafeZIPmembers全SHA ROOTPASS；requests02实际mkdir/upload/extractpreflight均PASS，8WAV/15method/3model/360fit和各解释器import PASS。OUT SHM/shared_frequency_profiles_fit_attempt02/run不存在、未launch。老launch要求GPU所有PID为空故不执行；fresh gate包装器05正在封，需注册Ditto99490与自身根/精确argv/seals，≥18432MiB free/stable2、所有GPU PID来自已绑定ancestry、每gen/tracker/scorer重新admission；保留等待行，不修改科学plan/core。包装器04 ROOT全读，代理新05修正-B脚本定位与自身GPU根。ROOT下一步review fresh05/policy/packet后实际launch。

全512：parity03 actual8 COMPLETE/exit0，summary171086B9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb/payload5490b94e…ebb4。ROOT root_full512_summary8_independent_readback_attempt01.json7a73956c…98aa；q0真实32/53/35/64/27/29/29/31，27queries/31lag均originalPCM原时钟。pred最大约1.5e−6，视觉embedding差约2/均值.25–.37；native−proxy说话人等权D0−6.827993/Cbest−.342909/B+.007348。copy-onlycompact8 receipt21433B68aa2942c9a5b9aa3722d8a65993aa3609b010f8eee8f866f9158eaede229432/output8total11037100B。ROOT逐文件/56数组验证receipt root_full512_compact8_arrays_independent_readback_attempt01.json05f4c5bd…fb7b。五时点两行视觉gridPNG2312925B2d88a9a3bb1dd48a0cf076eb09d65fdbfa81ae3d3ba835aa2ce51582cb37bc80/receipt367177f9…8d8 ROOTexact下载SHA/PNG检查+originalview PASS，不需重跑完整32图才实施校准。代理新decomposition04 output ed493aec…d8d/payload5a80806a…10cb尚未ROOT独立核验。

LatentSync heldout工程：本地B盘CloudDrive约1.265TB可用，官方HF资产commit87e9d506205fb2bedacdacaf7e1e1c3074e4ce56 UNet5072222488B正在B:/tts-exp-research-models/latentsync/...下载，最新4897931264B/96.6%，最终全SHA及Whispertiny75572083B未确认。源码bytedance/LatentSync commit a229c3948406bc2cf6eaf4873e662e70c6a04746，inferenceZIP1242248Bfe4149c29b6f07c91dfb7deab07184832beac7b75d4eaf2aa49f6290d2eb72ba，manifest14824B7d1b9fa1…961a/Githubtree14382Baf232f12…9d10。ROOT三文件sizeSHA/70uniquesafeZIP/CRC PASS，最后Gitblob一致性检查用错误metadata['tree']而实际keys为entries，读写科学产物均未发生；修为entries完成70member检查。source-stage5 requests index1767d58a…191与readonly storageaudit3 index63cccb7b…3882全部源码/request ROOTfullread+hashPASS但尚未dispatch。只读storage audit是缓存分类，不能把含rawNPZ/轨迹的syncnet_direct_eval整目录当可删；保留科学证据，复制+SHA+完整restore后才退休源。共享可再生pyframes经reference closure才清理。public-data NFS只读，禁止写。现有Muse VAE可复用path SHM/model_deploy_attempt03/models/musetalk/sd-vae，但需live exact config547B92d3dfb7…e7e/binary334707217B1b4889b6…ddc验证；不新复制335MB。

易失运行：NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003，SHM=/dev/shm/cross_tfg_audio_head_20261003。server root@connect.weste.seetacloud.com:12349/hostpinSHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；密码/API只在privatePTY50274/bridge2133 SecureStrings/env。dispatcher .codex/scratch/dispatch_requests_root_attempt02.ps1只派发reviewed JSON，响应request.json.response.json，outerok/code/timedout/trunc和innerstatus/seals分别核验。bridge8MiBcap/≤7MBchunks。literalW2L /root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python(有soundfile)，literalSync对应syncnet/bin/python(有PSF、无soundfile/requests)，不能resolvevenvsymlink。全PyUTF8；BM脚本必须pwsh7而非PS5。

## Verification

ROOT全512 summary8/compact8共56数组精确sealed读回+图像复核 PASS；共享wave96/plan360 root receiptc410c9af…82b已实际PASS maxRMS2.213451e−9、mask±3dB/帧mean3.14e−15/长度不变/finite/峰值<.376。wave单次串行medianRTF.004625非稳态GPU时延。此前v3 motion60 temporal360 exact root3ce1cd…a95c/openingSD+.001859156609/velocityRMS+.013704958482；motion60几何1440exact/ASR新20CER297chars0/全部40PCM归档pins079732…7a5 PASS，均不代替Cbest效应。未跑pytest/unit tests。配置/Router/实验指令/CodingSession/Task和samechat80/legacy0/newestCB直接读取身份、freshGit/Github canonical/noPR证据按checkpoint脚本核验。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；当前HEAD `bdf4654db5128b60d177b97aeb2d9362b3a37fa5` local/unpushed，无PR；[verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。
- openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md、transfer-iteration.md；.codex/scratch/checkpoint_CC_git_status.txt；上述runs/与scratch/receipt为权威可恢复产物。

## Observations

- [result] Muse240实际全部PASS/退出0，Ditto240实际32PASS，共享360预检PASS但尚未launch；全5128的视觉表示及嘴部外观在postprediction链路出现显著分歧。
- [decision] 保持ADR009冻结共享候选选择，并实现native512→frozen trackedcrop的校准surrogate后再学习；固定原自然音主评分及原时钟。
- [blocker] 三前端共享收益、40确认/heldout、可用方法及报告26仍未完成；Latent空间/依赖工程与模型asset最终SHA仍待完成。
- [next_step] ROOT立即review fresh共享admission05/policy/launch并实际运行360fit，同时下载Muse完整240ledger作配对分析。

## Relations

- continues [[Codex checkpoint - 2026-10-03T19-30-28Z - Actual fit freezes and native cross frontend evidence]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]