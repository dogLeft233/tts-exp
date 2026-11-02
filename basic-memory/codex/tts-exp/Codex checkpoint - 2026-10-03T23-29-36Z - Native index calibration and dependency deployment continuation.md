---
title: Codex checkpoint - 2026-10-03T23-29-36Z - Native index calibration and dependency
  deployment continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t23-29-36-z-native-index-calibration-and-dependency-deployment-continuation
started: '2026-10-03T23:29:36.1570199+00:00'
project: tts-exp
capture: deliberate
model: gpt-6.1-sol
branch: codex/cross-tfg-audio-head-20261003
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
agent: codex
username: dogle
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
git_sha: 48b0f787b1c6c2f2b2b693d56120b919bb8147ba
trigger: compact
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T23-29-36Z - Native index calibration and dependency deployment continuation

## Summary
完成八样本原音视觉链路的全字节回收与独立标量复核，定位稳定两帧特征索引差异，继续运行共享谱头筛选并部署两个确认架构。

## Story
原任务是解释 TTS 对中文 TFG 的同步增强机制，报告25已结论化；用户随后要求形成可用于多种 TFG 的自然音频增强头 h(x)，以增强音生成视频，播放及主 SyncNet 均恢复原自然音 x。目标仍 active，失败候选后继续改进，交付配置/权重/CLI/复现及未见40说话人与 heldout 架构确认。root 负责理论、冻结协议、BM、Git、运输和独立复核；实验代码及运行交给用户指定的 gpt-6-luna/max 子agent。近期 Alibaba Qwen ASR 授权已收到，既有100+20条转写 CER=0，尚无必要重跑。

现有 v3 尚未超过 RMS 的主 Cbest：完整 W2L 均值 RMS6.945、v3 6.839；Muse240行 RMS5.095、v3 5.082。保留两条迭代路线：12种局部共享频谱 profile 的360行 fit 筛选，以及正式视觉前向/lag contrast 校准后训练 v4。最后已核验 shared149/360、Ditto208/240均 PASS/0FAIL，尚未完成，不根据部分行选择或发表确认结论。新状态请求须使用实际 shared attempt03，不能误读已失败的 attempt02。

CPU8校准用同一原自然音 embedding、每行27 query×31 lag，8条 control 全部通过。summary SHA d1b0bb1af1855a5971add87a749871d5c7388e2a1b6ab85321730c48e28a964a、manifest SHA d2863654e75c60d91707dec9fb8776f5b2c08daa27c868ea1e13191aaf117669 已实际本地逐字节恢复。ROOT独立重算 JSON 曲线/标量/时钟/control PASS，最大标量误差7.55e-7（门槛2e-6）；hard precodec−official 的 Cbest +0.3921595、D0 +6.8709631、Canchor−6.8767928 是视觉链路校准，不是增强增益。102166827B archive SHA5a7ab7e41f3ec6f8392c9eaeb80ec3650d5c29d0ebc33ea454e6da37323938e0全部16分段已回收（旧007客户端timeout仍保留；新007–015全部实际PASS），混合目录安全恢复全部25成员/8NPZ后 bundle receipt a51f09d3e60068e5e83a3ec85900a68b691ad6e7d2a9b9a0f257f3cc5aa0f63e PASS。

原生 official lag=[0,0,0,1,1,1,1,0]，hard lag=[2,3,3,3,3,3,3,3]，差值均值+2.375，较早手算1.875已明确撤回。8条 selected crop AVI 均比 decoded full512多2帧。新全8数组诊断像素总体偏好相同帧，而 visual embedding 一致偏好+2 query（hard vs official均角9.944°，同帧71.539°）；官方 JPEG 重嵌入在相同帧完全重合。此事实定位特征/取帧索引问题，尚不能归因纯 codec。须实际审计 AVI→MKV→run_pipeline pyframes/crop 的 PTS 和 frame map，并检查五帧张量顺序；不对主评估做事后平移。依据实际 native forward 修正训练代理，然后继续训练与跨模型确认。

LatentSync六核心包隔离安装后，Kornia0.8.0/rs0.1.8现已实际pip exit0装入新的 kornia_overlay_attempt05，官方 pipeline/inference 仍缺 InsightFace。完整9403B报告 SHA812e7487c105da0a105b2b99073112da9259091bf5e168f2ad9ad78031e134ab实际读取恢复；新16模块CPU导入探查实际完成12PASS/4FAIL，9546B report SHA5a6728630905b866607e2a48b72dd0f8a11f370b2da8b39d341b160b83b5f09e待全文读回。独立 torch CUDA probe false不等同逐模块 CUDA gate。官方5.072GB UNet在B盘保留，内存流式官方SHA→weights_only→FP16派生权重路线仍准备中。官方入口实际FP32 load后 toFP16，须逐参数核验派生权重与该路径一致；尚未模型推理。

第五架构 SadTalker117文件官方源码已逐字节匹配 pinned tag/commit。官方 safetensors0.6.2 ABI3轮子实际安全提取14文件并 CPU import PASS（CUDA before/after false），再次源导入进入 skimage gap。其 imageio/pydub/kornia/facexlib 和 face_enhancer 的顶层 GFPGAN imports需要真实闭包；禁止增强选项仍有顶层依赖，不能依据可选标签忽略。4个官方推理资产总1.184GB实际 downloader启动 PID154309，待状态与完成全SHA封存。现有 overlay import probe 第一次实际upload失败 OSError: Failure，response保留，未执行probe；需新SHM namespace/空间修复，不宣称部署通过。

空间清理仅准备，尚未删除。三个精确可退役文件为已本地全成员恢复的447MB v2 archive、236MB embedding archive，以及与当前 MuseCLI无关且官方 pinned LFS可恢复的1.488GB训练 SyncNet 权重。ROOT完整审阅退休 worker，指出全局free_shm相等门槛会被并行实验/下载触发，要求新 attempt 移除该无关门槛，保留目标SHA/inode/mtime/size/noFD、冻结实际运行 source/asset组及单文件 unlink检查。root/data近满仍需实际定点恢复空间；保留唯一数据、活跃模型、音视频和运行行产物。

## Working State
HEAD 48b0f787b1c6c2f2b2b693d56120b919bb8147ba、branch codex/cross-tfg-audio-head-20261003，本地未push、无PR。该commit仅4文件：CONTEXT（93行）、differentiable_native_crop.py、audio_enhancement_eval/__init__.py、CPU parity03脚本。旧执行freeze bdf4654db5128b60d177b97aeb2d9362b3a37fa5和早期fit/dev d512876881fd630d4c8fb89920a031bdff72a144不重绑定。许多dirty/untracked属于旧研究；不stage-all。无新增或运行pytest/unit tests，科学评估和独立QC确实已执行。

机器局部：私人PTY50274/bridge2133需恢复时验证，服务器 root@connect.weste.seetacloud.com:12349（正确weste），密码/API key仅运行时，不存BM或文件。NS /root/tts-research-system-20261001/cross_tfg_audio_head_20261003；SHM /dev/shm/cross_tfg_audio_head_20261003。shared358391/ticks763311744、Ditto99490/ticks762772749，精确身份/GPU admission；不干扰未知GPU进程。Basic Memory Task25/Experiment24最新整篇读回PASS，status active/implemented；Task旧steps元数据尚需下一实质更新一并纠正。

## Changed Files
- scripts/audio_enhancement_eval/score_full512_precodec_parity_attempt03.py，source SHA ae5b7c123fdaa9302bcae18836cf081385b8b64333a9b6c743fd4382d47a638b；helper SHA1e89da5bdf2bee83938b717a084b55a699f1902925be3290d23317bfa562fa27。
- .codex/scratch/root_precodec_summary8_independent_qc_attempt01.py/.json独立JSON复核，receipt d977f46f4492e0a61d1aaa2a4233ad3f6a58fb1420eb326bb1bd4d3a9b774db1。
- runs/cross_tfg_audio_head_20261003/full512_precodec_bundle_readback_attempt02/ 已完整恢复，诊断source analyze_frame_alignment_attempt03.py SHAa75ee5a4693da163eb0603ff6ff5b3cf97df34eca896ba59d9c57fd9a6bd8e9a，diagnostic c5f4ee71ac5e645efd0766bf7ae819ae14e7d19b3922fe67fdb39d1112e17271。
- .codex/scratch/latentsync_dependency_overlay_attempt01/kornia_supplement_attempt03/ 保留实际Kornia安装及import qualification、16模块探查packet/response。
- .codex/scratch/cross_tfg_sadtalker_attempt01/assets_and_overlay_attempt02/ 保存7个实际上传/安装/源码重预检/权重启动回执；overlay_import_probe_attempt01保存真实失败。

## Verification
ROOT JSON曲线/8行control/时钟复算PASS；agent安全全归档和8NPZ恢复PASS。当前分解为fit RMS视觉校准，未见数据未开放。
Kornia实际pip PASS但model source import未闭包；Sad safetensors实际import PASS但skimage gap；依赖阻塞不能写作科学阴性。shared/Ditto尚部分结果，不选中间最佳。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)；当前 `48b0f787b1c6c2f2b2b693d56120b919bb8147ba` 为local/unpushed。[已验证远端commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。
- openspec/changes/develop-cross-tfg-audio-head-20261003/和data-protocol.json/development-data-seal.json是冻结协议；ADR009决定共享profile选择utility及model-equal/worst模型约束。
- Task25/Experiment24，update_cross_tfg_snapshots_attempt25.ps1及读回receipt保留当前typed状态。

## Observations
- [result] 全8校准数据已实际安全回收，ROOT曲线重建PASS，特征存在稳定+2 query discrepancy且纯像素对比未相同偏移。
- [decision] 以原自然音、原长度、原时钟主评估；视觉链路的+0.392校准差异不作为增强收益。
- [decision] v4先精确复现官方五帧输入与取帧映射，再优化lag contrast和跨模型utility，继续失败后的迭代。
- [decision] 只退役已精确恢复归档和可重新下载且运行闭包不使用的公共缓存，目标身份与FD复检后单文件处理。
- [blocker] 16import闭包、Sad依赖/4weights、LatentFP16流式实际部署、全360/240及native时间映射尚未完成；目标继续active。
- [next_step] 优先实际空间退役和依赖闭包，同步查明两帧feature/index差异并冻结v4实际训练，完成360选择/dev及heldout确认后交付可用增强头。

## Relations
- continues [[Codex checkpoint - 2026-10-03T22-53-03Z - Complete native calibration and fifth model source continuation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]