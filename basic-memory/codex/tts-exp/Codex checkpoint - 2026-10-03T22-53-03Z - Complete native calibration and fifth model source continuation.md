---
title: Codex checkpoint - 2026-10-03T22-53-03Z - Complete native calibration and fifth
  model source continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t22-53-03-z-complete-native-calibration-and-fifth-model-source-continuation
repository: dogLeft233/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repo_root: E:/Documents/tts-exp
started: '2026-10-03T22:53:03.7797555+00:00'
cwd: E:/Documents/tts-exp
git_sha: cad9c757329ddd1ea3d7c7c73e08c1c9ca8cdb2a
capture: deliberate
agent: codex
model: gpt-6.1-sol
username: dogle
hostname: LAPTOP-E133GSLE
status: open
trigger: compact
project: tts-exp
branch: codex/cross-tfg-audio-head-20261003
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T22-53-03Z - Complete native calibration and fifth model source continuation

## Summary
八样本原生视觉链路校准已实际完成且全部控制通过；共享频谱头筛选继续运行，LatentSync和SadTalker从已验证源码推进实际依赖与权重部署。

## Story
原目标是解释TTS改善中文TFG同步的机制，机制报告25已结论化。用户随后要求形成适配多种TFG的自然音频增强头h(x)：增强音只驱动生成，播放和主SyncNet评分恢复原自然音x；必须继续迭代失败候选，产出可复现可使用的方法。目标仍active。root维护理论、协议、BM、Git、运输与独立复核，实验代码及运行由gpt-6-luna/max子agent负责。最近ASR授权已收到，实际100+20条转写CER=0，尚无需替换ASR。

此前v3在完整W2L/Muse队列均未超过RMS Cbest基线；Muse240行独立复核结果identity5.044、RMS5.095、local5.083、v3 5.082，v3−RMS=-0.013。保留共享12个局部频谱profile的360行fit筛选，以及修正训练视觉前向和lag-contrast目标的v4两条路线。最新已读实际shared97/360、Ditto153/240均PASS/0FAIL，但这些计数约一小时前，需新status；不按中间分数选候选。完成360后按ADR009 model-equal和worst-model utility选择≤3个，再进入dev和未见40说话人/heldout确认。

新8样本CPU校准实际完成：同一原自然音embedding，完整31lag矩阵，proxy saved/reembedded、hard precodec512→224、decoded512→224、official saved/CV2/JPEG重嵌入等分支。预检路径及冻结来源两次错误均在加载模型前阻断并保留；自包含helper修复后preflight8样本/40输入seal PASS，运行93.272秒exit0/COMPLETE，所有控制通过。主summary214973B SHA d1b0bb1af1855a5971add87a749871d5c7388e2a1b6ab85321730c48e28a964a，manifest3220B SHA d2863654e75c60d91707dec9fb8776f5b2c08daa27c868ea1e13191aaf117669，source03 SHA ae5b7c123fdaa9302bcae18836cf081385b8b64333a9b6c743fd4382d47a638b。完整summary+8row+4control已实际远端读回code0/无截断，待本地逐字节恢复和ROOT数值复核；8原始NPZ约102MB待归档分段恢复。stdout中的precodec−official Cbest均值+0.398、范围-0.926至+1.330是视觉链路差异，绝非增强增益。它说明训练须处理实际codec、完整上下文和lag背景，不宜仅优化D0；正式v4方案仍待完整summary/数组分解后冻结。

LatentSync六个官方PyPI核心包已实际安装到独立overlay02，实际CPU导入10项中8项PASS，仅pipeline/scripts.inference缺kornia，CUDA未初始化、无权重加载或推理。源码requirements519B已核实确实锁定kornia0.8.0，kornia_rs0.1.8是必需间接依赖；两个小轮子已本地固定官方SHA，追加overlay包正在修复准备。官方5.072GB UNet保留在B盘canonical，pre-move全SHA通过，post-move虚拟盘mtime变化后第二次全读未完成。远端实际cgroup上限66.572GB/current36.031GB，约30.541GB可用；可尝试官方权重HTTPS流入内存、完整SHA、weights_only加载后生成显式标记的FP16派生权重，与官方入口loadFP32→toFP16逐参数精确一致，再实际推理。此路线尚未执行，不写成模型部署成功。

第五生成架构SadTalker官方tag v0.0.2-rc/commit28b8dfc2ebed9a00238090e5d46329cb77643207的117个源码文件已ROOT逐字节对照官方archive，实际上传/提取/导入预检PASS；实际import首先缺safetensors且其它依赖未齐。官方四个运行资产HEAD均200，总1183756176B，包括256 safetensors、crop mapping_00229、facexlib对齐和检测权重；尚未下载。它使用与W2L同mel家族，不增加frontend计数，但现有W2L/Ditto/Muse已覆盖3种frontend。后续配对3seed、crop默认、禁可选enhancer、保留i/25时间戳并丢弃native mux音轨恢复原自然F32。

远端没有已挂载可写云盘，root/data空间紧张。最新实际SHM4.235GB/data216.170MB/root3.154MB。旧447534356B v2归档已本地安全全恢复并远端SHA/无FD通过，236782637B embedding归档亦已本地恢复但待新远端audit。1.488GB公共训练SyncNet权重实际全SHA匹配官方LFS38fa63bad3ed2332f647c40a5dc616cb0e233db8579f698f62af4c41965c4da5，稳定/noFD20进程/无权限错误；官方Latent源码无引用，Muse仅app/download/trainingconfig引用，冻结CLI shim/adapter不加载它。audit所列一个deploymentcode根不存在，不能据此称整个运行闭包完整，子agent正补查app加载分支。未删除任何文件。完成真实闭包检查后可定点退役公共可再下载缓存与已恢复旧归档，为Latent FP16约2.54GB和Sad1.184GB腾空间。

## Working State
Git branch codex/cross-tfg-audio-head-20261003，HEAD cad9c757329ddd1ea3d7c7c73e08c1c9ca8cdb2a是local/unpushed/noPR。执行freeze bdf4654db5128b60d177b97aeb2d9362b3a37fa5，早期dev/fit d512876881fd630d4c8fb89920a031bdff72a144不重绑定；raw执行SHA与Git换行标准化分开。dirty/untracked含既有研究代码和tests，不能stage-all。没有新增或运行pytest/unit tests。

服务器root@connect.weste.seetacloud.com:12349，凭据仅runtime，不写笔记/文件。私有PTY50274、bridge2133、下载中B盘rehash会话均机器局部脆弱状态；恢复先验证存活。namespace /root/tts-research-system-20261001/cross_tfg_audio_head_20261003；SHM /dev/shm/cross_tfg_audio_head_20261003。shared worker358391/ticks763311744、Ditto99490/ticks762772749，使用精确进程身份与GPU admission gate。

## Changed Files
- scripts/audio_enhancement_eval/score_full512_precodec_parity_attempt03.py为实际CPU校准source，helper scripts/audio_enhancement/differentiable_native_crop.py独立seal。
- .codex/scratch/cross_tfg_audio_head_20261003/full512_precodec_shared_audio_attempt01/packet_attempt04至08保存actual预检、启动、完成、读回与未执行归档请求；08归档receipt的literal\\n写法已ROOT指出，需新请求修复后执行，保留旧稿。
- runs/cross_tfg_sadtalker_20261004/source_attempt01/及source_stage_and_import_preflight_attempt02.py与ROOT源码QC保存官方源码来源。
- .codex/scratch/latentsync_dependency_overlay_attempt01/import_probe_attempt01/及attempt02保存实际安装后导入/资源读回，kornia新包尚prepared。
- .codex/scratch/update_cross_tfg_snapshots_attempt24.ps1已实际整篇更新并读回Task24/Experiment23，但尚未反映最新CPU完成及Sad提取事实。

## Verification
- CPU finish_receipt1732B SHA9f7c1a63a86e3ab01d85bcb631626100fbe4b4d6c424722c65e031fff7316e92，exit0/COMPLETE/8rows，stderr0。
- CPU preflight18473B SHAa448295d20e6c5a1af8b54b7ecdb5f5aec281260e0a8f015135db6fce595f460实际字节读回PASS。
- Sad117源码独立QC实际PASS；worker099f23cb94b5ebdfe85d7b09ba2c52346df4e23ffd3fbac4075fb8c1b92c0a03实际上传SHA/提取seal/CPU导入gate通过，缺依赖不是科学阴性。
- Latent10项import报告14400B SHAdae00a045603475ea1c1f163b1645a9668e52c5b3404478b07e0ac3678dc6648，8PASS/2missingkornia，CUDAfalse。
- 公共SyncNet资产audit请求4fa6952a29710d0cbb98d1f7486bbe073c25a23bb91a014ff545aa8b52238418实际code0/fullJSON，官方SHA/size/stability/noFD通过，未删除。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，当前本地commit `cad9c757329ddd1ea3d7c7c73e08c1c9ca8cdb2a`未push；[已验证远端commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。
- openspec/changes/develop-cross-tfg-audio-head-20261003/及data-protocol.json/development-data-seal.json为冻结科学协议；ADR009控制共享profile选择。
- runs/cross_tfg_audio_head_20261003/musetalk_stageb_attempt04_readback_attempt02/与ROOTQC04 SHA8bdf45d17c672130055d14c6007df131c6eef63ef7770473f6644bf010c7e19a保存完整Muse240行复核。

## Observations
- [result] 八样本同原音CPU视觉校准实际COMPLETE/全部control通过，完整JSON实际读回，原始NPZ待完整本地恢复。
- [decision] 正式增强评价保持原自然音、原长度、原时钟；precodec视觉差异不能当增强收益。
- [decision] v4需要贴近正式codec并约束lag背景；必须审计变换后的全序列上下文，不以旧RMS帧缓存冒称exact前向。
- [decision] 公共可恢复训练权重与已全恢复归档可在精确路径/SHA/noFD/active closure通过后退役，保留所有运行数据和活跃模型。
- [blocker] Latent/Sad依赖、权重与actual inference未齐；v4目标待校准summary/数组分解，当前候选尚未形成可交付跨模型收益。
- [next_step] 先完整恢复并独立分解8样本校准、刷新shared/Ditto实际状态，同时完成Kornia追加与精确空间退役，然后冻结v4训练及共享profile dev，持续到未见确认和方法交付。

## Relations
- continues [[Codex checkpoint - 2026-10-03T22-13-03Z - Verified matrices and actual deployment repair continuation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]