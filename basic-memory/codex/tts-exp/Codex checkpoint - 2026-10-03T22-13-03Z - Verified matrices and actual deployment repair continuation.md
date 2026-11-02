---
title: Codex checkpoint - 2026-10-03T22-13-03Z - Verified matrices and actual deployment
  repair continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t22-13-03-z-verified-matrices-and-actual-deployment-repair-continuation
project: tts-exp
started: '2026-10-03T22:13:03.3114847+00:00'
branch: codex/cross-tfg-audio-head-20261003
status: open
username: dogle
repo_root: E:/Documents/tts-exp
trigger: compact
agent: codex
git_sha: cad9c757329ddd1ea3d7c7c73e08c1c9ca8cdb2a
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
hostname: LAPTOP-E133GSLE
model: gpt-6.1-sol
cwd: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T22-13-03Z - Verified matrices and actual deployment repair continuation

## Summary
共享自然音频增强头继续进行实际跨模型验证；MuseTalk全队列已独立复核，当前训练版未超过RMS基线，因此保留频谱族筛选与校准视觉链路后的下一轮训练两条路线。

## Story
项目起点是解释TTS为何改善中文TFG同步，最终机制报告25已完成。用户进一步要求可用于多种TFG的自然音频增强头：h(x)只驱动视频生成，播放及主SyncNet评分恢复原自然音x；要形成可复现、可使用、有泛化边界的方法，不能因失败候选而停止。当前目标仍active。root负责理论、实验协议、Basic Memory、Git、传输与独立复核；实验代码与运行由gpt-6-luna/max子agent执行。新提供的Qwen ASR授权保留在运行环境，已有100+20条转写验证CER=0，不因API可用而无目的重跑。

实际MuseTalk20clips/10speakers/3seeds/4methods的240行及一次生成重复已全部读回。ROOT复核251个归档成员、241个NPZ/1687数组、240个计划关联、原始N×31矩阵、25fps约分时间戳及FULL/guard20支持集。FULL Cbest为identity5.044、RMS5.095、local5.083、v3 5.082；v3−RMS=-0.013，不能当作可用的跨模型增强效果。ROOTQC04使用有记录的float32 reduction容差1.5e-5，实际最大差9.54e-6；早先QC01–03失败是ROOT几何/约分时间/过严容差假设，全部保留，未修改生产分数。120个ROI数组及4项像素测量也通过独立复核。

训练代理与正式视频评分存在大幅零延迟差异，已查明它不是预测贴回或RGB/BGR错误：实际CPU诊断10个时间点的96px预测经OpenCV放大贴回512px，保存图像完全相等。全8样本的新CPU视觉链路校准将重算proxy、precodec、decoded512、saved official224、JPEG等分支，固定同一原始音频embedding；须同时通过保存矩阵重建及CPU/GPU重嵌入容差，不能因处理完8个样本就标PASS。当前scorer03及packet02已实现、尚未实际preflight/forward。若precodec仍偏差，下一轮训练需完整原生codec前向/STE反向与lag-contrast目标，不能只优化D0。

共享12个有界局部频谱profile的360行fit实验保持既定cohort与RMS/相位/原时间约束，最新实际97/360 PASS、0FAIL；Ditto当前153/240 PASS、0FAIL。不得按途中分数选profile。完成360、消费完整矩阵后才按ADR009的model-equal与worst-model utility选≤3个候选，进入预冻结dev及最终未见40说话人/heldout模型验证。

LatentSync作为heldout生成模型已固定官方source和权重，70个源码Git blob通过；6个官方PyPI轮子30MB已分5段实际上传且逐段SHA通过。worker04的实际launch在进入worker前退出code1，诊断确认只因SOURCE_PREFLIGHT_FILE/MODEL_DEPLOY02_DIR误指旧目录，其余路径/哈希/不存在目标guard全通过；没有安装、模型加载或推理。子agent正在修复worker05的精确preflight路径，保留失败来源。B盘UNet部分文件完整SHA已匹配后重命名到canonical，但虚拟盘mtime改变，post-move全读SHA仍在执行，不能把pre-move验证称post-move验证。官方Latent入口要求stage2.yaml、相对Whisper路径及VAE repo-ID；独立shim须使用现有已验证VAE、每行独立cwd，恢复原PCM并验证25fps/PTS。

实际远端挂载清单没有可写云盘，root剩8MB、数据盘379MB、SHM5.248GB，公开NFS只读。UNet约5.072GB加Whisper、输出和余量无法直接塞进SHM；下一步以已全量恢复到本地的旧归档作精确SHA/引用闭包审计后定点清理，不能删除活跃Ditto/Muse/fit资产。第五生成架构选择SadTalker官方256 safetensors+crop mapping_00229，约0.82GiB候选体积，待官方HEAD/源码固定/依赖审计和实际运行；它与W2L同mel声学家族，不增加新frontend，但现有W2L/Ditto/Muse已满足至少3种frontend。

## Working State
Git branch codex/cross-tfg-audio-head-20261003，HEAD cad9c757329ddd1ea3d7c7c73e08c1c9ca8cdb2a，local/unpushed/noPR。当前执行冻结bdf4654db5128b60d177b97aeb2d9362b3a37fa5；更早运行d512876881fd630d4c8fb89920a031bdff72a144不重绑定。原始执行文件SHA是权威，Git AutoCRLF标准化不可混淆。dirty/untracked含既有研究代码和tests，不stage-all；没有运行pytest/unit tests。

远端root@connect.weste.seetacloud.com:12349，固定SSH host fingerprint见项目reference；密码/API key仅运行环境，不写文件。私有PTY50274与bridge2133属于机器局部脆弱状态，恢复后先确认仍可用，不输出环境secret。namespace /root/tts-research-system-20261001/cross_tfg_audio_head_20261003，SHM /dev/shm/cross_tfg_audio_head_20261003。shared worker358391/ticks763311744与Ditto99490/ticks762772749为当前身份；不得仅凭PID删除或停进程。

## Changed Files
- CONTEXT.md与3个fit repair来源已入Git cad9c75，CONTEXT93行；最新计数由实际status/BM快照确定。
- .codex/scratch/root_muse_roi_independent_qc_attempt04.py及对应JSON为ROOT分析复核，非实验模型实现。
- scripts/audio_enhancement_eval/verify_full512_prediction_paste_attempt01.py已实际CPU执行；score_full512_precodec_parity_attempt01/02/03.py为后续校准实现，尚未forward。
- .codex/scratch/latentsync_dependency_overlay_attempt01/ 与latentsync_unet_partial_finalization_attempt03.json保留安装前失败及postmove未完全验证的边界。

## Verification
- ROOT Muse+ROI独立QC04 receipt SHA 8bdf45d17c672130055d14c6007df131c6eef63ef7770473f6644bf010c7e19a，实际PASS。
- prediction_paste_probe_receipt.json SHA a5c11ead3d331d6c89783506d83bab16c7bb8e987531359e52b01ce30b1a1248，10/10 paste exact，channel swapped diagnostic MAE38.869。
- shared fit status05/ Ditto status09实际97/153 PASS，未完成与未选profile。
- launch_guard_readonly_diagnostic_attempt01_request.json SHA334ead9a703d79b6107d2509d3f54cb4d28094e3f55e513c0b21ced9add95732实际code0，两个错误preflight路径，其余guard通过。
- mount inventory01实际code0完整JSON覆盖所有已知根路径；后续03仅prepared，不重称新挂载事实。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，当前本地commit cad9c757329ddd1ea3d7c7c73e08c1c9ca8cdb2a未push；[已验证远端commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。
- openspec/changes/develop-cross-tfg-audio-head-20261003/ 与 data-protocol.json、development-data-seal.json为冻结协议。
- .codex/scratch/update_cross_tfg_snapshots_attempt23.ps1，Task23/Experiment22已整篇写入并直接读回；尚未反映最新97/153及worker04前置路径失败。
- runs/cross_tfg_audio_head_20261003/musetalk_stageb_attempt04_readback_attempt02/ 为实际原始矩阵归档及安全恢复；full512 ROI与compact readback路径见ROOTQC04输入。

## Observations
- [result] 当前v3在W2L与Muse均未超过RMS Cbest基线；继续迭代共享频谱族与校准后的训练目标。
- [decision] primary分数恢复原自然音x；候选保持相同样本时钟，禁止时长重映射或评分音频替换。
- [decision] 保存矩阵重建和CPU/GPU replay均是校准控制，不能用像素MAE或处理完成代替SyncNet parity。
- [decision] SadTalker增加生成架构覆盖，声学frontend仍记mel家族；不虚增frontend计数。
- [blocker] Latent worker04未进入安装，路径repair与空间定点释放仍需实际执行；canonical UNet的postmove全SHA未完成。
- [next_step] 先实际完成Latent worker05安装/import预检与8样本CPU视觉校准，再读取完整360 fit和240 Ditto结果，按固定utility进入共享候选dev与v4训练，持续至交付。

## Relations
- continues [[Codex checkpoint - 2026-10-03T21-34-34Z - Actual shared fit and native geometry continuation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]