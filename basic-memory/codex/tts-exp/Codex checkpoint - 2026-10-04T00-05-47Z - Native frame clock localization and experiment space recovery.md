---
title: Codex checkpoint - 2026-10-04T00-05-47Z - Native frame clock localization and
  experiment space recovery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t00-05-47-z-native-frame-clock-localization-and-experiment-space-recovery
username: dogle
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
model: gpt-6.1-sol
trigger: compact
started: '2026-10-04T00:05:47.0974409+00:00'
git_sha: 48b0f787b1c6c2f2b2b693d56120b919bb8147ba
capture: deliberate
agent: codex
status: open
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
project: tts-exp
branch: codex/cross-tfg-audio-head-20261003
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T00-05-47Z - Native frame clock localization and experiment space recovery

## Summary
定位八条原音视觉校准的两帧差异到 run_pipeline AVI 时间戳及 CFR JPEG 抽帧，完整筛选运行因 ENOSPC 结束，已定点回收两份可恢复归档并继续空间修复、精确失败重跑和实际 v4 训练。

## Story
用户要求形成可用于多种 TFG 的自然音频增强头 h(x)，增强音仅驱动视频，播放及主 SyncNet 均恢复原自然音 x；失败候选后持续改进，交付配置/权重/CLI/复现及未见40说话人与 heldout 架构确认。先前解释 TTS 中文增强的机制报告25已结论化，增强头目标仍 active。root 负责理论、冻结协议、BM、Git、运输和独立复核；实验代码及运行交给用户指定 gpt-6-luna/max 子agent。最新用户允许 Alibaba Qwen ASR；既有100+20条转写 CER=0，没有为了授权而重复ASR。

v3 尚未超过 RMS 主 Cbest（W2L RMS6.945/v3 6.839；Muse RMS5.095/v3 5.082），继续12种共享局部谱 profile 筛选及 native forward/lag contrast v4 路线。此前 shared149/360、Ditto208/240部分PASS已被实际最终状态取代：shared360行完成174PASS/186FAILED，Ditto240行完成239PASS/1FAILED且repeat失败，两个进程已结束。186共享失败及Ditto最后一行均有实际 ENOSPC stderr；repeat只有生成失败，比较值null，不能称为测得随机不一致。禁止挑174/239子集选择候选；修复空间后保留原360/240科学计划、精确复用已验PASS、重跑失败行和repeat、原consumer重建共同FULL/GUARD20支持集。

CPU8校准已全字节回收。ROOT独立读取25成员/176 arrays/56同原音矩阵全部PASS，float64 NumPy重构最大矩阵误差3.621507e-6（producer预设3e-5），主官方Torch分数不改。hard−official平均 Cbest +0.3921595、D0 +6.8709631、Canchor−6.8767928仅表示视觉链路差异，不是增强增益；lag差均值2.375帧，旧1.875已撤回。

新 clock_mapping_attempt03 已实际执行 exit0、8行 COMPLETE_READ_ONLY_CLOCK_MAPPING，604353B完整receipt SHA21c5fceb78f5afa9655f2f833776d07bb653447d849872fe778d263e64dcc0ca；ROOT从完整gzip响应恢复并核验全文。全8 full512 AVI→PCM MKV decoded pixels exact MAE0，PTS均0,.04,.08…；run_pipeline AVI仍N帧但PTS为0,.12,.16…；JPEG/crop为N+2帧且前五JPEG SHA是A,A,B,B,C。这将差异定位到 run_pipeline AVI timestamp→CFR JPEG抽帧，不能归因MKV首80ms或隐藏五帧中心。+2 signature最低值仍非逐帧精确映射。下一步核对正式TFG H264/PNG交付与此次DIVX校准的关系，按实际 delivered frame sequence 修正可微训练代理然后实际训练v4；主原音/时钟不做事后移位，不学习补偿评估器编码伪差的音素提前。

服务器root分区实际free0，/root/.cache/uv/archive-v0约6.27GB，data约202.8MB空闲，SHM清理前2.207GB。精确退休packet05已真实删除447534356B旧v2归档及236782637B embedding归档，两份本地全成员恢复/逐SHA验证仍在，观察释放684322816B。1.488GB公共训练SyncNet尚未删除：worker08因额外download_weights.bat引用守卫失败，文件保留。后续不重跑已成功06/07；先完整引用扫描并分类唯一公共目标。UV缓存仅审计准备中，需核对无安装进程/FD、运行env的symlink/.pth引用后使用实际UV清理或精确cache unlink，不触碰唯一实验数据或运行模型。agent第一审计draft的Muse/Future路径及OUT不一致已ROOT指出，要求新attempt02修正再执行。

SadTalker四官方资产共1183756176B已全部下载并ROOT独立fullSHA PASS，正式256 safetensors/mapping229/两个脸部权重可用，不需重下；实际overlay探查确认Muse的skimage/yaml/imageio/imageio_ffmpeg可复用，Sad safetensors独立import PASS。Kornia0.8/rs0.1.8也实际安装成功。Sad仍需yacs/pydub/facexlib/basicsr/GFPGAN的真实CLI导入闭包，随后实际推理。LatentSync完整16模块9546B报告已ROOT字节恢复 SHA5a6728630905b866607e2a48b72dd0f8a11f370b2da8b39d341b160b83b5f09e，12PASS/4FAIL都仅缺InsightFace；模型UNet/audio2feature/affine已import成功。官方5.072GB UNet在B盘保留，官方HTTPS RAM流SHA→weights_only→逐参数nativeFP32-load后FP16一致性仍未实际执行；空间修复后继续，不将部署阻塞写成科学阴性。

## Working State
HEAD48b0f787b1c6c2f2b2b693d56120b919bb8147ba、branch codex/cross-tfg-audio-head-20261003，本地未push、无PR。只提交4文件：CONTEXT93行、differentiable_native_crop.py、audio_enhancement_eval/__init__.py、CPU parity03脚本。旧执行freeze bdf4654db5128b60d177b97aeb2d9362b3a37fa5和早期fit/dev d512876881fd630d4c8fb89920a031bdff72a144不重绑定。dirty/untracked多数属于旧研究，不stage-all；未新增/运行pytest或unit tests。

机器局部：私有PTY50274目前prompt可用，bridge2133 runtime env仅当前进程，恢复需验证；服务器 root@connect.weste.seetacloud.com:12349（正确weste），密码/API key仅运行时不进文件/BM。NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003；SHM=/dev/shm/cross_tfg_audio_head_20261003。shared358391/Ditto99490已经结束；最新GPU空闲，启动仍fresh admission，禁止杀未知进程。typed BM Task26/Experiment25整篇读回PASS、active/implemented，HANDOFF30行SHA577950fc09290a2ff2259ec5e15038b5825eee86ed158229b1869105c0959f76；CONTEXT/HANDOFF尚需更新晚期clock和真实空间状态。

## Changed Files
- .codex/scratch/root_precodec8_raw_arrays_qc_attempt01.py SHA29ad2de139b25cca0527857c14fae3aa094efeea4c6dc68c8e406daaaa1bf540；receipt6e57206bafad01e7a9daffa7e71dbf17181969d77796a0d07149f359f7513d5b。
- .codex/scratch/cross_tfg_audio_head_20261003/pipeline_clock_mapping_attempt03/collect_pipeline_clock_mapping_attempt03.py21858B/b676da8cd46b1f8de92df3f540e738a1bb93c20118c023aa5cb62735aaa638d7；root_pipeline_clock_receipt_actual_attempt01.json完整raw receipt。
- .codex/scratch/latentsync_storage_retirement_attempt05/实际00–07回执；v2 unlink receipt3d2532218b86045a301414acef226bead821da358e2ee5d6bbb5113dac4ef469、embedding da767659053694f6337a1f7faab942ae35dcf5d5a975d78b1e21f1196d28ea86。
- .codex/scratch/cross_tfg_audio_head_20261003/live_status_attempt02/、failure_diagnostics_attempt02/、recovery_diagnostics_attempt02/保存真实最终状态和ENOSPC/失败repeat。
- .codex/scratch/update_cross_tfg_snapshots_attempt26.ps1及Task26/Experiment25回执；latent_16_import_report_actual_root_attempt01.json。

## Verification
真实ROOT raw array及曲线复算PASS；clock03完整size/SHA、全部8counts/PTS/dup/decodedpixel核验PASS。Sad四weights独立fullSHA PASS。两归档目标实际已unlinked，剩余公共SyncNet保留。shared/ditto全计划已终止有工程失败，主效应仍待精确重跑与完整独立复核。未开放未见40说话人。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)；当前48b SHA local/unpushed；[已验证远端commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。
- openspec/changes/develop-cross-tfg-audio-head-20261003/为冻结协议；ADR009为模型等权/最坏模型及主Cbest/D0/Canchor/lag contrast共同utility。
- BM Task26/Experiment25为可变任务/实验状态，immutable checkpoint链独立保存已核验快照。

## Observations
- [result] 全8视觉差异定位到run_pipeline AVI初始PTS缺口及CFR JPEG重复；MKVdecodedpixel/PTS正常。
- [result] shared186及Ditto1失败由ENOSPC引起；repeat尚未测得determinism。
- [decision] 保持原自然音主评分与时钟，修正训练代理实际视觉取帧，不事后改分数。
- [decision] 完整360/240共同支持集后再选择，不用幸存子集；保留旧失败产物和执行lineage。
- [decision] 继续实际候选训练、5架构部署、未见确认和可用交付，checkpoint不是结束任务。
- [blocker] root0free；共享/Ditto需要恢复；LatentInsightFace和RAMFP16、Sad实际闭包/推理、v4训练及确认未完成。
- [next_step] 实际恢复root缓存空间并精确重跑360/240失败和repeat，同步修正native proxy后运行v4训练，推进两个heldout模型推理及完整确认。

## Relations
- continues [[Codex checkpoint - 2026-10-03T23-29-36Z - Native index calibration and dependency deployment continuation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]