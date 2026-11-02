---
title: Codex checkpoint - 2026-10-04T01-20-14Z - Shared recovery execution and verified
  Latent runtime weights
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t01-20-14-z-shared-recovery-execution-and-verified-latent-runtime-weights
repo_root: E:/Documents/tts-exp
model: gpt-6.1-sol
git_sha: 03b2189f4b4c7ac88306da79d0181835b20a9dcb
trigger: compact
started: '2026-10-04T01:20:14.4537379+00:00'
branch: codex/cross-tfg-audio-head-20261003
agent: codex
username: dogle
project: tts-exp
repository: dogLeft233/tts-exp
capture: deliberate
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T01-20-14Z - Shared recovery execution and verified Latent runtime weights

## Summary
完整共享谱筛选恢复已实际运行至198/360全部PASS，LatentSync官方权重的FP16运行派生版已完成全SHA与重载校验；下一步真实native训练和多架构验证继续。

## Story
用户要求可适配多种TFG的自然音增强头h(x)：增强音驱动视频，原自然音x用于播放和主SyncNet，失败后继续改进直到交付权重、配置、CLI、复现和未见40说话人/heldout架构确认。报告25机制已结论化，增强头目标仍active。root负责理论、冻结协议、BM、Git、运输和独立复核；实验与代码由用户指定gpt-6-luna/max子agent承担。Qwen ASR/TTS和数据下载已授权；既有100+20转写CER=0，新的候选仍需新内容与视觉guards。

v3主Cbest未超过RMS（W2L6.839对6.945；Muse5.082对5.095）。正确匹配距离改善但错误匹配距离一起下降，使区分度丢失；v4同时奖励Cbest与远时移差、保护D0/Canchor，固定96步/12epochs与最后checkpoint，不按dev挑epoch。已完成8条原始full512/native时钟分析，JPEG窗口j对应source j−2；早段minimum-MAE monotone对应模糊保留。正式H264交付行仍须实际检查，不能把custom DIVX的shift直接推广。

服务器root空间真实恢复6.27GB后，shared旧360行174PASS/186ENOSPC失败进入独立recovery04。实际runner04 SHAac2c381d9238cbb4cd399478d561b54116757c2db2ff0b4b6186a1f9513f5e79、launcher07 SHAae06b4ee7b634a1d1575eaad54bb6818680533b0f163741bd84dfa75b7e7553c；原plan/rows/seed/科学代码保持不变。preflight逐项核对174旧PASS的1098文件引用和全部7组NPZ数组，fresh repeat强制新生成。真实worker202805在运行，最新198/360 PASS198/FAILED0，逐阶段GPU准入正常，尚未完整，不选幸存子集。Ditto240旧run239PASS/1ENOSPC失败、repeat比较null；子agent正在实现只补1条和fresh repeat、重建完整支持集的独立恢复。

LatentSync RAMstream07实际成功：官方5072222488B UNet SHA0a478e89eb660f82da4c35dbdde8a5adfb27f99d1b4e50edd03729e1e98316d3在RAM完整验证，1246个FP32 tensor逐个转FP16、全finite、原子写盘并CPU weights_only重载。派生2536318892B文件SHA243be78daffff133a10d8c191f999db996a31a25011ffea7554d40705e18e9de，receipt7451B SHA1e47de88b0c38fa5b00e6b3487362ec6cb1e71f14a1c8c9a1f80a783fa6d4502。尚未实例化模型/strict key-shape覆盖验证/真实推理，不能把派生验证称为模型通过。16模块12PASS/4缺InsightFace；offline18.47MB输入包和真实Cython构建packet04已ready待root审阅dispatch。SadTalker4官方资产完整已校验；默认face3dvis=false路径无PyTorch3D需求，继续真实最小依赖闭包与推理。

正式W2L clock02实际上传/preflight通过，collector因旧pyframes已移除exit1，未得到结论。root从远端取得真实10426B run_pipeline.py SHA3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf全文；fresh clock03已ready，仅复放两条冻结FFmpeg命令并按sealed轨迹裁剪比较，标签明确replayed JPEG，不改旧run/原音时钟/评分。其后尽快baseline forward与v4实际训练；旧context只有96/224 crop和patch，需复用已生成full512/tracks派生紧凑native context。

## Working State
HEAD03b2189f4b4c7ac88306da79d0181835b20a9dcb，branch codex/cross-tfg-audio-head-20261003，local/unpushed、无PR；执行freeze bdf4654db5128b60d177b97aeb2d9362b3a37fa5和早期fit/dev d512876881fd630d4c8fb89920a031bdff72a144保持。新commit只含CONTEXT和shared recovery两源码；legacy dirty/untracked不stage-all。CONTEXT94行，HANDOFF ignored/local，其Git行待修正。Task28/Experiment27全文读回PASS active/implemented，收录shared180和Latent完成；后续正式clock失败/最新198及03b需下一次实质性更新。

机器局部：私有PTY50274/bridge2133仅进程持有凭证；服务器root@connect.weste.seetacloud.com:12349，密码/API key不入文件/BM。NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003，SHM=/dev/shm/cross_tfg_audio_head_20261003。最新root free3734376448B/SHM2679410688B，shared新run191577840B。公共1.488GB训练SyncNet尚保留，精确引用/进程审计后可退役，非当前部署前置条件。三个Luna/max继续shared/Ditto、clock/v4、Latent/Sad部署。CodeGraph实际无可用index，已先尝试explore，不建index。sealed40保持未读。

## Changed Files
- scripts/experiments/cross_tfg_audio_head_20261003/recover_shared_frequency_fit_attempt04.py及launch_shared_frequency_fit_recovery_attempt05.sh：已commit、真实执行和复用PASS。
- .codex/scratch/latentsync_unet_ramstream_attempt07/ramstream_fp16_derivation_receipt_attempt07.json：7451B实际完整派生receipt。
- .codex/scratch/actual_run_pipeline_source_read_root_attempt01.py：远端真实冻结pipeline源码。
- scripts/audio_enhancement_eval/map_formal_w2l_delivery_clock_attempt03.py及fresh packet：ready未dispatch。
- .codex/scratch/latentsync_insightface_overlay_attempt04/packet_attempt01/packet_index_attempt01.json：ready未dispatch。

## Verification
ROOT核对恢复source/packet原始SHA、preflight与逐阶段run日志、198全PASS；Latent07完整receipt byte/SHA/readback、1246 tensor inventory/payload/finite和CPU重载；formal02实际失败、远端pipeline完整bytes/SHA；Git新commit精确3文件。此前CPU8独立25成员/176数组/56矩阵复核PASS，最大误差3.621507e-6小于冻结3e-5，官方Torch分数不改。未新增或运行pytest/unit tests；尚未进行v4训练、Latent/Sad推理或sealed40确认。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)；03b SHA local/unpushed；[已验证远端commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。
- openspec/changes/develop-cross-tfg-audio-head-20261003/；ADR009保持模型等权、最坏模型和Cbest/D0/Canchor/far共同utility。

## Observations
- [result] shared完整360计划恢复在运行，已198全PASS，174旧PASS经过文件及raw数组核验复用。
- [result] LatentSync官方UNet RAM完整SHA与FP16派生/重载成功，真实模型key-shape覆盖及推理待执行。
- [decision] v4使用实测native context和lag contrast联合目标，原音时钟不移动；完整支持集与fresh repeat后才选候选。
- [blocker] Ditto补跑、v4实际训练、InsightFace构建/Latent及Sad推理、内容视觉guard和未见确认未完成；目标active。
- [next_step] 实际dispatch ready InsightFace构建及正式W2L clock03复放，立即推进实测native baseline/v4固定训练，并补Ditto两次生成。

## Relations
- continues [[Codex checkpoint - 2026-10-04T00-36-49Z - Actual cache recovery and native mapping execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]