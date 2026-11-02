---
title: Codex checkpoint - 2026-10-04T00-36-49Z - Actual cache recovery and native
  mapping execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t00-36-49-z-actual-cache-recovery-and-native-mapping-execution
started: '2026-10-04T00:36:49.7456791+00:00'
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
capture: deliberate
agent: codex
status: open
project: tts-exp
git_sha: 48b0f787b1c6c2f2b2b693d56120b919bb8147ba
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
username: dogle
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T00-36-49Z - Actual cache recovery and native mapping execution

## Summary
实际恢复服务器 root 空间 6.27 GB，并完成8条训练片段原始帧到 SyncNet JPEG 的只读对应分析；继续原计划失败重跑、修正 native 训练代理及5架构增强头验证。

## Story
用户要求可适配多种 TFG 的自然音增强头 h(x)：增强音驱动视频，原自然音 x 用于播放和主 SyncNet。失败候选后继续改进，交付权重、配置、CLI、复现和未见40说话人/heldout架构确认。报告25的 TTS 机制已结论化，增强头目标仍 active。root 管理论、冻结协议、BM、Git、运输和独立复核，实验与代码交给用户指定 gpt-6-luna/max 子agent。Alibaba Qwen ASR/TTS及数据获取已授权；100+20条既有转写 CER=0，没有重复ASR。

v3 仍未超过 RMS 主 Cbest（W2L 6.839 vs 6.945；Muse 5.082 vs 5.095）。继续12共享谱 profile 完整筛选及 native/lag contrast v4。shared 原360行最终174PASS/186FAILED；Ditto原240行239PASS/1FAILED、repeat生成失败且比较值null。工程错误有实际 ENOSPC，不写成科学阴性或测得不确定性。恢复必须保留原计划、验SHA复用PASS、重跑失败及repeat、重建完整FULL/GUARD20，不能挑幸存子集选候选。

UV缓存真实审计22018 regular files、all nlink1、无symlink及实际env/进程/FD引用；实际执行worker fca319bf…e764，精确删除 /root/.cache/uv/archive-v0。2026-10-04T00:13:36Z→00:13:37Z，root free 0→6270697472 B；receipt2417B SHA27b99ce117ac53dd4ad78b7a78d5be6f1a01bb0627dc76576df44e8b2abd903c。运行环境、唯一实验数据、模型未删。此前两份可恢复归档已释放684322816B，完整本地成员仍在。公共1.488GB训练SyncNet仍保留；最新源引用扫描因bounded限额失败未删除，当前非部署前置条件。审计02源被子agent交付后改写，已把实际执行的60834568…9d01c源下载保存，禁止将新draft冒充旧执行。

8条 clock03 实际完整receipt604353B/21c5fceb…0ca：full512 AVI→PCM MKV像素MAE0/PTS0,.04,.08…，pipeline AVI为0,.12,.16…，CFR JPEG多2帧、首SHA A,A,B,B,C。新 mapper04实际 exit0/8行 COMPLETE_READ_ONLY_JPEG_SOURCE_FRAME_MAPPING，完整1866845B SHA926e08f84f2624fd8800974f2778c0ed875187aa6de065354c0db4701b0e8447，ROOT从03完整base64恢复。全8 q0的31帧窗口 JPEG j→source j−2；前段DP候选存在模糊，命名minimum-MAE monotone correspondence，不能称全序列精确身份。该差异发生于pipeline时戳→JPEG，主原音时钟不调整。v4选择实际对应的预测source j−2，保持crop几何/MFCC为native j，实际RMS全31lag forward后训练固定96步/12epochs；奖励ΔCbest和Δfar，单侧D0/Canchor guards及waveform/TV/norm约束，不再只奖励B。

Shared恢复02实际上传后launch因旧CORE路径不存在而preflight失败，没有推理。新恢复03包已准备，runner a8de821d…588a/launcher662799e6…5dcd：使用原TASK/scripts路径及冻结gpu_admission_wrapper05逐阶段run_logged准入，写新SHM recovery02。尚待ROOT审阅和真实dispatch，不能称恢复已启动。随后精确Ditto1+repeat重跑。

LatentSync实际16模块12PASS/4FAIL均缺InsightFace；Kornia/rs已实际import，UNet/audio2feature/affine成功。RAMstream06已真正启动但FAILED_PINNED_SOURCE_PREFLIGHT：引用的SHM acquisition receipt不在，未下载/未load，worker已结束。1251B receipt实际在B盘，SHA4f65ca…8388已核对；attempt07将上传它到独立control并新绑定，worker4f73a818…461f，尚未dispatch。保留官方5.072GB资产，只在RAM SHA验证后逐参数FP32→FP16保存root，不称已推理。SadTalker4官方资产合计1183756176B实际下载/ROOT独立fullSHA PASS，真实CLI最小闭包yacs/pydub/facexlib/basicsr/GFPGAN仍待安装/推理。独立架构部署与未见确认继续，不开放sealed40。

## Working State
HEAD48b0f787b1c6c2f2b2b693d56120b919bb8147ba、branch codex/cross-tfg-audio-head-20261003，local/unpushed、无PR。执行freeze bdf4654db5128b60d177b97aeb2d9362b3a37fa5、早期fit/dev d512876881fd630d4c8fb89920a031bdff72a144不重绑定。legacy dirty/untracked不stage-all；HANDOFF ignored/local。CONTEXT94行需更新新实测空间/恢复状态。Task27/Experiment26实际全文读回PASS active/implemented，尚未收录06失败和mapper完整执行。

机器局部：私有PTY50274/bridge2133 runtime凭证仅进程可用；正确服务器root@connect.weste.seetacloud.com:12349，密码/API key从不写文件/BM。NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003；SHM=/dev/shm/cross_tfg_audio_head_20261003。原shared358391/Ditto99490已结束；新GPU工作须fresh admission，不杀未知进程。三个Luna/max子agent继续实际重跑/v4/deploy。CodeGraph shell实际报告无可用index（即使存在目录），继续rg且不建index。

## Changed Files
- .codex/scratch/uv_cache_retirement_attempt01/retirement_receipt.json：实际空间恢复；04_download_receipt_transport_repair_root_attempt01.json仅修正下载路径。
- .codex/scratch/uv_cache_audit_attempt03/executed_audit_uv_archive_cache_attempt02.py：60834568…9d01c实际审计源码保存。
- .codex/scratch/root_pipeline_jpeg_source_map_actual_attempt01.json：8条mapper完整原始receipt；源packet pipeline_clock_mapping_attempt04。
- .codex/scratch/latentsync_unet_ramstream_attempt06/：实际失败报告与上传worker；attempt07是未dispatch的fresh修复。
- .codex/scratch/update_cross_tfg_snapshots_attempt27.ps1及Task27/Experiment26读回。

## Verification
ROOT核验实际cache执行源码、retirement receipt、空间数值；mapper完整byte size/SHA与8行窗口/早期模糊；Latent06实际process dead、未下载。先前CPU8 raw arrays QC仍PASS：25成员/176 arrays/56矩阵、NumPy重构最大3.621507e-6小于冻结3e-5；主Torch分数不改。未新增或运行pytest/unit tests，未见40说话人未打开。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)；48b SHA local/unpushed；[已验证远端commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。
- openspec/changes/develop-cross-tfg-audio-head-20261003/；ADR009规定模型等权/最坏模型与Cbest/D0/Canchor/far共同utility。

## Observations
- [result] 实际root恢复6270697472 B，唯一实验和模型保留；186共享和Ditto1失败可开始精确恢复。
- [result] 8条native JPEG窗口支持source j−2代理，早段DP模糊保留；原音时钟不移动。
- [decision] 原360/240共同支持集及fresh repeat完成后才选候选，固定最后训练步并保持sealed确认集。
- [decision] 用户指定Luna/max负责实验代码，root继续主线和独立QC；checkpoint不结束任务。
- [blocker] 新恢复包/v4未实际训练，Latent06缺receipt路径失败，InsightFace和Sad闭包及5架构/未见确认未完成。
- [next_step] 实际dispatch修复Latent07与带逐阶段GPU准入的shared恢复03，同步执行native v4 forward及固定96步训练，随后Ditto精确重跑与5架构确认。

## Relations
- continues [[Codex checkpoint - 2026-10-04T00-05-47Z - Native frame clock localization and experiment space recovery]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]