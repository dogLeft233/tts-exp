---
title: Codex checkpoint - 2026-10-17T10-34-05Z - Four speaker preprocessing qualified
  partly and coordinate gate failed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t10-34-05-z-four-speaker-preprocessing-qualified-partly-and-coordinate-gate-failed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 10:34:05.610915+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-17T10-34-05Z - Four speaker preprocessing qualified partly and coordinate gate failed

## Summary
四身份原始素材独立QA和持久校验通过，Brown完整预处理已产出、Patrick第一阶段已封存，但Brown跨录制坐标门实际失败，尚未准入任何新生成器训练。

## Story
用户最初检验输入音量是否影响SyncTalk生成与SyncNet，May训练/驱动/评价三轴先导已封存在报告50/51。最新要求扩至四新身份各6–8分钟自然素材，同人不同原录制测试，全部资格通过后自动执行官方完整训练以及三档驱动、三档评价27格；后续训练已明确授权，无需再问一般许可。当前spec固定24模型（4人×seed42/2026×三训练增益）、65536 rays/FP16、60k基础与累计102k完整epochs，保持完整初始化与实际optimizer更新预算一致。

本轮补齐官方授权BFM转换、四人420秒素材的全量独立检查和F盘SHA交付，并在首次几何拟合前冻结ADR038实施合同。官方train BFM/联合BA，heldout固定训练身份/焦距并冻结拟合点细化姿态，再仅用训练相机配对估计相似变换；共同坐标是独立录制测试成立的必要条件。Brown的三个split关键点、光流、投影恒等式均通过，但训练BFM→BA全局相似变换旋转p95=6.6750268°、平移p95=0.0148960m，超过冻结2°/0.005m阈值。这个是真实准入失败，不是可删断言的数值误差；不得放宽门槛或宣称已训练。下一步诊断官方BA对非刚性几何/相机的改变是否使单一相似变换假设不成立，并修复可验证的跨录制几何绑定，必要时显式修订方法且保存旧失败。

## Working State
持久权威根F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017，包含selection/source_provenance、全部60个原材料文件4001168448字节、BFM/许可/转换资产、源码与资格receipts。四人全部原PCM逐样本、42000帧SHA/PTS、DOUBLE三档增益、split/上下文、跨录制去重通过；源是HDTF自然录制处理镜像，不冒称raw字节。Patrick测试482平方上采样512、Cathy源test4个既有满幅样本保留为限制。素材本身无需重下。

Brown与Patrick全部8250 JPEG/lms/BiSeNet完成，各41251文件F独立SHA通过。Brown五个竞争框人工图审为背景/手，官方preds[0]保持不变，F/SherrodBrown_face_adjudication.json闭合；Patrick全8250无review flags。Debbie/Cathy待继续。stage1_worker pid207722因2GiB空间reserve停止于Debbie，现SHM已回到约3.2GB空闲，但启动前查实际进程/receipt，不重复启动。Windowspersist也随失败停止，保留attempt03记录。

Brown frozen AVE六个recording×gain全量缓存通过，原音频mel和完整重复forward逐值一致、编码器权重不变。GT/torso/眼系数8250帧完成，训练专属背景三split一致，PNG/array眼系数control差0。train原版BA及val/test固定点BA完成。光流mask纠正为portrait的255-G三通道；第一轮train flow最终断言误把float32存储与int32+float32提升成float64的参考比较。原数组不改，独立重验前2500刚性轨迹/原版选点，并fresh NNRT重算0/1/2/2500/4500/5999帧逐值一致通过，旧源码和失败log留存。

易失Ubuntu根/dev/shm/synctalk-multispeaker-data-20261017，SSH wjj@192.168.50.2，实际Python /home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python。root盘仅约536MB；SHM16GiB，必须边封存边回收。stage2_worker attempt02 pid217869在Brown pose_qa已失败。查看prepared/SherrodBrown/pose_qualification.json、train_fitted_gauge.json和stage2_SherrodBrown_pose_qa.log。flow缓存已全部流式gzip到F并解压重算raw SHA后删除SHM缓存，tracks/pose输入保留；observer pid115392随后在等待Patrick时随stage2失败停止。stage1远端tar只在F解包全文件SHA后回收。May50旧SHM6051020800字节在F和既有百度盘独立读回验证后已回收，CUDA/GCC依赖保留。

训练外层dataset/audio-context绑定、preload0/1像素/rays等价、完整ray预算资格、训练/渲染/27格评分仍待实施。官方preload1仅float32 RGB+RGBA就44GB，超过31GiB主机；ADR038只允许验证后preload0，不能缩rays冒充官方完整流程。AlexNet LPIPS backbone尚待官方缓存准备。没有选择任何改变AMP/scaler更新策略的方法；必须记录原版实际updates/skips并比较。

## Changed Files
本轮未提交新目录scripts/experiments/synctalk_multispeaker_full_20261017与tests/experiments/test_synctalk_multispeaker_full_20261017.py。重点：preprocessing_contract_v1.json、geometry.py、face_tracker_frozen.py、flow_stream.py、verify_flow_outputs.py、prepare_layout.py、bundle_frozen.py、qualify_pose.py、supervision.py、ave_cache.py、stage2_worker.py、audit_official_source.py，以及流式持久化/空间回收工具。既有stage1/material脚本在同目录。spec已补留出预处理合同，新增ADR038。共享树另有大量无关dirty/untracked文件，完整git status存.codex/scratch/multispeaker_checkpoint_git_status_20261017.txt，不做清理或批量提交。

## Verification
实际Ubuntu环境8个unittest通过，含正/负重复检测、竞争脸门、正确flow alpha、已知相似变换与heldout投影恒等、已知刚性点三角化。Windows首次缺SciPy失败保留。官方固定commit43079fb3f349675f6cb33031bb07262f7566cdd5新ZIP逐文件核查68个Python文件：network/renderer/utils及所有预处理源码精确匹配；仅4个CUDA backend构建兼容修改和provider已有real-test/full-val/verified-AVE绑定差异。F/qualification_state/official_source_audit.json可复核。新源ZIP SHA132e99351402e7087b6585b8331bafcaeb590442aaec4b0712f18b5c7a39cc89。冻结几何合同SHA f472a5a9ef0bf50f2d85bb535407eb83025efe2eca86be2cf294f4e685851e51。

Brownlandmark RMS median/p95 train1.486/1.680、val1.503/1.780、test1.305/1.532px；flow median/p95 train1.025/2.185、val1.734/2.480、test0.771/1.361px；heldout投影恒等最大1.14e-13px。只有gauge旋转/平移失败，不能用其它通过项替代。BFM mat/转换和材料F全量SHA通过；Brown/Patrick stage1逐文件交付通过。Startup Router/实验/任务指令与skills均已读；本checkpoint同chat与legacy搜索无后页，前一checkpoint全文agent/session精确匹配。只读GitHub API确认仓库、无本分支PR，当前SHA不能在GitHub解析，标为本地未推送。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)
- 本地未推送SHA：`2b71639c3b555293375d248350999377168d515d`
- scripts/experiments/synctalk_multispeaker_full_20261017/protocol.json
- F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/material_persistence_verification.json
- F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/stage1_persistence_receipt.json
- .codex/scratch/multispeaker_geometry_diagnosis_codegraph.txt

## Observations
- [result] 四人原材料全部独立QA及持久SHA通过；Brown/Patrick第一阶段全量持久通过；Brown AVE/监督合成/关键点光流通过，新TFG训练数仍为0。
- [decision] 固定训练身份/焦距/背景与训练专属共坐标；65536 rays/FP16完整预算和24共同初始化模型，不因空间缩减实验。
- [decision] 冻结资格阈值不按结果放宽；float32 flow比较修复保留原数组、失败源码/log和独立重算证据。
- [blocker] Brown训练相机相似对齐旋转6.675°及平移14.896mm超过2°/5mm，当前跨录制坐标方法未合格；其余资格和训练runtime未闭合。
- [next_step] 先诊断Brown BFM与BA相机差异、保留失败并实现可独立验证的跨录制坐标绑定，再恢复其余身份预处理和完整训练准入。

## Relations
- continues [[Codex checkpoint - 2026-10-17T09-27-55Z - Four speaker full training spec and official BFM acquired]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[038-synctalk-training-owned-geometry-and-lazy-loading]]