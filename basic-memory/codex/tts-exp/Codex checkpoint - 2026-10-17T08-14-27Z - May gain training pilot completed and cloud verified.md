---
title: Codex checkpoint - 2026-10-17T08-14-27Z - May gain training pilot completed
  and cloud verified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t08-14-27-z-may-gain-training-pilot-completed-and-cloud-verified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 08:14:27+00:00
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

# Codex checkpoint - 2026-10-17T08-14-27Z - May gain training pilot completed and cloud verified

## Summary
May三档等初始化/等更新训练和九格原音轨测试已完成，报告50与Task验收收口，全部原始结果和最终报告已分别通过F盘及百度盘独立读回核验。

## Story
本线程从现成May增益推理和官方编码器归一化审计开始，随后用户要求保存训练spec、准备自然视频数据并核实官方来源与充分性。最新请求明确授权May −6/0/+6 dB同初始化、同训练预算先导，先看学习/验证曲线再做三档测试，统一原始音频评分。这个授权已取代先前“不训练”的限制；当前请求现已完整完成。

沿固定原版SyncTalk/AVE构造受控第一阶段训练，避免把不同初始化、随机采样或AMP跳步混入电平差异。三个生成器从共同随机完整state加载，各60000次成功FP32 AdamW更新、4096 rays/step、max_steps16，官方train_step/loss不改；AVE冻结核验缓存，未用现成May生成器warm start，未做lip/torso/camera阶段。执行次序0/−6/+6，master seed42，固定图像顺序与每步随机域，实际GPU采样ray记录全程相同。固定最终60000 EMA作验证/测试，不按test选checkpoint。

核心发现是训练电平与测试电平都有条件响应，但评分端点与口部误差不同向。九条val MSE曲线均通过预定±2%后段平台筛查，变化−1.57%～−0.57%；这不是数学收敛或数据量充分证明。固定test0的C按训练−6/0/+6为9.199/8.909/8.568，相对训练0为+0.290/−0.341；同模型把test0改+6时C分别+0.079/+0.127/+0.277。固定test0训练−6的D0反而+0.069、训练+6的D0反而−0.098，不能把C排序解释为原时钟更准。全部36个主clip offset−3；主分析后补评同四clip真人参照，C8.809且也均−3，不把共同偏移归因增益训练，也不把高于真人C称为感知优于真人。

## Working State
权威结论为[[50-synctalk-may-gain-training-pilot]]（concluded）；[[训练 May 电平交叉先导]]已done，4步骤全部勾选、status观察恰一条，并从Task Board移除。研究地图、实验README和SYNC-C-SCORES均已读回确认链接50；评分在单列诊断节，不混入AISHELL自然/TTS排行榜。准备Task [[准备 SyncTalk 训练电平先导数据]]同样done；166秒数据充分性仍UNVERIFIED。

May素材原官方42518文件一致，但新时间划分不是官方默认。监督4150/550/1100帧（166/22/44秒），全为自然讲话视频/自然语音标量增益，无TTS或生成监督。已暴露单英语身份、继承全序列pose/背景预处理，不作完全未见/无污染确认队列。正式测试完整1100帧/格，评分四个预定10秒clip；共同GT-owned ROI、FFV1/BGR0、原PCM16、FP32 SyncNet，guard20主/full次，C/Dmin/D0/B/offset完整保留。单seed和CUDA原子梯度不保证训练间逐位复现；重复推理不能估计训练噪声。seed2026、多身份与数据量消融没有执行。

持久根F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017；remote/保存全部原始运行。delivery_manifest.json证明643文件6049784599 bytes与Ubuntu逐文件SHA一致。may_training_pilot_verified.tar.gz为4301034008 bytes，SHA32f6b648e2236693e97a7eb1ea0ce483f80ea30a68cb2508b6893fb090278b22。Windows B盘上传后，Ubuntu独立读回九个云分片、各SHA及按序重组完整SHA全部通过；canonical cloud_readback_receipt.json为attempt05，路径/home/wjj/百度网盘/tts-exp-research-models/synctalk-may-training-pilot-20261017/chunked_attempt02。原整包也留在云端，但只将已完整核验的分片集计为正式云备份。首次可见性超时与慢读替换保留于cloud_transport_attempts.json，科学输入/训练/评分未重跑或改写。

最终报告、spec、代码、曲线和凭证另存may_training_pilot_delivery_metadata_v1.zip：918094 bytes，SHAe13f84eaac66d1eac27a440079365f99cb265bcc598e0afe04a7eff1de8f4e2d，Ubuntu独立读回并逐成员SHA核验49项通过（metadata_cloud_readback_receipt.json）。E盘薄交付在runs/synctalk_may_training_pilot_20261017，包含曲线、九格观看预览、QA与交付receipt。主视频F盘保留，H264/AAC预览不是正式评分输入。

Ubuntu wjj@192.168.50.2 的/dev/shm/synctalk-may-training-pilot-20261017、/dev/shm/synctalk-gain-training-data-20261017/dataset和/dev/shm/synctalk-may-gain-inference-20261016仍作为易失工作缓存保留，SHM不算备份。共同初始tensor SHA15e74e38abec740a7e9790917658da13a0d3a0c0bf426faeb0c119b252eabbc1，实际ray文件SHA1753fd3a19d5ce59a9491bdfa1ee06ccb92292ce4f985862e5e211ab88a7ef7a。训练supervisor历史PID14831已完成退出，备份和读回自有进程已结束，GPU恢复约322MiB桌面占用。只清理了两个自建读回SHM临时目录约5.20GB；训练/数据原件和历史实验未删除，根盘仍不足1GB，共享服务与环境未改。

重要读法：原始experiment_complete.json中的persistent_backup_verified=false和ROOT/status.json的awaiting_delivery是当时生产阶段快照，不被覆盖。现在持久交付应看外层F盘delivery_manifest/cloud_readback/metadata_cloud_readback与done Task，不能据旧快照重复启动实验。原supervisor拒绝复用已执行身份；后续实验必须新run目录和独立协议，不能改这次已封存训练源码。

当前分支仍codex/cross-tfg-audio-head-20261003，本任务代码/测试未跟踪，已有9个tracked修改来自共享工作树其他工作，未全量stage/clean/commit或创建PR。GitHub只读API确认仓库dogLeft233/tts-exp、当前分支PR为空；当前SHA解析422，保留本地证据不构造commit链接。

## Changed Files
- scripts/experiments/synctalk_may_training_pilot_20261017/：冻结协议/共同初始化/训练验证/监督进程、持久同步、九格渲染评分、独立QA、曲线/像素响应/预览、事后真人参照和传输核验；原训练与评分执行源保持封存。
- tests/experiments/test_synctalk_may_training_pilot_20261017.py：不等初始化/预算/rays、缺臂、变评分PCM的拒绝控制。
- basic-memory/docs/experiments/50-synctalk-may-gain-training-pilot.md；实验README、SYNC-C-SCORES诊断节、研究地图、训练Task与Task Board。
- 数据准备代码及封存仍在scripts/experiments/synctalk_gain_training_data_20261017与F:/Datasets/tts-exp/synctalk_gain_training_20261017。

## Verification
三个optimizer最终状态step均60000、EMA60，actual frame/ray schedule相同。独立核验9900测试RGB/原PCM、288保存val浮点帧和最终曲线、36矩阵273420距离及共享裁剪；距离最大误差4.425e−6，端点2.689e−6，曲线MSE差2.274e−11，全部通过。train0/test0首250帧重复的float/RGB/距离逐位相同。主实验失败/排除0；真人事后4矩阵另验最大误差3.589e−6，不并入主分母。

5项控制拒绝unittest实际运行通过；数据准备阶段5项及全量跨端QA见已完成数据Task。网络和loss同官方源码，80源文件运行快照明确捕获于训练期间而非冒称启动前封存，最终逐文件确认不变。全程FP32且每步finite约束，未只凭日志声称成功更新。

视频补充分析（测试前封存）显示每模型±6相对0有1100/1100 RGB和float hash变化，口部8bit绝对变化均值1.194–1.460；固定test0比较训练−6/+6相对0为3.052/2.879。只有像素响应解释，无盲评或几何质量结论。学习曲线/分数热图和九格预览代表帧已视觉检查布局与标签。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)
- 当前SHA仅本地证据：`2b71639c3b555293375d248350999377168d515d`
- 正式报告：basic-memory/docs/experiments/50-synctalk-may-gain-training-pilot.md
- 本地薄交付：runs/synctalk_may_training_pilot_20261017/
- 完整持久结果及云核验：F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/

## Observations
- [result] 用户授权的May三档训练、九格原音轨测试、学习曲线资格、独立QA及持久交付全部完成；报告50 concluded、训练Task done。
- [result] 固定test0的C9.199/8.909/8.568；九条val MSE平台筛查通过，但原D0/口部误差不同向，真人与生成共同−3帧偏移。
- [decision] 同完整随机初始化/实际rays、各60000 FP32成功更新、固定最终EMA，不按test选checkpoint或延长个别臂。
- [decision] 只作单身份单seed工程条件响应，数据充分性UNVERIFIED，SyncNet高分不等于原时钟/感知改善。
- [decision] 原生产marker保持不可变，持久资格由独立外层receipt证明；云传输维修不改科学产物。
- [next_step] 后续若继续确认这次排序，先做第二共同seed复现以估计训练噪声；本次不自动扩训。

## Relations
- continues [[Codex checkpoint - 2026-10-17T03-53-39Z - May data verified and matched gain training running]]
- relates_to [[50-synctalk-may-gain-training-pilot]]
- relates_to [[训练 May 电平交叉先导]]
- relates_to [[准备 SyncTalk 训练电平先导数据]]
- relates_to [[研究主线与证据地图]]