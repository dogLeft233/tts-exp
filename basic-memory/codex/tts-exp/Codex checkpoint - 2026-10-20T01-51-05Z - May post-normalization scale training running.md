---
title: Codex checkpoint - 2026-10-20T01-51-05Z - May post-normalization scale training
  running
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t01-51-05-z-may-post-normalization-scale-training-running
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T01:51:05.603991+00:00'
username: dogle
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

# Codex checkpoint - 2026-10-20T01-51-05Z - May post-normalization scale training running

## Summary
用户授权的新May后归一化尺度实验已冻结、通过14缓存和4真实入口更新/fullstate资格，HuBERT低尺度模型正在从共同完整随机initial正式训练；两模型/12条件结果尚未完成。

## Story
先前报告55完成原录音gain−6/0/+6与native输入norm的72格，提示推理归一化使整体gain响应近消失，但质量指标混合。用户进一步明确此前相对gain并非绝对电平，并要求“试试用低于归一化音量的数据训练，用高于归一化音量的数据驱动”。因此本次固定同HuBERT-base或wav2vec2-base权重，先做完整split的native float32 z-score，再乘0.5/1/2，直接进encoder；不二次归一化、不clip、不改内部LN/GN/模型权重。倍数相对标准化输出，约RMS0.5/1/2，并非原录音±6dB或旧Patrick RMS0.1。

仅新增两低尺度模型（每encoder一组scale0.5），各12000成功FP32更新，总24000新formal；完整random generator initial SHA f9cd6cfadbc72812507e293e660b27e74a61f0556373162c487ffca6f5148e7b、seed42、原AdamW beta0/.99 eps1e−8、60k LR前缀、4096rays/maxsteps16、EMA每1000/decay.95/fixed finalEMA12和实际frame/ray/LR序列保持。两scale1标准训练模型从报告54按完整758成员及fullstate freshSHA核验后原件导入，不重训、不warmstart新训练。每个固定generator交叉test scale0.5/1/2，共12条件24十秒clip；两个standard/standard cell（4clip矩阵）精确复用，新增10cell/20clip/5000帧。新尺度特征已真实提取，不能拿旧归一化前gain缓存冒充。

## Working State
权威持久根F:/Datasets/tts-exp/may_postnorm_scale_20261020。协议SHA0110e9fd52fb4f6fe8c0133e191095b4037a73cd46e8f8496c1672e99513ec08，17新Python源在execution_sources与execution_source_pins冻结，不能修改已执行GPU源或重复运行run.py。入口是workflow_launcher.json、active_controller.json、workflow_failure.json、dispatch、jobs与remote execution。Host解释器为C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/env/Scripts/python.exe；Windows父launch245640/实际host290328为启动时身份，未来看实际文件/进程。远端Ubuntu wjj@192.168.50.2必须SSH独立参数-b 192.168.50.1，避免PowerShell把未引用的-b192.168.50.1错误解析。新remote /dev/shm/may-postnorm-scale-20261020，原GPU flock /dev/shm/synctalk-multispeaker-data-20261017/preparation_GPU.lock。

新remote只读symlink至原may-encoder-norm-pilot的official/data/historical；不改共享venv/CloudDrive。原CUDA扩展因新隔离路径自然重编译，约5分钟内资格尚未发生正式更新，后正常完成，无数值失败/重放。SHM准备时2.1GB可用，新部署约110MB；正式前持久F约38GB可用，E约90MB，不清理旧科学资产。新媒体和全部新fullcp都放F。主机所有job分配唯一远端dispatch marker并detach持有GPU锁，后续只按wrapper PID/startticks及原子execution.json观察；SSH生命周期不作为训练完成，明确不重试不确定的科学dispatch。完成后逐成员F SHA核验，并每2000步持久完整断点；首真实nonfinite/lost/job失败封存state/trace/job并停止所有未来job，不自动replay/restart/换gain/调精度或scaler。

已实际进入HuBERT低尺度正式训练；startup_live_readback.json独立完整1300行有限loss/1300成功/0skip，实际frame/source/ray/LR与标准模型前缀精确。这只是启动前缀，不能当epoch完成。末次active快照1700成功步、loss0.0169217009、84.8秒，后续会自然增长；不得固定旧PID或快照当现状。wav2vec2低尺度正式训练尚未开始，将在前一组自然完成并F全验后运行。源中各模型训练验证每2000，所有fullcp保留。

自动流程run.py顺序为两组资格→独立CPU review→两新正式训练→独立全state/trace review→四个固定generator的新尺度render（跳过两个已导入标准cell）→20新SyncNet矩阵→ffprobe全20视频时钟→本地analysis独立5000帧RGB/PCM/ROI/crop/selectedfloat/距离QA→report候选图与CSV。run.py最后只写numerical_delivery_ready_pending_visual_and_memory，不冒称最终交付。**届时本chat仍须实际view_image三图、核完整12格/交互与candidate、更新BM实验/Task/地图/正式报告，再做fresh全成员manifest和final receipt**；当前尚无新完整训练模型或视频/评分结果，任务不是done。本轮没有创建自动化；后台host会自动到候选交付，勿无授权向别的chat发消息。

## Changed Files
未提交scripts/experiments/may_postnorm_scale_20261020/共17模块：protocol/scaling/features/review_features/deploy/runtime/train/review_training/review_baselines/job_wrapper/host/run/evaluate/review_media_clock/consumer_common/analysis/report。训练源仅相对55改训练feature的postnorm input_scale和固定val尺度，原loss/优化/采样/精度不变，cpu_source_review记录精确diff资格。测试tests/experiments/test_may_postnorm_scale_20261020.py。CONTEXT保持99行、HANDOFF60行当前头部改为本实验运行，旧55交付及旧研究保留；没有stage/commit/push或清理其它dirty文件。机器局部.codex/scratch的build/publish/checkpoint辅助脚本属于未提交部署材料，F源capsule才是本次真实执行依据。

Basic Memory新实验tts-exp/experiments/may-归一化后低尺度训练与高尺度驱动对照implemented/TRAINING，Task tts-exp/tasks/运行-may-归一化后输入尺度训练推理对照in-progress/current_step2。scope_publication_receipt与running_memory_publication_receipt记录四笔记读回，Task只有一个Observations/一个status，地图和板当前段同步，不写逐epoch日志。前一报告55和54及封存F/E不改；旧Patrick coarse7/fine/Chris/Cathy/原24full保持暂停，不重建已删除巡检，不归档chat。

## Verification
4实际CPU测试passed：z-score后调尺度（能检出误做二次标准化）、scale1逐位保持/scale2无clip、坏输入与非有限拒绝、12格完整配对及缺格重复拒绝。17模块AST语法与所有已部署source freshSHA通过，测试文件SHA7d2c59b0266a898d230480e2e93d680e35631aeb0cfd31ad47991f2fc4abe0ff。14缓存独立float32公式/输入SHA/输出shape与finite全部passed；6标准cache和原件SHA exact，8新缓存实际encoder forward，没有再用pre-normalization gain代替。

实测训练低/标准RMS0.4999898681/0.9999797361、test低/标准/高0.4999858041/0.9999716082/1.9999432164，实际峰值可大于1，内部浮点无clip。特征诊断仅描述：训练0.5相对1的relativeL2 HuBERT0.00044009、wav2vec2 0.00963824，不能据此预判质量。feature_scale_diagnostics.json没有额外encoder重算。

两新组真实资格2×2=4success/0skip，model/optimizer/scheduler/EMA/counters/RNG/gradient完整往返与采样通过，随后独立CPU state/trace audit，0GPU重算；资格状态不warmstart正式initial。原父758成员、两个standard fullcp model/optimizer/gradient/EMA finite/step12000/EMA12、远端17403源数据成员726508911 bytes及10支持metadata/原PCM和四核心源freshSHA通过。证据见parent_import_review、baseline_state_review、feature_independent_review、qualification_independent_review、core_and_remote_data_readback、support_metadata_readback。资格不证明完整稳定性。

未运行/尚待：两新组完整12000、20新视频/矩阵、独立最终QA、实际图检和结果发布；没有用候选代码或fixture当结果。原单May/单seed/短预算边界保持，SyncNet与嘴部像素误差不同于人类感知。期间一次只读Python查询错误假定父目录有deployment_receipt.json而FileNotFound，0mutation/0GPU；正确文件runtime_deployment_receipt和actual部署不受影响。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)；HEAD若GitHub不能解析则是本地未推送，不能构造commit链接。
- 前一正式报告55与54，F新根protocol/active_controller/execution_source_pins为恢复入口。

## Observations
- [result] May后归一化scale0.5训练与0.5/1/2推理实验已真实启动；14缓存/4入口更新/fullstate资格passed，HuBERT低尺度首1300正式成功前缀通过，无最终评分结果。
- [decision] 调幅放在同native z-score之后，明确实际RMS/峰值；标准模型与标准cell核验后复用，低尺度模型从同随机initial起，保留原预算和采样。
- [decision] 不恢复Patrick/旧队列，不修改已冻结新GPU源，不自动replay/调精度；新全部cp保留，当前无自动化。
- [next_step] 只观察现有run.py和remote execution推进；完成到numerical_delivery_ready_pending_visual_and_memory后实际看图、核12格交互、发布BM并封存交付，遇首真实故障先审计停止。

GitHub只读核查：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}

## Relations
- continues [[Codex checkpoint - 2026-10-20T01-23-30Z - May low gain normalization interpretation]]
- relates_to [[运行May归一化后输入尺度训练推理对照]]
- relates_to [[May归一化后低尺度训练与高尺度驱动对照]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1068条；大量其他任务改动，不批量stage/clean/commit。
