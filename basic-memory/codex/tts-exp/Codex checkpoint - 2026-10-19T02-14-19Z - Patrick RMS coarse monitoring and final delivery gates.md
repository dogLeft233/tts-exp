---
title: Codex checkpoint - 2026-10-19T02-14-19Z - Patrick RMS coarse monitoring and
  final delivery gates
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t02-14-19-z-patrick-rms-coarse-monitoring-and-final-delivery-gates
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T02:14:19.899337+00:00'
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

# Codex checkpoint - 2026-10-19T02-14-19Z - Patrick RMS coarse monitoring and final delivery gates

## Summary
Patrick 的同 AVE 逐录音 RMS 三档独立粗训练已通过入口资格并真实运行；本轮只读核验训练及评测交接进程健康，未启动或重放任何训练任务。

## Story
原目标是检验训练、驱动与评价音量三个轴。Patrick 原未归一化 coarse10 和27格已交付，原 CONTROL_FAILED 不等实际更新仍保留；Brown 原 FP16 失败与四身份24完整模型目标未闭合。用户随后要求暂停旧 fine 并关闭 Ubuntu，已在三档 fine14 完整 state 持久后执行。最新人类授权另立 Patrick 三档粗训练并在完成后评测，明确选择同一个 AVE 前加逐完整录音 RMS；旧 fine、Chris 和 Cathy 继续暂停。

新前端先施加 −6/0/+6 dB，再以 float64 计算完整录音 RMS 并缩放到0.1，float32 进同一原 AVE。无减均值、epsilon、clip 或 limiter，保持原时钟、架构和权重；内部 FLOAT WAV 可超过1，样本读回 exact。各录音三 gain 的归一化样本和特征实际逐位同，仍按授权独立训练三个模型，不能复用0档模型代替三臂执行。原完整随机初始化、seed42、每档 coarse1–10×6000 attempts、原 FP16/defaultGradScaler/guard/loss/AdamW/EMA 和65536 rays/maxsteps16 保持，无旧训练 warmstart、补步或精度降级。

本次 heartbeat 只读复查新合同14冻结源、当前 receipt/execution、实际进程及完整 live trace 前缀。训练健康，评测与交付观察器等待真实终点，无科学结果或代码变化，因此健康进度保持静默。最新运行产物始终优先于此快照和历史 PID。

## Working State
持久根 FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；OUT=FROOT/trials/patrick_ave_rms_coarse_v1。合同 SHA5cc4d22b57a32e5b6c3672d1e782dd40914b33850da5ae9b1e819b34ebb17859；14源在 OUT/sources 冻结，当前重新核验 exact。权威入口为 contract.json、receipt.json、controller_launcher.json、deployment_repair02_receipt.json、posttraining_followups_launcher.json。共同完整随机 initial SHA d6ce1f369bd2ee1695288c93f1602fcf181f0329313c710388e6b7c70071be5c；总180000 formal attempts，资格6次更新不进入正式训练初始状态或计数。

机器局部且易失：Ubuntu wjj@192.168.50.2，SSH 绑定 -b192.168.50.1 防 Meta 代理假在线；ROOT=/dev/shm/synctalk-multispeaker-data-20261017，RUN=ROOT/trials/patrick_ave_rms_coarse_v1，隔离工具链 /dev/shm/synctalk-may-gain-inference-20261016。当前 boot a257ef30-7bdc-406b-aaa1-891b50b0dacf。共享 venv 与 CloudDrive 不修改，SHM 不当持久备份。

本轮最新只读快照：Windows controller217680 status training，gain0/coarse1，current.relative=training/PatrickLeahy/seed42/gain_+0/coarse/ep0001/attempt01；remote job15249/flock child15259 实际 running，commandSHA a48d5ea19963c0a68676b0dbd49cb395b2dbda4b12c16da1c43ca35f38f8c8e2。完整 live 前缀4403 attempts/4403 successes/0 skips，loss 全有限，末行 global4403/loss0.00017949506582226604/scale262144；这是前缀，不是完整 epoch。startup_live_prefix_readback.json 是更早冻结的1130行，不能冒称最新进度。

score handoff190252 waiting_for_actual_coarse10，score_jobs_started0；delivery handoff219064 waiting_for_actual_new27，passed=false 表示等待。三 Windows 进程均存在，未发现 controller_failure.json、score_handoff_failure.json 或 delivery_handoff_failure.json。PIDs 会随真实作业轮换，不能用硬编码旧 PID 的 ps 非零断言训练失败。

只有三档真实 coarse10 才写 coarse_ready_for_27_grid.json 并冻结 score_27_v1 的三份真实 fullcpSHA/EMA60/core/data/source/helper 及精确 ready。score_handoff_candidate_v1/candidate_receipt.json 绑定20候选源；patrick_rms_score_handoff_v1.py SHA7684937819965c7de85ccaf3410ae54e58e28b040d1d55b42cc6b5572ba8272e。先9固定帧重复 FP16 与 FFV1/PCM 像素资格，再9新视频/27格/162矩阵。保持 source-owned ROI guard20 主/full 次，0 PCM16/±6 float64 原单位，SyncNet FP32/TF32off，原时钟，H264 只预览；F全成员和 cloud 全包独立 Linux 读回后才窄退休媒体。

delivery_candidate_v1/candidate_receipt.json 绑定5源；patrick_rms_delivery_handoff_v1.py SHA2b70b7129fe8a777feb1639f3fdfd7438a70d3c6cff741a22d634a22a10c6c7b。实际新27后按列 float64 重建162矩阵、六clip等权、PNG/head/PCM及scorearchive/cloudreceipt检查，并输出 score_27_v1/delivery_v1/ 的CSV、曲线、patrick_27_grid.png、report.md和候选交付manifest。到 numerical_delivery_ready_pending_visual_and_memory 时，本chat仍须 view_image 真实图、核候选receipt/manifest、发布BM结果并收口新Task，才通知最终交付。候选代码和合成fixture不算结果。

首 terminal/lost 须封存后停止本新版本全部新job，不自动换gain/replay或盲重启。每闭合epoch完整F验证后，只有新同臂完整state核验才允许轮转中间cp；final/异常/全部trace保留。原旧coarse CONTROL_FAILED、Brown failure SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72 与旧fine14完整state不动。原24full/fine17未完成，不归档chat或恢复旧队列。

## Changed Files
本轮未修改冻结训练/评分/交付源，只有只读监测和此次handoff。既有本任务未跟踪源码在 scripts/experiments/synctalk_multispeaker_full_20261017/：ave_rms_normalization_v1.py、prepare_patrick_ave_rms_v1.py、patrick_rms_runtime_v1.py、qualify_patrick_rms_entry_v1.py、train_patrick_rms_epoch_v1.py、patrick_rms_controller_v1.py，以及 coarse_preview_rms_v1.py、coarse_preview_rms_controller_v1.py、patrick_rms_score_receipts_v1.py 和新RMS score/delivery handoff、review/plot/report。对应两个测试文件在 tests/experiments/test_ave_rms_normalization_v1.py 与 test_patrick_rms_score_receipts_v1.py。恢复和独立审计源在 .codex/scratch/ 与 OUT 对应receipt目录，不能把机器局部未提交文件当已推送代码。

## Verification
本轮 read-only 合同及14源 SHA 精确，三活动host进程与remote执行状态健康，4403行 live trace 记账与finite loss通过。没有GPU重算、checkpoint全tensor重载、重新资格、重启controller或重复epoch。

前序已执行、此次未重跑：3 RMS CPU tests 与3 score receipt拒绝测试及新源 py_compile 通过；六条件各两次实际AVE repeat exact，独立CPU归一化公式 maxerror0。三gain×2真实FP16入口为6success/0skip，full model/optimizer/scheduler/scaler/EMA/counters往返及采样exact；qualification/independent_receipt_review.json fresh13 archive成员+3 standalone fullcp/rawresult/trace passed。入口资格不证明完整训练数值稳定性。41447原inputpins、stage1 archive41251和derived16562全成员、8250原公式mask恢复exact证据见 full_member_restore.json、stage1_support_restore.json、restore_continuation_receipt.json。

旧bootstrap_failure.txt 是import缺原geometry、trainer构造前0attempt/0optimizer的首次准备失败；deployment_repair02补齐原依赖后同合同14源的attempt02资格job13408成功。保留原失败，后续repair02/controller_launcher优先。命令长度、stage1 tar前缀和只读liveprobe quoting失败也保留，不是当前formal failure，不能据此重复训练或覆写独立review。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)：本轮只读API确认canonical URL、当前branch无PR，HEAD远端解析422，故不构造commit链接。
- 当前Task tts-exp/tasks/运行-patrick-的-rms-归一化-ave-三档粗训练并评测：in-progress/current_step2；实验与ADR048保持最新授权。仅当前头部/唯一observations，不逐epoch写文档日志。
- 详细实现、资格/恢复失败及旧暂停证据见上一同chat不可变checkpoint及OUT权威文件。旧coarse27 SHA0c727888b1fbeb09e573ab26bc7129d453aa03b45de345b460d66973ac6fac1a供未来新减旧描述比较，不能当新结果。

## Observations
- [result] 本轮合同14源与实际训练/后处理进程只读核验健康；4403/4403/0有限loss是live前缀，没有新闭合epoch或27格分数。
- [decision] 仅推进用户明确选择的同AVE逐录音RMS0.1新Patrick三档粗训练及新评测；旧fine14/Chris/Cathy持续暂停。
- [decision] 健康进度静默，仅最终真实交付、真实故障或确需输入通知；不把SyncNet变化外推感知改善或总体训练gain因果。
- [next_step] 继续按实际receipt/execution/ps核验闭合epoch与自动score/delivery交接；真实新27候选交付后实际看图并发布BM结果/收口新Task。

## Relations
- continues [[Codex checkpoint - 2026-10-19T02-06-50Z - Patrick RMS AVE coarse qualified and training running with evaluation handoffs]]
- relates_to [[运行Patrick的RMS归一化AVE三档粗训练并评测]]
- relates_to [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- relates_to [[048-synctalk-patrick-same-ave-rms-coarse]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1055条；大量其他任务改动，不批量stage/clean/commit。
