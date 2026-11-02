---
title: Codex checkpoint - 2026-10-19T02-06-50Z - Patrick RMS AVE coarse qualified
  and training running with evaluation handoffs
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t02-06-50-z-patrick-rms-ave-coarse-qualified-and-training-running-with-evaluation-handoffs
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T02:06:50.197048+00:00'
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

# Codex checkpoint - 2026-10-19T02-06-50Z - Patrick RMS AVE coarse qualified and training running with evaluation handoffs

## Summary
用户明确选择同AVE加逐录音RMS，独立Patrick三档coarse10已经真实启动，归一化与六次FP16入口通过，后续27格及独立报告观察器已运行等待真实终点。

## Story
原四身份完整训练因Brown FP16失败仍未完成；Patrick未归一化coarse10/27格已交付，旧fine后按用户要求完整暂停在三档epoch14并关闭Ubuntu，旧断点和所有失败原件F保留。用户于2026-10-19重新授权“加入音量归一化编码器运行Patrick粗训练，仍三档，完成后评测”，并明确从RMS/LUFS/换HuBERT中选同一AVE加RMS。这是独立实验，不恢复旧fine/Chris/Cathy。

Ubuntu现新boot a257ef30-7bdc-406b-aaa1-891b50b0dacf、V10016GB可用，原SHM空。将原工具链/运行时、数据/资格/初始化从持久证据恢复到SHM，按旧41447个pins逐字节核验而非降低旧门。新前端先gain再逐完整录音float64 RMS→0.1、float32进原AVE，保持原时钟、架构与权重；六条件各两次完整前向及独立CPU公式复核通过，main/test各三gain波形样本/AVE特征均逐位相同。仍独立运行三个训练，不复用0档模型。

真实每gain两次FP16入口共6success/0skip，完整model/optimizer/scheduler/scaler/EMA/counters roundtrip exact、实际采样/trace同，原完整随机initial未改；入口不证明完整数值稳定性。F fresh独立核13archive成员+3standalone checkpoint/rawresult/trace。正式coarse10控制器和两个后处理观察器现真实运行，0新评分结果；已授权工作持续推进，不因正在训练而停止或误报完成。

## Working State
FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；OUT=FROOT/trials/patrick_ave_rms_coarse_v1；ROOT=/dev/shm/synctalk-multispeaker-data-20261017，RUN=ROOT/trials/patrick_ave_rms_coarse_v1。contractSHA5cc4d22b57a32e5b6c3672d1e782dd40914b33850da5ae9b1e819b34ebb17859，14源精确冻结在OUT/sources并部署读回。coarse seed42、0/−6/+6每epoch6000，1–10各60000/arm、总180000formal；原共同完整随机initial SHA d6ce1f369bd2ee1695288c93f1602fcf181f0329313c710388e6b7c70071be5c，非任何旧训练warmstart。FP16/defaultGradScaler/guard、loss/AdamW/EMA、65536rays/maxsteps16、原scheduler/数据/随机域保持；无补步/precisionfallback/scalertune/limiter。归一化目标0.1（−20dBFS），不减均值/epsilon，拒绝退化输入；内部浮点波形可超过1，不做clip，FLOAT WAV样本回读exact。

正式host217680，当前0gain/coarse1/job15249/child15259，commandSHAa48d5ea19963c0a68676b0dbd49cb395b2dbda4b12c16da1c43ca35f38f8c8e2；真实receipt/current.relative/execution/ps优先，PID会轮换。startup_live_prefix_readback.json frozen已核1130attempts/1130success/0skip、loss有限，首成功global1，scope仅完整live前缀不是epoch完成。controller逐闭合epoch F全archive成员+完整checkpoint/trace持久，并核实际采样/LR/success；首terminal/lost封存停止本版本全部新job，无自动重放；中间state仅新同臂完整state核验后轮转，final/异常及全部逻辑trace保留。

score handoff190252，源OUT/score_handoff_candidate_v1/patrick_rms_score_handoff_v1.py SHA7684937819965c7de85ccaf3410ae54e58e28b040d1d55b42cc6b5572ba8272e，status waiting_for_actual_coarse10；delivery handoff219064，源OUT/delivery_candidate_v1/patrick_rms_delivery_handoff_v1.py SHA2b70b7129fe8a777feb1639f3fdfd7438a70d3c6cff741a22d634a22a10c6c7b，status waiting_for_actual_new27。OUT/posttraining_followups_launcher.json为实际启动证据。候选未通过真实final渲染资格，不把代码/fixture当实验结果。3真实coarse10才coarse_ready_for_27_grid；handoff将冻结3fullcpSHA/EMA60/core/data/source/helper、精确复制ready，9固定帧重复FP16/FFV1PCM像素资格后9新视频/27格/162矩阵。原source ROIguard20主/full次、0PCM16/±6float64原PCM单位、SyncNetFP32/TF32off/原时钟、H264只预览。cloud整包独立Linux回读及F全验前不退媒体。final delivery观察器实际162矩阵按列重建/等权聚合/PNG head PCM及scorearchive/cloudreceipt检查后产CSV/图/新减旧报告，留visual_chart_review和BM publication待本chat实际完成。不可盲重启任何控制器。

## Changed Files
新源在scripts/experiments/synctalk_multispeaker_full_20261017/：ave_rms_normalization_v1.py、prepare_patrick_ave_rms_v1.py、patrick_rms_runtime_v1.py、qualify_patrick_rms_entry_v1.py、train_patrick_rms_epoch_v1.py、patrick_rms_controller_v1.py；对应测试tests/experiments/test_ave_rms_normalization_v1.py。原训练数学与冻结源未改，版本化selector只改变数据入口。

评分/交付候选为coarse_preview_rms_v1.py、coarse_preview_rms_controller_v1.py、patrick_rms_score_receipts_v1.py（对应3测试）、patrick_rms_score_handoff_v1.py、final_review_patrick_rms_v1.py、plot_patrick_rms_v1.py、report_patrick_rms_v1.py、patrick_rms_delivery_handoff_v1.py；F候选目录分别20和5源SHA绑定。report只在真实结果后出文件，最后仍需view_image图、发布BM/收口新Task并一次性通知结果。不得对原旧coarse表再通知。

恢复/部署/CPU QA脚本在.codex/scratch/*patrick*rms*20261019.py，F OUT内保留receipt与独立QA源。所有这些仓库代码仍为本机未跟踪文件，dirty tree大量其他任务，不批量stage/clean/commit。Gitstatus --short本轮1055条、既有9tracked修改；branch codex/cross-tfg-audio-head-20261003，HEAD 2b71639c3b555293375d248350999377168d515d。无新PR/提交。

## Verification
3RMS CPU tests（增益抵消/波形比例/退化拒绝/无clip去均值）+3score receipt拒绝测试（旧非RMS、错阶段gain-driver checkpoint EMA帧序）通过，新源py_compile通过。实际normalized AVE全部repeat exact，independent_normalized_input_review.json全六条件公式maxerror0、原样本数/特征形状与SHA通过，0额外GPU。资格job13408/attempt02真实6更新和fullstate往返通过；13 archive成员和3 fullcp freshSHA、rawresult物理字节/跨gain trace精确，0GPU重算/全tensor重载。普通训练当前仅前缀，不冒称任何完整epoch。

restore_continuation_receipt.json、stage1_support_restore.json及full_member_restore.json支持全部41447 pins、stage1 archive41251/derived16562全成员和8250原公式masks字节同。原官方4模型执行源、15core源、protocol/admission/初始RAW state与旧数据SHA均复原，未改共享venv/CloudDrive配置。

保留准备失败：初restore命令行长度限制、stage1 tar目录/身份前缀拒绝，均0optimizer；manifest文件+严格路径映射修复。bootstrap_failure.txt对应最初入口job12546的import缺geometry，trainer/dataset构造前0attempt/0optimizer；deployment_repair02补齐同原geometry依赖后同14源/同contract新attempt02 job13408，不重放旧job或optimizer，deployment_repair02_receipt/controller_launcher优先。首次只读prefix quoting失败也另存live_prefix_probe_attempt01_failure.json；资格独立审核本身已通过，后分离prefix读取修复，不覆盖原review。勿将旧准备failure当当前训练失败。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)；本轮只读GitHub API确认canonical仓库/branch PR空。HEAD远端不可解析，保留本地SHA `2b71639c3b555293375d248350999377168d515d`，不构造commit链接。
- OUT/contract.json、receipt.json、controller_launcher.json、posttraining_followups_launcher.json、qualification/result.json、qualification/independent_receipt_review.json、independent_normalized_input_review.json、startup_live_prefix_readback.json。
- Task tts-exp/tasks/运行-patrick-的-rms-归一化-ave-三档粗训练并评测、Experiment tts-exp/experiments/patrick-同-ave-逐录音-rms-归一化三档粗训练与-27-格评测、ADR048 tts-exp/docs/adr/048-synctalk-patrick-same-ave-rms-coarse已回读；Task current_step2/in-progress，地图更新。

## Observations
- [result] RMS真实六条件/独立公式、六次真实FP16更新/fullstate入口及独立receipt QA通过，正式coarse1运行，后处理观察器已启动等待真实终点。
- [decision] 用户选择同AVE+RMS0.1；保留旧fine14/Chris/Cathy暂停，独立新模型/预算/评测，不合并旧失败或用q更新warmstart。
- [decision] 新phase按实际更新验收，不补步；原coarse CONTROL_FAILED/Brown失败与旧fine14全state保持。当前0新27/感知结果，单身份单seed不推总体因果。
- [blocker] 三档coarse10、新render资格/27格、独立最终报告/图目视与BM发布尚未完成；原24full/fine17未完成。
- [next_step] 以最新receipt/execution/ps核验217680及后处理190252/219064的真实闭合epoch和最终handoff，保持健康静默；真实新27及CPU报告后实际view_image并发布结果/收口新Task，不恢复旧训练。

## Relations
- continues [[Codex checkpoint - 2026-10-19T01-29-30Z - Patrick fine safely paused and Ubuntu powered off with new normalized coarse experiment authorized]]
- relates_to [[运行Patrick的RMS归一化AVE三档粗训练并评测]]
- relates_to [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- relates_to [[048-synctalk-patrick-same-ave-rms-coarse]]
- relates_to [[暂停 Chris 并精调 Patrick 三档及27格]]
