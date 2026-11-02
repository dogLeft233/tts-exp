---
title: Codex checkpoint - 2026-10-18T02-25-05Z - FP16 preupdate guard qualified and
  upstream density overflow isolated
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t02-25-05-z-fp16-preupdate-guard-qualified-and-upstream-density-overflow-isolated
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 02:25:05.236339+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-18T02-25-05Z - FP16 preupdate guard qualified and upstream density overflow isolated

## Summary
独立 FP16 更新前保护在新旧 Torch 环境各通过10项测试；零更新诊断将持续非有限梯度定位至无界密度指数及其反向，正式重训尚未启动。

## Story
四身份24模型与三轴27格原目标仍未完成。原Brown/+6/ep5失败被完整保留。用户最新明确授权保持FP16、另立版本修复更新前检查、继续追查上游，要求验证后再重训，替代此前等待范围决定的状态。此前官方旧依赖的独立重放也产生NaN loss，并捕获首梯度/密度Inf/低scale/失败state；原适配的非有限loss检查位于optimizer/scaler更新之后，暴露了保护缺口。

新增fp16_preupdate_guard_v1对loss、模型参数与buffer、optimizer state、scale及float32逆scale、实际unscale后的梯度检查，阻止已知非法更新。正常AMP skip保留，默认scaler政策未改；健康4步与原实现模型/AdamW/scaler/scheduler/EMA逐位一致。若原optimizer本身产生非有限值，postcheck仅报警，不承诺回滚；precheck失败后必须终止独立attempt，禁止catch后继续。

新旧环境各12组冻结状态、零optimizer/scaler update诊断：在官方global27043低scale状态，loss仍有限，但sigma指数Inf，有限incoming梯度与Inf相乘产生NaN，另有巨大有限乘积转回FP16溢出。scale .5与1均有28个参数tensor梯度非有限；诊断性detach_sigma后均全部有限，detach_eye_regularizer后仍非有限。两次forward存在CUDA采样差异，故下一步在同一次forward图上重复backward，仅置零密度logit梯度作更严格因果核查。

官方首次密度Inf global26624精确输入点在两环境结果一致：density logit104.0625，sigmaInf；zero_audio仍117.8125/Inf，zero_eye为-41.9375/finite；只换ep4 eye_att_net权重为22.53125/finite，只换ep4 sigma_net为74.1875/finite，只换audio attention仍108.75/Inf。该点眼部attention范数759.9163，证据指向眼部条件与sigma_net增长的相互作用，但不能推断所有帧或+6因果，也尚未证明为什么训练轨迹逐渐增长。无界exp及其梯度是当前已定位传播机制，不是完整解释所有历史skip。

## Working State
ROOT=/dev/shm/synctalk-multispeaker-data-20261017；FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；SSH wjj@192.168.50.2。新产物controller_qualification/fp16_preupdate_revision_v1下current_guard_tests、official_guard_tests、current_upstream、official_upstream；官方upstream session22702已exit0。新产物尚需F独立全SHA持久，合同已F保存。运行脚本ROOT/fp16_revision_v1；原runtime_exec.sh和隔离官方run_official.py保持。

版本合同fp16_preupdate_revision_contract_v1.json：formal_training_allowed=false，formal_optimizer_updates=0，fixture update cap32，upstream zero-update model case cap32。当前fixture实际16更新合计，仅微型Linear测试；TFG更新0。新upstream已24/32案例，剩8预留同图4与真实模型guard4。禁止无预算扩展、FP32或静默改sigma/scaler。

原失败archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72；ep4 SHA0e2223445603b688d6c8feae30e8717bb90dacf66226599bced99306f2eb67f7。旧审计完整F归档preserved_snapshot_02/verification.json为68文件174419530B，manifestSHA2bd65e00cebe72fd5f0aa233edbb4073c51340db4b9107255a6c6da6f9d744f1；analysis_02/evidence_summary.json SHAe68ba8418b3d9e787b15148386741201fb82c5ee04edf759c5051dcf0e360c23。原版本supervisor/observer/job均退出不重启。原实验IMPLEMENTED/EXECUTION_FAILED、四身份数据passed不变。旧报告《SyncTalk FP16 非有限训练失败的适配与官方环境核查》保留。父Task48625B，只改头或另立子卡。现有核查Task blocked待更新为本次授权范围；需记录typed decision和更新map/spec头。实际heartbeat PAUSED，未擅自恢复；其prompt仍需同步本次授权。

## Changed Files
未提交scripts/experiments/synctalk_multispeaker_full_20261017/fp16_preupdate_guard_v1.py SHA f9d05523686e6fd4272f904762add21e5e4dd0c38fc637f199c19b1c7574eb70；tests/test_synctalk_fp16_preupdate_guard_v1.py SHA20d133bcb112859842dec6d42b2b69d98eace8f375332534d956b220e0a4994d；fp16_preupdate_revision_contract_v1.json；diagnose_fp16_upstream_v1.py SHA7656b3f609259084df478b8619debe296a747ae0b82dec0f88408e1db1fae20f。新upstream源hash记录于结果，不能声称在旧合同前已冻结。原math和controlled_training.py不改。仓库1038项dirty含既有工作，禁止整体stage/clean。远程SHM/run低余量，不写大型checkpoint。

## Verification
Torch2.5.1+cu121与1.12.1+cu113各10/10 guard测试通过，含scale2^-128逆数溢出、有限梯度unscale新Inf、非法model/buffer/optimizer/loss、绕过入口、正常AMP skip和健康完整状态逐位对照。两环境各12/12新上游零更新案例完成。未做正式TFG更新、尚未做同图核查/真实模型guard集成，未证明训练稳定。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，GitHub只读当前分支无openPR。
- 本地SHA `2b71639c3b555293375d248350999377168d515d`，未核验已推送。

## Observations
- [decision] 用户接受独立FP16保护修订；原版本与失败不可覆盖，先资格再考虑重训。
- [result] 已知unscale导致非法更新的入口已被测试拦截，正常更新与AMP skip保留。
- [blocker] 上游无界密度指数仍会溢出；保护成功不等于稳定训练合格。
- [next_step] 完成同一forward图的密度梯度干预与真实模型保护集成，随后F全SHA保存并同步版本决策和任务。

## Relations
- continues [[Codex checkpoint - 2026-10-18T02-04-06Z - Official FP16 environment reproduces nonfinite loss with captured states]]
- tracks [[核查 SyncTalk FP16 适配与官方环境]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]