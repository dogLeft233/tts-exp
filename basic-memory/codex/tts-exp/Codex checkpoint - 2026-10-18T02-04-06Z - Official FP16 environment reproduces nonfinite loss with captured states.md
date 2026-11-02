---
title: Codex checkpoint - 2026-10-18T02-04-06Z - Official FP16 environment reproduces
  nonfinite loss with captured states
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t02-04-06-z-official-fp16-environment-reproduces-nonfinite-loss-with-captured-states
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 02:04:06.108353+00:00
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

# Codex checkpoint - 2026-10-18T02-04-06Z - Official FP16 environment reproduces nonfinite loss with captured states

## Summary
官方旧版依赖及原数学源码的独立FP16重放已在global27981复现NaN loss，并封存首梯度异常、首密度Inf及失败前完整state；原失败记录未改。

## Story
四身份24模型和27格原目标仍未完成。用户明确要求先核查新增适配、在官方原环境对照，先不改FP32，保留失败。新审计任务《核查 SyncTalk FP16 适配与官方环境》已建立并读回，父任务和研究地图当前头部已链接。原supervisor105372/observer133984/job610686未重启，所有数字均属独立诊断。

此前current环境5680步重放只捕获梯度异常、未出现非有限loss，12个零更新重复表明同输入同loss的梯度仍非逐位确定。现已完成Ubuntu18.04.5用户空间/glibc2.27/Python3.8.8/Torch1.12.1+cu113/cuDNN8302/GCC7.3/CUDA11.3.109隔离环境。37固定官方源文件逐字节核验，shencoder/gridencoder/raymarching均以原C++14 flags编译并实际CUDA运行。宿主kernel7.0和driver580.178.04保留，不声称完整旧机器复原。Python platform的glibc2.10启发式显示不准确，os.confstr实测2.27。

同捕获state直接调用原Trainer.train_step，在新旧环境均出现scale2^24下MmBackward0 NaN，而unit multiplier FP16梯度有限；这是零更新诊断，不改变训练GradScaler政策。低scale状态的晚期密度点probe均有限，但其density_grid已有66Inf，故不能用晚期权重替代首次产生Inf的现场。官方独立重放采用同ep4完整模型/optimizer/scaler/scheduler/EMA状态，仍固定我们的数据/采样适配以比较环境；不是零适配的官方默认数据管线复现。

官方重放6000上限实际完成3980次，global27981前向loss NaN后按诊断合同在backward前停止。firstscaledgradient global26028、scale16777216；firstoccupancygrid Inf global26624、7Inf、mean_density Inf；firstscale<1 global27043。末global27980 loss .0109028453、scale2.938735877055719e-39、scaledgrad全部有限、实际optimizer成功计数从27823到27824；下一步渲染大面积NaN。需优先查GradScaler reciprocal/unscale溢出及失败state参数，同时用已捕获首密度真实输入定位exp溢出。尚不能将此新重放的链条冒称丢失的原失败步精确现场。

## Working State
ROOT=/dev/shm/synctalk-multispeaker-data-20261017；FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；SSH wjj@192.168.50.2。AUDIT=controller_qualification/fp16_adapter_environment_audit_attempt01。原failure.tar.gz SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72，ep4 SHA0e2223445603b688d6c8feae30e8717bb90dacf66226599bced99306f2eb67f7前后不变。

官方重放路径AUDIT/official_environment_replay，receipt state diagnostic_stopped、completed_attempts3980，SSH session38756已exit0（harness捕获异常）。first_nonfinite_scaled_gradient.pth SHA f6bf3f40079dc8020c88750c6eb85cb84440a7ef8472ae2c53c3bbb600bd794b；first_nonfinite_occupancy_grid.pth SHA a82218a90f28b8e97b47aeb6495aa2243b02aa0072491c2bbf1e1418f5924945；first_scale_below_one.pth SHA e20c5338abe73a9a00dfd951ed0ab70875a98158f317a57b958b08cb760d5949；nonfinite_forward_loss.pth SHA6b25015a17fc66b65bfdc7c9b46f0dfab9618448419f6d2babaf94e68c7b5af4。首密度输入first_nonfinite_density_inputs.pth SHAac72831d0338f24d29c9d26bccbb61a59b348353a08b5e1b6d156ed259410e2a，保存精确x/encoded_audio/eye/sigma，配合同次occupancy完整state（其batch字段是上一步，不能作事件输入）。此批新产物仍需F复制和独立全SHA核验。

预算保守上限12004 optimizer attempts，已用资格4+中止320+current5680+official3980=9984；不可随意使用未用余额重跑。零更新cap24已22，剩余2计划对精确首密度输入做current/official各一组。current/F既有25文件独立SHA已通过diagnostic_preservation_verification_01.json；其他官方probe/新source/inputs/replay尚需后续持久验证。root约160MB/run100MB/SHM余量几十MB，实际df优先，不清共享env/旧数据。

官方launcher /home/wjj/tts-audio/synctalk-official-env-audit-20261018/run_official.py；Python/site/37官方源/扩展build位于/run/user/1000/synctalk-official-env-20261018；cuDNN大库位于ROOT/official_env_audit_torchlibs。均机器局部易失环境。Py1.12 shim仅移除不支持weights_only关键字和step计数接口，完整ep4恢复资格0forward0update passed。原数学源未改。旧新prefix采样/像素/rays一致，LR个别组差8.67e-19，loss微差，不能称逐位重建。

最新实际automation.toml显示synctalk-27 PAUSED，先前工具曾返回ACTIVE，原因未查。用户可见没有暂停请求；不得把状态说成ACTIVE或擅自覆盖可能UI手动暂停，先查证。正式24模型/27格继续停机，数据资格保持。

## Changed Files
scripts/experiments/synctalk_multispeaker_full_20261017下新增诊断v2/v3、probe_fp16_event.py、probe_fp16_occupancy.py、run_official_fp16_replay_v2.py及版本合同v1-v5；v4/导出脚本部分仅remote待持久。diagnose_fp16_failure_v3.py SHA766d906fe17fd8bbb0569942b86af67a08e6dca680d3a7853b18de96a24ab022。其他下载/环境/持久/记忆脚本在.codex/scratch。全部未提交；仓库1037dirty项含他人/既有工作，不整体stage/clean。

## Verification
官方完整state恢复通过；同state首梯度异常新旧环境直接原数学均复现；两晚期密度probe有限；官方3980次独立重放随后真实NaN已捕获。原失败archive与ep4保持。还未核定首密度exp算子及末步unscale/参数破坏的因果链，根因结论待补。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，GitHub只读当前分支openPR为空。
- 本地SHA `2b71639c3b555293375d248350999377168d515d`，未验证已推送。

## Observations
- [decision] 所有对照保持FP16；失败与新诊断分别保留，不改原attempt。
- [result] 旧环境也出现非有限loss，单纯恢复旧依赖不能保证此条件稳定。
- [blocker] 原失败步完整state丢失，新重放可查机制但不等于精确重建原事件。
- [next_step] 用已捕获精确密度输入和末步scaler/参数证据定位首发算子及NaN传播，完成F全SHA持久后更新审计任务。

## Relations
- continues [[Codex checkpoint - 2026-10-18T01-35-21Z - FP16 adapter replay evidence and isolated official environment staging]]
- tracks [[核查 SyncTalk FP16 适配与官方环境]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]