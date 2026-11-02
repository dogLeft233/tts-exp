---
title: Codex checkpoint - 2026-10-18T02-42-58Z - FP16 guard evidence preserved and
  identity scope clarified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t02-42-58-z-fp16-guard-evidence-preserved-and-identity-scope-clarified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 02:42:58.852028+00:00
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

# Codex checkpoint - 2026-10-18T02-42-58Z - FP16 guard evidence preserved and identity scope clarified

## Summary
双环境FP16保护和同图密度梯度诊断已完成并持久保存；持续溢出的身份范围仅证实Brown，其余三人未进入完整训练，不能推断其正常或必然失败。

## Story
原目标仍是四身份、两seed、三训练gain的24完整模型与训练/驱动/评分27格。Brown seed42 +6 coarse ep5失败后，用户要求官方原环境核查，并明确接受独立FP16更新前保护版本，先验证再重训，保留原失败和原实验。官方旧依赖对照也复现非有限loss；原scaler在2^-128时float32逆scale溢出，可让原本有限scaled gradient在unscale产生非法值而未被旧检测拦截。

独立fp16_preupdate_revision_v1保留原数学和正常AMP skip，检查loss、state、scale倒数和实际unscale梯度，拒绝已知非法更新。双环境各10/10单元测试、实际ControlledTrainer trace/state对照和真实NeRF拒绝均通过。postoptimizer检查不是回滚；拒绝后禁止同attempt继续。正式epoch/supervisor尚未接线或重训，保护合格不等于完整训练稳定。

后续同一次forward图仅置零密度logit梯度，两环境非有限参数梯度tensor均由28降至0，保留其他64维geofeature梯度。冻结首发点眼部输入置0或换ep4 eye_att权重可消除该点exp溢出；只置零audio仍溢出。CPU只读检查确认Brown三gain眼部数组SHA相同，七维输入全有限且捕获行与官方provider映射精确，eye_att/sigma权重沿已有state增长。尚未隔离是哪项loss/空间区域驱动增长，不能宣称上游根因完全查清，也不能把局部诊断当作正式修复方案。

用户现在问是否只有一个人的梯度有问题。已核对父Task和正式报告：正式完整训练只推进至首人Brown seed42，+6发生持续溢出并失败；0与-6 ep5各6000attempt/5998成功，也有AMP跳步。PatrickLeahy、ChrisVanHollen、CathyMcMorrisRodgers仅完成数据/加载/共同初始化等准入，尚无同等完整训练梯度轨迹。官方旧环境复现仍是同一Brown，不能算第二身份。当前应回答“只在一个身份上证实，其他三人未知”，不能给1/4失败率。

## Working State
权威最新结果为 [[SyncTalk FP16 更新前保护资格与密度溢出因果核查]] 与 [[044-synctalk-fp16-preupdate-guard-revision]]，早先02:25 checkpoint是未完成同图/控制器验证时的旧快照。F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/controller_qualification/fp16_preupdate_revision_v1/preserved_snapshot_01保存38文件259505B，manifest SHA92016a548dca0ce27ca5ddf060f111396380f3f6f9c8df096e19fcf191f6e71e；readonly_supplement01保存CPU只读结果、源和最终文档pins，result SHA1a4de96f85b3e50fdb339bf69f0000dfde6d722d172fdd0878bda40ecde93bfe。原failure.tar.gz SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72保持。

本修订预算实际32/32零TFG更新模型案例，24/32微型fixture optimizer更新，0TFG optimizer/scaler更新；旧审计9984/12004尝试独立计数。下一轮数值工作需新冻结有界诊断合同，不能悄悄扩充已用完案例。原supervisor/observer退出不重启，实际heartbeat PAUSED已保留且prompt已同步，无新正式训练。ROOT=/dev/shm/synctalk-multispeaker-data-20261017，SSH wjj@192.168.50.2，缓存易失且空间紧张；其他三人仅部分监督cache，加载前必须显式全量恢复并验证。

## Changed Files
先前完成未提交源位于scripts/experiments/synctalk_multispeaker_full_20261017/：fp16_preupdate_guard_v1.py、controlled_training_fp16_v1.py、diagnose_fp16_upstream_v1.py、qualify_fp16_guard_real_model_v1.py、audit_fp16_eye_state_readonly_v1.py、qualify_fp16_controller_fixture_v1.py及冻结合同；对应tests/test_synctalk_fp16_preupdate_guard_v1.py与安全显式controller入口。原controlled_training和官方数学未改。原执行fixture字节另保存在F，改为安全collection入口后仅验证导入，不冒称新入口重跑GPU。

## Verification
报告记录两环境各10/10 CUDA测试、同图梯度干预、真实state拒绝、控制器update/skip/update精确状态与trace、F逐SHA验证。旧环境controller第一次因缺importlib_metadata导入失败，隔离补依赖attempt02通过；原析构器log_ptr警告保留。六源AST、目标diff检查及安全入口import通过。此次回答只做记忆/文档与Git核查，无新GPU测试或训练。Git当前有1039项dirty，含大量既有工作，未stage/clean/commit。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，只读GitHub查询当前branch无openPR。
- 本地HEAD `2b71639c3b555293375d248350999377168d515d`，未验证已推送，不构造commit链接。

## Observations
- [decision] 保持FP16，原失败和原版本不可覆盖；独立保护已资格，完整稳定性门仍未通过。
- [result] 持续梯度溢出只在Brown得到实测和双环境复现，其他三人的完整训练梯度状态未知。
- [blocker] 同图因果定位密度梯度传播，但促成眼部/密度权重增长的loss与区域贡献尚未隔离。
- [next_step] 基于现有冻结状态设计并冻结最小loss/区域梯度归因对照，保持FP16和独立预算，稳定性资格前不启动24模型重训。

## Relations
- continues [[Codex checkpoint - 2026-10-18T02-25-05Z - FP16 preupdate guard qualified and upstream density overflow isolated]]
- tracks [[核查 SyncTalk FP16 适配与官方环境]]
- relates_to [[SyncTalk FP16 更新前保护资格与密度溢出因果核查]]
- relates_to [[044-synctalk-fp16-preupdate-guard-revision]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]