---
title: Codex checkpoint - 2026-10-18T03-04-19Z - Other identity FP16 trial running
  with controller handoff preserved
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t03-04-19-z-other-identity-fp16-trial-running-with-controller-handoff-preserved
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 03:04:19.998141+00:00
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

# Codex checkpoint - 2026-10-18T03-04-19Z - Other identity FP16 trial running with controller handoff preserved

## Summary
用户新授权的Patrick/Chris/Cathy独立FP16粗训练已启动Patrick真实更新；受保护的原失败版本保持，主机路径修复以单独接管源保存且未重启GPU。

## Story
原目标是四身份两seed三训练gain的24完整模型及训练/驱动/评分27格，Brown在+6 coarse ep5出现非有限loss后失败。用户先要求核查适配与官方环境，随后明确保持FP16、另立更新前保护版本。保护资格和官方环境对照已完成，但眼部/密度权重为何增长仍未完全隔离，保护不等于上游数值修复。用户最新要求先把Patrick、Chris、Cathy训练跑起来，授权独立身份诊断训练。

本轮固定PatrickLeahy→ChrisVanHollen→CathyMcMorrisRodgers，各seed42、0/-6/+6dB、原coarse10epochs×6000attempts，共最多540000；fine和seed2026不在本轮。保持FP16/defaultGradScaler、65536rays/maxsteps16、原loss/AdamW/EMA及每人共同完整随机初始化。独立入口接入已资格guard，首次AMPskip保存完整真实state/gradient，terminal失败先写原因再捕获state。首terminal失败停止该身份剩余条件，完整封存后推进下一人；lost job停止控制器且禁止重播。正常phase末原实际成功更新配对门仍适用，不补步，不将试跑算原24完成。

原host控制器的Windows Path拼接远端checkpoint路径问题在首epoch结束前发现；独立controller_revision02 wrapper用PurePosixPath修复，冻结原7源不改。仅停止Windows166604，保留旧receipt并用164828接管同一个运行job709978/child709985/训练709986。GPU进程未停，未重放更新。新wrapper只允许此首次epoch接管情境；不能当通用恢复器反复运行。

## Working State
权威Task [[试跑 Patrick Chris Cathy 的 FP16 粗训练]] 与 ADR045。OUT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1，remote RUN=/dev/shm/synctalk-multispeaker-data-20261017/trials/fp16_other_identities_trial_v1；SSH wjj@192.168.50.2。Windows164828及remote709978/709985/709986为易变状态，应读receipt.json、controller_revision02日志、jobs execution与实际ps，不盲重启。当前Patrick seed42 gain0 coarse ep1真实训练，Chris/Cathy待运行。02:59:03UTC持久前缀1897成功/0skip，SHA72c102dca1da05b54da6da3f74c7a68ba0a9d69c9d07e1d046550e8314960ed4；稍后实时tail3832成功/0skip。均不是完整epoch或身份稳定性结论。

独立contract SHA787d917d259eeca89311016737893348578335e3623f217858797926311893bd；7源跨端精确，原12host sourcepins核对。V2wrapper SHA4a03e7c1cf501d89c97207bcc07ee14035348c61a595a1c7f1a2a4e3eaf41d38，handoff.json单独记录不改原contract。源与qualification、state、trace、曲线持久保存在OUT；远端SHM易失。自动化synctalk-27实际PAUSED沿用，独立控制器自行工作，不冒称heartbeat ACTIVE。

Brown16500监督已按F完整归档全SHA+每成员fresh验证后，对16428非控制GT/torso持独占Linux lease退休，72controls和原素材/模型/失败保留；以后Brown输入诊断必须full恢复。Patrick full16500恢复passed，接管再次全部逐SHA验证。Chris/Cathy仍需控制器显式full恢复。当前空间紧张，后续每人切换仍先独立归档全验再退休缓存，不能用旧observer。中间checkpoint仅新同臂完整state持久验证/消费后轮转，coarse final/异常state与全实际trace曲线保留。

原failure archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72不变；原supervisor105372/observer133984/job610686不重启。原科学任务仍未完成，四身份数据准入仍passed。Brown密度exp/眼条件因果证据及官方旧环境复现见既有报告，不将此轮暂时无skip当作Brown根因证明。

## Changed Files
scripts/experiments/synctalk_multispeaker_full_20261017/新增other_identities_trial_v1.py、other_identities_trial_remote_v1.py、controlled_training_trial_v1.py、train_other_identity_epoch_v1.py、qualify_other_identities_trial_v1.py、other_identities_trial_contract_v1.json、other_identities_controller_handoff_v2.py；tests/test_synctalk_other_identities_trial_v1.py。仓库源未提交，F及remote冻结源为本次执行实据。新增Task/ADR045并更新研究地图、TaskBoard和原核查Task；本次更新当前Task头部及ADR测试数，F controller_revision02/documentation_snapshot保存读回和测试源码pins。

## Verification
新controller资格fixture实际4微型optimizer更新，原vs新update/skip/update trace及完整model/optimizer/scaler状态精确，firstskip和terminal capture通过；7资格文件独立SHA复制验证。3CPU测试通过：terminal后继续下一人且保留失败、lost job禁止重放、Windows生成POSIX远端checkpoint路径。训练入口与原入口diff审核、7源AST/跨端SHA和12原host pins通过。原失败archive再校验不变。没有重跑全仓库tests；Git有1040项dirty包含大量既有工作，未stage/clean/commit。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)；GitHub只读查询当前branch无openPR。
- 本地HEAD `2b71639c3b555293375d248350999377168d515d`未验证已推送，未构造commit链接。

## Observations
- [decision] 用户最新授权先三身份独立FP16粗训练，保留原失败和原实验；正常AMPskip策略不改，非法更新由已资格guard拒绝。
- [result] Patrick真实FP16更新正在运行，持久1897步前缀全部成功且0skip；Chris/Cathy排队，不能称其已通过。
- [verification] 主机路径修复以独立V2接管原remote job，未中断GPU或重放更新，原7源冻结不变。
- [blocker] Brown上游权重增长根因仍未完全查明；三身份完整粗阶段结果和首epoch持久化均待实际回执。
- [next_step] 核验Patrick首完整epoch独立F持久化与controller后续gain调度，按真实失败/完成回执继续Chris/Cathy；不得重复启动现有job。

## Relations
- continues [[Codex checkpoint - 2026-10-18T02-42-58Z - FP16 guard evidence preserved and identity scope clarified]]
- tracks [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- relates_to [[045-synctalk-other-identities-fp16-diagnostic-training]]
- relates_to [[核查 SyncTalk FP16 适配与官方环境]]
- relates_to [[SyncTalk FP16 更新前保护资格与密度溢出因果核查]]