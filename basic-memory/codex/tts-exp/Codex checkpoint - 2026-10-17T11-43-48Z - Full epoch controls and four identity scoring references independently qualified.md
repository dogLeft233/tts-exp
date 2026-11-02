---
title: Codex checkpoint - 2026-10-17T11-43-48Z - Full epoch controls and four identity
  scoring references independently qualified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t11-43-48-z-full-epoch-controls-and-four-identity-scoring-references-independently-qualified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 11:43:48.470055+00:00
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

# Codex checkpoint - 2026-10-17T11-43-48Z - Full epoch controls and four identity scoring references independently qualified

## Summary
Brown全量数据的独立F读回、原loader90项/资源18项通过；原完整训练状态恢复、持久轮换和四人72路GT评分核验完成，训练/渲染/27格代码已补齐，全四人准入尚未完成，实验TFG更新仍0。

## Story
用户从May先导扩展到四身份各420秒自然HDTF、同人不同原录制测试，已授权所有门通过后执行4×两seed×三增益24个官方完整模型和三driver×三scorer27格。完整预算FP16/65536rays、10+7完整epochs累计102000 attempts、同完整随机generator初始化与同实际成功更新要求不变；原AMP skips如实记录，阶段末不等就CONTROL_FAILED，不改scaler或补步。ADR038/v1坐标失败保留，ADR039冻结训练共享参考点v2不放宽阈值。

本轮把预处理、资源、训练控制与真实实验结果分开。完整state恢复用原Trainer.train_one_epoch和toy参数实际验证：保存完整官方checkpoint后，恢复optimizer/scheduler/scaler/EMA、renderer非注册counter以及post-val EMA collected，继续运行逐state等同连续运行；fine新constructor原counter reset和coarse当前LR保留。真实CUDA overflow测试确认只有实际AdamW调用计入更新，finalEMA必须应用后深拷贝再restore。Windows到Ubuntu持久传输/逐成员读回/旧checkpoint cache安全轮换用合成bytes实际走完，未训练TFG。

完整执行合同已加spec，outer coordinator逐完整epoch轮换三档并独立比较实际frame/ray序列，阶段末预算不等停止。原test_step、portrait blend和uint8截断用于正式渲染，FFV1/原PCM16保真；评分代码复用固定source ROI和三评价gain音频特征，独立NumPy重建全部距离及端点，等权聚合六clip→seed→身份→四身份。代码存在不代表这些正式阶段已运行。训练coordinator当前没有自动中断续跑入口；若进程故障，应依据已有receipts/锁定源补版本化恢复，不能重开覆盖。

## Working State
持久根F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；易失ROOT=/dev/shm/synctalk-multispeaker-data-20261017，SSH wjj@192.168.50.2。Task与spec已实际BM更新并read_note回读，分别in-progress与IMPLEMENTED部分准备，均不是结果完成。先读Task、spec、ADR038/039及真实JSON。四人原材料/stage1全部8250帧/F每人41251成员/score references通过；Brown五脸部副框人工裁决已fresh图片/lmsSHA再验，其余三人无flags。Patrick原test482平方上采样/Cathy原4满幅样本与预训练暴露限制保留。

Brown v2/监督/AVE已全通过，F三个归档全部fresh SHA/完整解压逐成员复验和数值门通过，见data_identity_preflight/SherrodBrown/verification.json。Brown监督仅恢复72控制文件，full训练前仍须16500文件恢复。90项原provider preload0/1 pixels/rays/条件/整录制attention等价、18项原FP16/backward资源probe通过；lip原矩形3844rays，粗/patch65536，默认upper7维和fine flip保留。原scale65536 lip梯度非有限诊断保留，不冒充实际更新预算通过。

最新远端stage2_v2_worker pid233261 attempt02在Patrick AVE：Patrick全量v2及监督已通过，main -6 AVE7751帧日志已有；其他两人stage2待顺序继续。stage1/score_reference worker均完成，别重复启动。Windows runtime_qualification_worker pid110572/session22551等待Patrick verified derived归档及cache退役后自动72控制恢复→绑定→90loader→18capacity→F全成员读回，准备worker不训练。Brown runtime archive49文件SHA94e091a6324d0028dcd24f1a6c410fd8c25b6bb8c33c0378640357119f0b6d9e；attempt01 tar硬链接类型检查失败保留，attempt02 hard-dereference后逐字节通过。旧capacity replacement watcher已停不可恢复。

F derived/qualification/flow/canonical observers仍工作，stage2等待每人F全成员验证及GT/torso退役后才继续；实际ps/receipts优先。root约527MB，SHM最新1.7GB，F约8.5GB。Brown canonical视频SHM已验证后退役，仅F原数保留；不能假设待训练监督仍完整。现工具可restore control/full/test，test仅3000个GT/torso，后续渲染无需恢复6000训练cache。仅验证F/百度盘后退役本任务缓存，不动未验证原数/CloudDrive配置。正式视频持久交付空间仍需规划。

四人评分参考fresh18路audio forward各通过，实际FP32 MFCC张量/音频features精确；72GT矩阵用独立float64 NumPy显式pad/欧式norm+原eps1e-6重建，maxmatrix4.151e-6、endpoint2.377e-6，offset全部一致。F qualification_state四个independent_reference_qualification和persistence proof已SHA读回。首次float64 MFCC字节比较失败因末位差最大9.238e-14，failure log保留；diagnose_mfcc_repro确认实际FP32输入0差。修为精确FP32比较、记录double误差，未放宽实际网络控制。不能称double字节相同。

## Changed Files
本轮新增未跟踪scripts/experiments/synctalk_multispeaker_full_20261017的完整状态/控制/epoch执行、paired_budget.py、persist_runtime_qualification.py、runtime_qualification_worker.py、qualify_all_data.py、persist_training_epoch.py、training_coordinator.py、render_model.py、score_identity_seed.py、validate_scores.py、qualify_score_references.py、assemble_results.py；恢复工具支持test-only，reclaimer支持明确post-training独立receipt。当前源/测试/spec/Task共76文件F execution_sources/execution_admission_attempt06全部字节读回，前attempt04/05保留。当前Windows coordinator/data admission脚本在本地，不应在准入前启动。render/score/report源码已上传ROOT；training epoch及helpers已上传。scratch辅助资格fixture/io传输和MFCC诊断是资格产物，不是实验模型。

## Verification
实际Ubuntu17个unittest7.799秒全过0skip，含原完整checkpoint连续/重启、fineLR/EMA/scaler、实际AMP skip、independent originaldistance和formalRGB/PCM。F controller_qualification/io_state_rotation_attempt01.json实际跨平台读回与只在新完整state验证后旧cache轮换通过，保存新state；__io_qualification仅合成bytes无模型/optimizer。独立score参考72矩阵通过并持久。全部源码AST和76文件F snapshot SHA通过。训练/渲染/27格正式执行、端到端media/score/交付QA尚未发生，汇总/plot代码未跑真实结果。

Git root/branch/SHA直接再次核验；GitHub只读fetch确认canonical仓库，search_prs该head无PR；当前SHA仍本地未验证推送，不构造commit链接。共享dirty树含大量无关untracked，未清理/全stage/commit。原generator train_one_epoch/network/renderer/loss未改。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)
- 本地未验证推送SHA `2b71639c3b555293375d248350999377168d515d`
- F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/execution_sources/execution_admission_attempt06/source_manifest.json
- F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/runtime_qualification_worker_receipt.json
- scripts/experiments/synctalk_multispeaker_full_20261017/qualify_all_data.py

## Observations
- [result] Brown独立F全档案preflight、90loader、18full-budget backward通过；四人72评分参考矩阵及17runtime单元控制通过，实验TFG更新0。
- [decision] 原FP16/GradScaler、default upper7维、fine flip和EMA数学保持；末位double MFCC差单列，实际FP32网络输入必须逐字节精确。
- [blocker] 全四人stage2/持久/runtime准入未闭合，正式24模型/生成27格/最终QA交付尚待实际执行；后续视频空间与协调器故障恢复需继续处理。
- [next_step] 复用stage2和Windows runtime资格observer完成余三人，全部独立门通过后运行qualify_all_data.py与已授权24模型完整训练，再生成/评分/独立核验并持久交付27格。

## Relations
- continues [[Codex checkpoint - 2026-10-17T11-28-27Z - Brown loading and full ray backward qualified with training controls prepared]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[039-synctalk-shared-training-flow-geometry]]