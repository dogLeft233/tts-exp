---
title: 运行Patrick的RMS归一化AVE三档粗训练并评测
type: task
permalink: tts-exp/tasks/运行-patrick-的-rms-归一化-ave-三档粗训练并评测
status: blocked
current_step: 2
project: tts-exp
started: October 19, 2026
tags:
- synctalk
- rms
- patrick
latest_execution_version: patrick_ave_rms_coarse_repeat_v1
hold_reason: explicit_user_pause_after_verified_coarse7
---

# 运行Patrick的RMS归一化AVE三档粗训练并评测

当前状态：blocked/current_step2，用户主动暂停（不是数值失败）。执行版本patrick_ave_rms_coarse_repeat_v1，详见[[Patrick RMS归一化同初始化独立复跑]]。

用户最新明确要求“完成这一轮后先保存断点并中断实验”，覆盖此前继续coarse10/评测授权。当前+6gain coarse7已自然训练、验证、完整保存结束，三档均完成coarse1–7，共21/30完整epochs，各42000attempts。最新0/−6/+6累计成功更新为41991/41990/41991，普通AMPskip按实际记账；更新不等保留、不补步，phase最终资格尚未验收。

F三档最新coarse7完整RAW checkpoint、全部model/optimizer/scheduler/scaler/EMA/counters及原逻辑trace/curves均保留。三份最新training archive全部成员与standalone fullcp再次freshSHA通过，最新+6ep7 raw6000trace/finite loss与7个跨gain18000行sampling/LR block独立复核passed；未新GPU重算或加载全tensor。持久证据OUT/user_pause_v1/paused.json、remote_pause_verification.json、controller_exit_classification.json及independent_reviews；暂停时间2026-10-19T12:43:35.174715+00:00。三份cpSHA：0=497916c17d6966bfb35062f634b96fabb0d90a13facfd8e5bfde0759602c3d0a；−6=e042410f62a58b8c61e676f81c9322f20602294ba701534fd06528b4ac90e7bb；+6=788fd29c2563b658c749b9dcc3973611b7c5f8077ad5c5de578b8d9dbc5eba43。

用户pause_request在当前作业运行时写入，允许它完整结束但禁止下一job；host229476在F持久和block核验后由原pause guard退出，0GPU信号/0host信号。原controller_revision02_metadata_handoff_v1/controller_failure.json含预期用户pause AssertionError，原样保留并独立分类为显式暂停，不是训练数值terminal。Linux无本任务训练/评分进程，formal job恰21个、无coarse8 job、无新score_27_v1。OUT/active_controller.json已指向user_pause_v1/paused.json，原authority在previous_authority.json；旧root receipt/metadata故障和冻结19源/contract b44d67ce4daa7076e88648adefceb663118dd877a5b2a527a7bcf2d5439ee53b保持，不重启旧host、不重放。未来恢复需人类新授权和逐receipt新审计handoff，不能删除pause flag后盲重启。

本repeat粗训练和新27格目标均未完成，0新视频/评分，不能称科学成功/失败或RMS总体因果结论。评分helper仍仅score_receipt_candidate_v1的6CPU测试候选，未接入实际renderer或GPU资格。旧RMS数值失败/恢复probe116全skip、旧coarse CONTROL_FAILED/Brown失败全部保持；旧fine14/Chris/Cathy和原24full仍暂停。自动巡检在暂停发布核验完成后删除，不自动继续训练/评测；不归档chat或标总目标done。

OUT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1。

## Requirements
### Requirement: SHALL 独立冻结RMS入口并通过真实资格
#### Scenario: 新源和数据恢复
- **WHEN** F原归档完整成员freshSHA通过，固定新contract/source/data/core/初始化pins并部署读回
- **THEN** 六档main/test实际AVE和重复、标量归一化/时钟公式、三gain六次真实FP16入口及full-state roundtrip和实际采样全部通过，才启动formal。

### Requirement: SHALL 运行Patrick三档完整coarse10并精确持久记账
#### Scenario: 正式执行
- **WHEN** qualification passed且contractSHA匹配
- **THEN** 三档各60000attempts，原FP16数学和数据政策不变；每闭合epoch全state/trace/job F鲜验，跨gain采样/LR/实际success核验，不补步；首terminal/lost封存并停止新job，不盲重跑。

### Requirement: SHALL 完成新27格评测及与旧结果并列交付
#### Scenario: 三档coarse10完成
- **WHEN** 三份full checkpoint/EMA60和真实9固定帧重复/像素资格通过
- **THEN** 原六clip、9新视频、27格/162矩阵和原PCM评分合同，独立矩阵/等权聚合/媒体持久读回通过后交付CSV/图/报告，与旧未归一化coarse并列；保留旧false/失败，不称感知或总体因果结论。

## Observations
- [status] blocked
- [result] 已在当前+6coarse7完整训练/验证/F全验后暂停；三档各7轮、21/30完成，最新三fullcp及全部trace保留，无coarse8/新评测。
- [decision] 只执行本新实验，不恢复Patrick旧fine或Chris/Cathy；RMS固定0.1不按test选参。
- [blocker] 用户主动中断实验，等待新指令；不是数值terminal，原粗训练/新27格目标未完成。
- [next_step] 保持暂停并停止巡检，不自动训练或评分；后续仅按人类新授权审计恢复。

## Relations
- project [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- relates_to [[暂停 Chris 并精调 Patrick 三档及27格]]

- implements [[048-synctalk-patrick-same-ave-rms-coarse]]
