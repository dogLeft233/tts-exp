---
title: 046-synctalk-first-identity-early-score-preview
type: decision
permalink: tts-exp/docs/adr/046-synctalk-first-identity-early-score-preview
status: accepted
project: tts-exp
decided: '2026-10-18T09:12:43.266817+00:00'
---




# 046-synctalk-first-identity-early-score-preview

## Decision
用户要求一个人完成后先计算分数。先在本轮首个完成三档coarse10的身份上提供独立粗训练阶段预览，不等24完整模型；不把coarse模型说成fine17完整模型。固定seed42、epoch10 EMA和既有六段test，三训练×三驱动×三评价27格，先展示固定driver/scorer0的三训练档，再完整27格。后续正式完整模型结果另列。

## Context and consequences
现有render_model.py严格要求fine17和fine qualification，render_delivery_supervisor等待全部24模型，不能直接启动或修改原门。另立coarse预览实现，核验EMA提取/模型计数器、原FP16 test_step、FFV1/PCM像素与音轨读回、FP32SyncNet TF32off、source-owned ROI guard20主/full次及独立距离聚合QA。旧全局renderer/full false记录保持，不改pass。

若三档实际成功更新不相等，保留CONTROL_FAILED，预览只描述实际模型，不作训练gain受控因果结论。固定模型和原test支持集，不按分数选checkpoint/换条件/改预算。未完成或含非有限state不渲染，不用半途checkpoint替代。

优先调度评分，但不能抢占活跃GPU epoch。现controller使用非阻塞GPU锁，另起等待锁的评分进程可能导致下一训练job锁失败；必须先资格并协调在完整epoch/identity边界的controller handoff。现有训练未修改或中断。空间先实测、资格持久方案，退休输入显式恢复。自动跟进按此新请求恢复ACTIVE，完成首份经QA预览时通知，常规等待保持安静。

## Implementation qualification
执行入口独立early_score_preview_v2；先保留v1 CPU EMA映射资格失败（未启动真实渲染），按实际torch_ema全部parameters映射修正后3CPU测试通过。三档真实epoch10完整state/EMA/counters与9固定帧重复FP16渲染、无损PCM/RGB读回通过，0训练更新。V3训练controller在Patrick三档完成处等待，独立评分controller处理9视频/27格；当前完整27格及独立最终QA已完成，见 [[SyncTalk Patrick coarse10 seed42 27格描述性预览]]，以Task当前头部和F receipts为准。三档59985/59985/59984实际成功更新，原CONTROL_FAILED保留，资格pass只指正确读取和评价实际模型。

## Observations
- [decision] 按最新用户要求将首身份粗训练预览提前，原正式科学合同和失败状态保留。
- [evidence] 请求已固定在F trials/fp16_other_identities_trial_v1/early_score_request_v1/request.json，SHA941e1b9dade0285a9c90c8edab5852704e3910d2c337e9acd5a6e0afd7801b54；此请求文件保留创建时排队状态，执行现已产生经独立复核的首张driver0/scorer0三档预览，见Task当前头部与first_table_independent_review.json。
- [next_step] 完整预览已持久交付，按V3 release继续Chris/Cathy；保留原训练控制失败与全部历史版本。

## Relations
- affects [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- relates_to [[045-synctalk-other-identities-fp16-diagnostic-training]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]