---
title: 暂停 Chris 并精调 Patrick 三档及27格
type: task
permalink: tts-exp/tasks/暂停-chris-并精调-patrick-三档及-27-格
status: paused
project: tts-exp
current_step: 3
started: October 18, 2026
pause_reason: Explicit user pause and Ubuntu poweroff request
paused_at: '2026-10-18T15:14:14.055840+00:00'
---




# 暂停 Chris 并精调 Patrick 三档及27格

## Current verified execution
V3已固定携带parent_receipts/coarse_qualification.json，SHAa492c2d70d1406b92a8148d97cab64804ed9bbb8e21539f8155c695b4d119ad4，部署读回通过。v2首次0gain入口资格job821455在optimizer/trainer构造前因远端缺少主机phase验收文件失败，0attempt/0optimizer/0formal；job/case archive均保留，不重启或重放该job。V3按新源修复元数据接线后重新独立资格，Patrick16500复用并fresh逐SHA核验，0重复上传。

Chris已在完整epoch边界暂停：0/−6 ep2、+6 ep1均F最新完整state逐SHA核验，job816708 rc0/795.109秒，+6 ep2未启动；原V3 Windows181520已停止，不重启。OLD/controller_revision04_pause_for_patrick_fine_v1/paused.json为权威暂停证据。

当前新执行版本patrick_fp16_fine_v3，OUT=FROOT/trials/patrick_fp16_fine_v3；contractSHA7d6276b92c406a8496de8f230c66db0eed37351fec259ca8af7c8c3b7ed073b9，7源二进制部署freshSHA通过，host controller177316。v1原contract/source与零GPU状态保留，因资格设计错误要求前两步都成功更新，在执行GPU前被v2替代。v2仍固定各gain原前两帧，只修正资格对正常AMPskip的记账判定：6资格尝试、最多6实际更新，真实success/skip逐行精确、finite state和full-state roundtrip；不要求人为消除skip，不改scaler/precision，不能称完整稳定性已证实。4CPU拒绝错parent/phase/gain与lost重放/正常skip精确计数测试通过。旧guard及FP16数学不变。

容量切换只退休已完整F逐成员freshSHA的重复cache：closed_cache_retirement_receipt在fine_v1，724个闭合Patrick/Chris日志/采样/验证/firstskip副本及coarse入口资格媒体缓存，257822309逻辑字节，Linux独占lease通过，F全原字节保留。另将Chris最新3份remote checkpoint副本冷置，F三个最新完整checkpoint逐SHA保留，chris_checkpoint_cache_retirement_intent/receipt在fine_v2；未来Chris恢复必须先从F显式恢复这些checkpoint、监督和所需remote trace/资格媒体，不假定原remote路径仍有文件。数据空间预测passed：free456818688+Chris可退1386229760−Patrick需1609936896=233111552B，保持原180MiB门，真实df优先。capture cache observer182424仅在闭合case/epoch的F全成员freshSHA通过后，以独占lease退休qualification checkpoint/firstskip remote副本，F state与archive全保留、marker保留；不动正式当前checkpoint/trace/curve或原失败。

fine17评分候选patrick_fine_score_v3.py及patrick_fine_score_controller_v3.py已准备；render receipt的旧coarse标签已修正，新增patrick_fine_score_receipts_v1.py拒绝错phase/arm/checkpoint/EMA/帧序，F fine_score_candidate_revision01保留旧候选，revision02绑定新源；最终score manifest必须含helper。3CPU receipt测试及既有3EMA核心映射测试通过，尚未冻结final checkpoint或做真实fine17 GPU资格，0新fine视频/分数。另有compare_patrick_coarse_fine_v1.py候选及3CPU轴/clip配对拒绝测试，真实旧coarse27 SHA/聚合通过，尚无fine27实际比较；F coarse_fine_comparison_candidate_v1保存source/test/receipt。

本次写入时真实receipt摘录：用户2026-10-18明确要求先暂停训练，并在完成后关闭Ubuntu；此指令覆盖原自动推进fine17/评分。Patrick三档已在fine14完整epoch边界保存，fine11–14共12/21 epochs完成；user_pause_v1/paused.json passed，三档最新完整RAW model/optimizer/scheduler/scaler/EMA checkpoint与原job/trace均F逐成员和SHA核验。0/−6/+6累计成功83964/83962/83962，各global84000。Windows177316已精确冻结后停止，capture observer182424在无活动子操作时停止，0GPU signals，旧source/controller receipt和全部历史保持；暂停boundary observer178756只等待并封存原+6 ep14，没有新optimizer/job/重放。ep15未启动，fine17/新27格均未完成，原CONTROL_FAILED保持。完整断点与恢复入口见F OUT/user_pause_v1/paused.json和poweroff_preflight.json；关机命令已返回rc0，SSH明确进入shutdown/PAM nologin，直连Ethernet(index6,192.168.50.1)已Disconnected/0bps；最终证据在同目录poweroff_receipt.json passed。普通Meta代理TCP/ICMP曾误示仍可连，失败探针完整保留；未重复下关机命令。用户明确恢复前不启动训练、精调或评分。

关机会清空SHM，未来恢复必须从F重建并核验runtime、监督/所需数组和三份最新完整state，再按逐receipt新版本handoff调度fine15–17；不能重启旧controller177316、假定remote路径存在或重放已完成optimizer。Chris保持原0/−6 coarse2、+6 coarse1暂停，Cathy未启动。原24完整模型目标仍未完成，保留chat；heartbeat已设PAUSED，不归档或标done。凭据只在运行时使用，不写文件或记忆。

## Requirements
### Requirement: SHALL 在已启动Chris完整epoch结束且持久断点核验后暂停后续调度
#### Scenario: 当前job完成
- **WHEN** pinned job816708 completed rc0且epoch/job全部F成员及最新三gain checkpoint SHA通过
- **THEN** paused.json passed，+6 ep2未启动，保留0/−6 ep2与+6 ep1续训入口；旧controller不重启、不重放。

### Requirement: SHALL 先通过独立fine入口资格，再执行Patrick三档FP16 fine11–17
#### Scenario: 数据与入口资格
- **WHEN** Chris完整监督归档freshSHA及独占lease切换、Patrick full16500恢复，三档各前两帧GPU资格通过
- **THEN** full model/optimizer/scheduler/scaler/EMA/counters roundtrip和采样/lip交替一致；qualification/result.json绑定contractSHA，formal才可启动；6资格尝试不算formal。
#### Scenario: 正式精调与失败
- **WHEN** 统一21个fine epochs执行或首terminal故障
- **THEN** 每epoch6000尝试和三gain采样原序通过；完整state/trace/异常F持久；首故障停止本fine版本，不重跑；父CONTROL_FAILED保留，fine按真实实际更新验收不补步。

### Requirement: SHALL 固定fine17 EMA重新完成27格及独立QA持久交付
#### Scenario: 三档完整fine17
- **WHEN** 三档fine17 fullstate存在且finite与计数/EMA验证通过
- **THEN** 原六test、三train/driver/scorer gain共9新视频27格；FFV1PCM/guard20主full次/原PCM评分合同，独立像素及矩阵/聚合QA后交付，与原coarse结果并列，不替换原记录。

## Observations
- [status] paused
- [decision] 最新用户授权优先Patrick精调，Chris/Cathy暂缓；保持原目标未完成和heartbeat ACTIVE。
- [blocker] fine17三档及评分GPU资格/27格未完成，原CONTROL_FAILED保持。
- [next_step] 观察pinned Chris job完整封存，完成监督切换及Patrick三gain真实入口资格。

## Relations
- project [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- implements [[047-synctalk-patrick-fp16-fine-and-safe-pause]]
- relates_to [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]