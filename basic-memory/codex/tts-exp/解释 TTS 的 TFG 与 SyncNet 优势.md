---
title: 解释 TTS 的 TFG 与 SyncNet 优势
type: task
permalink: tts-exp/codex/tts-exp/解释-tts-的-tfg-与-sync-net-优势
description: 恢复历史实验并执行可区分机制的因果干预，明确 TTS 原生评分优势的生成器、评价器和视觉运动来源。
status: done
current_step: 4
started: '2026-09-30'
steps:
- '[done] 从 Git、CONTEXT、OpenSpec 与脚本恢复机制研究路线，完成本地 BM 设置。'
- '[done] 恢复服务器与百度网盘历史 run，核对实际 summary、报告、控制和独立验证。'
- '[done] 冻结新来源、RMS、方向范数、物理移植与外部复核实验，完成主双脸2560行及独立消费。'
- '[current] 完成音素均值谱形×内部轨迹与插值对照、独立可见口型及新产物持久归档，整合报告25。'
blockers: []
context: 研究主线已完成：正式报告25、11776声学及54口型族独立复算、完整源证据恢复、最后explicit小段包portable恢复全部实际通过。
project: tts-exp
tags:
- tts
- tfg
- research
- task
execution_phase: concluded_science_full_archives_and_portable_delivery_verified
completed: '2026-10-02T17:57:56.0858034+00:00'
---

# 解释 TTS 的 TFG 与 SyncNet 优势

## Objective and established conclusion

用户授权恢复历史研究、远程重跑和数据/Qwen使用，要求持续探明TTS增强TFG尤其局部声学原因。ROOT负责主线、理论、BM和Git；三个gpt-6-luna/max代理负责代码和实验。科学主线已完成，正式报告 [[25-tts-tfg-mechanism]]。

AISHELL-1中文/cloud Qwen/Wav2Lip链上，输入gain→log-mel→真实编码器方向→生成动作是正向增强来源，男40/女20说话人分别核验。固定A男声方向g20 C+.161[+.094,+.232]、方向−半径+.142[+.077,+.212]，内部mel+.171[+.101,+.245]。男native q00 RAW/RAW→q11 FIXED/FIXED paired-gap缩减+.204[+.119,+.295]同时改变生成和评分音；固定A男性自然生成G excess+.185[+.108,+.266]单列；女性paired-gap+.245[+.118,+.388]为另队列。

局部mu/rho因果干预定位recipient谱比轨迹与发声/时序结构的相容性。在保持自身源参数/时钟/逐帧SP功率/全局RMS时，Natural→TTS rho−SELF_GRID fixed-A/raw/g15 C−.192[−.336,−.047]，joint−grid两方向负；FULL35不同phone负端点主要n/ŋ及少量早tɕ/晚a˥˩，约350ms覆盖包含共发音，guard15phone区间含0。54几何族唯一联合99%正项为N→T q01−SELF_GRID meanap+.001518908[+.000085767,+.002952050]，开口增加与固定音频对应下降可并存。端点不合并为原生差距中介比例或人类感知质量。

## Authoritative evidence and completed verification

800视频/1600评分、11776族完整712文件/160phone maps实际恢复，ROOT recon11全部665600局部比较与完整bootstrap PASS；planned10304可估1472短Tg20保留40分母。11776图表8资产/41PDF页/45QA图已ROOT查看。

全800tracking独立来源/时钟验证、Av163文件/2080数组/630token重建及完整fullvisual tree恢复已通过。Primary05/independent07均实际exit0，44数组/54端点/20k共同bootstrap残差0；本机math05跨NP版本最大派生残差8.88e−16在原门内，54区间零分类一致。54图表CSV/3PDF页/3PNG与ROOT实际6图片QA通过。失败attempt和original支持集保持，所有源未删除/退休。

实验 [[音素平均谱形与内部轨迹的因果分解]] 已CONCLUDED；ADR004数值场实现完成，ADR005保持native/fixed-A/localG/motion分离。最新BC checkpoint保存实际源、SHA和运行定位；正式报告/source registry为当前权威，旧checkpoint是不可变历史。

## Completed delivery

最后explicit小段包已实际builder VERIFY、fresh sidecar-only stage与完整portable restore PASS。该冻结包保存报告/BM/源码/图表/验证回执和Task ACTIVE-before-pack快照；本Task现在DONE，由包外completion receipt绑定最新BM/Git、正式报告、全部归档和portable恢复，避免归档写入自身完成状态的循环。

CONTEXT与正式报告是当前研究入口。权威结论限定AISHELL-1中文/cloud Qwen/Wav2Lip及各自固定评分音/支持/队列；全部旧失败和planned缺口保留，所有源均未删除或退休。

## Observations

- [status] done；current_step4/4，完整科学验证、证据恢复与最后portable交付已实际通过。
- [result] 正向编码器方向路径及局部recipient相容性已由完整因果对比、独立数学复算和动作测量明确定位。
- [result] 最后portable小段包已完整恢复，最新完成状态由包外completion回执绑定。

## Relations

- relates_to [[25-tts-tfg-mechanism]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[004-canonical-float32-score-field]]
- relates_to [[005-separate-score-motion-and-native-estimands]]
- relates_to [[Codex checkpoint - 2026-10-02T17-00-12Z - Final visual archive running and evidence synthesis ready]]