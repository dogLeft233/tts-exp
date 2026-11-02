---
title: FlashHead Pro 中文20人可听音量九评分器配对实验
type: experiment
permalink: tts-exp/experiments/flash-head-pro-中文-20-人可听音量九评分器配对实验
status: CONCLUDED
run_id: flashhead_chinese20_rms24_20261030_attempt01
tags:
- FlashHead
- server192
- paired-comparison
---

# FlashHead Pro 中文20人可听音量九评分器配对实验

## 问题与冻结设计
用户要求用新SoulX-FlashHead Pro重做中文20人自然语音/TTS视频和九评分器配对比较，并明确要求把音量归一化到可听水平。仅采用已确认的中文20源20N+60既有云端TTS；英文不纳入，不重新合成TTS。主分层沿用12个同句克隆参考，8个扩展同人参考另列；三draw不作独立人。

新run为 /home/data2/wangjiajun/av-evaluation/jobs/flashhead_chinese20_rms24_20261030_attempt01。protocol.json在生成前冻结，SHA256 0534b6e8f8c97217806c3f3e5571ef5e827425c3a25aae12c440c7dab02eefb0。inputs.json逐条绑定原件/派生WAV SHA、增益、峰值及统计身份。原始规范音频不改，整段乘单一增益再round到PCM16，统一RMS −24dBFS，无压缩、裁切、时间编辑或评分补静音。生成与评分必须使用同一派生PCM。

固定3.png、512x512、25fps、Pro/stream、seed42，关闭compile。加载一次官方pipeline，但每片调用原get_base_data重置种子和运动状态；官方生成函数体不改。完全相同输入/图片/种子允许复制视频并记录alias，不把其宣称独立生成。评分MP4保留视频流、ALAC音轨逐样点比对驱动WAV。

九端点沿用部署：SyncNet原pipeline FP32原始距离矩阵重算C；CoGenAV全片Eq1；PEAVS全80固定排序单次batch64+16；SyncReward BF16中心4.75s；AVS_u论文定义复现frame mean；MTD/VocaLiST作者K5原31候选；Synchformer/SparseSync整数中心5s P0。短输入无法满足原窗口时UNMEASURABLE，不填充凑数。保留实际时移控制，控制失败不等于科学阴性。每人先均值三T，再T−N；各指标完整配对人数单列，bootstrap10000、signflip100000敏感性、每分层Holm9，无跨指标合成分数或感知真值结论。

## 执行与证据
状态CONCLUDED：80条视频、九评分器、原始归约和本地统计复算已完成。完整结论见 [[76-flashhead-chinese20-nine-scorers]]。同句参考12人九项均未通过Holm9；全20人Sync-C+0.212、CI跨零，PEAVS差值−0.452971、Holm9 p=0.04131。prepare80/80 PASS，RMS范围[−24.000007666,−23.999993547]dBFS，最大峰值.857666，增益4.476793–4.477421dB；41种唯一音频。4个针对增益安全/时序/人级配对/Holm的测试通过，Python编译通过。

服务器手册当次重读，全部工作在个人目录；nohup后台单A6000 UUID、torch60%显存、CPU4线程、顺序阶段。其他用户程序不管理。原始部署入口 [[server192-nine-scorers]] 保持完整历史。

源码 scripts/experiments/flashhead_chinese20_20261030/{prepare,generate,batch_adapters,controller,analyze}.py；远端同名目录 experiments/。本地及远端均为新工作文件，未提交。运行结束后需回读所有原始产物、复算和写最终结论，不能把部署单片或当前生成状态当作完成。

## Observations
- [decision] 用户本轮明确统一可听增益，故新N/T全部重生成重评分；不拼接旧低增益自然分数。
- [decision] 只有原生满足窗口的样本进入相应指标完整配对统计，其余保留不可测原因。
- [next_step] 从报告76及本地不可变证据包复核本轮已完成结果，无需重新部署或重跑。

## Relations
- uses [[server192-nine-scorers]]

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Freeze Chinese20 audible-gain full experiment and start generation | October 30, 2026 | user |

| Complete80 media/nine scorers, raw audit and local statistics readback; conclude exploratory experiment | October 30, 2026 | user |

## 最终支持与补充分析

Synchformer/SparseSync各33/80媒体、7人完整配对；其他七项均80/80、20人配对。SyncReward实际官方短片流程支持内部片段补齐，主分析全20人，原生完整4.75秒8人子集差+0.031901且CI跨零。此前4.75秒最短输入假设已在该模型首次评分前按官方源码澄清，绑定syncreward_native_support_binding.json；不改变输入媒体。PEAVS额外用同一特征回放batch20，差值与64+16一致到6位小数，0新特征前向。

本地 results/soulx_flashhead_pro/chinese20_rms24_20261030_attempt01/：2234文件SHA、80媒体导出和25非空指标分层的统计复算全部PASS。309272893字节证据包SHA256为7cf9916be529e8f0f4099bd346a4cae168bb590bd254af9266cfee213ee7a700。本人GPU/后处理进程已退出。图已查看，科学结论和所有边界见报告76。
