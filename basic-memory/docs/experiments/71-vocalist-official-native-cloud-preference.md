---
title: 71-vocalist-official-native-cloud-preference
type: experiment
permalink: tts-exp/docs/experiments/71-vocalist-official-native-cloud-preference
status: concluded
scientific_status: SCORER_PREFERENCE_ONLY
run_id: vocalist_official_cloud_rescore_20261028_attempt01
---

# 71 原版VocaLiST官方方法：云TTS与自然音频驱动视频的原生偏好

状态：CONCLUDED（评分器偏好已完成；感知同步优劣未建立）。权重、真实GPU部署、全组评分、独立矩阵复核均完成。

## 问题与支持集

检查完整VocaLiST与Sync-C是否给出相同TTS优势。使用此前中文云端qwen-audio-3.1-tts-plus同组20来源：20自然视频及每来源3次TTS对应60视频，共80媒体、60配对，独立单位为20来源。不是EARS英文组，不是固定视频换音轨，也不是MTDVocaLiST。
原生视频与各自原生音轨沿用冻结inputs.json及sync_pairs.json，来自cloud_mtd_synchformer_20261028_attempt01及更早cloud_av_preference20_20261027_attempt05。原有RMS目标0.0376838172675746已保留；本次不额外归一化，不改时长、音轨、帧率、不做最佳时移校正。Sync-C沿用原组精确配对值，TTS胜48/60=80%，平均差约+0.745（Sync-C单位）。

## 官方执行与部署

原版作者源码commit c265b59bcf7cd8559a265aaf454a9333ec24cb0f，官方[GitHub Release](https://github.com/vskadandale/vocalist/releases/tag/interspeech2022)的独立002 ZIP含唯一vocalist_5f_lrs2.pth。ZIP 882753219字节，解压961635703字节；80106561参数models.model.SyncTransformer严格加载，torch2.5.1+cu121、V100、FP32真实推理成功。未重新训练。此前“官方权重不可得”的判断已纠正，原因是漏查Releases。
直接执行作者test_lrs2.py的Dataset、calc_pdist和eval_model原文AST函数体；未更改作者模型、掩码、边界平均或指标公式。全512 S3FD JPEG帧复用正式MTD运行的同组裁剪，但用原版Dataset重新读取：cv2 BGR、Resize96、除255、取下半48×96，不使用MTD RGB0–255张量。16k音轨逐条确认PCM数值相同，作者Mel80/FFT800/hop200/Slaney及16列窗口。全80模型前向新运行；保留全重叠窗口及作者边界-1000无效候选掩码。
K5主端点预先冻结：31候选（±15帧），预测在零点±1帧的窗口比例。K7–15采用作者相邻窗口得分平均，正确容差仍±1帧。K值相关，不能当成六个独立评价器。运行环境不是作者2022环境的逐位复现，但官方调用和公式已经直接执行验证。

## 结果

分数表示官方检索正确窗口比例；平均差统一为TTS−自然，以百分点计，越高越好。胜负及冲突均按严格符号，平局单列，冲突列分母固定60。

| 指标 | 自然均分 | TTS均分 | 平均差（百分点） | TTS胜 | 自然胜 | 平局 | 与Sync-C冲突 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K5（主） | 4.001% | 3.063% | -0.937 | 18/60（30.0%） | 42/60 | 0 | 36/60（60.0%） |
| K7 | 3.468% | 2.531% | -0.937 | 15/60（25.0%） | 42/60 | 3 | 39/60（65.0%） |
| K9 | 3.655% | 2.193% | -1.463 | 12/60（20.0%） | 45/60 | 3 | 42/60（70.0%） |
| K11 | 4.144% | 2.005% | -2.139 | 9/60（15.0%） | 45/60 | 6 | 42/60（70.0%） |
| K13 | 4.567% | 1.665% | -2.902 | 6/60（10.0%） | 54/60 | 0 | 42/60（70.0%） |
| K15 | 4.998% | 1.390% | -3.608 | 3/60（5.0%） | 57/60 | 0 | 45/60（75.0%） |

K5冲突36对的方向：Sync-C认为TTS更好、VocaLiST认为自然更好33对；反向3对。
K7、K9、K11排除平局后的冲突率分别39/57=68.4%、42/57=73.7%、42/54=77.8%；平局不代表一致。

10000次来源bootstrap，seed20261028，20来源而非60独立样本：

- K5：平均差95%来源bootstrap区间 [-1.915, -0.021] 个百分点。
- K7：平均差95%来源bootstrap区间 [-1.682, -0.243] 个百分点。
- K9：平均差95%来源bootstrap区间 [-2.421, -0.439] 个百分点。
- K11：平均差95%来源bootstrap区间 [-3.231, -1.018] 个百分点。
- K13：平均差95%来源bootstrap区间 [-3.950, -1.790] 个百分点。
- K15：平均差95%来源bootstrap区间 [-4.685, -2.549] 个百分点。

严格参考子集12来源/36对：K5 TTS胜15/36=41.7%，自然21/36，平均差−0.777个百分点，95%区间[−1.977,+0.368]跨0；冲突24/36=66.7%。全组K5区间上界仅略低于0，不能夸大稳健性。

## 控制与边界限制

真实部署smoke和对应artifact测试PASS；两个CPU端点测试PASS。全80保存矩阵的480官方端点、360配对端点、all20/strict12胜负及冲突计数独立重算PASS，未重复GPU推理。首3自然样本±6400样本（±400ms）的6项物理PCM干预，检测的相对偏移变化全部为预期±10帧，误差0；这是时差变化追踪通过，不等于K5主端点已被证明能衡量感知质量。所有控制输出独立保存，不用于原生评分校正。
关键诊断：去掉首尾各20行以避开无效候选掩码后，20自然及60TTS的整体曲线峰值全部在−3帧。这是模型估计，不证明物理延迟精确120ms。K5仅接受零点±1帧，因此这组数据处在较低值区，原始K5自然4.001%、TTS3.063%。
自然98个正确窗口中仅27个在guard20内部，约72.45%正确窗口在首尾；TTS207个中仅37个在内部，约82.13%在首尾。guard20平均K5分别1.2624%和0.7972%；12/20自然及42/60TTS内部正确率为0。首尾本身也可能有真实信息，这不是边界错误的因果证明，但说明正式小分数受边界贡献及低值效应明显影响，不能将负差直接称为自然口型更准。
scores.json中的global_offset_frames是包含作者边界掩码的全矩阵均值峰值，可能被−1000掩码推向0，不能解读为真实零偏移；偏移诊断以independent_audit.json的guard20字段为准。没有借诊断移动任何原生媒体。
原模型LRS2是英文域，本组中文；已知时移控制不能排除域偏好。原版与MTD有教师学生关系，二者结果不是独立真值投票。

## 可复现产物与判断

run：runs/vocalist_official_cloud_rescore_20261028_attempt01，两机均保存summary.json、paired_scores.csv/json、scores.json、matrices/、protocol.json、independent_audit.json、boundary_floor_diagnostic.json、known_shift_controls.json、source_snapshot/及deployment_receipt.json（本机部署收据副本）。原媒体及控制媒体留Ubuntu。
源码scripts/experiments/vocalist_official_cloud_rescore_20261028.py，独立重算scripts/experiments/vocalist_official_results_audit_20261028.py，部署smoke及对应tests。全流程无需API密钥。
结论限于此评分器：官方原版K5总体偏向自然，TTS胜30%，与Sync-C冲突60%；K7–15也没有平均TTS优势。它补充了评分器分歧证据，但明显低值与边界效应使它不能独自裁决SyncNet偏置或真实视觉同步优劣。

## Relations

- relates_to [[VocaLiST原版部署]]
- relates_to [[VocaLiST原版官方Release权重补齐 2026-10-28]]
- relates_to [[70-mtd-official-native-cloud-preference]]
