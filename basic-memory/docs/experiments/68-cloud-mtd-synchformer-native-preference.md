---
title: 68-cloud-mtd-synchformer-native-preference
type: experiment
permalink: tts-exp/docs/experiments/68-cloud-mtd-synchformer-native-preference
status: concluded
scientific_status: EXPLORATORY_SYNCHFORMER_SUBSET_PASS_MTD_CONTROL_FAILED
---

# 68 — 云TTS与自然原生视频的MTDVocaLiST及Synchformer评分

✅ 2026-10-28计算及独立标量复核完成。MTDVocaLiST连续评分CONTROL_FAILED；Synchformer官方5秒配置在长度合格子集控制PASS。不能将前者的自然偏好当同步真值，也不能将后者6来源外推全组。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 两模型原生偏好与时移控制报告 | 2026-10-28 | 用户明确授权两模型评分 |

## 问题与cohort

使用既有中文云Qwen qwen-audio-3.1-tts-plus cohort，而非新EARS：20来源自然视频N各1条、云TTS驱动视频各3条，共80原生媒体、60个T−N配对。继续使用原生各自音轨，无交叉音轨、云API重抽或TFG重生成。沿用已有媒体与音频电平处理，本轮未额外音量归一化。完整原Sync-C配对来自cloud_av_preference20_20261027_attempt05/paired_scores.json，20自然及60云视频的输入来自avs_u_zh_pilot_20261027_attempt01/inputs.json。

## 方法与可复现入口

Ubuntu权威run /home/wjj/tts-audio/tts-exp/runs/cloud_mtd_synchformer_20261028_attempt01；Windows小结果E:/Documents/tts-exp/runs/cloud_mtd_synchformer_20261028_attempt01。脚本scripts/experiments/cloud_mtd_synchformer_20261028.py，对应tests/test_cloud_mtd_synchformer_20261028.py。模型见[[MTDVocaLiST部署]]、[[Synchformer部署]]。使用官方预训练网络与权重strict load；以下连续评分归约为本项目明确的adapter，不是作者直接提供的TFG评分程序。

MTD：作者README student_thin_200_all，官方test_stu与train_MTD均torchvision RGB0–255/JPEG/Resize96，取下半脸48×96；16k mel80、FFT800、hop200、Slaney55–7600Hz，dB ref20与[-4,4]。S3FD在256分辨率定位后框缩回原512，非作者人脸框逐点复现。原生全片每帧5帧窗口，作者lastframe=min_length−5约定；音频起点int(80*f/25)。主分对齐位置raw logit均值，辅分逐窗口sigmoid均值；没有最佳lag搜索或DTW。注意mean sigmoid不等于sigmoid(mean logit)。

Synchformer：官方LRS3 checkpoint23-12-23T18-33-57，patch_config、test transforms、prepare_inputs、forward，FP16 autocast遵保存cfg，5秒14段×16帧、25fps、短边256/中心224裁剪。音轨float PCM MOV适配避免Short.mean错误；28条均断言读取后与原WAV逐点一致。主分21类softmax的零时移P0，辅分负期望绝对时移。PyAV12.3/Torch2.5.1兼容环境，不称bit-exact官方运行时。

80媒体中52条不足5秒。原计划统一2秒的Synchformer实测因官方学习位置编码固定198tokens而失败（2秒72tokens），记UNMEASURABLE；没有插值权重、重放片段、冻结嘴帧或补静音凑数。native5s_amendment.json在任何Synch得分前冻结，仅28原生>=5秒媒体评分，其中形成18完整T−N配对、6来源，均属strict-reference12内。其余42对不能用当前官方原生5秒配置直接评分。MTD原中央2秒口径保留，随后全原生支持扩展与相同评分公式在其产生前写mtd_full_support_amendment.json。原protocol.json未回写。MotionFormer相对import需源码cwd，execution_repair.json记录修复。

控制：20自然视频固定，只对真实全PCM做±6400采样点（400ms）零填充平移，crop/视频不变。预设至少80%控制分数下降。MTD全组40项；Synch仅10条合格自然，共20项（其中6配对来源12项也全部下降）。无最佳时移校正。

## 主结果

平均分差均为T−N，越高越好。T与N平均按相同配对分母统计，N在每个对应TTS配对中重复；CI按来源聚类bootstrap10000次seed20261028，60或18不是独立来源数。冲突=两模型T−N与旧Sync-C的T−N严格异号，平局另列，本表均无平局。

| 指标 | 配对/来源 | N均分 | T均分 | 平均T−N | T胜 | N胜 | 与Sync-C冲突 | 真实时移控制 |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| MTD全片平均logit（主） | 60/20 | -4.540270 | -5.692770 | -1.152499 | 0/60=0.0% | 60/60=100% | 48/60=80.0% | 17/40，失败 |
| MTD全片平均sigmoid（辅） | 60/20 | 0.181890 | 0.101095 | -0.080795 | 3/60=5.0% | 57/60=95% | 51/60=85.0% | 29/40，失败 |
| Synchformer官方5秒P0 | 18/6 | 0.984113 | 0.997050 | +0.012937 | 15/18=83.3% | 3/18=16.7% | 3/18=16.7% | 20/20，通过 |

MTD logit差来源bootstrap95%CI[-1.401726,-0.907873]；sigmoid[-0.103683,-0.057716]。Synch P0差CI[+0.001997,+0.032509]，约+1.294个百分点；只有6个来源，区间不能作强确认性推断。Synch辅分负期望绝对时移：自然−0.009072秒，T−0.001157秒，T−N+0.007914秒，方向15胜3负、冲突3/18与P0一致。

主冲突方向：MTD全片48例全部为Sync-C偏T、MTD偏N，反向0。辅sigmoid51例包括同向48、反向3。Synch3例全部Sync-C偏T、Synch偏N，反向0；在该18对子集旧Sync-C全部偏T。

## 支持集与敏感性

相同18对上：MTD logit T胜0/18，T−N−0.938873，旧Sync-C冲突18/18；Synch15/18偏T。这排除了两个模型分母不同是此18对子集方向分歧的唯一原因，但MTD控制失败仍不能作真值。

MTD strict12同句参考子集36对：T胜0/36，平均−1.131557，冲突33/36=91.7%。中央2秒旧设计：T胜11/60=18.3%，平均logit差−0.897671，冲突43/60=71.7%，控制14/40失败；与全片强度不同，说明时间支持会影响这类汇总，不能把中心2秒与旧全片Sync-C当完全相同端点。

## 验证与边界

三条实际测试通过：原生长度支持不padding/stretch、PCM平移非循环且不缩放、冲突方向/平局/来源分组。独立结果复核不调用生产统计函数：从120组MTD全片逐窗口logit与48组Synch类别logit重新核算均值/sigmoid/softmax，共168输出组，复算配对均差/胜负/冲突与主表一致；receipt independent_result_audit.json PASS。两模型严格加载、native输入格式检查、Synch28/28 PCM逐点一致均通过。无新GPU重跑用于此独立标量复核。

可支持：Synchformer在此长片段子集有与Sync-C大体一致的TTS优势；MTD的平均分类分数方向相反。不可支持：所有独立指标都反对TTS、SyncNet已经被证明有偏置、或Synchformer已证明全20来源/人类感知口型改善。Synch零偏移概率是未作中文校准的类别置信度，并非物理同步误差真值。MTD真实错位控制失败，不能凭极强自然偏好否定TTS同步。两个分数单位不同，不能比较平均差的绝对大小。

## Relations

- follows [[云TTS与自然原生视频的MTDVocaLiST和Synchformer评分 2026-10-28]]
- uses [[MTDVocaLiST部署]]
- uses [[Synchformer部署]]
- relates_to [[AVS_u中文域同步控制与自然TTS原生比较 2026-10-27]]