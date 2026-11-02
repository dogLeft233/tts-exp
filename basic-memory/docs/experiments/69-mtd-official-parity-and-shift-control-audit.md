---
title: 69-mtd-official-parity-and-shift-control-audit
type: experiment
permalink: tts-exp/docs/experiments/69-mtd-official-parity-and-shift-control-audit
status: concluded
scientific_status: OFFICIAL_FORWARD_PARITY_AND_OFFSET_CHANGE_PASS_RAW_SCORE_ADAPTER_UNQUALIFIED
---

# 69 — MTDVocaLiST官方调用与时移控制审计

✅ 2026-10-28完成20自然来源、40次真实PCM平移及3来源原分辨率裁剪对照。修正实验68的解释：未通过控制的是项目自定义的零时移平均logit，不是证明MTDVocaLiST无法识别时差。模型能恢复全部已知时间变化。旧数值保留，不把旧0/60 TTS胜出作为官方同步偏好证据。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 官方Dataset/网络/检索与控制审计 | 2026-10-28 | 用户要求检查官方调用和预处理 |

## 问题与复现

用户要求解释实验68的MTD全片平均logit控制17/40降分是否源于测法。原raw回放稳定失败：BAC009S0002W0122_N_0原始−4.151115，音轨后移6400samples后−3.677114，反而升高。复现命令是读取runs/cloud_mtd_synchformer_20261028_attempt01/mtd_full_scores.json，按20N×2实际平移重算lower并assert>=32；实际17/40，AssertionError REAL_CONTROL_FAILED。同例重复回放同值。

采用diagnosing-bugs流程，依次排除分数定义、模型类、裁剪、音频索引/边界。没有调整参数直到旧分数变好；原数据、旧协议、旧结果均未覆盖。

## 来源与可运行入口

官方源码 https://github.com/xjchenGit/MTDVocaLiST commit5795416f9277094d3ae6215596a436a0a01f79ad；官方test_stu.py https://github.com/xjchenGit/MTDVocaLiST/blob/main/test_stu.py 。权重同实验68的pure_MTDVocaLiST.pth，strict加载。

Ubuntu run /home/wjj/tts-audio/tts-exp/runs/mtd_official_audit_20261028_attempt01；Windows小结果与时移矩阵E:/Documents/tts-exp/runs/mtd_official_audit_20261028_attempt01。脚本scripts/experiments/mtd_official_audit_20261028.py，tests/test_mtd_official_audit_20261028.py。

```bash
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/mtd_official_audit_20261028.py pilot
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/mtd_official_audit_20261028.py pilot --full512
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/mtd_official_audit_20261028.py naturals
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/mtd_official_audit_20261028.py finalize
```

审计对作者test_stu.py做AST，只提取原封不动的Dataset和calc_pdist类/函数体，并提供作者相同globals。重新按旧检测框从原视频导出JPEG，使用作者Dataset读取，而非再写一份声称相同的前处理。直接用作者calc_pdist计算每窗口31个候选时移（±15帧）矩阵。结果在results.json、results_full512.json、audit_summary.json；官方源码SHA在official_source.json；实际音轨只在audit fixtures中改写并恢复原始，未改源音频。

## 调用和前处理数值核验

| 核验项 | 实测 |
|:--|:--|
| 官方Dataset图像张量 vs 旧缓存，给定同一裁剪框 | 20/20来源逐点一致 |
| torchaudio.load vs 原soundfile波形 | 20/20逐点一致 |
| 官方Mel vs 旧构造 | 20来源最大绝对差0 |
| 原生逐窗口logit重新前向 vs 旧存储 | 最大绝对差1.40667e−5 |
| README student_thin_200_all vs test_stu student_thin_200 | 两类均strict加载；首3来源共93窗口logit最大差0 |
| batch1 vs batch31 | 首3来源最大差约8.11e−6 |

视频RGB0–255、Resize96、取下半48×96，音频16k Mel80/FFT800/hop200/Slaney、dB与[-4,4]、int(80*f/25)索引均匹配作者测试。不要改为原版VocaLiST的BGR/255。两类主要差别为返回额外中间特征，未改变最终logit。

## 并非完全照搬的两处

1. 旧人脸检测先缩到256后缩框回512，作者Wav2Lip预处理在原分辨率检测。首帧旧框[136,112,358,374]，原512框[137,97,358,372]，确实不同。重跑首3来源所有原生帧的原512检测与JPEG前处理：6/6已知平移仍被准确追踪，零lag边界剔除平均logit控制仅1/6降分（旧框0/6），因此此差异没有修复已复现的失败模式。512 S3FD关cudnn.benchmark避免算法搜索巨额workspace，峰值约2.42GB；未改变网络或权重。
2. 官方主要评价每窗口31shift argmax落在±1帧的比例（K5），并对相邻窗口平滑得到K7等。实验68自行汇总零时移logit或sigmoid，并不是作者官方终点；严格地说，实验68不是端到端官方评分协议。

## 时移控制为何失败

先导固定为字典序前3个自然来源，未按TTS偏好筛选；之后扩至全部20。为避开官方−1000无效候选与真实PCM零填充边界，定位与诊断主统计均剔除两端各20帧（0.8秒），该支持下31候选均有效。

| 输入 | 20来源中模型估计的原生/扰动峰值 | 相对原生变化 |
|:--|:--|:--|
| 原始自然音轨 | −3帧 | 0 |
| 音频前移400ms | +7帧 | +10帧 |
| 音频后移400ms | −13帧 | −10帧 |

真实PCM控制40/40恢复正确变化，最大变化误差0帧。这里检验的是已知平移的相对变化，不假设原生绝对时差为0。−3帧是MTD估计，不等于已证明真实物理延迟120ms；可能同时含生成链与估计器偏差。首3原生旧SyncNet av_offset均−2帧，也提示不能把原生视频当零偏移真值。

原生零lag已经偏离该模型的匹配峰，且分类logit不是随时间错位绝对值单调变化的距离。因此原生与再错位的零lag分数可能都处于低分区，后者仍会更高。边界剔除后原始零lag平均logit仅12/40控制降分，故并非只由首尾静音填充造成；不剔边界的旧17/40结果仍数值成立。

用于定位原因的固定偏移对照：仅用前3来源固定音频窗口lag+3帧，随后在未参与该选择的17来源验证，实际在矩阵列18取分、没有每条自适应搜索或改写媒体。40/40控制降分，其中留出17来源34/34降分，见holdout_diagnostic_protocol.json。这个对照是解释评分位置选择的诊断，不是可替代原始时间轴的科学评分，也没有用它重算或筛选TTS优劣。

官方原生K5均值仅0.045995，原因之一是其±1帧零偏移判据与这些样本的−3帧主峰不一致。不能声称“换成官方K5后零时移评分已全部通过”；通过的是已知时间变化恢复。这两个终点必须分开。

## 修正的结论

- 工程层：核心模型与前处理数值核验通过；人脸框有已知实现差异，但3来源完整对照不支持它为主因。不能宣称全部官方流程逐点复现。
- 测量层：旧“零时移平均分类分数降分”控制没有资格用来否定整个模型的时差敏感度。应分开报告未校准分数、模型估计偏移和已知偏移恢复。
- 偏好层：实验68的MTD raw均值T胜0/60、辅sigmoid3/60仅描述那个adapter，不是官方模型已经判定自然口型更准确。不能用它加强SyncNet偏置论；本轮未重算60条TTS的官方K5或另选分数替代。
- 已知边界：模型能恢复时间平移并不证明其绝对偏移无偏，也不证明局部发音口型、人类感知或跨语言泛化准确。后续比较两臂须预先固定有效评分定义并区分原生非零峰与时移搜索，不能为提高得分逐条校正后再称原生改善。

## 验证与状态

三个实际测试PASS：已知shift retrieval不误当原生同步；边界峰不污染interior；带非零原生offset的变化控制正确通过且拒绝恒定输出。全部20源前向和60个作者候选矩阵已完成，full512另3源9矩阵。最终统计直接从存储矩阵重算并核对逐源结果，不借用旧偏好统计。旧raw控制保持失败，这是对测量解释的修正，不伪造旧端点通过。审计fixtures是证据而非临时调试日志；无运行时key、无新增API/训练/TFG。

## Relations

- corrects_interpretation [[68-cloud-mtd-synchformer-native-preference]]
- relates_to [[MTDVocaLiST部署]]
- follows [[云TTS与自然原生视频的MTDVocaLiST和Synchformer评分 2026-10-28]]