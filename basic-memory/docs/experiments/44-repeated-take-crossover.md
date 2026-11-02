---
title: 44-repeated-take-crossover
type: experiment
permalink: tts-exp/docs/experiments/44-repeated-take-crossover
status: concluded
scientific_status: ORDINARY_INSTANCE_MISMATCH_REPRODUCED_PILOT
project: tts-exp
date: '2026-10-15'
run: repeated_take_crossover_20261015_attempt01
cohort: AISHELL1_8speakers_1text_each_4realizations
experimental_unit: speaker_source
scope: single_Wav2Lip_local_Qwen_pilot
head_research: paused
---

# 44-repeated-take-crossover

同文本独立发声的类内/跨类交叉失配对照（2026-10-15）

## 结论

**只换成同类的另一段同文本发声，就会产生很大的 Sync-C 下降。** 自然驱动视频从自配音轨换成另一段自然发声，平均下降 **3.650**；TTS 驱动视频换成另一次独立 TTS，下降 **4.275**，两类均为8/8来源下降。跨自然/TTS的降幅分别是 **3.963/4.708**，相对类内的额外降幅只有 **0.314/0.433**，描述性区间均跨零。

因此，原N/T四格的“对角高、非对角低、强交互”并不是TTS来源独有的模式。论文应将它解释为**具体发声实例的配对依赖**，补类内重复参照与局部时间控制，再讨论来源特有作用。这里的差值不作为旧实验可直接扣除的失配常数，也没有做等效性检验。

## 问题与冻结设计

用户2026-10-15授权本实验，检验原交叉降分是否只是更换具体发声实例的正常下降；增强头、旧训练/中心/审计保持暂停。

- 8个AISHELL-1中文说话人，每人1个目标文本，真实不同utterance ID的N1/N2，以及同一独立参考句、同文本、两个预定seed的Qwen自克隆T1/T2。
- 自然条目有不同PCM；最大lag/RMS归一化波形相关0.084–0.155。语料没有正式重录/session标签，不称新采独立确认集或无上游训练污染。
- 同固定静态图image3、同Wav2Lip GAN和SyncNet V2。4来源16视频起步，按工程完成/预算扩到8来源32视频；没有按评分方向扩样或挑样。
- 全部canonical mono16k PCM16，句级RMS目标0.0376838172675746。保留原录音时长/时钟，没有裁静音、填充同长或时间warp；相同文本不等于局部音素时刻相同。
- 每来源完整4×4矩阵，源PCM直接评分，不解码MP4中的AAC作评价音轨。视觉/音频特征分别缓存，只改变矩阵配对。
- 每来源所有16格使用全部四段音频、四视频和31lags（±15帧）的共同有效query支持。查询数79–121，起始索引15，绝对anchor区间从[0.60,3.76)到[0.60,5.44)秒；没有边界零嵌入参与评分。
- 先对query求均值曲线，再取B=median(curve)、Dmin=min(curve)、C=B−Dmin、D0=curve[0 lag]。保存原128距离数组、64嵌入数组、完整曲线和offset。
- 统计单位为8个speaker/source，等来源权重；2000次来源bootstrap，seed20261015，普通95%描述性区间。128格和音素不是128/1967个独立样本。

## 实际执行与版本

云端Jan22模型注册成功但合成返回404 `Model not exist.`，没有生成音频。执行前provider修订改用LAN现有 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，无安装/下载。本地FP16首个T1发生非有限采样概率错误；保留失败01，0个波形时冻结protocol_v3，独立FP32/SDPA attempt02成功。文本、参考、seed、采样策略均未因失败或分数改变。

两seed为2026101501/2026101502各加100×speaker数字，do_sample=true、temperature0.9、top_k50、top_p1。full ICL含独立参考音频及其实际转写；S0006参考为两个不同句明确拼接。16段原24k FLOAT波形全部读回，两TTS的PCM均不同。该结果是**新本地Qwen provider**，不并入旧云TTS队列。

TTS成功GPU用时44.499+269.161秒，约5.23分钟；Wav2Lip32视频+评分共326.3秒，约5.44分钟。MFA CPU约4分14秒；Qwen ASR32次约243秒，0重试。所有实际任务已正常退出，未运行实现tests/pytest/compileall。

## 主结果

均值Sync-C（越高代表lag曲线的最佳谷值更突出）；行是驱动视频，列是评价音轨。

| 驱动\评价 | N1 | N2 | T1 | T2 |
|---|---:|---:|---:|---:|
| N1 | 5.681 | 2.299 | 1.568 | 1.718 |
| N2 | 2.262 | 6.179 | 2.400 | 2.182 |
| T1 | 1.350 | 2.323 | 6.653 | 2.400 |
| T2 | 1.509 | 1.977 | 2.046 | 6.343 |

损失定义为同一驱动的“自配C−替换音轨C”，正值表示下降。跨类为另一类两音轨均值；额外损失=跨类损失−类内损失。

| 驱动类别 | 自配C | 类内异实例C | 跨类C | 类内损失（95%区间） | 跨类损失（95%区间） | 额外损失（95%区间） |
|---|---:|---:|---:|---|---|---|
| 自然 | 5.930 | 2.280 | 1.967 | 3.650 [2.561,4.494] | 3.963 [3.537,4.399] | 0.314 [−0.579,1.656] |
| TTS | 6.498 | 2.223 | 1.790 | 4.275 [3.483,5.002] | 4.708 [4.375,5.101] | 0.433 [−0.101,1.006] |

额外损失正向来源为自然3/8、TTS5/8，异质性明显。当前区间仍允许中等来源特有差异，不能把“区间跨零”写成两者等价或没有来源效应。

自配TTS−自然C为 **+0.568 [0.096,1.100]**，7/8正向，保留了这批本地provider的原生评分优势。自配D0均值自然12.307、TTS13.176，TTS反而高0.869；D0较高代表零lag距离较大。两个端点不是同一含义，均非人类嘴型准确性。

10/128格的最佳lag在±15帧边界，全部为非对角；32自配格无边界命中。每格offset和完整距离曲线均保留，没有据此删格或调整lag。

## 独立内容与时间诊断

Qwen Audio 3.0 ASR Flash对32段源PCM独立识别，不给目标文本/hotwords，不据识别结果重采样或删来源。自然两次发声的识别文本一致或为他/她同音字。数字汉字/阿拉伯形式造成较大原始CER；另有S0006/T1年龄读法、S0008技术词/下称以及S0014专名的识别差异，尚无人工听辨裁决。S0020/T1虽10.8秒，ASR未检出整句重复/遗漏。原假设和全部原始转写保留；“同内容”指同输入/语料文本，不宣称每音素经人工确认完全一致。

Mandarin MFA2.2.17在与TFG完全相同的32段canonical PCM上对齐，32原TextGrid、48角色对、1967匹配音素行保存。各来源先计算时差，再等来源平均；不是池化音素统计。

| 配对 | 首音素onset绝对差均值(s) | 各自去onset后的音素中心MAE(s) |
|---|---:|---:|
| N1–N2 | 0.056 | 0.497 |
| T1–T2 | 0.230 | 0.414 |
| 自然–TTS四种配对等权 | 0.347 | 0.692 |

局部时差在类内已经很大，跨类平均更大；单一global lag无法吸收这种变化。MFA边界是估计，不用于证明精确因果份额。每role有7个unknown-phone标签（各8录音合计），平均phone标签覆盖0.971；未知段未当静音隐藏。评分anchor区间内recognized-phone union覆盖，按来源平均N1/N2/T1/T2为0.703/0.766/0.790/0.774。它是强制对齐诊断，不能替代语音真值或改变原评分支持。

## 证据边界与论文影响

普通实例替换的大幅降分已在自然→自然、TTS→TTS两种类内对照中实际复现。它足以否定“原N/T非对角下降只可能来自来源偏好”的解释方式，支持把发声实例/局部时钟作为核心评测控制。

本pilot不分解纯评分器来源偏好、生成器质量份额或视觉改善；额外跨类差仍混合局部时间、发音、voice和谱形等差异。类内基线不是可跨cohort/model直接搬用的常数。只有一个TFG、8人/8句、既有AISHELL训练语料和单个本地TTS，不作跨TFG或未见数据确认；LibriSpeech未使用。后续若要论文主张来源特有效应，应在新采/明确独立的重复录音上先控制局部事件时钟与处理sham，再配独立同步判断。本次不据此恢复增强头。

## 产物与验证

根目录 `runs/repeated_take_crossover_20261015_attempt01/`，原件同时保留在LAN；秘密仅runtime无回显输入，没有写文件/argv/BM/log。

| 权威产物 | SHA256 |
|---|---|
| protocol_v3_local_qwen_fp32.json | f86fa38175cd8adecadc2c638b6f90f5862824d57d21c545a9fd24e9b1f3369d |
| matrix_inputs_full8_v3_localqwen_fp32.json | 1ccc73a749d3027bb39d81b09100f389512a0300aac927812044c6a88435449d |
| matrix_scores.json | 66615d8910e6d3455902370d087981b4782de72338aa8944fc9ce4c35ab084ae |
| audio_manifest.json | 7aab8a8edd0b369ee1452c9f18f598b60a9f92e0ab17a662645b346e8ab5638b |
| video_manifest.json | 24ef07738ad0e556d30d5aee83ded2fe36a3ccc2c25bad2de31aa040bf7854ab |
| root_actual_qa.json | 2e0721cbe90f1642d0a4a507cd90671f1946133af9d609d709048fd28e64c438 |
| timing_diagnostics_canonical32/timing_summary_canonical32.json | cdd41398f41efb520017c17513ae497883ee5b7728fa4004ba1d8a839ae7e1e6 |
| analysis_delivery_attempt01/delivery_manifest.json | d94e65b7f24c586dad2097a0b97ce1ab6f3c7d76b8cb3f35a8bc5601484b26bd |

ROOT一次实际QA核验32源/视频身份、16原TTS、64嵌入、128距离数组、共同支持、全部C/B/Dmin/D0/offset与来源bootstrap。NumPy重建float32距离相对原Torch最大绝对差3.815e−6；曲线归约/统计重算误差0；PCM16量化后的最大RMS偏差1.100e−6，resampling样本时钟误差0。四个实际render/score阶段OS rc0。MFA原件和ASR结果均已本地读回，不把远端manifest视为媒体备份。

专用源码在 `scripts/experiments/repeated_take_crossover_20261015/`。图表与长表在 `analysis_delivery_attempt01/`：`sync_c_heatmap.svg`、`paired_losses.svg`、`mean_C.csv`、`mean_D0.csv`、`source_contrasts.csv`、`contrast_summary.csv`、`duration_support.csv`、`offset_by_cell.csv`。复现以冻结配置和实际阶段receipt命令/cwd为准；原已完成目录不重复launch。失败云请求、本地FP16及MFA不支持--version的探针均保留。

ROOT已实际查看两图：heatmap通过；paired-loss原图底部CI与文字重叠，单次绘图修复后的 `paired_losses_readable_v2.svg` / `.png` 可读性通过（SVG SHA `658571cd73433e7c5cb3bb7ba898f7b2dad38a0fe75e49a85efadaee0a5175b1`）。该修复仅用保存的analysis JSON重绘，不重跑评分；原图和原delivery manifest保持不变。

当前7个专用源码已提交本地Git `5a8d006892aae7ce937c4b1dc48fc8e64ec48c24`，未推送。首版posthoc分析器a46beabb的执行字节副本未保存，不能由当前绘图修订版1201184冒充；`plot_v2_receipt.json`明确记录此窄provenance缺口。原评分器、原数组、ROOT独立复算和当前可重算工具均保留，该缺口不改变本报告数值，但首版分析器不能逐byte复现。BM/run媒体为本地与LAN副本，尚未认证云备份。

## Observations

- [status] concluded
- [conclusion] 8来源类内替换已产生自然3.650/TTS4.275的大幅Sync-C损失；跨类额外损失0.314/0.433区间跨零，原交叉强交互需普通实例失配参照。
- [result] 32视频/128格/64嵌入/全部原距离数组及来源统计完成，ROOT实际QA通过，32段ASR和MFA诊断完成。
- [boundary] 同输入文本不统一局部时钟；ASR有数字/专名等未人工裁决差异；新本地provider、单TFG小pilot不作纯来源偏好、视觉质量或泛化结论。
- [decision] 保持增强头暂停；论文将发声实例配对和局部事件时钟列为必要控制，不从N/T交互直接归因评分偏好。

## Relations

- follows [[独立自然发声与Qwen重复合成的交叉失配实验 Spec 2026-10-15]]
- completes [[repeated-take-crossover-20261015]]
- evidence_for [[研究主线与证据地图]]
- evidence_for [[论文实验数据整理与写作主线]]
- qualifies [[独立40人同长交叉与评分边界探索 2026-09-29]]
