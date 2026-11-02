---
title: 74-avs-boundary-context-vs-silence-mouth-state
type: experiment
permalink: tts-exp/docs/experiments/74-avs-boundary-context-vs-silence-mouth-state
status: concluded
scientific_status: SCORER_BOUNDARY_CONTEXT_SENSITIVITY
date: '2026-10-28'
run_id: avs_boundary_content_cloud_20261028_attempt01
training: false
tags:
- experiment
- avs-u
- boundary
- silence
- mouth-state
- cloud-tts
---

# 74-avs-boundary-context-vs-silence-mouth-state

✅ 同组80媒体的内容状态检查、固定目标位置干预、整段上下文扩展与独立复算完成。当前AVS_u实现存在明显的边界/上下文敏感性；静音或闭嘴本身不足以解释首尾高分。尚未估计各机制占比，也未将效应单独定位到位置卷积层。

| 修改日期 | 修改 | 授权 |
|---|---|---|
| 2026-10-28 | 首次：80媒体原生重放、状态分层、两类上下文干预、独立审计及最终报告 | 用户要求验证首尾高分来自静音闭嘴还是编码器边界 |

## Material Passport
- 类型：code experiment，探索性评分器机制验证；非人评或感知同步真值。
- 状态：CONCLUDED；SCORER_BOUNDARY_CONTEXT_SENSITIVITY。
- Run：runs/avs_boundary_content_cloud_20261028_attempt01（Windows与Ubuntu相同相对目录）。
- 数据：既有中文云Qwen/自然音频驱动的Wav2Lip GAN视频，20N+60T，20独立来源；严格同参考12来源另报。沿用原RMS、源PCM及源视频，无TTS/TFG重生成、交叉音轨、DTW或训练。
- 执行：academic-research-suite experiment-agent code runner；ROOT从原始输入、landmark、embedding独立重建，不导入producer。

## 问题与先前线索
报告73和追加逐帧检查发现，AVS_u全片T−N为−0.007529，共同中央窗为+0.009466；两臂均降分，自然降得更多，17/20来源的窗口效应向TTS移动。自然被排除首尾比例更大，且尾部高分突出。这是本次机制实验的已知线索，不冒称盲法新发现。

本轮分开检验：内容状态分层后高分是否仍在；目标内容完全不变时，改变其编码边界位置与周围上下文是否足以改变分数。所有结论适用于本项目明确实现的AVS_u：官方AV-HuBERT VSR Base LRS3 433h骨干和逐帧cosine归约，不是作者发布的完整官方评分程序，也不能外推所有AVS实现。

## 冻结协议
protocol.json/source_frozen.py在新模型前向及landmark分析前固定。视频F=59–195帧，25fps，80条全部满足设计。FP32 eval，复用原AVS音频logfbank/stack4/每帧LN与AutoAVSR四点口部裁剪、灰度88变换；全80原生A/V embedding必须与旧缓存最大绝对差≤1e−5，才对该条做干预。

### 固定目标、等长自然上下文
每clip以t=(F−5)//2选择同一5帧目标，即200ms。三段输入均为25帧/1秒，分别从t、t−10、t−20开始，让目标处在局部0:5、10:15、20:25。三条件的目标音频前端帧张量和口部视频张量逐元素相同，时序配对不变，输入总长度相同；周围来自原片的真实内容和边界位置改变。

这是一秒短输入压力测试，不是原生长片评分协议。它检验边界/上下文效应，不能孤立成纯绝对位置效应，也不能将效应大小直接外推原片。

### 整段前后补充上下文
原配置conv_pos=128帧（25fps下约5.12秒，半宽约64帧），因此在原输入两侧各扩展75帧，使原首尾越过此局部位置卷积边界范围。两种方式分别运行：
- reflect：按时间反射原输入帧；
- repeat：重复原端点帧。

音频/视频前端帧使用相同映射；原F帧在75:75+F逐元素不变。这里补的是预处理后的输入帧，不是合成新波形或真实延长语音。两种方法改变合成上下文和总长度，Transformer全局注意力仍可影响中间，故同时计算中心变化，不能把全部变化归为局部卷积。

固定原始head15、center15、tail15帧（每段600ms），按原帧索引计算native/reflect/repeat分数。核心调整量：
DID = (扩展后首尾均分−原首尾均分) − (扩展后中央均分−原中央均分)。
负值表示首尾相对中央的额外高分减弱。

### 内容状态
- 低能量：每40ms RMS < 全片RMS的0.1；敏感性0.05/0.2。只是能量代理，不当作经验证VAD。
- 闭嘴代理：MediaPipe FaceMesh内唇13/14距离除以78/308宽度 <0.05；敏感性0.03/0.08。像素坐标先乘源图宽高；仅是几何代理，不是音素或人评真值。
- 分quiet_closed、quiet_open、active_closed、active_open四状态；active仅表示非低能量。
- 观测首尾−中央按同clip同状态可用帧比较，空支持保留NA/coverage，不按分数删样。

主统计分N/C两臂，每来源先平均可用重复draw，再等权来源；10000次来源bootstrap、seed20261028、边际95%描述区间。不是多终点确认性检验，不做中介百分比。

## 固定目标的位置结果
20来源/臂；目标同一5帧内容，周围输入上下文不同。

| 指标 | 自然均值 | TTS均值 |
|---|---:|---:|
| 目标在开头的AVS_u | 0.386541 | 0.366078 |
| 目标在中间 | 0.074082 | 0.061294 |
| 目标在结尾 | 0.258243 | 0.202189 |
| 开头−中间 [95%CI] | +0.312458 [+0.281260,+0.344939] | +0.304784 [+0.273215,+0.336122] |
| 结尾−中间 [95%CI] | +0.184161 [+0.144911,+0.225172] | +0.140895 [+0.122238,+0.160546] |
| 首尾均值−中间 [95%CI] | +0.248310 [+0.215270,+0.281393] | +0.222839 [+0.202435,+0.243164] |

目标原始声音、嘴形及长度不变，仍能因编码位置/周围上下文产生较大分数变化。不能只凭原首尾常有静音/闭嘴解释此结果。

## 整段上下文交叉检查
| 同一原始帧的区域分数 | N头 | N中央 | N尾 | T头 | T中央 | T尾 |
|---|---:|---:|---:|---:|---:|---:|
| native | 0.252234 | 0.075631 | 0.308607 | 0.227299 | 0.070630 | 0.187637 |
| reflect75 | 0.045078 | 0.058942 | 0.089190 | 0.069535 | 0.055080 | 0.066804 |
| repeat75 | 0.150037 | 0.070220 | 0.261698 | 0.158911 | 0.069820 | 0.172681 |

| 调整量DID | 自然 | TTS |
|---|---|---|
| reflect75首尾变化−中央变化 | −0.196598 [−0.222276,−0.169702] | −0.123748 [−0.138381,−0.109267] |
| repeat75首尾变化−中央变化 | −0.069142 [−0.104008,−0.031400] | −0.040861 [−0.065488,−0.013528] |

原首尾内容保留不动，补充上下文仍削弱首尾相对高分；中间也有变化，reflect中央N−0.016689、T−0.015550，已通过DID单列。repeat效果明显较弱，不能把所有改变都解释成纯序列位置或指定层故障。

严格12来源的固定目标两种位置效应与reflect DID均保持方向且区间不跨0；repeat严格层N/C DID区间均跨0。完整数值在summary.json，不能宣称所有分层与上下文类型均同等稳定。

## 内容状态的证据
原生数据中，仅保留非低能量且嘴部张开（E0.1/L0.05）的同状态帧，首尾−中央仍为：
- 自然 +0.137968，95%[+0.092791,+0.181305]，20来源；
- TTS +0.040360，[+0.018712,+0.063629]，19来源，S0031无成对状态支持，保留缺失而非造值。

因此原生首尾优势不只存在于低能量闭嘴帧。但这是状态分层的观察比较，仍可能有音素和语境混杂。

### 追加：固定相同内容状态的干预
ROOT在部分前向及首条QA完成后、未读取全组效应汇总时固定target_state_analysis_amendment.json；不是最初协议，也不伪装为预注册。原始能量/嘴形mask在所有编码条件间完全固定，未增加GPU计算。所有9阈值组合与4状态保留，有限支持明确列出。

主要active_open结果：
| 同一状态、同一目标帧 | 自然 | TTS |
|---|---|---|
| 开头−中间 | +0.317873 [+0.262898,+0.382546]，20来源 | +0.289615 [+0.247670,+0.335634]，18来源 |
| 结尾−中间 | +0.187978 [+0.142444,+0.236923]，20来源 | +0.138112 [+0.112239,+0.163865]，18来源 |
| reflect75 DID | −0.140961 [−0.183818,−0.097292]，20来源 | −0.034735 [−0.057837,−0.013614]，19来源 |
| repeat75 DID | −0.044895 [−0.081527,−0.008499]，20来源 | −0.000599 [−0.027236,+0.027770]，19来源 |

固定目标的开头/结尾增分在两臂全部9种阈值组合都保持正区间，N来源覆盖12–20、T15–20。reflect active_open DID在9设定中N8项、T6项负区间；repeat active_open T全部跨0。不能把这项不一致隐藏，也不能由阈值筛选一个最有利结果作为新主端点。

这些结果足以反对“只是静音/闭嘴内容导致首尾高分”的单一解释，支持编码边界/上下文是重要来源。尚未用严格内容×边界完整因子设计估计静音、闭嘴及上下文各自因果占比，不能称已证明边界独占或静音完全无贡献。

## 工程核验与独立审计
- 全80原生A/V编码重放与旧缓存最大差1.7285347e−6，低于冻结1e−5；ROOT用报告73实际旧特征独立核对，不只信任producer记录。
- 所有固定目标输入张量及扩展原区逐元素相同；全部映射/裁片边界独立验证。
- FaceMesh 9442/9442帧有效，保存原始四点及比例；42张QA图已回传，执行器/ROOT查看代表性关键点图与N/C目标上下文联系图，非人类同步评判。
- ROOT不导入producer，从80NPZ重算6条件cos、区域/目标分数、lip ratio、四状态9阈值、两层来源CI，以及追加同状态干预。最大端点误差6.66e−16、统计1.33e−15；Ubuntu与Windows均PASS。
- Ubuntu独立读取80份原PCM重建每帧RMS，与保存值逐项一致；Windows原PCM不在相同Linux路径，未虚称本地重复读PCM。
- 执行器Ubuntu/Windows各5真实artifact测试PASS；ROOT在Windows运行producer5项+独立审计1项，共6项PASS；py_compile通过。
- 初次在任何模型前向前因python_speech_features vendor路径未注入而退出。修复仅移动import至既有loader之后，不改环境、阈值或科学设计；原日志/source_frozen与execution_repair.json、source_executed.py分别保留，未回写冻结源。
- 完整约403MB输入/特征/映射/QA/小结果在本地E:/Documents/tts-exp与Ubuntu /home/wjj/tts-audio/tts-exp同run；自有GPU进程已退出，外来5652MiB进程未动。

代码：scripts/experiments/avs_boundary_content_20261028.py；独立审计/图脚本scripts/experiments/avs_boundary_content_independent_audit_20261028.py；对应tests同名。
关键产物：protocol.json、source_frozen.py、source_executed.py、execution_repair.json、inputs.json、items/*.npz/.json、items.json、analysis_rows.json、summary.json、target_state_analysis_amendment.json、content_stratified_interventions.json、parent_independent_audit_ubuntu.json、parent_independent_audit_windows.json、parent_audit_source.py、chain_status.json。
图：boundary_context_diagnostics.png/.pdf。

## 结论与下一步边界
当前实现不能把首尾的高AVS_u直接解释为更准确的口型对应；同内容的位置/上下文变化足以造成大幅提分，且非低能量张嘴帧也出现。它解释了为何全片均值易受首尾比例影响，并支持报告73对窗口支持的谨慎处理。

这不证明中央窗评分就是感知真值，也不证明SyncNet无偏或TTS真实口型更好。若继续定位层级，可区分视觉前端时间padding、位置卷积和共享Transformer上下文；本轮未启动模型层消融、重新训练或将派生评分推广至其他数据。

## Relations
- follows [[73-common-window-cloud-curve-decomposition]]
- relates_to [[AVS_u、AVS_m、AVS_v原论文与本项目适用性核对 2026-10-27]]
