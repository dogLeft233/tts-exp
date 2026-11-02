---
title: 49-may-hubert-gain
type: experiment
permalink: tts-exp/docs/experiments/49-may-hubert-gain
status: concluded
tags:
- synctalk
- hubert
- may
- gain
- normalization
---

# 49 — May原音频上的官方HuBERT增益复核

日期：2026-10-16。✅ 编码器实际前向完成，0训练、0新视频、0SyncNet。用户提到“原装音频编码器”，本轮已提出路径澄清，在未取得更明确选择时，按此前被描述为近似不敏感的HuBERT路径进行补测；不能把本结果冒充DeepSpeech或所有官方编码器的结论。

| 修订 | 日期 | 内容 |
|---|---|---|
| Initial | 2026-10-16 | 相同May全段纯增益、原始HuBERT权重与官方归一化、完整表示和12秒匹配区域复核 |

## 结论

**官方HuBERT路径确实近似抵消纯波形增益，但不是严格不变。** 同报告48实际使用的May −6/0/+6dB驱动，在相同12秒区域的HuBERT表示相对L2变化约0.0093%/0.0090%；全段约0.0088%/0.0074%，重复误差0。

这里是完整官方前处理+编码器的响应，不是仅看配置推测，也不是HuBERT网络本身的严格数学不变性。前端的整段零均值/方差归一化大幅抵消增益，分母epsilon留下微小尺度残差；模型内浮点运算和非线性可继续改变此残差。未将小但非零的变化舍入为“完全不吃音量”。

本次May渲染用的**原版AVE是另一条路径**，报告48已经实测视频和评分响应；它没有被换成修改过的音频编码器。默认DeepSpeech的完整TF图仍未验证。

## 冻结输入和运行方式

- 完全复用报告48实际驱动文件：`/dev/shm/synctalk-may-gain-inference-20261016/render_attempt01/{gain_m6,gain_0,gain_p6}/data/aud.wav`。全段约242.923秒、mono16k DOUBLE，纯标量关系和无削波在报告48已逐样本验证，不重新重采样/量化或做额外归一化。
- 模型为官方指定 `facebook/hubert-large-ls960-ft`，revision `ece5fabbf034c1073acae96d5401b25be96709d8`，权重SHA256 `9cf43abec3f0410ad6854afa4d376c69ccb364b48ddddfd25c4c5aa16398eab0`。
- 保留官方 `do_normalize=true`。完整波形先归一化，再沿固定SyncTalk commit `43079fb3f349675f6cb33031bb07262f7566cdd5` 中 `data_utils/hubert.py` 的原helper切20秒段并编码、拼接，非截12秒重新归一化。
- 使用报告47已验证的严格权重加载：只兼容映射旧weight_norm参数名，全部权重逐值核对，缺失/额外参数均空，有效位置卷积权重最大误差0；eval/no_grad，不改变官方前处理、不做模型训练。
- 每档实际运行两次完整官方helper，保存首次输出及重复误差。18个保存阶段数组的dtype/shape/SHA、finite、RMS/peak均重新验证。长音频hook仅记录最后一个官方chunk的CNN/投影阶段；processor和hidden覆盖完整全段，不把hook误称全轨。
- 输出完整hidden为[12145,1024]，去奇数尾后paired cache为[6072,2,1024]，三档长度相同。报告48视频img_id5520–5819对应hidden行[11040:11640]共600行；实际与paired cache[5520:5820]展开逐位相同，不重新选择相位或偏移。

## 实际数值

相对0dB基线，relative L2定义为||H_gain−H_0||/||H_0||：

| 驱动档位 | processor相对变化 | 全段hidden相对变化 | 同一May12秒hidden变化 | 同12秒maxabs |
|---|---:|---:|---:|---:|
| −6dB | 0.006545% | 0.008757% | 0.009302% | 0.000833809 |
| 0dB | 0% | 0% | 0% | 0 |
| +6dB | 0.001658% | 0.007366% | 0.009009% | 0.000893962 |

同12秒hidden余弦：−6dB约0.999999995674，+6dB约0.999999995942。高余弦本身不能证明相等，本次同时保存最大差和relative L2。每档完整hidden重复最大绝对差均0；微小增益响应真实高于本次重复误差。

processor结果与整段官方公式 `(x−mean(x))/sqrt(var(x)+1e−7)` 在三档逐值一致，最大公式误差0。若epsilon为0且忽略浮点舍入，正标量增益会数学抵消；这里实际epsilon不是0，所以不能宣称严格不变。

只有一个May来源，不做显著性或跨身份结论。亦未把HuBERT小变化与AVE向量变化量直接当作同尺度性能指标。

## 生成器边界

现有May checkpoint的AVE音频MLP接收512维特征；HuBERT输出1024维、网络与接口不同。直接替换编码器并随机初始化适配层不能构成有效的HuBERT生成对照。因此本次只验证“原装HuBERT链是否保留增益”，没有声称HuBERT生成的视频/SyncNet分数不变。后续若要检验生成层，需要匹配的HuBERT-trained checkpoint；本次未训练、不混接权重。

## 证据与产物

- Ubuntu运行目录：`/dev/shm/synctalk-may-hubert-gain-20261016/`，属于易失SHM；原始输出在 `attempt01/`，主复核为 `independent_analysis.json`，12秒实际数组为 `hidden12s_gain*.npy`。
- 持久本机目录：`E:/Documents/tts-exp/runs/synctalk_may_hubert_gain_20261016/`；包含冻结protocol/input_spec、全18阶段数组与统计、同12秒数组、重复记录、独立复核和完整压缩包。
- 运行程序复用报告47的 `audit_hubert.py`；新CPU读回脚本是 `scripts/experiments/synctalk_may_gain_inference_20261016/analyze_hubert_may.py`。source-clock合同unittest已通过，拒绝不足支持的切片。
- canonical statistics SHA256：`dcf2ae37c50f8f6527ff13c6f360e11d0de73a45d7e7a3724cf973c16b1114f8`。没有新资源下载，根盘保持约354MB，未删除其他实验记录。

## Observations

- [status] CONCLUDED：原装HuBERT全段真实前向及同May片段响应已复核，0训练/0视频。
- [result] 同12秒−6/+6dB hidden相对L2为0.009302%/0.009009%，重复误差0；近似不敏感而非严格不变。
- [conclusion] 增益主要被官方整段波形归一化抵消；该结论不属于原版AVE或尚未跑完整图的DeepSpeech。

## Relations

- follows [[48-synctalk-may-gain-inference]]
- extends [[47-synctalk-amplitude-frontend]]
- relates_to [[研究主线与证据地图]]