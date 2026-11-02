---
title: 62-gn-baseline-preserving-diagnostic
type: experiment
permalink: tts-exp/docs/experiments/62-gn-baseline-preserving-diagnostic
status: concluded
execution_status: SUCCEEDED
scientific_status: BASELINE_PRESERVING_GN_MEDIATOR_DIAGNOSTIC
inputs: report58 same May train/test first20.005s; post-waveform-normalization gains0.5/2
outputs: F:/Datasets/tts-exp/encoder_gn_baseline_preserving_20261020_v1
model: SHA-pinned facebook/hubert-base-ls960 and facebook/wav2vec2-base; frozen
code_paths:
- scripts/experiments/encoder_gn_baseline_preserving_20261020
- tests/experiments/test_encoder_gn_baseline_preserving_20261020.py
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
---

# 62-gn-baseline-preserving-diagnostic

✅ 完成160次冻结编码器前向、1280个完整层比较、独立CPU复核。0训练、0TFG、0SyncNet。保持各自原基线后，关闭GN的epsilon尺度响应，两模型所有观测层精确归零；把GN出口扰动统一为相对Frobenius幅度0.001，wav2vec2末层方向响应仍为HuBERT的3.22倍（train）/13.06倍（test）。GN是此输入增益响应的入口，但出口扰动大小与gamma来源不足以单独解释末层差距；后续路径对扰动方向及工作点的响应也不同。

## 为什么这次要保持基线

报告60直接互换首层GN的gamma/beta，交换本身使末层基线方向偏移约63–74度，混合了正常工作点改变和音量敏感性改变。本轮让每个编码器始终沿用自己原来的基线，只替换GN出口随音量变化的增量。对GN变体theta定义：

`injected(a) = original_GN(1) + [GN_theta(a) - GN_theta(1)]`。

所以a=1时，干预后GN出口逐值等于原始GN基线。增量在CPU float64相减后转float32；所有模型输入保持scale1，hook位于首GN输出、GELU之前。模型参数、epsilon属性和后续计算全部保持原样；功能式GN产生目标，未改写模型或权重文件。theta可改epsilon或gamma；beta保持原值，且固定beta在括号差分中抵消。这是对GN响应通路的干预，不是推荐实际部署的替换GN。

输入沿用报告58：May train/test全split先波形标准化，再各取前320080采样（20.005秒）；0.5/2增益作用于此后，不再波形归一化。两模型源revision、三文件SHA及父实验manifest逐项绑定protocol。HuBERT revision dba3bb02fda4248b6e082697eee756de8fe8aa8a；wav2vec2 revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8。FP32/eager、eval、无梯度、TF32关闭、确定性计算，单本机RTX4060运行，峰值reserved约1.01GiB，无Ubuntu调用。

## 条件与指标

每模型×片段40次前向：基线与重复2；自然GN增益回放2；epsilon0/1e-7/1e-3各两个增益6；两个代表性零响应基线控制2；供体gamma/自身固定排列gamma各两个增益4；自然减半/加倍方向各四档等幅剂量8；两类随机方向×两固定seed×两剂量×正负16。共2模型×2片段×40=160。功能式GN目标生成不另计为完整模型前向。

自然增益回放在首GN注入原始scale0.5/2的GN输出，末层与报告58真实缩放波形的缓存逐值相同，证明hook重放了本轮要解释的效应。首层卷积无bias且正增益响应可缩放的前提沿用报告58实测。

GN剂量`r = ||delta z||_F / ||z_baseline||_F`，各模型相对于自己的GN基线范数归一化，**不是相同绝对幅度或相同GN角度**。末层方向变化`d = sqrt(mean_t ||u_changed(t)-u_base(t)||²)`，u为逐帧单位特征；它是单位方向改变量的RMS，不是平均角度或相对特征L2。另报告GN单位方向RMS和`d / GN方向RMS`，避免把等Frobenius剂量误读为等角度。不同模型坐标未对齐，仅比较自身方向变化的大小。

## 关闭或改变epsilon响应

下表为减半方向、保持原基线的末层方向RMS。原始两模型首GN的epsilon本来就同为1e-5。

| 片段 | 模型 | eps0 | eps1e-7 | 原eps1e-5 | eps1e-3 |
|---|---|---:|---:|---:|---:|
| train | HuBERT | 0.00000000 | 0.00000435 | 0.00042290 | 0.01896954 |
| train | wav2vec2 | 0.00000000 | 0.00006167 | 0.00630448 | 0.13458551 |
| test | HuBERT | 0.00000000 | 0.00000594 | 0.00057555 | 0.02291892 |
| test | wav2vec2 | 0.00000000 | 0.00020570 | 0.02228542 | 0.17932493 |

eps0时8个非基线条件、所有8个观测层都逐值等于原基线，而原基线自身完全保持。这排除了报告60“改GN导致基线换了”的混杂，支持GN的epsilon依赖响应是这两checkpoint/片段对正全局增益的必要入口。降低epsilon大幅减弱响应，增大则增强；不能因此说两个模型原始epsilon不同。其差异还依赖GN输入的卷积逐通道方差、gamma和后续路径。

## 自然响应与出口等幅对照

自然RMS减半：

| 片段 | HuBERT GN相对L2 | wav2vec2 GN相对L2 | GN幅度W/H | 末层方向W/H | 每单位GN幅度响应W/H |
|---|---:|---:|---:|---:|---:|
| train | 0.00061228 | 0.00272776 | 4.455 | 14.908 | 3.346 |
| test | 0.00086695 | 0.00314764 | 3.631 | 38.720 | 10.665 |

这可写作“末层响应比=GN幅度比×条件响应比”，只是定义带来的代数分解，**不是因果贡献百分比**。

将各自自然方向缩放到相同GN相对剂量0.001：

| 片段 | 自然增益方向 | HuBERT末层方向RMS | wav2vec2末层方向RMS | 末层W/H | GN方向RMS W/H | 每单位GN方向响应W/H |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.5 | 0.00069156 | 0.00222874 | 3.223 | 2.153 | 1.497 |
| train | 2 | 0.00069069 | 0.00216912 | 3.141 | 2.160 | 1.454 |
| test | 0.5 | 0.00066414 | 0.00867455 | 13.061 | 1.757 | 7.432 |
| test | 2 | 0.00066431 | 0.00537047 | 8.084 | 1.762 | 4.588 |

等相对Frobenius剂量后差距依然存在，不能只用GN出口扰动更大来解释。需同时承认GN的角度/范数组成不同：半尺度方向的GN单位方向RMS比train约2.15、test约1.76；相对于此实际方向变化，末层条件响应比仍约1.50/7.43。两模型被注入的是各自自然方向，不是同一已对齐方向，因此这些数值描述各自工作点与方向的联合响应，不能唯一归于某一层权重或当作全局Jacobian范数。

四档剂量1e-4/3e-4/1e-3/3e-3均已完成。半尺度方向末层W/H在train为3.15–3.34，test为9.99–13.06；加倍方向为train3.08–3.19、test6.08–9.42。小剂量test1e-4仍为9.99倍，按GN方向RMS归一后的条件响应比5.69，差异没有只出现在某一个大剂量上。完整曲线见diagnostic_summary.png与paired_model_ratios.json。

## 差距沿后续层如何变化

下表为test、半尺度自然方向、GN相对剂量0.001，每个阶段都对自己的原基线计算方向RMS。Conv7为第七卷积块输出，Tn为Transformer第n层输出；阶段观测较稀疏，不声称确定唯一发生层。

| 阶段 | HuBERT | wav2vec2 | W/H |
|---|---:|---:|---:|
| gn | 0.00059174 | 0.00103995 | 1.757 |
| conv7 | 0.00092985 | 0.00217498 | 2.339 |
| projection | 0.00076100 | 0.00153051 | 2.011 |
| T1 | 0.00082452 | 0.00137529 | 1.668 |
| T4 | 0.00079735 | 0.00127504 | 1.599 |
| T8 | 0.00062118 | 0.00522936 | 8.418 |
| T9 | 0.00063911 | 0.01108639 | 17.347 |
| last | 0.00066414 | 0.00867455 | 13.061 |

GN已经有角度组成差异；卷积尾部仍有差异。wav2vec2在T4至T8/T9间进一步增大，HuBERT在此区间较稳定或减小。不是GN一处制造固定倍数后沿全网原样传递。这里的跨阶段方向量在各层不同空间计算，不能解释成能量守恒或严格乘法放大链。

## 保持基线的gamma交换

只替换响应增量中的gamma，beta及基线保持原样。供体按原512通道编号交换，无跨模型通道对齐；另用固定seed20261020对自身gamma排列作对照。下表为半尺度末层方向RMS。

| 片段 | 接收模型 | 原始响应 | 供体gamma响应 | 自身gamma排列响应 |
|---|---|---:|---:|---:|
| train | HuBERT | 0.00042290 | 0.00055071 | 0.00010328 |
| train | wav2vec2 | 0.00630448 | 0.00165586 | 0.00183521 |
| test | HuBERT | 0.00057555 | 0.00076330 | 0.00016732 |
| test | wav2vec2 | 0.02228542 | 0.00599934 | 0.00401489 |

供体gamma使HuBERT响应上升、wav2vec2下降，但没有互换响应水平；test双方供体臂仍相差约7.86倍，train约3.01倍。自身排列也能显著压低wav2vec2，因此不能称HuBERT gamma具有独特减敏效果。gamma确实影响GN响应的强度与通道分布，但本轮不支持“仅gamma来源决定两模型差距”。此臂有意排除了beta改变基线的效应，不能外推beta全无作用。

## 固定随机方向探针

两类：完整[T,C]独立高斯；512个随机通道系数乘各自`GNbaseline-beta`时间信号。每类两个预定seed20261020/20261021，同seed在各自坐标生成，未做语义或通道对齐；剂量3e-4/1e-3、正负各一次。它们是人工GN探针，不一定可由真实音频产生。下表取0.001剂量，均值与范围来自2seed×2sign四条件，误差条不是置信区间。

| 片段 | 方向族 | HuBERT输出/实测GN剂量：均值[范围] | wav2vec2：均值[范围] | 均值W/H |
|---|---|---:|---:|---:|
| train | iid_gaussian | 5.5998 [5.1413, 6.5639] | 55.7428 [48.3660, 60.1562] | 9.954 |
| train | channel_modulated | 0.4414 [0.4076, 0.4752] | 1.2144 [0.9034, 1.5811] | 2.751 |
| test | iid_gaussian | 5.8585 [5.1242, 6.7539] | 30.3641 [20.4859, 45.6651] | 5.183 |
| test | channel_modulated | 0.3720 [0.3617, 0.3818] | 1.3845 [1.0554, 1.7679] | 3.722 |

这说明差距不只存在于某一条自然减半方向，同时具有很强的方向依赖。用正负探针的单位末层输出计算odd=(u+−u−)/2、even=(u++u−)/2−u0。wav2vec2 IID的even/odd在本轮约0.195–1.243，部分非线性很强；通道调制方向全模型约0.004–0.056。因此报告有限剂量响应，不称所有探针都位于线性区，也不从四个方向估计完整Jacobian/显著性。probe_odd_even.json保留全部结果。

## 验证与复现

独立analyze.py不导入producer/metrics，检查1280层的全部逐帧矩、81920个采样向量、160个完整末层。相对L2重建误差0；解析GN样本目标最大误差3.975e-06；16个原始随机方向全数组hash重建相同；所有等幅/随机GN采样目标逐值复现；剂量最大相对误差1.818e-06。8自然回放末层与父缓存逐值相同，16基线/重复/代表性anchor控制通过；每个theta的a=1目标逐值等于原基线。8个eps0条件全部观测层无变化；两模型state_dict前后SHA精确一致，源权重未修改。

三项数值单测通过：任意变体基线anchor不变、原生anchor精确回放与等幅剂量、方向指标正缩放不变/零误差。producer和metrics与运行前protocol SHA一致，父helper与manifest绑定。源、测试及consumer快照在source/；逐调用末层数组、64帧层样本、完整逐帧矩、GN参照/方向台账均保留。manifest.json逐文件SHA且交付时重新读取核验，未声明云备份。producer环境numpy1.26.4复现随机探针，绘图使用独立CPU运行时，不增加GPU调用。

复现顺序：在protocol所列torch/transformers/numpy环境运行run.py（只接受不存在的新输出目录，禁止覆盖本冻结结果），随后analyze.py和finalize.py。源脚本使用本机绝对数据路径；移植时需在新实验命名空间显式改路径。父报告58提供标准化输入与原缓存，父report54权重目录提供固定模型；本kit未打包大权重/全部原数组，也不宣称是独立离线数据包。源hash可核对已执行版本。权威根：F:/Datasets/tts-exp/encoder_gn_baseline_preserving_20261020_v1。

## 结论边界

在本轮冻结条件里，GN的epsilon响应是全局音量进入表征变化的必要入口；GN响应大小、通道方向组成及后续网络在原工作点上的敏感性共同影响末层差距。基线控制和出口等幅对照削弱了“直接互换GN产生基线偏移”和“只因GN扰动幅度更大”的解释，但并未唯一分离前卷积、gamma、后续每层权重及其相互匹配的因果份额。两checkpoint、两段May、两seed/方向族、一个排列；没有总体显著性、语义/音素或感知质量验证。尤其不能把该编码器端点结论等同于报告56 Sync-C训练分差的因果解释，后者需要另行下游实验，本轮严格未重训。

## Observations
- [status] concluded；160冻结前向/1280层独立复核；0训练/TFG/SyncNet。
- [result] 保持原基线后关闭GN epsilon响应，两模型所有观测层精确不变。
- [result] GN等相对幅度0.001，半尺度方向末层W/H train3.22、test13.06，GN角度组成与后续条件响应均有差异。
- [result] gamma供体交换未互换响应；人工随机方向也有差异且明显方向依赖/部分非线性。
- [limitation] 不把有限方向条件响应称全局Jacobian，不量化下游训练分差的因果百分比。

## Relations
- follows [[60-encoder-groupnorm-swap]]
- extends [[58-encoder-layer-epsilon-diagnostic]]
- relates_to [[56-may-postnorm-scale]]
- supports [[研究主线与证据地图]]