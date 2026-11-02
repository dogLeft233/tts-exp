---
title: 60-encoder-groupnorm-swap
type: experiment
permalink: tts-exp/docs/experiments/60-encoder-groupnorm-swap
status: concluded
execution_status: SUCCEEDED
scientific_status: FROZEN_ENCODER_PARAMETER_TRANSPLANT_DIAGNOSTIC
inputs: report58 same May train/test first20.005s chunks; scales0.5/1/2
outputs: F:/Datasets/tts-exp/encoder_groupnorm_swap_20261020_v1
model: report58 SHA-pinned HuBERT-base and wav2vec2-base; only firstGN gamma/beta
  transplanted in memory
code_paths:
- scripts/experiments/encoder_groupnorm_swap_20261020
- tests/experiments/test_encoder_groupnorm_swap_20261020.py
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
---

# 60-encoder-groupnorm-swap

✅ 已完成40次冻结编码器前向、832个完整层比较；0训练、0TFG、0SyncNet。交换首层GroupNorm的gamma/beta会明显改变两模型本身的特征方向，但没有互换两模型的RMS敏感性。模型间末层时间几何CKA增加，但自身参数打乱也产生相似效应，不能当作变成供体的语义表征。

## 交换的究竟是什么

用户要求互换HuBERT与wav2vec2的首层GroupNorm，观察各自交换前后及彼此之间的方向。两者GN的512组、512通道、affine=True、epsilon=1e-5相同，且GN无BatchNorm式running statistics，因此完整模块的有效互换等价于按原通道序号交换已学习gamma与beta。保持各自卷积、后续网络、输入和epsilon不变；只在内存换参，不写原权重文件。另设自身gamma/beta用同一固定seed20261020排列共同打乱的对照，不搜索最优排列、不做跨模型通道对齐。

每模型两段May音频，沿用报告58先全split标准化、后取前320080采样20.005秒，再乘0.5/1/2。三臂native/swapped/self_permuted共36完整前向，另4个默认重复，总40。模型为facebook/hubert-base-ls960 revision dba3bb02fda4248b6e082697eee756de8fe8aa8a、facebook/wav2vec2-base revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8；同报告58三文件及权重SHA全核对。模型eval、FP32/eager、TF32关闭、无梯度与optimizer；本机4060峰值reserved见completion.json，没有Ubuntu调用或重训。

## 交换本身对方向的影响

下表固定输入尺度1，比较末层全部1000帧的单位特征方向。角度是逐帧夹角的平均，非平均余弦的反余弦。

| 片段 | 接收模型 | 交换后对自身原模型余弦 | 交换后平均夹角° | 自身打乱后平均夹角° |
|---|---|---:|---:|---:|
| train | hubert_base | 0.322874 | 69.553 | 68.331 |
| train | wav2vec2_base | 0.239391 | 75.815 | 71.217 |
| test | hubert_base | 0.405648 | 63.433 | 63.489 |
| test | wav2vec2_base | 0.276798 | 73.535 | 68.176 |

这是一项很强的表征干预，远大于单纯RMS减半的微小变化。自身打乱参数也造成类似偏移，说明结果包含打破已学习通道匹配的效应；不能仅凭偏移把接收模型叫作供体模型。first-conv输出在换参前后逐值相同，所有非GN参数SHA不变，确认变化由GN换参进入。

## 交换后对RMS的响应

每臂分别以自己的尺度1为参照；每格为平均方向夹角° / 全特征相对L2变化%。这是音量响应，与上一节换参本身的基线偏移不同。

| 片段 | 接收模型 | native | swapped | self_permuted |
|---|---|---:|---:|---:|
| train | hubert_base | 0.014622 / 0.04444% | 0.036016 / 0.07733% | 0.043288 / 0.08804% |
| train | wav2vec2_base | 0.227198 / 0.61907% | 0.248650 / 0.48412% | 0.149110 / 0.28825% |
| test | hubert_base | 0.021376 / 0.05938% | 0.042274 / 0.09651% | 0.043748 / 0.09442% |
| test | wav2vec2_base | 0.417371 / 1.97346% | 0.330161 / 0.64017% | 0.172431 / 0.38296% |

测试片段HuBERT换参后方向响应0.02138→0.04227度，wav2vec2为0.41737→0.33016度，仍相差约7.8倍。wav2vec2训练片段方向响应反而从0.22720升至0.24865度，所以不能称接受HuBERT参数就统一减敏。其相对L2在两个片段下降，是另一种范数加权端点，应分开描述。自身打乱后的wav2vec2方向响应比供体换参更低，表明减敏并非HuBERT供体参数的特异证据。

首层GN的相对L2在换参后两边都下降：测试片段HuBERT0.08669%→0.07040%，wav2vec2 0.31476%→0.22678%；但HuBERT末层响应反而上升。因此后续网络及换参后的工作点也是决定因素，不能把所有差异归于GN参数大小。解析GN公式在24个尺度对照上重建最大误差1.461e-8。

将每帧单位特征的RMS改变量Delta u展平，交换前后该改变量的整体余弦，测试半尺度HuBERT为+0.02267、wav2vec2为-0.01171，接近0；说明换参还改变了音量响应方向，不能看成只把原响应乘一个系数。这里每个差值都在同一接收模型坐标中计算。所有0.5/2结果见terminal_full_geometry.json。

## 两模型相比于彼此

两个独立训练模型的768维坐标不天然对齐。直接同维余弦保留为描述数值，不能解释为语义相似度。主要补充单位化特征的中心化线性CKA：比较同一批时间帧之间的几何关系，允许正交旋转和整体尺度变化。完整末层使用全部1000帧，既无采样也无拟合旋转；逐层补充CKA使用预先固定128个时间点，分开存储，不混用数值。

| 片段 | HuBERT参数臂 | wav2vec2参数臂 | 原坐标余弦 | 单位方向CKA |
|---|---|---|---:|---:|
| train | native | native | 0.010917 | 0.353863 |
| train | swapped | native | -0.009436 | 0.344872 |
| train | native | swapped | -0.004445 | 0.499907 |
| train | swapped | swapped | 0.005969 | 0.665686 |
| train | self_permuted | self_permuted | 0.002651 | 0.612569 |
| test | native | native | 0.008341 | 0.384882 |
| test | swapped | native | -0.007241 | 0.383884 |
| test | native | swapped | -0.005458 | 0.558763 |
| test | swapped | swapped | -0.004093 | 0.717724 |
| test | self_permuted | self_permuted | 0.002488 | 0.661368 |

测试片段两者均交换的CKA从0.38488升到0.71772，训练从0.35386升到0.66569；但两者各自打乱也升到0.66137/0.61257。只让wav2vec2接收HuBERT GN时，其对原HuBERT的CKA提高；只让HuBERT接收wav2vec2 GN时，其对原wav2vec2几乎不变或略降。没有对称的“互换表征”现象。

这些CKA增高确实依赖共同时间排列：固定时间打乱控制降到约0.011–0.019。但时间结构接近不等于音素或语义保留更好，尤其交换参数已经严重改变基线。补充全轨迹参与率秩，HuBERT测试39.72→12.29，wav2vec2 7.92→9.88；并非两者统一塌缩。没有ASR、音素标签、生成或SyncNet验证，不能给出语义/感知优劣判断。

## 验证与产物

独立CPU复核832个完整层比较的全部逐帧矩、106496个采样向量及32个完整末层比较。所有12个独特native条件及4重复末层与报告58逐值一致；4个默认重复的所有层逐值一致。换参张量逐值核对供体/排列，前卷积不变，两模型最后精确恢复原state_dict，原权重文件没有修改。另26项全部末层CKA用特征协方差公式独立重建，最大误差2.78e-16。几何单测2项通过，包括正交旋转保持CKA而改变原坐标余弦，以及正增益不改变单位方向。

代码scripts/experiments/encoder_groupnorm_swap_20261020/{run,geometry,analyze,full_geometry,finalize}.py；测试tests/experiments/test_encoder_groupnorm_swap_20261020.py。protocol.json在运行前绑定源、输入、参数网格和比较定义。full_geometry.py追加全帧endpoint分析，没有新增编码器forward；128点CKA在旧逐层JSON保留，正式末层表以terminal_full_geometry.json为准。源和测试快照、raw末层数组、逐层样本/逐帧矩、参数副本、原始统计均保留；manifest.json逐文件SHA，未声明云备份。pretrained helper按原SHA引用报告58代码。

## 结论边界

这次直接验证了：已学习GN仿射参数确实影响特征方向及尺度响应，但按原通道编号互换gamma/beta不足以互换两个编码器的行为。通道所依赖的卷积滤波器、后续网络以及二者共同训练形成的匹配不能忽略。自身打乱对照只有一个预定排列、只有两段May和两个checkpoint；不估总体显著性或供体特异效应。仍不能从这次冻结编码器干预推断报告56训练分差的因果贡献，也不能建议直接换GN部署。

## Observations
- [status] concluded，40冻结前向、832层比较，0训练/TFG/SyncNet。
- [result] 换GN仿射参数大幅改变自身特征方向，RMS响应没有互换。
- [result] 跨模型末层CKA增高也见自身打乱对照，不等于语义向供体迁移。
- [limitation] 原通道未对齐；两段May/单固定排列，不作总体或下游质量结论。

## Relations
- follows [[58-encoder-layer-epsilon-diagnostic]]
- relates_to [[56-may-postnorm-scale]]
- supports [[研究主线与证据地图]]