---
title: 58-encoder-layer-epsilon-diagnostic
type: experiment
permalink: tts-exp/docs/experiments/58-encoder-layer-epsilon-diagnostic
status: concluded
execution_status: SUCCEEDED
scientific_status: FROZEN_ENCODER_MECHANISM_DIAGNOSTIC
inputs: report56 same May train/test whole-split normalization, first 320080 samples
  of each
outputs: F:/Datasets/tts-exp/encoder_layer_epsilon_20261020_v1
model: facebook/hubert-base-ls960 and facebook/wav2vec2-base, report56 frozen SHA
  pins
code_paths:
- scripts/experiments/encoder_layer_epsilon_20261020
- tests/experiments/test_encoder_layer_epsilon_20261020.py
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
---

# 58-encoder-layer-epsilon-diagnostic

✅ 完成。两模型的尺度残差从首个 GroupNorm 开始出现；wav2vec2 在此已大于 HuBERT，随后卷积与部分后段 Transformer 进一步扩大差异。仅将首层 GroupNorm epsilon 设为0，或按输入尺度平方补偿 epsilon，即使两模型从该层到末层恢复逐值一致。本轮60次冻结编码器前向、0训练、0TFG、0SyncNet。

## 问题与设计

报告56发现：标准化后波形乘0.5重新训练 SyncTalk，wav2vec2 组的 Sync-C 约增加0.092，HuBERT约0.0015；固定低尺度训练模型再提高推理输入尺度的效应很小。本轮只诊断编码特征的尺度响应来源，不运行下游生成、评分或重训。

使用相同的两套权重及其配置/预处理器，逐文件SHA核对。HuBERT revision dba3bb02fda4248b6e082697eee756de8fe8aa8a，权重062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad；wav2vec2 revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8，权重3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21。原split音频先做同公式float32全段标准化，再取各自前320080采样（20.005秒），严格对齐父实验首个编码分块。每模型两片段，四epsilon×三尺度，另每片段一次默认重复、两次epsilon补偿，共60次完整前向。四个默认末层缓存与父实验逐值一致。

epsilon为0、1e-7、默认1e-5、1e-3；输入尺度为0.5、1、2。唯一修改属性是 feature_extractor.conv_layers[0].layer_norm.eps；所有权重、其余norm、时钟、分块、eval模式保持。补偿对照为输入乘a，同时该层epsilon设为1e-5*a²，与默认输入尺度1比较。每组epsilon都有自己的尺度1参照，并另测改变epsilon本身的基线漂移，不把它当原模型无损修复。

本机RTX4060执行，PyTorch2.5.1+cu121、Transformers4.49.0、FP32/eager、TF32关闭，2CPU线程、CUDA进程显存上限40%；峰值reserved以completion.json为准。0Ubuntu调用，不干扰另一会话的May TTS训练。

## 残差从哪一层开始

下表是输入0.5相对于1的全坐标相对L2，单位%。训练/测试各只取一个与原缓存严格绑定的20.005秒片段，不等于报告56整条音频的汇总。

| 记录点 | HuBERT训练片段 | wav2vec2训练片段 | HuBERT测试片段 | wav2vec2测试片段 |
|---|---:|---:|---:|---:|
| 首层GroupNorm | 0.06123 | 0.27278 | 0.08669 | 0.31476 |
| 第7个卷积块输出 | 0.06220 | 0.52458 | 0.08814 | 0.58465 |
| 投影线性层 | 0.04292 | 0.43446 | 0.06272 | 0.47540 |
| Transformer第4层 | 0.04779 | 0.36169 | 0.07055 | 0.39538 |
| Transformer第8层 | 0.03682 | 0.54564 | 0.05433 | 1.07668 |
| Transformer第9层 | 0.03867 | 0.51773 | 0.05660 | 2.20663 |
| Transformer第12层 | 0.04444 | 0.61907 | 0.05938 | 1.97346 |

首层无偏置卷积满足Conv(a*x)=a*Conv(x)：所有32个epsilon/尺度对照的逐坐标比例残差为0。该处50%或100%的原始相对变化仅是输入缩放，不能直接当作归一化未抵消的残差。经过首层GroupNorm后才出现小的、非零的尺度残差；其模型间比值训练约4.46倍、测试约3.63倍。到末层则约13.9倍、33.2倍。并非每层都放大；测试片段wav2vec2在Transformer后半部尤其8→9层变化明显，HuBERT整体维持小残差。

## epsilon干预与公式验证

对首层卷积某通道的输出z及正尺度a，GroupNorm输出为 gamma*(z-mu)/sqrt(v+epsilon/a²)+beta。epsilon=0时整体正尺度可消除；epsilon随a²一起改变时，也与默认尺度1的分母相同。本轮两种控制都在真实模型完整路径上得到逐值一致，而非仅低于阈值。

| 首层epsilon | HuBERT测试末层相对变化% | wav2vec2测试末层相对变化% |
|---|---:|---:|
| 0 | 0 | 0 |
| 1e-7 | 0.000616 | 0.013660 |
| 默认1e-5 | 0.059383 | 1.973465 |
| 1e-3 | 2.578445 | 16.010330 |

上表固定输入0.5对该epsilon自己的尺度1；不是跨epsilon基线比较。另8个补偿条件中的200个首层GN及后续记录点也全部逐值一致；epsilon=0的8个条件另有200点一致。首层残差按实际逐通道方差与gamma/beta独立公式重建，32条件预测相对L2与实际最大绝对差1.343e-8。

这把此前epsilon候选解释推进为本轮严格配对输入上的实际干预证据：首层epsilon是本轮尺度残差的入口，后续网络传播这项残差；并非HuBERT有归一化而wav2vec2没有。

## 为什么相同epsilon产生不同响应

两模型的卷积输出方差分布及学习到的GN仿射参数不同。不能简单说wav2vec2全部通道方差更低：其训练片段方差中位数0.07360，HuBERT0.06544；但最低1%分位wav2vec2约0.002609、HuBERT0.011133，低尾不同。所有通道方差均大于1e-5，epsilon仍能产生可测影响。

在训练片段，将gamma设1/beta设0的解析核心响应分别为HuBERT0.05633%、wav2vec2 0.15163%；使用真实仿射权重后的实际GN响应为0.06123%/0.27278%。wav2vec2最强10个通道贡献其GN尺度差平方能量的92.46%，HuBERT为74.26%；测试片段为90.10%/75.84%。这里只是解析能量分解，没有交换两模型参数，也没有声称某通道代表特定音素。详细通道ID、方差、gamma、beta见channel_attribution.json。

## 数值核验与可复现性

独立CPU消费者复核1352个层比较的全部逐帧矩、43264个采样向量、52个完整末层数组比较；复建相对L2最大差2.082e-17。每层指标均基于全部坐标，不是采样估计；采样数组是额外检查。4默认缓存桥接逐值相同，4默认重复全部记录点逐值相同，两模型运行前后state_dict SHA保持。26个记录点含12层Transformer及单独保存的末层端点，末层端点与第12层重复。

代码：scripts/experiments/encoder_layer_epsilon_20261020/{run,analyze,finalize}.py；数值单测tests/experiments/test_encoder_layer_epsilon_20261020.py两项通过。原始前向protocol、stage_samples、full_frame_moments、last_hidden_state、逐通道统计与控制均在F根保留。run.py协议SHA冻结后未修改；作图环境复用原有matplotlib及bundled Python/Pillow，未改科学运行环境。第一次作图因缺matplotlib/Pillow失败，仅更换出图Python后完成；没有模型重跑。completion.json是前向完成权威，qa.json是CPU核验。

复现需相同父数据、冻结权重和依赖：原run.py在新的独立OUT执行；本目录v1已完成，禁止直接覆盖重跑。原输出共约0.66GB，无云备份声明。完整manifest按文件SHA绑定，报告和代码副本一并保存。

## 结论边界

已解释这两个checkpoint、这两段May输入、正尺度0.5/2的编码尺度残差从哪里进入及如何传播。没有重训、没有新视频或SyncNet，不能据此证明报告56的0.092分差由某一层或epsilon完全解释，更不能称降低epsilon改善口型/感知。修改epsilon本身会改变尺度1基线：epsilon0下HuBERT测试末层漂移0.02007%，wav2vec2漂移0.52283%；它是诊断干预，不是已验证部署修复。跨语料、更多说话人及其他checkpoint未检验。

## Observations
- [status] concluded；60 frozen encoder forwards，0训练/TFG/SyncNet。
- [result] 首层GroupNorm引入尺度残差；epsilon0及平方补偿的所有后续记录点逐值一致。
- [result] wav2vec2首层残差更大，后段Transformer进一步传播；HuBERT响应保持较小。
- [limitation] 两段May、两个固定checkpoint；不替代原训练分差或感知因果验证。

## Relations
- follows [[56-may-postnorm-scale]]
- supports [[研究主线与证据地图]]