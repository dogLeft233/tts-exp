---
title: 008-fixed-lag-and-rms-reference-head-objective
type: decision
permalink: tts-exp/docs/adr/008-fixed-lag-and-rms-reference-head-objective
git_sha: 4a2422c7b12afaacda9b702d3321682e1bcac6bb
decision: 下一共享频谱头使用固定零时差距离/单位向量cosine及有界RMS参照的规范化正则，保留旧候选和实际跨前端比较。
status: open
project: tts-exp
repository: dogLeft233/tts-exp
decided: '2026-10-03T11:15:43.9639278+00:00'
---

# 008-fixed-lag-and-rms-reference-head-objective

## Decision and context

下一训练候选改用固定源时钟零时差目标，配合同历史音频的有界RMS视频与波形参照。增强音只驱动TFG，SyncNet仍使用原自然音。旧fit24与fit96、原240及派生300评估不修改；新候选使用独立runtime、plan、cache、checkpoint和dev生成。

独立Wav2Lip开发集10新男声×2clip×3seed原音FULL均值：RMS对identity +0.234，localcompression +0.267，fit24 +0.256，fit96 +0.217。fit24相对RMS Cbest +0.022主要来自B +0.021，D0几乎不变；fit96对RMS D0 +0.057、Dmin +0.019、B +0.003、Canchor −0.054。旧median-minus-softmin目标的历史训练收益未转化成开发集改进。这是改变目标的开发证据，不是确认集或跨TFG结论。

## Rationale and specification

R(x)定义为与候选相同stack、head禁用、local禁用，global RMS target0.0376838172675746、stage peak .99、post-RMS target与final peak .99全部一致的有界RMS输出。它与已冻结dev rms_fixed的peak=null控制分别命名；记录实际peak-active及两者数值差异。zero-init候选h(x)须与R(x)实际同N/clock且maxabs≤1e−7，初始奖励与正则近零。

历史8全音、固定portrait、center32帧/27query、所有31lag合法、自然PCM16的1024维音频向量和既有fullcontext来源固定。每行以R(x)运行同生成/视觉编码器一次并封存detached视觉向量、D0、cos0、norm与波形；cache不得用dev或confirm音频。候选前向保留到音频头的梯度，冻结生成器和SyncNet权重。

最大化奖励 J = .5*(D0_R−D0_h)/max(D0_R,1e−6) + .5*(cos0_h−cos0_R)。最小化 −J 加 .1*mean(log(norm_vh/norm_vR)^2)、.02*MSE(h,R)/max(mean(R²),1e−8)、.002*(mean|Δfrequency logmask|+mean|Δtime logmask|)。固定零lag，不奖励B、最优lag或事后重定时；所有denominator、epsilon和finite/zero边界写入冻结计划。

旧绝对波形MSE在约.038 RMS下数值很小，新相对RMS能量规范化让改动惩罚具备明确尺度。norm约束限制仅由向量长度变化获得收益，cosine约束关注方向；它们不代替独立运动、内容或跨模型验证。逐步诊断B/Dmin/D0/Cbest/Canchor、cos/norm、所有正则、梯度与更新。下一候选拟同8clip/96steps、Adam1e−4、±3dB正STFT残差，技术review后才实际执行。

## Alternatives and consequences

继续延长旧softmin训练已有fit96开发退步，因此不作为下一主方案。仅选择当前localcompression作为终稿仍缺跨前端/确认/运动内容证据，它继续作为有用对照与候选。直接优化Cbest或自由时间偏移会混入错位分离与时钟改变，本方法固定时钟。共享waveform/config优先；如迁移失败，继续指定历史fit范围扩展或低维频谱/包络控制，不读取未见确认集来调参。

## Evidence and affected work

权威300 ledger及独立audit：`runs/cross_tfg_audio_head_20261003/stageb_joint_attempt02_readback/`；ROOT receipt `.codex/scratch/root_stageb_consumer05_actual_readback_attempt01.json`。源plan/results/manifest SHA分别838eebee47a56f789f02c352879e54b9877ed15b9bb62be25e0820cb65176503、0cbda0bfed7225e272c479b99485d914d6ae8ae281224e6aff57b4ca96acfaf5、5f46add13690760c7b59d4d1b1a4a68e4161493e15afa4a81c648d649cd9b02a。九FULL COMPLETE，九guard20因同一短clip INCOMPLETE保留分母，无dev bootstrap CI。

## Observations

- [decision] 新候选以有界RMS参照、固定零时差和unit-normalized方向优化，保留旧训练/统计/音轨协议。
- [rationale] fit96开发D0退步而B几乎不变，fit24额外收益主要来自B；改目标直接对准源时钟匹配并约束norm/波形偏移。
- [consequence] fresh缓存及source/plan/checkpoint/开发结果独立封存；只有实际跨前端、运动/content和未见确认支持时才能交付共享方法。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[006-portable-waveform-enhancement-head]]
- relates_to [[007-original-pcm-scoring-for-waveform-head]]