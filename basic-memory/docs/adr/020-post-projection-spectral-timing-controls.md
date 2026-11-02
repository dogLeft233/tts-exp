---
title: 020-post-projection-spectral-timing-controls
type: decision
permalink: tts-exp/docs/adr/020-post-projection-spectral-timing-controls
status: open
project: tts-exp
---

# 020-post-projection-spectral-timing-controls

## 背景

用户要求解释局部声学特征如何增强TFG，并形成可用于多模型的自然音频增强头。冻结Fit40的160条门控诊断显示：正确门控相对±512ms错位的Cbest均差接近零，零延迟D0约改善0.010，并含小幅单位特征方向改善。旧static_only将动态门置零，同时改变平均谱形和全局投影比例。DirectD0/GateDC新训练保持独立身份，尚未实际启动。

当前九参数头根据整个频率×时间掩码的最大绝对值实施一次全局径向3dB投影。GateDC在投影前替换门控，其变化不只来自时间结构。ADR011的共同尺度组件分解针对另一套共享频谱profile；这里需要对当前实际已投影掩码另做控制。

## 选择与理由

在完整训练与实际checkpoint独立核验后，冻结theta和输入，从实际dynamic头取得最终投影后的M(f,t)。做三项score-only控制：每频点时间均值mean_t M广播到全T；M沿时间循环移位+64帧；循环移位−64帧。以8ms hop计，移位为±512ms。每个控制直接用其掩码重建波形，不重新投影，不训练或更新theta。

均值控制保留原频点平均dB谱形，移位控制保留每频点的值多重集、均值和极值。这样分别测量保留平均谱形后删除时间变化，以及保留完整掩码分布后改变时间对应的功能效应。它们是独立反事实控制，不把GateDC的差异转成局部时序的百分比，也不假设视频或SyncNet输出可线性相加。

## 执行与数值契约

保持同一STFT/window/source complex spectrum、完整N与源时钟、delta-ISTFT残差合成和原固定RMS基线。mean/roll掩码仍在原3dB范围内。保留全部257频点统计，用float64独立归约报告均值误差；float32运算容差明确记录，均值不称逐位精确。roll按每频点排序数组检查值多重集逐元素一致。

记录原投影比例与mask统计、最终WAV/F32 SHA、N/RMS/finite和构造STFT相位检查。ISTFT后的波形重新做STFT所得相位可因窗口一致性变化，单独测量，不与构造相位混称。循环接缝、源/目的帧索引、其可能影响的样本范围和query支持都要明确记录。保留全体主评分支持；边界敏感性单列，不能删行制造效应。

第一批限于已曝光Fit40/20sp，全部40×dynamic与GateDC同theta参考配对输出，未来artifact SHA和PID在实际生成前保持null。
评分实施固定为80条同theta原始dynamic/GateDC参考前向，加240条投影后均值/正负移位控制，共320条Fit-only诊断记录。原runner的80条GateDC账本只有聚合分数和波形SHA，缺少逐query矩阵；额外80次参考前向因此必须导出完整27×31距离和相关特征，核对原PCM SHA及既定同机数值容差后才能作接缝局部差值。参考资格失败保留全部记录并定位，不删参考或放宽阈值。

此评分沿用固定Wav2Lip神经前向与SyncNet、32个全音频时钟输入chunk和缓存几何/自然音频嵌入，不写或编码外部TFG视频文件；它仍有模型前向计算。主结果保持全部27个query，接缝及窗口传播的支持敏感性另报，不能用局部子集替代完整结果。240个控制分母与80个必要参考分母分别报告，全部源、权重和theta不更新。GateDC按有效transition均值再乘有效mask，首帧及无效/静音过渡仍有结构，不称为全时常数。
任何native补充实验须在读其效应前冻结新计划、自然音轨和完整失败分母。未来Dev20的主要三臂identity/RMS/DirectD0与本控制分开；Confirm保持封闭。

## 备选与后果

继续只比较dynamic与static_only/GateDC可提供功能消融，但投影比例与平均谱形一并变化，解释不够精细。投影前roll也不能保证最终掩码统计严格相同。各臂重新投影会重新引入尺度变化。对动态残差独立裁剪会改变分解，因此这些路径不用于本控制。

新增helper及资格只读原head/runner，保持原候选和旧160结果不可变。控制的完成与训练完成分开核验；只有实际结果支持时才形成新的机制结论。部署方面，精确训练GPU绑定的研究CLI用于严格重复；最终release capsule另作Fit-only CPU/异GPU数值资格后启用，保留完整原音交付。

## 证据指针

- 既有Fit门控结果：[[冻结增强头的门控时序与频谱贡献]]。
- 当前训练source：scripts/audio_enhancement/shared_recipient_transition_head_v3_d0_gatedc_attempt01.py；scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_d0_direction_gatedc_attempt02.py。
- 实际暂存回执：.codex/scratch/fit40_d0_gatedc_training_packet_attempt02/source_receipt_readback_attempt01/。
- source-only新控制helper由Signal子代理准备，最终version/pin待ROOT完整审阅；不预填结果。

## Observations

- [decision] 对实际已投影掩码做频点均值与±64帧循环移位，保持原theta/投影/重建/时钟/RMS，不再次投影。
- [rationale] GateDC同时改变全局投影尺度与平均谱形；投影后控制可分别测量平均谱形和时间对应的功能效应。
- [alternative] 保留GateDC为功能消融，但不用于计算纯时序贡献比例；独立裁剪或重投影不用于本设计。
- [consequence] 先完成Fit-only数值/接缝/支持核验，实际结果另报，未来Dev和Confirm身份不变。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]
- relates_to [[011-common-scale-mask-and-envelope-ablation]]