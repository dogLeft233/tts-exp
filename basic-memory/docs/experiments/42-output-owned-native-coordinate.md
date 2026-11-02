---
title: 42-output-owned-native-coordinate
type: experiment
permalink: tts-exp/docs/experiments/42-output-owned-native-coordinate
status: concluded
scientific_status: NATIVE_FUNCTIONAL_CONTROL_SUPPORTED_FULL_SCORE_NOT_IMPROVED
project: tts-exp
cohort: seen_DEV_S0022_S0024_two_utterances_MuseTalk
experimental_unit: utterance_paired_within_model
---

# 42-output-owned-native-coordinate

## 结论

**修正输入行与输出帧的所有权后，MuseTalk两句均出现预设的双向口部功能响应：正方向增强构音类别开合对比，负方向减弱。** raw及标准化的Cp/Hp四项全部通过，2/2句复现。保持H264观察方式也得到2/2；旧映射在同一观察方式下为0/2。因此，报告41中的Muse失败包含具体的输入作用域错位，不能概括成模型没有可控构音方向。

这条正方向没有提高原自然音轨下的正式full Sync-C：两句分别−0.169、−0.012，原时钟D0均恶化。功能控制坐标与评分目标已经分开。当前成果是限定的原生控制方法，下一步处理真实前端的共享切片约束与评分编码链；尚不扩大波形训练、TFG矩阵或确认队列。

## 研究路线与问题

报告39完成固定RMS的分离度、局部稳定性、径向能量以及可合成phone谱形筛查；所测单因素没有形成双人稳定增强。报告40在原自然时钟上得到共同的输出功能坐标：构音类别×固定模型参考时间。报告41将同一构音轴注入三TFG原生表示，完整双向门仅1/6格。

随后沿实际forward定位Muse：prompt row j由逐帧2D UNet生成同index输出；旧干预改audio core I，端点却读I+4，14个Cp帧全部没有直接暴露。此次只修这个映射，不根据新结果选择相位、剂量或符号。ROOT负责理论和一次实际科学QA；代码、六短视频与读出仍由原GPT-6 Luna/max子代理完成。

## 冻结设计

- 两句为BAC009S0022W0123、BAC009S0024W0122，各一位已见DEV说话人；原MFA支持与报告41相同。每句zero/plus/minus，共6次新MuseTalk渲染，seed20261003。
- 同一fixed-RMS自然驱动，16k mono FLOAT32，N=51376/56561；原自然PCM用于最终换轨与评分。音频内容、样本数和原视频j/25时钟均固定。
- 原audio core I映射到prompt rows I+p，p仍固定4；在映射行的真实baseline方向上，施加同natural9FIT类别轴、同对称低秩旋转，角度上限0.05rad。`high`指高开度a，`low`指i/y/u。这是原生类别方向控制，不把native norm当声学功率。
- 保持逐行norm、token内部Gram、zero、非有效修改行和clock；不裁剪、不后验翻号。有效行S22为[28,29,44,45,57,58,59]，S24为[31,32,42,43,53,54,55]；跳过单帧token的提案行不算修改mask。
- 每臂保留全部预编码PNG。primary是同IMAGE模式检测器直接读取PNG的开度；H264解码几何为编码对照，正式SyncNet仍用既定H264链和原自然PCM。

作用域映射不改变时间轴；p只是借用旧RMS official offset的固定参考，不解释为已确定生理延迟。三个token联合移动，不能把局部响应比值解释成每token隔离Jacobian。

## 端点与支持

全部原S17支持、绝对17个lag[-8,8]、三臂common-valid、原自然duration和层级权重保持。每句3个token（高开度1、低开度2），80/88帧共同有效，tracking均100%，无支持损失。

令C(l)为高开度−低开度的眼间尺度归一化开度对比：token帧均值→basephone等token权→class等phone权。Cp=C(4)，Hp=C(4)−mean(C(0),C(8))。raw与每臂在共同mask上独立标准化的z并列。冻结门为两句全部raw/z Cp/Hp满足plus>zero>minus，容差1e-9。速度仅描述，无真人口型准确度真值。

## 结果一：可复制的原生口部功能控制

| 句子 | 臂 | raw Cp | raw Hp | z Cp | z Hp |
|---|---|---:|---:|---:|---:|
| S22 | zero | 0.047696 | 0.092593 | 1.357364 | 2.635087 |
| S22 | plus | 0.062214 | 0.107111 | 1.753244 | 3.018492 |
| S22 | minus | 0.034882 | 0.079779 | 0.997736 | 2.281932 |
| S24 | zero | 0.016221 | 0.029592 | 0.453434 | 0.827194 |
| S24 | plus | 0.025820 | 0.039191 | 0.721160 | 1.094618 |
| S24 | minus | 0.005522 | 0.018893 | 0.153865 | 0.526421 |

plus的raw Cp/Hp分别增加0.014518、0.009598；minus分别减少0.012813、0.010699。两句四端点均有预设双向顺序。由于本次只改Cp所有者且C(0)/C(8)的primary帧未变，raw ΔCp与ΔHp相同；这也是作用域控制的结果。

为了区分所有权修正与更换观察方式，另用既有六条H264轨迹计算同一冻结功能端点，没有重新检测或评分：

| 输入操作 | H264几何双向门 | 预编码PNG几何双向门 |
|---|---|---|
| 报告41旧I行操作 | 0/2 | 原件已删除，缺失 |
| 本次I+4所有者操作 | 2/2 | 2/2 |

这支持所有权映射的解释。旧操作缺PNG，不能补造完整2×2实验，也不把两句描述性screen写成泛化确认。

primary开度速度RMS（眼间尺度/秒）为S22 zero/plus/minus=0.491218/0.507396/0.481836，S24=0.513285/0.513665/0.516744。功能对比方向可控，速度变化却不构成共同质量排序；不能写成“更快更准确”。

## 结果二：正式评分与功能目标分离

所有Δ相对本次同路线zero；Sync-C使用官方float32曲线，3位小数。full与guard20按既定query支持计算。

| 句子 | 臂 | full ΔC | full ΔDmin | full ΔD0 | full ΔB | guard20 ΔC | guard20 ΔD0 |
|---|---|---:|---:|---:|---:|---:|---:|
| S22 | plus | −0.169 | +0.104 | +0.110 | −0.065 | +0.118 | +0.152 |
| S22 | minus | −0.072 | +0.081 | −0.015 | +0.009 | −0.080 | +0.189 |
| S24 | plus | −0.012 | −0.010 | +0.016 | −0.023 | +0.398 | +0.111 |
| S24 | minus | +0.032 | −0.046 | −0.052 | −0.014 | +0.259 | −0.066 |

两句plus的功能指标提高，full C却下降，D0均恶化；guard20 C的正增益也没有对应D0改善。S24 minus单行通过full joint评分门，但其构音功能变弱，且S22未复现，不能事后选负方向成为方法。C=B−Dmin的代数分解保留，不能把B或距离变化写成生理贡献比例。

## 结果三：编码会把局部输入效应传播到非所有者帧

ROOT逐体核对504张实际PNG：每臂7个owned帧确实变化，全部非owned帧RGB与zero逐字节相同，primary开度也完全相同。随后六个既有512 H264仅各做一次软件RGB24解码，不重渲染、评分或检测，得到：

| 句子/臂 | 非owned变动帧/总数 | 非owned RGB字节平均绝对差 | 最大差 |
|---|---:|---:|---:|
| S22 plus | 73/73 | 0.324043 | 16 |
| S22 minus | 64/73 | 0.095311 | 16 |
| S24 plus | 81/81 | 0.317878 | 16 |
| S24 minus | 68/81 | 0.086512 | 17 |

因此，非owned的H264 landmark变化有实际像素变化作为来源，不能归因逐帧UNet跨帧响应。编码前后开度RMS差为0.00376–0.00523，最大单帧约0.017345。上述512像素诊断尚不量化后续224缩放/评分链对SyncNet的贡献；六条匹配FFV1/yuv420p链评分诊断已另行冻结。FFV1保留RGB→YUV及4:2:0抽样，不称RGB无损，也不替换正式H264端点。

## 结果四：共享前端约束解释了native控制与波形方法间的缺口

实际Muse AudioProcessor的Whisper特征为50Hz、5个hidden-state tuple、384维。每25fps输出行j取10个上下文位置，flatten索引m的c=m//5、layer=m%5，对应同一底层primitive `(segment=0, time=2j+c−4, layer)`。窗口重叠使同一primitive出现在多个prompt行；这些身份来自源码切片，不是按数值相等猜测。

源映射逐位重建两份accepted zero。令A为primitive→prompt的gather，现有独立行扰动δ的最近切片一致分量为AA⁺δ，即对同primitive的所有副本取均值。一次CPU计算与ROOT独立向量化实际数组QA一致：

| 句子/臂 | 残差能量/原δ能量 | owned保留能量/原owned能量 | owned保留范数比 | 非owned占投影能量 |
|---|---:|---:|---:|---:|
| S22 plus | 0.612775 | 0.175759 | 0.419237 | 0.546106 |
| S22 minus | 0.612465 | 0.175926 | 0.419435 | 0.546038 |
| S24 plus | 0.613029 | 0.174998 | 0.418327 | 0.547776 |
| S24 minus | 0.613116 | 0.174936 | 0.418254 | 0.547833 |

约61.3%的原扰动能量违反共享切片一致性；一致投影在owned行仅保留约17.5%能量，同时每臂影响24个非owned行。六token投影后的signed类别方向仍保持plus正/minus负。这里的17.5%是owned保留，不是全投影能量；全投影约占原δ能量38.7%。

这确定了下一坐标应包含**共享frontend primitive、conditioning所有者、输出功能事件**。AA⁺只是必要的切片约束，不是Whisper编码器像空间或波形可达证明；投影也不保持原来逐行norm/Gram。不能把它直接作为纯分离度候选或声学增强头投入训练。

## 可靠性、时间成本与执行终局

输出所有者CPU一次rc0，ROOT核对6f32数组/2clock、12token-arm：公式最大误差0，norm相对误差≤1.419e-9、Gram≤7.097e-10，14Cp直接覆盖；zero、非有效行和clock精确不变。

六renderer于UTC2026-10-14 21:45:52–21:50:39终局rc0/6PASS；六readout于22:01:45–22:03:25终局rc0/6PASS，实际进程已退出。保留504PNG、6H264及6自然PCM mux；ROOT已核验全部实际PTS与完整原自然PCM。ROOT响应QA一次检查12矩阵/6primary+6H264轨迹/504RGB/2884数值比较，最大几何误差0；float64均值与官方float32曲线的舍入差≤2.152e-6，B/D/C由官方曲线精确恢复。

后续只做已有产物的CPU几何对照、H264像素诊断和primitive投影；ROOT投影QA四臂指标误差0、能量正交分解相对误差≤1.14e-16。没有新MFA/ASR/API、encoder、训练、安装或实现测试。readout01与consumer前置字段/路径错误均保留，局部修复后成功，不把0处理失败当科学阴性或重跑理由。

## 来源与恢复

完整精度表、冻结协议、effect、ROOT QA、像素与primitive诊断及compact readout见 [科学产物目录](assets/42-output-owned-native-coordinate/)。这些小文件不替代完整媒体；完整媒体仍为machine-local scratch，不宣称灾备完成。

本地根：`.codex/scratch/native_attribute_response_probe_20261015_attempt01/`。

| 关键原件 | SHA256 |
|---|---|
| output_owned_musetalk_protocol_attempt01.json | 561e81a747b62c02af7a2465eac9457436013ffba45286be15d8c6cf40c9123e |
| output_owned_geometry_effect_attempt01.json | bb5fa06e5fd6b1d8a8f1340752b1fdad4c4d6cc733434cb0ba95a20e06fc4967 |
| root_actual_output_owned_response_qa.json | f470d1aa4a926cafb65f73555e6957797bf3b7176b7bbaf8a30348593d3f32d9 |
| root_actual_output_owned_pcm_clock_qa.json | 563e81724232c35cb4e52e6957d2177ed2b701549a06a1f3b6f7ce897645a3e7 |
| root_h264_functional_comparison.json | 39703dfebf283ce098186a031354ebf4003ac4c31f6036b212f28e7e86e7d159 |
| whisper_primitive_projection_attempt01.json | 0ee1129a7201932ee1b99ab8ea0dcdb6cc4fb44660c442c56e8ccdce1121e6d4 |
| root_primitive_projection_qa.json | 44e4eb5ecbf7fa6d0553231c10e02e2638e19430c0e74376db090e3aa3dbd551 |

完整媒体：`output_owned_musetalk_render_readback_attempt01/output/`，543文件197713752B。compact读回：`output_owned_musetalk_readout_attempt02_readback/readout/`，67文件887654B，manifest SHA1b4f454463a4387b470c744b0f9269b991e2c4e4fd0164132c40a1c87a655c4f。远端媒体在`/dev/shm/cross_tfg_native_attribute_response_probe_20261015_output_owned_musetalk_render_output_attempt01/`；远端readout在项目`.codex/scratch/native_attribute_response_probe_20261015_output_owned_musetalk_attempt01/readout_attempt02/`。不得重复launch已完成任务。

## Observations

- [result] 固定RMS/原内容/原时钟下，Muse输出所有者构音干预两句raw/z Cp/Hp完整双向门通过；same-H264 observer也通过。
- [result] plus原自然音轨full Sync-C两句下降，功能增强与评分增益分离；未形成替换音轨增强方法。
- [result] 非owned PNG精确相同而解码H264像素改变；共享Whisper切片又使独立prompt行扰动约61.3%能量不一致。
- [decision] 下一坐标必须显式包含共享前端primitive与输入/输出所有权；先小codec诊断和源内可实现性，不扩大TFG训练或改冻结符号。
- [next_step] 完成最多六条已有PNG的匹配codec评分诊断，并据其结果及真实frontend可微路径冻结下一最小波形可达性方案。

## Relations

- follows [[41-native-articulatory-response]]
- builds_on [[40-functional-response-coordinate]]
- extends [[接收端局部构音响应坐标]]
- tracked_by [[native-axis-factor-pilot-20261013]]
- indexed_by [[研究主线与证据地图]]
- contributes_to [[开发跨 TFG 的自然音频增强头]]
