---
title: 三类TFG音频前端的谱缩放与时间上下文
type: experiment
permalink: tts-exp/experiments/三类-tfg-音频前端的谱缩放与时间上下文
status: concluded
project: tts-exp
repository: dogLeft233/tts-exp
scientific_status: active_frontend_source_transform_audit_complete
scope: Active V100 Wav2Lip, MuseTalk V1.5, Ditto v0.4 HuBERT routes bound to initial
  Fit40 720 rows
tags:
- tfg
- frontend
- mechanism
---

# 三类TFG音频前端的谱缩放与时间上下文

## 问题与证据范围

自然音频增强头目前在三类TFG上的效果方向不同，需要弄清实际输入表示、增益映射和时间上下文，避免把同一声学干预视为三模型接收相同特征。本记录是已部署路径的源码、配置、现有ONNX常数审计和既有720行时滞结果的只读核验。没有运行新的模型forward、视频生成、SyncNet、选择器或优化更新，也没有运行测试。

适用cohort为已冻结Fit40/20说话人、每人2clip，Wav2Lip/MuseTalk/Ditto各240官方评分行、合计720。输入为16kHz mono FLOAT32，原音与25/1有理时钟不变。源码事实支持输入变换和上下文结构；具体特征响应与下游收益须由后续干预测量。

## 实际前端

| 模型 | 部署路径与变换 | 每个输出帧的局部取样 | 完整上下文及增益映射 |
|---|---|---|---|
| Wav2Lip | child adapter→官方inference/audio；librosa mono16k，预加重0.97，center STFT800/200，80mel，55–7600Hz，20log幅度、−100dB floor、ref20dB、仿射后clip[−4,4] | 16mel列，起点floor(3.2i)，列增量3/3/3/3/4；列跨度200ms | centered窗并集相对40ms帧时刻约前25–35ms至后202.5–212.5ms，边界padding/尾部末16列复用。未floor/clip区+6dB约+0.48mel单位。hparams rescale字段未在active调用中用于waveform归一化 |
| MuseTalk V1.5 | active adapter08→AudioProcessor→Transformers4.39.2 Whisper；16k mono，最多30s分段pad480000点，80power-mel400/160，log10/max−8 floor/(log+4)/4；do_normalize未启用 | 50Hz hidden轴以2i为anchor取[2i−4,2i+6)十位置，再拼接5层hidden（4 encoder层+初层） | 每段全局双向self-attention，encoder实际attention_mask=None；局部十token窗口不限制waveform感受野。未floor区+6dB约+0.15特征单位，无逐waveform均值/方差标准化 |
| Ditto HuBERT | offline StreamSDK→wav2feat→HubertStreaming→ONNX；Python前不做RMS/peak归一化 | chunksize(3,5,2)，6480点/405ms crop，每200ms推进，block起点−125ms至+280ms；输出[-14:-4]十向量两两均值成五个25Hz特征；ceil(N/640)帧 | active ONNX输入固定affine (x−2.586069967946969e−05)/0.20523720979690552；下游audio2motion80帧/3.2s、overlap10帧/.4s。固定输入affine已知；后续内部normalization、学习感受野、精确token相位和响应仍属待测 |

Whisper配置d_model384、4 encoder layers、1500 positions；所谓30s上下文包括pad后的完整segment。Ditto部署配置虽然有cuda字符串，active loader/运行包使用ORT1.20.1 CPU，不能据配置把HuBERT声明为CUDA执行。

Ditto active graph为model_assets/ditto_v04_hubert/ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx，1,460,740,880B/SHA04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697。实际全SHA与历史readback_input_constants_audit.json对应graph一致；Constant410/412→Sub411→Div413给出上述affine。由此纠正“ONNX输入归一化完全未知”的旧宽泛表述。固定affine不是逐片段RMS标准化。

## 既有时滞证据与解释

已有initial720共同FULL支持包含120contexts/12537queries，ROOT全量重算在《原生模型固定时滞与增强头距离曲线》。W/M/D context-mean bestlag众数分别+1/−4/−2帧，RMS→step060有111/120 context的argmin不变，变动均1frame；Ditto step060 −2为29/40，旧28/40转录已纠正。bestlag符号与official offset相反，非独立时钟校准。

已建立的机制线索是增强主要改变匹配曲线深度和背景对比度，而非大范围整体平移。三模型不同谱尺度、归一化与时间上下文提供可检验的异质响应来源；这些源码事实本身不把原始TTS效应归因给某一个frontend步骤，也不直接等同唇部运动或人类感知改善。

目前全局flux门控头频谱mask有很强DC/静态成分，已有门控时序与线性设计研究支持静态与动态参数共线。W收益、M/D固定D0恶化说明单模型静态谱偏好尚未形成通用增强。冻结20轮三模型实际联合优化优先完成；目标、原PCM时钟与封闭Dev/Confirm不随本诊断改变。

## 下一项机制干预方案
以下为PROPOSAL_ONLY，需冻结20及真实finalcenter选择之后执行。复用旧encoder-probe的packer/clock/trajectory代码，不另建框架。旧8clip列表与当前Fit40仅4条重合，不能当当前8-Fit实验资格；缺失的是S0052–55的W0122。未来roster预先固定为family03的S0052–55每人两条真实原音、按原冻结first-batch有序rows：BAC009S0052W0121/BAC009S0052W0495、BAC009S0053W0121/BAC009S0053W0495、BAC009S0054W0121/BAC009S0054W0495、BAC009S0055W0121/BAC009S0055W0495。此选择来自冻结batch、无效果筛选，必须逐sourcePCM pin与最终center roster交叉验证。

这8clip×3frontends×natural原音/full真实最终候选/DC-only三条件=72 feature units。候选必须来自真实sealed finalcenter，不能人工theta或未选择快照；若finalselection NOT_ADMITTED，可仅使用其规则记录的refinementseed作provisional机制诊断，不称为admitted或portable。旧probe仅复用代码脚手架，fresh condition/runtime plan须更新为W2L Python3.8/Torch2.1.2cu121、Muse/Ditto Python3.10/Torch2.5.1cu121（D ORT CPU），不能沿用旧统一cu124/Python3.10.12 gate。

DC-only从候选已经投影有界的mask_db(f,t)取时间均值mask_DC_db(f)，按gain=10**(mask_DC_db/20)作用于同一原STFT，保留乘法时相位、重建同N/时钟并将最终RMS匹配full候选。若用自然log幅度表达，M=mask_db·ln(10)/20，才有log|Y|=log|X|+M。禁止再次radialtanh投影，因其非幂等会缩小mask并混入新的剂量差异。该条件是逐片段DC滤波，不是跨数据固定EQ。

现有mask analyzer按config.sample_ids检查，但真实center以40-row point-plan roster声明样本、config省略此字段。窄兼容版本应从真实center rows验证40roster，保留head/helper/source pin与full-WAV一致性；输出已投影mask供DC推导。此代码准备不等于已运行feature实验。

读出native query map上的feature均值、逐维时间中心化轨迹和相邻query delta方向/大小，Muse保留完整padsegment编码。full与DC-only相同RMS，区分平均谱加权和时间残差对前端的贡献；feature结果另需视频/原音评分检验，才决定方法效益。现有168-unit probe没有完成：W因缺shared_frequency_profile_v1在feature前exit1，M/D未派发，旧Muse合同floor2q窗口不当作active[-4,+6)窗口。
## 实际核验与权威产物

Signal audit02：.codex/scratch/frontend_audio_source_audit_attempt02/audit.json，22574B/rawc8ef5b67dfb48bfa1610950fdd3860c4096561564b40b5adaca7794380aecaba/self5d40c186a3f9a00814b958d0fa64778cfba15db831905858536daac3f8b1873a。

source_capture_manifest.json，20833B/raw52d1604ee824ea87133cd9f67e8e41d7fc26ae8c55ef1ef60e892bc0b0417c43/self6e58de77b78dc1e6e87323af71805351d4b5ef7242d6b41a8153d220d2d83e70，25完整source/config captures共603176B，模型大weights仅路径/size/SHA pin，未复制。Muse UNet实际size3400074924B；Ditto runtime config正确路径为model_assets/ditto_v04_hubert/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl。

ROOT independently fetched 23 remote text/config bodies并逐pin核对，.codex/scratch/root_frontend_actual_source_review_attempt01/ QCselfe09309b1db38947a1c72dc387b80698ecaba09ba8afd489472d4bb933a6cdbeb。ROOT随后完整读audit02和manifest、重新hash全部25captures及7linked本地证据、对照独立23body overlap PASS；.codex/scratch/root_frontend_audit02_review_attempt01/root_actual_qc.json self707aed52debe43dce79586c7463315524e8b3a0a43cacea1772cebd8a1ada6e5。大模型assets未在本次ROOT QA重新hash，明确继承既有资格与Signal active ONNX fullSHA。ROOT实际阅读W mel/窗口、active AudioProcessor/Whisper feature+encoder全attention关键函数、Dittochunk/audio2motion源码；没有新score或效果统计。

## Observations

- [result] 三模型active输入谱尺度和完整上下文不同，W局部mel、Muse全segment Whisper、Ditto固定affine的HuBERTchunk均有精确source/config证据。
- [result] Ditto graph输入固定affine已由同SHA图常数审计建立；配置cuda字符串不代表当前ORT GPU执行。
- [verification] ROOT全部25source captures/7linkedpins及独立23remote body overlap实际核验PASS；结论仅active源码变换与既有曲线证据，不新增效果宣称。
- [decision] 冻结20联合优化优先，之后以相同N/时钟/RMS的full-vs-逐片段DC滤波区分平均谱加权与时间变化，mask_db不二次tanh投影。
- [next_step] 完成真实20轮/finalcenter，再运行复用8Fit/72unit的active前端feature干预，随后把机制用于下一候选类并在原音换轨评分中检验。

## Relations

- relates_to [[原生模型固定时滞与增强头距离曲线]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]
- relates_to [[自然音频增强头的局部门控与参数可辨识性]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[开发跨 TFG 的自然音频增强头]]