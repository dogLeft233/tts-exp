---
title: 近静态谱形通过局部谱成分改变三类TFG的时变条件特征
type: note
permalink: tts-exp/experiments/近静态谱形通过局部谱成分改变三类-tfg-的时变条件特征
status: concluded
scientific_status: concluded_descriptive_provisional_fit8_frontend_only
cohort: fixed Fit first batch eight clips four speakers
endpoint: original-clock local PCM gain and native frontend feature geometry
models:
- wav2lip_gan_bac9a81e
- musetalk_fixed_portrait_attempt16
- ditto_hubert_fixed_portrait_attempt18
selection_feedback: false
tags:
- audio-head
- mechanism
- native-frontend
---

# 近静态谱形通过局部谱成分改变三类TFG的时变条件特征

## 问题与结论

当前step060自然音频增强头的投影mask几乎为静态谱形，但这是否意味着它只给TFG前端加一个恒定偏置？固定8段Fit音频的实际测量给出明确答案：近静态谱形足以产生显著随时间变化的条件特征响应。整体RMS匹配后，原时钟40ms局部增益仍随自然语音的谱成分变化。这是当前增强头的前端机制结论；它没有把这些特征距离换算为Sync-C收益或整个TTS增强效应。

## 已执行设计与支持集

cohort为S0052–S0055四说话人，每人W0121/W0495两段，共8段，固定为Fit首batch。此step060是初始720四候选全NOT_ADMITTED后的最小违例初值，不是最终选择的方法。natural/full/DC三臂原72项加独立RMS appendix24项，共96项：三TFG每模型32个真实原生前端NPZ。所有结果仅作描述，不反馈seed、20联合SPSA、selection、admission或封闭Dev/Confirm。

DC取full的postprojection mask_db逐频率时间均值，作用于同源complexSTFT并以同窗/iSTFT恢复原N，匹配full实际RMS，没有第二次tanh或peaklimiter。RMS对照为原PCM乘单一float32增益，target=.0376838172675746、peak_ceiling=None。原自然PCM与样本数不改，分帧为ceil(N/640)、16kHz、25Hz原时钟，保留尾帧真实有效N。频带[0,500)、[500,1500)、[1500,3500)、[3500,8000]Hz，640点对称Hann/rFFT；尾帧以窗前缀并补零，谱power按使用窗能量归一。activity floor1e−6只作标志，不删音频/帧。

## 实际结果

先按每段计算，再每说话人两段平均、四说话人等权。无推断置信区间或显著性检验；8段点与描述性离散度均保存在JSON/图。

| 模型 | full与DC之差 / full与RMS之差 | DC与RMS之差 / full与RMS之差 | full与DC相对RMS响应方向cos | full−RMS响应平方量的时间中心化份额 |
|---|---:|---:|---:|---:|
| Wav2Lip | 0.100741 | 1.023204 | 0.995256 | 0.780148 |
| MuseTalk | 0.241803 | 0.956470 | 0.969548 | 0.965381 |
| Ditto | 0.124446 | 1.004190 | 0.991782 | 0.977296 |

feature变化分解为query均值与query中心化残差，逐clip为精确正交平方量分解。三个模型绝对特征norm不互比，表内norm ratio也不是可相加的因果份额。DC与full高度同方向，同时大多数feature变化为时变分量，说明静态输入谱形与静态embedding偏置不能混为一谈。

24个natural/DC/RMS WAV独立读验，共886个40ms帧完整保留；本支持集中没有低于activity floor或RMS分母为0的帧。DC/RMS局部增益定义20log10(frameRMSdc/frameRMSrms)：等说话人平均均值−0.038762dB，平均每clip帧间标准差0.455143dB，平均每clip第5/95分位−0.391333/+0.949170dB。全局RMS相同仍存在局部增益变化。

DC/RMS四频带mean-frame power对数比的等说话人均值为−0.378383、+1.299831、+2.232494、+2.062493dB。自然音频四频带power占比与局部增益的每clip Pearson相关，再按说话人等权，分别为−0.950450、+0.554115、+0.652467、+0.621967。这是组成数据上的描述性关联，四相关不表示四个独立因果贡献，也未赋予声母/韵母、清浊或具体音素标签。

## 解释与对方法的影响

理论推理：对静态频率增益H(f)和全局RMS系数λ，局部能量比近似为 λ²∑f |H(f)|²P_t(f) / ∑f P_t(f)。即便H不随时间变化，只要自然语音局部谱成分P_t变化，局部能量与模型条件轨迹仍会变化。实际886帧增益与谱成分关联，以及三原生前端数组的时变响应，与这一路径一致。

当前头主要体现跨频带重分配。下一代可考虑以局部谱成分控制增强幅度或频带门控，并用三TFG实际生成分数确认跨模型取舍。尤其Ditto的DC响应也近full，而初始step060的Ditto ΔSync-C为负，单纯删去小时间残差未直接处理这一冲突。冻结20联合SPSA不因此修改；此研究note不宣布新方法被选择，也不单凭前端几何断言口型质量改善。

## 工程核验与来源

ROOT已对三模型各实际SSH完整33文件raw pins、32NPZ输入身份/F32/shape/finite/queryorder、48 comparisons和528标量独立重算PASS。W/M/D权威QC分别为`.codex/scratch/root_provisional_{wav2lip,musetalk,ditto}_feature_qc_attempt01/root_actual_qc.json`；self为275529c5ee54ccbbd8b7c2bead798fd64a89c1ebb03783326c89ad8b8f639e76、84b440fac88f0461a3257eff33ac24d15ad41200388364e93858de8a63d31454、3cd82b02230222b0631d244b4919a1f01953039b98b6384606f3b995dea8e576。

ROOT40ms独立分析使用stdlib RIFF解码与独立NumPy实现，24个raw/PCM pins、886帧和192标量PASS：`.codex/scratch/root_step060_local40ms_actual_qc_attempt02/root_actual_qc.json` self9339c2b493bd4683a804577ee3db65dfff55bc2214f8dbaea7124046bd84e5d7。JSON内WSL `/mnt/e/Documents/tts-exp/`明确映射到同本机`./`，raw bytes仍逐一核对。attempt01因未映射路径停止且保留，无科学计算改变。

Signal实际分析`.codex/scratch/step060_native_feature_analysis_attempt04/step060_native_feature_descriptive_summary_attempt01.json`166211B/rawda7c422e98c8078a58a21ca510fa30ec116aea12ebf90450de1e2fff6923f50c；source13003B/e28d1fe577365bd44e07db9b8e72e1460c339732ec86dfc44f47520bc813470c。首图保留。仅图修订已完成并由ROOT目检/rawpin核验PASS：.codex/scratch/step060_native_feature_figure_revision_attempt01/step060_native_feature_descriptive_figure_attempt02.png 525553B/3ed503d277a05ce58fa3bcb415374d3d9b90a9a11ee27c0842695cb1b4f9883d，C/D说明已移入图外caption，不与数据重叠。实际renderer .codex/scratch/step060_native_feature_figure_revision_attempt02/render_step060_native_feature_figure_attempt03.py 8857B/d20cdecdc437dff13c5ba4394fb50fd46459e8a3bf29bc176cd2f112dd931218；曾误报13207B已按实际更正，SHA与source未变。ROOT回执 .codex/scratch/root_step060_cosmetic_figure_review_attempt01.json。数值JSON未改，无新model/score。`.codex/scratch/step060_native_feature_analysis_attempt04/step060_native_feature_descriptive_figure_attempt01.png`526310B/6e326fc4ba9ef58bc0754add234b1f4d8ebcf7a2730d6e81b47be5610b73ffee，ROOT已目检；C/D底部说明与基线附近数据轻微重叠，另行仅图修订，数值JSON不改。

Ditto使用同pin ONNX graph的显式CPU-only普通wrapper，worker07修复force_ori_type=True丢device导致的CUDA请求/CPU fallback，源/constructor实际核验通过；模型权重与计算输入输出没有更换。MuseTalk为CPU FP32诊断，V100实际生成路径FP16，因此本结果不宣称数值等同生成时精度。无新模型forward、视频、SyncNet、optimizer或tests用于40ms归约；原96项forward是先前实际执行的测量。

## Observations

- [status] concluded
- [result] 固定Fit8/四说话人/三TFG共96真实前端测量及ROOT逐数组重算PASS；近静态谱形造成主要为时变的条件响应。
- [result] full/DC相对RMS响应方向平均cos W/M/D=.995256/.969548/.991782；几何ratio不作为因果收益百分比。
- [result] 原时钟886帧中DC/RMS局部gain平均clip标准差.455143dB，低频占比相关−.950450；全局RMS相等不使局部增益恒定。
- [decision] 结果只用于描述性机制；保持冻结20联合优化与封闭Dev/Confirm，不以机制数组替代跨模型实际score准入。
- [next_step] 用冻结三模型实际联合优化/四中心评分检验跨模型取舍，再按局部谱成分探索下一代输入条件门控。

## Relations

- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[自然音频增强头的局部门控与参数可辨识性]]
- relates_to [[三类TFG音频前端的谱缩放与时间上下文]]

## 三类前端的实际归一化与上下文

这是当前已封存源码/配置的只读解释，不增加模型forward、音视频效果或TTS总体因果结论。权威23项actual source/config bodies为 `.codex/scratch/root_frontend_actual_source_review_attempt01/root_actual_qc.json`，self `e09309b1db38947a1c72dc387b80698ecaba09ba8afd489472d4bb933a6cdbeb`；完整raw capture 716955B/SHA `66e42d335b36eab0a8741d2b43d79a67f1c7f9dbf62ec0bb14055f4c90668f9c`。该QC没有重hash大型ONNX资产，资产身份以已有native/runtime/feature receipt为准。

| 前端 | 当前实际处理 | 对机制的含义 |
|---|---|---|
| Wav2Lip | 16kHz；preemphasis .97；800样本/50ms STFT、200样本/12.5ms hop；80 mel [55,7600]Hz；magnitude→dB再固定范围归一化/裁剪。实际load_wav→melspectrogram路径未应用hparam rescale。 | 线性preemphasis重加权频谱而不消除输入增益；整体gain可表现为mel/dB偏移，floor/clipped cells另有非线性。局部mel windows把频带变化交给生成器。 |
| MuseTalk | Whisper FE未传do_normalize，默认false；400样本/25ms FFT、160样本/10ms hop、80 power-mel；log10/floor/max−8 floor与(log_spec+4)/4。每完整30s chunk先经Whisper encoder，再切50Hz hidden states中[-4,+6)十token上下文供25fps输出。 | 这条路径不执行每段波形方差归一化。远离底噪floor时gain g约加0.5log10(g)到最终FE cells；encoder上下文可使谱形变化传播超出返回十token窗。现有诊断是CPU FP32，native为FP16。 |
| Ditto | Python Wav2FeatHubert没有输入响度归一化；6480样本/405ms graph chunk、3200样本/200ms advance、2000样本/125ms left pad；十个20ms输出两两平均为五个40ms条件。pinned graph输入affine为(x−2.586069967946969e−05)/.20523720979690552。 | 固定affine保留绝对输入增益关系，不是按每条音频估计均值/方差。仅描述已审计图运算，不外推未审计的ONNX内部机制。 |

全局RMS约束固定整段总功率。对近静态频带增益H与整段标量λ，局部帧功率近似满足 `G_t²=λ²·Σ_b |H_b|²P_(t,b)/Σ_b P_(t,b)`。原帧谱成分P_(t,b)随时间变化，故同一静态谱形也会改变局部增益和条件序列；该式为频带近似，不宣称STFT/iSTFT与有限窗波形逐样本恒等。886个原时钟窗的实际重算支持这一描述。

三模型前端差异给出当前跨模型冲突的可检验解释：同一谱形/局部功率重分配在局部mel、长上下文Whisper和分块HuBERT中形成不同激活响应。已有step060 W/M正、Ditto D0/Cbest负与此相容；前端特征响应本身不决定SyncNet收益，实际冻结20仍须完成。

方法延续须复用历史而不重复造类：V3已有positive/negative relative-flux两个局部门控；`shared_recipient_local_power_spectral_head_v1.py`已有9shape+2envelope的11参数构造，其CPU声学资格见2026-10-07T13-38-29Z checkpoint与 `runs/cross_tfg_audio_head_20261003/local_power_signal_qualification_attempt10/`，不是已证明跨TFG收益。下一轮若需要扩大函数族，应区分优化重参数化与真正新增输入条件，如各频带独立功率/flux或共发音上下文；当前冻结20不改。