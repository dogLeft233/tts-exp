---
title: 已测TFG的音频归一化调用链
type: note
permalink: tts-exp/docs/reference/已测-tfg-的音频归一化调用链
status: active
project: tts-exp
scope: historical TTS-vs-natural TFG roster; source and configuration survey
checked_date: 2026-10-18
tags:
- tfg
- frontend
- normalization
- source-survey
---

# 已测TFG的音频归一化调用链

## 范围与结论层级

按 [[SYNC-C-SCORES]]、[[02-tfg-cross-model]] 和 [[研究交接总览]] 确定范围：早期9模型、后续5模型、Hallo3补充和Seedance2.0补充，共16种实际有TTS/自然双臂记录的TFG。纳入正向、近零及负向模型，避免仅选择TTS受益模型。SadTalker、Sonic、InfiniteTalk、FantasyTalking、Wan2.2等不因上一轮通用调研而加入本历史表；SyncTalk的纯幅度实验另见47–51，不当作TTS双臂实验。

这是源码与配置核查，不新增音频编码、TFG视频、SyncNet、训练或实现测试。当前能直接绑定本地已核验实际运行源的是Wav2Lip、MuseTalk、Ditto后续V100路径；EchoMimicV2和SoulX-FlashHead的上游commit与部署文档所记一致；LeapTalk另取部署文档所记5d8fef8完整commit。但后续部署pin不自动证明7月旧容器同一版本。其余为已保存commit的公开调用链和对应公开模型配置，未恢复全部历史容器/patch/config。下表“是/无”首先描述所列路径，不将当前公开源码冒充全部旧实验的实际运行快照。

## 2026扩展与配置纠正

后续[[2026年TFG的响度与输入尺度处理]]核查了10个2026公开实现，明确区分−23LUFS与波形标准化。Tencent中文wav2vec2-base配套配置实际do_normalize=False，不能从中文HuBERT或英文base-960h外推；此前聊天关于InfiniteTalk此开关True的断言已纠正，其显式LUFS处理另算。本表8个标准化正例均为英文base-960h路径，原表分类未改。

## 判别口径

这里的“音量归一化”专指按当前音频或当前音频片段计算统计量，再消除整体尺度，例如 `(x-mean(x))/sqrt(var(x)+eps)`、峰值归一化、RMS或LUFS匹配。它与以下操作分开：

- 固定常数的PCM整型到float转换，例如除32768：单位转换，保留音量比例。
- 固定训练统计的affine `(x-mu_train)/sigma_train`：仍保留输入增益差异。
- Wav2Lip/Whisper频谱固定缩放和floor/clip：会改变增益传递，但不等于消除增益。
- LayerNorm、GroupNorm、BatchNorm：必须看位置、归一化维度、bias、epsilon、训练/推理模式；仅凭模块存在不能宣称整个模型对音量不敏感。

## 历史已测模型的路径表

| 模型 | 编码输入 | 逐音频/片段尺度归一化 | 核查事实与历史绑定边界 |
|---|---|---|---|
| Wav2Lip | log-Mel → 自训练CNN | 无，实际在用路径 | `audio.py`固定dB映射和clip[-4,4]；hparams虽有rescale=True，active inference没有调用波形rescale。实际源见三类前端笔记。 |
| MuseTalk1.5 | Whisper-tiny | 无，实际在用路径 | active AudioProcessor调用Transformers4.39.2，do_normalize默认False且调用未开启；log-Mel `(log_spec+4)/4` 不消除音量。 |
| Ditto v0.4 | 导出HuBERT | 无逐片段标准化，实际在用路径 | Python直接分块；已核验同SHA图入口固定 `(x-2.586069967946969e-05)/0.20523720979690552`。不能套独立HF HuBERT的do_normalize=True。 |
| IMTalker | wav2vec2-base-960h | 是，公开调用链 | 实验使用App入口；公开`app.py` DataProcessor.process_audio调用FeatureExtractor，配套cbsjtu01/IMTalker配置do_normalize=True。旧容器patch未全量恢复。 |
| V-Express | wav2vec2-base-960h | 是，公开调用链 | inference加载Wav2Vec2Processor；pipeline.get_audio_embeddings处理waveform后再编码。checkpoint配置True。 |
| Hallo2 | wav2vec2-base-960h | 是，公开调用链 | AudioProcessor显式调用FeatureExtractor。历史公平对照关闭Kim_Vocal_2人声分离，不能把关闭分离误写成关闭wave标准化。 |
| JoyVASA | 历史报告为HuBERT；代码另支持wav2vec2/中文HuBERT | 公开路径无显式波形尺度归一化 | wrapper librosa.load→pad→DitTalkingHead.extract_audio_feature→自定义audio_encoder；没有调用HF FeatureExtractor/Processor，pad_audio仅反射/复制padding。具体旧checkpoint的audio_model参数未读回，不能按train.py默认值定实际模型。网络内部归一化保留。 |
| LatentSync | Whisper，具体tiny/small依版本配置 | 无，公开调用链 | Audio2Feature→transcribe→load_audio/log-Mel；PCM除32768是固定单位换算。历史checkpoint变体未在本次恢复。 |
| EchoMimicV2 | Whisper-tiny | 无，部署记录同commit的公开路径 | commit38c86809与部署记录一致；Audio2Feature→Whisper原生load_audio/log-Mel，无逐音频标准差/RMS匹配。 |
| SoulX-FlashHead Pro | wav2vec2-base-960h | 是，部署记录同commit的公开路径 | commit9bc03de与部署记录一致，FlashHeadPipeline.preprocess_audio调用FeatureExtractor；固定checkpoint配置True。 |
| LeapTalk | wav2vec2-base-960h | 是，已记录部署pin的公开路径 | 取5d8fef8版本，pipeline.preprocess_audio及_preprocess_audio_embeddings都调用FeatureExtractor；stream缓存窗口决定统计支持。该pin是较新部署记录，非已证明7月原容器SHA。 |
| AVTR-1 | 专用导出HuBERT | 外层未见；导出图内未知 | Python run_hubert直接送入audio_batch；当前公开Normalizer用于运动so3/kp/exp固定统计，不是波形标准化。未下载/解析实际HuBERT ONNX，不能判整条路线有或无自适应输入标准化。官方notice称HuBERT-base衍生，而I/O为1024维；不据名称冒充未改动标准HF模型。 |
| AvatarForcing | wav2vec2-base-960h | 是，公开调用链 | utils/dataset.py TextImageAudioPairDataset调用wav_feature_extractor，再把input_values送入Wav2VecModel。 |
| StableAvatar | wav2vec2-base-960h | 是，部署文档所列repo的公开路径 | neosun100实现pipeline按当前sub_vocal_input_values调用Processor；FrancisRing权重包配置True。该repo为部署使用版本，不把它与其他上游/分支自动当同源码。 |
| Hallo3 | wav2vec2-base-960h | 是，公开调用链 | config指定base-960h，AudioProcessor调用FeatureExtractor。 |
| Seedance2.0 | 闭源 | 未公开 | 本项目有6对英文API双臂记录，但内部编码器/归一化没有可核查实现，不补猜测。 |

按所核查路径计8条逐输入标准化、6条没有显式逐输入标准化/仅固定缩放、2条内部不明；这不是16个旧容器均已重建的比例统计，也不是音量干预的实际TFG响应统计。

## 已读取的直接源码入口

所有公开源码均保存小体积正文和SHA，不下载模型权重；精确commit、path、bytes、SHA见本次receipt。以下行号对应本次保存的源码：

- IMTalker：`app.py:131–140,188–189`，FeatureExtractor初始化和实际App读音频路径；[固定commit源码](https://github.com/bigai-nlco/IMTalker/blob/bd91867e93c3880db271579c14ef5ccb2e0d404b/app.py#L188)。
- V-Express：`inference.py:165–166` 和 `pipelines/v_express_pipeline.py:375`；[固定commit源码](https://github.com/tencent-ailab/V-Express/blob/6814f7db362dc9e26644162034b4d28c5277651a/pipelines/v_express_pipeline.py#L375)。
- Hallo2：`hallo/datasets/audio_processor.py:73,109–110`；[固定commit源码](https://github.com/fudan-generative-vision/hallo2/blob/58a9aa6c9f66817a6e084f3874cfc01ac24fed3e/hallo/datasets/audio_processor.py#L109)。
- JoyVASA：`src/live_portrait_wmg_wrapper.py:362–368`、`src/modules/dit_talking_head.py:78–93,201–214` 和 `src/modules/common.py:pad_audio`；[固定commit源码](https://github.com/jdh-algo/JoyVASA/blob/916a90f8de490e8648fee460c1200bd5d9a795af/src/modules/dit_talking_head.py#L201)。
- FlashHead：`flash_head/src/pipeline/flash_head_pipeline.py:205–219`；[已记录commit源码](https://github.com/Soul-AILab/SoulX-FlashHead/blob/9bc03de06bb0de82cd6bc477804512ae06144bf2/flash_head/src/pipeline/flash_head_pipeline.py#L205)。
- LeapTalk：`inference.py:273–275,818–819,992`，pipeline `:269–277`；[已记录commit源码](https://github.com/zhangrongxiang/LeapTalk/blob/5d8fef8dac3d1c9d3b3d5d9cceb6c76acc81e2c1/flash_head/src/pipeline/flash_head_pipeline.py#L269)。
- AvatarForcing：`utils/dataset.py:275–277,304–307`；[固定commit源码](https://github.com/KlingAIResearch/AvatarForcing/blob/63b73e6c0f7bb42180ca6d7e1bf11c1de1a80b39/utils/dataset.py#L304)。
- StableAvatar：`inference.py:475–476`，`wan/pipeline/wan_inference_long_pipeline.py:730–731`；[部署所列repo源码](https://github.com/neosun100/StableAvatar/blob/f63d80287ab3bffa4745f7b0794c1f90f5d1cdde/wan/pipeline/wan_inference_long_pipeline.py#L730)。
- Hallo3：`configs/inference.yaml:16`，`hallo3/sgm/utils/audio_processor.py:70,103–104`；[固定commit源码](https://github.com/fudan-generative-vision/hallo3/blob/e342dcec7ec17cdf26da6c97679c0414c9738a3c/hallo3/sgm/utils/audio_processor.py#L103)。
- EchoMimicV2/LatentSync：各自`audio2feature.py:audio2feat/_audio2feat`调用transcribe，配套`whisper/audio.py:load_audio/log_mel_spectrogram`；[Echo同部署commit](https://github.com/antgroup/echomimic_v2/blob/38c86809efa041884c774ee31d984a9577c0e0aa/src/models/whisper/whisper/audio.py)、[LatentSync固定commit](https://github.com/bytedance/LatentSync/blob/a229c3948406bc2cf6eaf4873e662e70c6a04746/latentsync/whisper/whisper/audio.py)。
- AVTR：`components/hubert.py:44`直接转交audio_batch，`avtr1_motion_generator.py:Normalizer`明确为motion统计；[固定commit入口](https://github.com/avaturn-live/avtr-1/blob/eb1e5a89ed7bec3debe712b090efd676b3a420bd/src/avtr1_renderer/components/hubert.py#L44)。

## 标准化配置与公式

本次保存了facebook/wav2vec2-base-960h、cbsjtu01/IMTalker配套包、FrancisRing/StableAvatar配套包的preprocessor_config.json，三者159B、同SHA `b225d617c025463b9e157e06afea8b90dc7078fc70b013c533328423e0486b4a`，均do_normalize=True。各HF固定revision及原URL在`config_and_historical_receipt.json`。其它base-960h路径由代码声明同checkpoint建立公开链条，但未取回每个历史容器的配置body。

HF波形标准化为 `(x-mean(x))/sqrt(var(x)+1e-7)`，沿当前送入片段计算。忽略epsilon且无裁剪时，正标量增益会被抵消；epsilon使它不是数学严格不变。不同TFG按全句、流式缓存窗或视频窗口调用，因此标准化支持也不同；本调查不把它们当完全相同算子。

实际Wav/Muse/Ditto的精确变换与旧source/config核验见 [[三类TFG音频前端的谱缩放与时间上下文]]。尤其Ditto是固定affine，不能由独立HuBERT实验的HF前处理外推。

## 对机制研究的含义

归一化主要消除全局尺度，仍保留局部能量包络、相对频谱、音素时长/局部时钟、发声实例和上下文差异。已有历史TTS正向分差既出现在没有逐输入标准化的路径，也出现在公开实现有标准化的IMTalker/V-Express/FlashHead/LeapTalk等路径。研究上应按实际接收链分层，不能把全部TTS评分差统一归给整体RMS，也不能由源码归一化断言实际生成/SyncNet对任何音量都完全不敏感。

历史原生双臂还同时改变了驱动音频和评价音频；本调查没有分离生成端与评价端，也没有新增视觉/感知证据。LibriSpeech相关历史结论仍按地图排除独立资格。若后续需要验证归一化的因果作用，优先相同权重/相同输入切换一个明确前处理开关的小比较，再决定是否渲染；当前不自动启动。

## 产物与保存

本地`runs/tfg_audio_normalization_audit_20261018/`保存12repo的commit/tree索引、选定源码正文、配置、实际下载receipt和历史LeapTalk源码；来源正文逐文件SHA汇总见`artifact_manifest.json`。没有远端状态修改、权重下载或GPU任务，没有新增统计显著性判断。初始两个猜测文件路径返回404后，沿已下载tree定位实际文件，不能把404视为模型能力证据。

## Observations

- [result] 公开已测TFG路径可分为逐输入标准化、固定缩放/无显式标准化、导出或闭源内部不明三类；不能由编码器家族名替代调用链。
- [correction] Ditto实际图入口固定affine已经有同SHA常数证据，旧“HuBERT必然抵消音量”与“Ditto内部归一化完全未知”均过于宽泛。
- [limitation] 多数历史旧容器的精确source/config快照未恢复，公开版本事实不能冒充旧批次已逐字节追溯。
- [decision] 不为本次源码调查启动权重下载、音频编码或TFG/SyncNet；AVTR导出图和闭源Seedance保留明确未知项。

## Relations

- relates_to [[研究交接总览]]
- relates_to [[三类TFG音频前端的谱缩放与时间上下文]]
- relates_to [[52-encoder-direction-commonality]]
- relates_to [[SYNC-C-SCORES]]
- relates_to [[02-tfg-cross-model]]