---
title: PhoVis VocaLiST与Synchformer代码权重发布核查 2026-10-28
type: research
permalink: tts-exp/docs/reference/pho-vis-voca-li-st-与-synchformer-代码权重发布核查-2026-10-28
tags:
- synchronization
- release-audit
- phovis
- vocalist
- synchformer
---

# PhoVis、VocaLiST与Synchformer发布可用性核查（2026-10-28）

本次核查官方论文、作者仓库、发布权重和推理源码；没有安装、完整下载权重或运行评分。只用Range 0-1023请求验证下载端点，不把链接可访问称作模型已可推理。E盘已满，小文件保存C，不删除现有工作。

## 结论

|方法|官方代码|权重|本项目落地判断|
|---|---|---|---|
|PhoVis|本次未找到公开官方实现|本次未找到参考构建资产/RF评分器发布|不是可立即下载使用的评分器；还依赖原版同步参考视频|
|VocaLiST|公开，训练/测试/模型均有|README给Google Drive，但本次匿名浏览目录未显示文件|代码可研究，当前官方权重获取尚未确认|
|Synchformer|公开，训练/单视频example.py齐全|LRS3及AudioSet完整同步checkpoint实际206可访问，各约1.13GB|最明确的即用候选；未实机验证推理和中文适用性|
|MTDVocaLiST（补充）|作者公开蒸馏学生代码|GitHub release实际206，53278751字节|可取得的VocaLiST家族替代，但不是原VocaLiST权重|

## VocaLiST

[官方仓库](https://github.com/vskadandale/vocalist)，核查commit c265b59bcf7cd8559a265aaf454a9333ec24cb0f。
[README](https://github.com/vskadandale/vocalist/blob/main/README.md)给[官方权重目录](https://drive.google.com/drive/folders/1-g4qHUNNcCZpmSqEflKMxPMvwnn9e88N)。
本次GET200，浏览器标题VocaLiST_Weights，但匿名页面显示空目录提示，没有可见文件；不据此断言永久失效或作者未发布。[2022年作者issue](https://github.com/vskadandale/vocalist/issues/1)曾确认发布完成。
[test_lrs2.py](https://github.com/vskadandale/vocalist/blob/main/test_lrs2.py)期待vocalist_5f_lrs2.pth，提供模型加载与不同时间窗口offset准确率，不是现成MP4到Sync-C脚本。自定义队列需保持官方预处理和窗口规则写薄适配，不能自称作者官方整段分数。
输入16kHz/25fps、Wav2Lip式96x96脸部帧，再取下半脸；requirements是Torch1.10/torchaudio0.10/CUDA11.3等旧环境。
代码README明确CC-BY-NC及复用组件各自许可，不称无条件商业开源；未确认另有权重许可证。

## Synchformer

[官方仓库](https://github.com/v-iashin/Synchformer)，核查commit b66668a1521d7567cc760e5544b2b5b53179b687。
[README发布表](https://github.com/v-iashin/Synchformer#pre-trained-models)列LRS3、VGGSound、AudioSet模型以及配置；完整同步模型包含音频和视觉特征提取器。
LRS3完整权重23-12-23T18-33-57.pt，本次HTTP206，Content-Range总长1131113925；
AudioSet完整权重24-01-04T16-39-21.pt，本次HTTP206，总长1131153989。
端点base为 https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/ ，后接对应ID目录及ID.pt。
LRS3配置实际可GET，max_off_sec2、offset21类、crop_len_sec5；因此是±2s、0.2s间隔的离散网格，不把它当毫秒级精度。最终精度须检查具体配置和控制。
[example.py](https://github.com/v-iashin/Synchformer/blob/main/example.py)支持--vid_path、--exp_name、--offset_sec0及--device，自动取配置与checkpoint，输出offset类别概率。输入25fps、16kHz、短边256。示例可能重编码，必须审计实际音频，不把默认媒体处理直接等同本项目无损PCM路径。
[LICENSE](https://github.com/v-iashin/Synchformer/blob/main/LICENSE)为MIT代码许可；第三方和数据/权重条件另核对。README提示PyAV版本影响复现，8.1或9.0与较新版本不一定同结果。
优先LRS3语音模型作为适用性先导，AudioSet可为后续不同训练域对照；不能看结果择优后只报告赢家。

## MTDVocaLiST补充

[作者仓库](https://github.com/xjchenGit/MTDVocaLiST)，commit5795416f9277094d3ae6215596a436a0a01f79ad。
[官方release](https://github.com/xjchenGit/MTDVocaLiST/releases/tag/v1.0)，[学生权重](https://github.com/xjchenGit/MTDVocaLiST/releases/download/v1.0/pure_MTDVocaLiST.pth)实测Range返回206，Content-Range bytes0-1023/53278751。README给SyncTransformer(d_model=200)、load_state_dict与test_stu.py。
它是VocaLiST蒸馏学生，属于同一家族，不当额外独立投票。根树未见LICENSE，不能沿用另一仓库许可或臆称MIT。

## PhoVis适用性修正

[Amazon官方页面](https://www.amazon.science/publications/perceptual-synchronization-scoring-of-dubbed-content-using-phoneme-viseme-agreement)，[官方PDF](https://cdn.amazon.science/04/22/1d72d09b4cfdbe76580f794ad6ab/perceptual-synchronization-scoring-of-dubbed-content-using-phoneme-viseme-agreement.pdf)。
作者论文§3.1假定lead/lag问题已不存在或由其他算法纠正；§3.5从original version视频及音频建立每片段phoneme-viseme reference，再评价配音。电影片段ASD筛选使用SyncNet；不是全流程完全独立于SyncNet的时移裁判。
感知映射还使用约8200个Prime Video专家标注样本训练随机森林；通用landmark模型不能替代整套官方评分资产。本次未找到官方代码、字典生成实现、RF权重与数据发布。负面结论限定本次查证范围。
因此收回上轮“优先立即部署PhoVis”的建议。静态头像队列无真实原版口型，严格原方法不直接适用；可在有自然同步参考视频队列借鉴空间音素-视素诊断。

## 下一步建议

先部署Synchformer LRS3及可取得的MTDVocaLiST做小型资格验证：已知整体时移、局部时移，以及保持时间结构的声学扰动。通过后再比较现有自然/TTS原生视频。保持评分与自动重对齐分开，中文可靠性单独验证。本次只完成调研，未启动部署或评分。
