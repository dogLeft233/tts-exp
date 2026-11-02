---
title: 女性说话人上的 TTS 来源方向独立复核
type: experiment
permalink: tts-exp/experiments/女性说话人上的-tts-来源方向独立复核
status: concluded
project: tts-exp
started: '2026-10-01'
---

# 女性说话人上的 TTS 来源方向独立复核

## 问题与设计

核对 AISHELL 官方 metadata 后发现，现有 fresh cal10与eval40全部标为M。因此在任何fresh评估效果汇总被root读取前，独立预声明官方F说话人的复核，检验同一已拟合来源方向的外部转移。保留原主队列，复核不替代主结果。

权威协议：openspec/changes/tts-sex-replication-20261001/spec.md。选择S0124–S0143起的20个F说话人，各两句满足相同mono16k/3–10s/Han规则；不足时记录并按F ID顺延。40句Qwen同模型自克隆。原男性cal d/u/k/RMS及随机方向不重拟合；固定primary face，共640预定评分行。20speaker×2、99%四项联合门槛与随机特异性、raw/unit、缺失分母原样应用。

## 状态与来源
输入与Qwen已经完成，actual-F32-dose修复已封存，CPU预检attempt02已exit0/PASS，GPU首run仍待单GPU队列执行。原80男性双脸主实验完成并经独立消费；本女性复核保留独立40 clips/20 speakers，不由其效果重拟合男性axes/k/RMS。

官方speaker.info（2800 bytes）本地runs/recovered_20260930/local/qwen_source_axis/hf_source/speaker.info，SHA569cef88f7fb1c3173cf3a45b00b4659b427d90af120af91513a2c6f151f7e08。[官方固定revision metadata](https://huggingface.co/datasets/AISHELL/AISHELL-1/blob/bbe295d530192a4cd41644b711c9aecd087df653/resource_aishell/speaker.info)。

MFA attempt05全部80 grids完成，38/40 pair合格，S0133_BAC009S0133W0121及S0142_BAC009S0142W0122均unknown_or_oov_phone，保留不重抽样，不为latent主分母删这2条。Local runs/recovered_20260930/remote/female_eval_20261001/mfa_attempt05/：eligibility raw9a9f17cb90800fbd9cf929f499fe980e0c39b9984156385d30141b8961df3c5a，protocol rawc3641d50cc7fca1145119d12382342f92f54dc1217c7c54fc831a6465b44e1fd，producer raw381cb9dcc683faa80faeb2d3659b5f9266098d078bd3aaf1ee77ba2e448c1931。潜变量仍全部40×16 primary640计划行，视觉端点另有MFA资格。
## 数据覆盖修订

实际准备发现固定publisher HF wav tree只含100个speaker archive（S0002–S0101），没有预声明S0124–S0143。保持speaker/选择规则不变，改用完整官方OpenSLR SLR33原包补足waveform；full data_aishell.tgz需匹配官方MD5 2f494334227864a8a8fec932999db9d8，resource包MD5 957d480a0fcac85fc18e550756f624e5，同时保存SHA256、inner archive/WAV hashes与既有metadata/text比较。镜像只能作为transport，完整publisher checksum一致后才使用。协议修订已保存本地Git bd5e08e（未push），作出时fresh effects仍未读。

[官方校验表](https://www.openslr.org/resources/33/checksum.md5)；[官方数据与镜像入口](https://www.openslr.org/33/)。

## Observations
- [decision] 女性外部复核保持S0124–S0143×2/40clips/20speakers，不替换两个MFA OOV样本，latent整体意向cohort仍40。
- [result] Female CPU02 actual PASS40/20/640；GPU03 actual COMPLETE40clips/640rows/0fail、worker exit0、2948个actualforward均benchmarkFalse/cuDNNdetTrue/detalgTrue。
- [verification] 440个video/crop/JPEG帧计数共47835完全相同，lip windows45635=N−5；640scoreNPZ/40contexts。protocol ce03790adf556e881ff7bafebf08270e12d3266aa534684d70e47893d0976e44，rows8efd0d0ac56e34e647836c8f6b8a01aa27d826b3252356f88f094840414e88a5。
- [next_step] 独立消费40/20完整结果：重算F32剂量、raw/unit四支持域、完整A null和20k99CI，再报告女性原生及RMS/source方向响应；完成GPU不是科学结论。
## Relations

- relates_to [[TTS 来源方向与声学机制确认]]
- advances [[解释 TTS 的 TFG 与 SyncNet 优势]]

## 当前状态
独立consumer04已实际完成，科学状态CONCLUDED；后续全支持域时间null为另行追加分析。source6942b636e43d267ae2c2215950d1b40542788e7d59378a6520e407ac0eaec0a7，actualPID330749/exit0。40clips/20speakers/640rows，0failed/0missing，完整V×A640重建，40context与三random臂120剂量/KKT全合格。报告`runs/recovered_20260930/independent_validation/female_consumer_attempt04/independent_female_summary.json`3,504,578B/SHAcd31ca9541d690ebd34b557215209e094d3a0431aaedaf9a2af4fe45a51062d6，root实际下载、SHA与读取均PASS。04仅改自然/TTS结果单位hash字段名，与PCM16→float64/context逐元素及SHA严格相同；02/03工程失败保留。

女性40/20，单primary face、guard20、20k等权speaker bootstrap、99%CI：natural生成RMS G raw C +.250266[+.116708,+.417378]、unit +.034505[+.017046,+.056129]，C_anchor完全相同。TTS G raw -.043149[−.106908,+.018759]、unit -.009409 CI跨0。natural评分E raw -.023237 CI跨0、unit -.004764[−.008940,−.001395]；I两geometry CI跨0。

原生T−N raw C +.530425[+.260720,+.870099]；q11处理差+.258817[−.003165,+.614622]；配对gap缩小+.271608[+.137158,+.424423]。unit原生+.065583[+.032701,+.099224]、处理+.025922 CI跨0、配对缩小+.039661[+.018510,+.060905]。这是女性新说话人上生成端音量路径与gap缩小的重复。尚未以本分析估计全效应解释比例。

CAL男性拟合静态来源方向移植到女性：plus−baseline raw -.004892 CI跨0、plus−randommean+.008648 CI跨0；plus−minus raw+.029382 CI跨0，unit+.006876[+.000233,+.012397]为次级条件阳性。主四联合gate均false，eligible true；保留次级响应，不能把它升级为普遍优于baseline。现有consumer内部search_gain沿用旧C错误，C/anchor/G有效；fresh全支持域时间null侧车将独立从矩阵重建D0−D并保留legacy/full-A两匹配边界。

- [result] 女性新20说话人上RMS生成端增益raw/unit重复，配对原生T/N gap显著缩小。
- [result] 静态来源方向主四gate未通过；unit plus−minus次级响应为正。
- [verification] root实际下载SHA/read完整640行独立科学PASS，不由GPU完成或exit0推断。

## 完整女性时间配对复核

独立cached-null02实际exit0并由root gzip下载、完整SHA与JSON读回。报告 `runs/recovered_20260930/independent_validation/female_cached_null_addendum_attempt02/independent_female_cached_null.json` 9,541,203B，SHA1bfa675eafce3595ac07b8a8d88d6d7f4efe75901ad092a3b8b447ddaf3da20a；源码7d78462a1dbefda4b94b541af74d2ff2bc735470b6cf8b37324add169dc76746。原640 rows、320 native score artifacts与40clips/20speakers全部封存核验，80个N/T permutation域、100 full-domain Aπ；legacy/full-A observation/null各自边界匹配，四支持集无短域缺失。20k配对speaker bootstrap、99%CI、同π跨四格。

full-A guard20 raw生成G observed +0.250 [+0.117,+0.417]，mean-null +0.020，时间配对excess **+0.230 [+0.105,+0.385]**；unit excess **+0.031 [+0.015,+0.051]**。N G在legacy/full-A × raw/unit × 四supports × C/C_anchor的32种组合excess下界均正。T G raw excess−0.039 [−0.098,+0.020]、unit−0.009 [−0.018,+0.001]。

原生T−N guard20 raw excess+0.340 [+0.118,+0.566]，q11 excess+0.095 [−0.110,+0.325]；配对gap shrink excess **+0.245 [+0.118,+0.388]**，unit **+0.036 [+0.017,+0.055]**。全部32组合配对gap shrink excess下界也均正。把gap操作差及随机错配基线同时纳入配对bootstrap，不做两CI直接相减，不宣称中介比例。

attempt01在report字典构造KeyError C后退出，另有launcherCR行退出127；旧文件保留。02仅同时初始化G和T_minus_N的metric字典并绑定新attempt，统计数学未修改。女性与男性一致支持句级RMS生成路径及正确时间配对响应；仍分别解释SyncNet、可见运动与音素/人类感知。

- [result] 女性独立full-domain时间null完成；N G及T/N gap shrink在全部32组合excess为正。
- [verification] 640行、320缓存评分文件、40/20、完整SHA与legacy/full-A匹配重建由root实际读回。