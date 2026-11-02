---
title: WORLD 音素局部响应与声学剂量
type: experiment
permalink: tts-exp/experiments/world-音素局部响应与声学剂量
status: concluded
project: tts-exp
tags:
- research
- world
- local-acoustics
---

# WORLD 音素局部响应与声学剂量

## 研究问题与当前状态

在固定受体音素时钟和全局RMS的WORLD九臂双向移植上，定位哪些音素/阶段的生成响应改变，并对齐实际谱形、F0/UV/AP剂量。当前CONCLUDED：v6完整独立评分后，local08全部16640计划端点与327文件已完成、下载验封并独立复算。14560可估端点使用共同99% max-|t|区间，2080短T guard20端点保持INCOMPLETE_SUPPORT；结果明确区分完整计划家族和可估子集，最终报告25继续整合新五格与独立视觉。

## 读取EVAL前冻结的设计

`openspec/changes/diagnose-world-local-acoustic-response-20261001/spec.md` raw SHA5fa49ef882ec173639dded57f843991cb529b2bd3dfe5f3ae92e99f4f30d7963。16个exactlabels只用20条CAL/10人≥10distinctclips/≥5speakers选定，保留声调/送气/长度：a˥˩、i˥、i˥˩、i˧˥、j、n、o˥˩、o˧˥、t、ts、tɕ、w、ŋ、ɕ、ʂ、ʈʂ。inventory raw62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2/payload54a51ff1a9af891271b0eaed797dc0bb7508a338ef7e38f922b8f9292fa40f5d。

局部G_i=d(V_RT[i],A_RAW[i+3])−d(V_X[i],A_RAW[i+3])，固定anchor column18/k3；positive只指移植视频embedding更靠近未变RAW评分音，不是global median-C的可加分解。每个SyncNet row的五lipframes各16实际mel列取set union，mel center=m/80；音素权重及early/middle/late三等phase用exact recipient半开区间；边界±40ms、mixed-phone RF及50ms STFT overlap另报，−85ms仅MediaPipe视觉端点。所有arm含EX mel starts必须与recipient一致。

primary local贡献sum(wG)/|S|，包括没有selected label的零权重窗口，保持计划80clips/40speakers；缺phone不缺clip，conditional token均值和曝露counts另作描述。full/valid/guard20/guard15全保留。short S0099_BAC009S0099W0123的T guard20空，相关端点INCOMPLETE，不切换guard15、不39人救援。

原16384 phone-local endpoints（8arms×2dirs×2geometry×4supports×16labels×4whole/phases×observed/excess）在任何EVAL效果前经静态审阅扩展：`spec_addendum_attempt03.md` SHA0faf10ad357b0ce2a655e2eb2612ae2db9282c889cc4fa4025b40bfcf743ad4c加256 unweighted全窗口G，同一16640 planned max-|t| family/同20k×40 speaker draws。time excess使用100full-A permutations，各RT−X共享同π、labels/phases共用，PCG64 seed20260930+orderedEVALID。空支持端点留planned mask，estimable subset interval范围明确，不污染N/full/g15。STFT扩展仅描述，无另外CI/p或posthoc family。

## 剂量与解释边界

剂量同phone×phase×support保存，melcenter映射最近5ms WORLD center、tie earlier且必须相同exact recipient token，不将重复IPA同phase但不同token跨边界join。保存WORLD帧features、mel/WORLD owner与phase masks、windowunion/multiplicity/source paths/hash以重建每窗口共位剂量。频带AP，voiced mismatch/0→1/1→0，jointvoicedF0cents，normalizedSP Hellinger及withinphone adjacent logshape change，SP power不变量全部报告。

EX联合转F0/voicing/AP，F0/AP子臂仅共同有声域，不将EX减子臂称为纯UV。EX_EM/SP_EM局部能量匹配及E是预定控制，保留残差/bound-hit不丢失败样本。WORLD的voiced-state是声源估计，不能把对应清辅音标签上的WORLD有声帧直接等同音系浊音。操纵arm可作物理家族因果对比；phone贡献和dose关联只用于定位/描述，不拟合或按EVAL选择labels/phase/lag/频带、不声称中介比例。

## 验证与执行门

完整160contexts/12160rows/5760raw-unit matrices/1600videos、零失败的sealed v6 run以及独立consumer04 PASS、物理QA04 SHAe78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d为前置。score/json/code/input/member seals先核，随后才读arrays。保存shared bootstrap indices/counts/max_t draws和16640×40 speaker值以独立重算，不保存20k×16640冗余。

local08源码/运行绑定由实际metadata及完整summary.input_seals自动核验；能量处理前剂量用EX_EM→EX、SP_EM→SP、E→RT参数坐标，最终WAV/RMS/gain audit和独立物理QA另验。失败06/07因参数schema和QA sample_id工程错误保留，fresh08实际预飞160/12160 PASS，统计完成退出2只表示预定短T支持不完整、stderr空。未运行pytest。

## Observations

- [decision] 音量基线已完成，局部物理家族及phone/phase响应为当前主线，实施前固定16labels和16640联合家族。
- [decision] 空guard20端点保留计划分母，phone absent零权重不是科学阴性，mixed RF/overlap明确报告。
- [decision] 分开物理arm因果、音素定位、声学dose描述；映射后谱形稳定性不同于native TTS天然平滑。
- [result] 全部327产物/108834694B逐成员验封，14560/16640端点独立bootstrap复算通过。
- [next_step] 完成五格平均谱形/内部轨迹/SELF_GRID及独立视觉，整合报告25。

## Relations

- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[校准音素内谱形动态与清浊覆盖]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
## 完整局部结果与解释

80条/40人、160方向全部进入固定分母，16640端点中14560可估、2080仅因S0099_BAC009S0099W0123的T guard20空而保留不可估。20k×40的PCG64重抽样索引和counts逐位重建，全部可估speaker均值、SE、共同max-|t|及区间独立重算：临界值4.947547631337023，重算差7.11e−15，max-t draws最大误差3.64e−14、CI最大误差1.11e−15。区间覆盖可估子集；计划家族保持不完整，没有39人救援或切换端点。

全部可估区间中237负向、0正向、14323跨零，N→T 177负向、T→N 60负向。BOTH85、SP68、SP_EM80、E4；EX/EX_EM/F0/AP没有联合显著响应。152个phone-local负向区间均在full/valid（两support在本数据上重复，相当于76个不同比较），其中/n/64、/ŋ/78、/tɕ/6、a˥˩晚段4。N→T占138，T→N占14；另有85个不分phone的all-window端点负向。guard15的phone-local联合区间全部跨零；不能将full/valid定位写成去边界后的稳定音素专属效应。

这给出清楚的局部定位：谱形移植损害集中于覆盖鼻音/n、ŋ/的短时输入窗口，部分/tɕ/早段与a˥˩晚段也响应；能量匹配SP_EM保留鼻音局部损害，反应有raw和unit、observed和full-A time-excess证据。生成器以五视频帧对应的mel列集合读入音频，一个音素覆盖窗口还含共发音和邻音。phone贡献是全谱形干预的共位定位，不是对该phone单独随机干预，不可直接解释为鼻音中介比例或单一共振峰因果。原生TTS的增强需要协调的短时输入结构解释，后续事前冻结五格将区分平均log谱形、内部残差及网格插值。

全部327文件安全解包与inner326-member inventory验封PASS；完整统计`runs/recovered_20260930/independent_validation/world_local_response_attempt08/package02_unpacked/`，summary raw 0b7fe22c5f13ea3ab1dff1cf7a8a1ad6f1e21e0136e0334d854deae1e82e8ad7、bootstrap NPZ由summary metadata自动绑定。独立复算receipt`local_response_independent_bootstrap_reconstruction_attempt01.json` raw 841210d86e9ae05b3b49ee8a1dc2ea0440973f8cb161b33865e05cc53336bb88；包括planned mask、每endpoint40speaker向量、20k共同counts/indices/max-t、160dose caches与窗口覆盖。不要按shell折行摘要抄写bytes/SHA，inner inventory文件实际45134B。

- [result] 谱形局部损害在full/valid的鼻音/n、ŋ/窗口最集中，能量匹配后保留；guard15没有联合显著phone-local响应。
- [verification] 全16640计划/14560可估的共享bootstrap和237负向区间独立重算，327文件完整逐字节验封。