---
title: SyncTalk FP16 更新前保护资格与密度溢出因果核查
type: experiment
permalink: tts-exp/docs/experiments/sync-talk-fp16-更新前保护资格与密度溢出因果核查
status: concluded
tags:
- synctalk
- fp16
- numerical-qualification
---

# SyncTalk FP16 更新前保护资格与密度溢出因果核查

独立版本 `fp16_preupdate_revision_v1` 已验证能阻断已知的非有限状态和反缩放非法更新，保持FP16与原GradScaler正常跳步政策。两套环境各10项保护测试通过；实际ControlledTrainer记账对照、真实NeRF状态拒绝均通过。上游冻结状态的同一forward图干预确认：非有限梯度由密度logit分支传播；这不是已经完成稳定训练，正式重训尚未启动。

## 用户范围与版本

用户明确要求“保持 FP16，另立版本修复更新前的数值检查，并继续查清上游为什么持续溢出。先验证修复有效再重训，原失败记录和原实验版本都保留”。接受的工程决策见 [[044-synctalk-fp16-preupdate-guard-revision]]。本轮修改仅独立guard/controller模块和资格脚本，没有编辑原controlled_training.py、官方network/renderer/utils数学源码、原attempt或原数据资格。

更新前检查loss、模型参数与buffer/密度计数、optimizer state、正有限scale及其float32倒数、实际unscale后的梯度；实际optimizer入口再次检查。已有scaled非有限梯度仍按原GradScaler跳步，不耦合三档跳步、不钳制梯度、不调整scale增长/衰减参数、不切FP32。发现非法值终止该新attempt；拒绝事件独立写numerical_failure.json，不补造完成trace行。postoptimizer检查只能发现更新后异常，不承诺回滚；unscale可能已改变grad/internalstage，拒绝后禁止原对象继续训练。完整重训必须另有统一稳定性资格和新版本入口，不能从旧ep4往旧attempt补步。

## 资格结果

| 检查 | 当前Torch2.5.1+cu121 | 官方依赖Torch1.12.1+cu113 |
|---|---|---|
| CUDA guard单元资格 | 10/10，0skip | 10/10，0skip |
| 健康4步与原版完整状态对照 | 模型/AdamW/scaler/scheduler/EMA逐位一致 | 逐位一致 |
| ControlledTrainer实际记账fixture | update/skip/update的trace与state精确一致 | attempt02精确一致 |
| 真实captured density_grid Inf | forward/optimizer调用0，state SHA前后相同 | 相同 |
| 完整有限模型state人工置scale2^-128 | unscale前拒绝，optimizer0、state不变 | 相同 |

最后一项明确为复制state后的合成阈值测试，不冒称历史临界state。真实模型测试使用原Trainer附件，控制器fixture使用实际原attach_controls/trace/计数逻辑；尚未运行完整epoch/supervisor重训。原训练数学中的exp在FP16 autocast下本来输出FP32，本轮没有将训练改为FP32。

旧环境第一次controller导入失败于Numba缺importlib_metadata，0数值运行；保留official_controller.log。仅隔离环境补importlib-metadata4.13.0和zipp3.15.0，URL/whlSHA存old_environment_metadata_dependency.json后attempt02通过。该fixture使用Trainer.__new__，退出时原析构器报告未设log_ptr；结果断言/trace/state已全部通过且exit0，警告日志原样保留，不当正式训练行为。首次错误shell转义只产生SyntaxError、未安装，随后用独立脚本执行。未改共享venv。

## 持续溢出的传播机制

在官方独立重放捕获global27043完整state上，原FP16 forward仍给有限loss，但density sigma出现Inf。incoming dL/dsigma有限，exp反向的sigma×gradient产生Inf×0=NaN；还观察到有限乘积超过65504，回到FP16 logit梯度时溢出。直接用.5与1的loss multiplier仍有28个参数tensor非有限；这个诊断不调用scaler.update，不是修改正式loss scale。

更严格的同图核查只执行一次forward，保留原loss图，先正常backward，再仅把sigma_net输出第0列（密度logit）的回传梯度置0，保留64维geofeature梯度。两环境baseline都有28个参数tensor非有限；置0后全部参数梯度有限。当前图第0列37NaN/9Inf、其他列0非有限；旧环境39NaN/9Inf、其他列0非有限。每个环境两次backward使用同一loss对象及相同SHA，消除了跨forward ray容量/顺序差异。此处置0是因果诊断，不是训练修复；不能部署为“删除密度学习”。

因此该捕获状态下继续降低scale无法消除已经产生的Inf×0。它解释了进入密度非有限状态后为何AMP可能持续skip，而不是证明每一次历史skip均由同一点产生。首个高scale2^24梯度异常的早期state与后期持久溢出有区别，旧报告仍保留。

## 首发密度点的局部因果

官方global26624首个Inf的精确x/encoded_audio/eye全部有限。两环境以下结果一致：

| 仅诊断的输入/权重干预 | density logit | sigma |
|---|---:|---|
| 原state/原输入 | 104.0625 | Inf |
| audio输入置0 | 117.8125 | Inf |
| eye输入置0 | -41.9375 | finite |
| audio和eye均0 | -8.484375 | finite |
| 仅换ep4 eye_att_net权重 | 22.53125 | finite |
| 仅换ep4 sigma_net权重 | 74.1875 | finite但极大 |
| 仅换ep4 aud_ch_att_net权重 | 108.75 | Inf |
| 全部ep4模型权重，仍用捕获的晚期encoded_audio | 13.1875 | finite |

该点原眼部attention范数759.9163，仅换ep4眼部attention后216.3778；原眼部条件分量范围约[-76.889,24.006]。原官方代码的eye_att与密度exp无界，当前证据定位为眼部条件放大与sigma_net权重共同把该点logit推入指数溢出。不是输入NaN，也不是仅音频分支或新Torch版本才会发生。

边界：只针对固定捕获state/点。尚未沿完整训练轨迹隔离哪项loss梯度、数据/眼部尺度或optimizer演化最先促成eye_att/sigma权重增长，不能宣称+6本身有因果作用、所有帧同因或完整适配绝对无问题。下一步首先对现有完整state和原数据眼部映射做只读审计，再冻结必要的有界对照；不自动更改sigma参数化、学习率或scaler政策。

## 保存、预算与原始失败

`F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/controller_qualification/fp16_preupdate_revision_v1/preserved_snapshot_01`逐SHA核验38个源/合同/结果/日志文件，共259505B（不计非权威pycache）；manifest SHA92016a548dca0ce27ca5ddf060f111396380f3f6f9c8df096e19fcf191f6e71e。保护源SHAf9d05523686e6fd4272f904762add21e5e4dd0c38fc637f199c19b1c7574eb70，上游诊断源SHA7656b3f609259084df478b8619debe296a747ae0b82dec0f88408e1db1fae20f，同图/真实保护源SHA7eccc285bd0ec5509543767462361bc5ca099ae677c0d1bc9ad1bddf0e491ec2。初合同与supplement01/02逐次冻结，初24案例执行前没有声称已有后续源pin。

新版本预算：32/32模型零更新案例，0TFG optimizer/scaler更新；微型fixture实际24/32 optimizer更新（两套环境单元对照16+controller对照8），与旧审计9984尝试分别记账。原失败archive再次SHA核验55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72不变。旧环境大state证据仍在先前preserved_snapshot_02，不重复上传/清理。原24模型/27格、phase预算与数据准入未改。

执行后的controller fixture原字节另保存于scripts/.../qualify_fp16_controller_fixture_v1.py，tests/test_synctalk_fp16_controller_integration_v1.py仅改为显式运行入口，普通测试collection不启动GPU；入口import通过。执行来源仍以snapshot的原始文件及pin为准。

## 眼部输入与已有权重的只读追查

后续CPU只读检查不调用模型forward/backward/optimizer/scaler，不扩展32个模型案例。Brown三训练gain的bs.npy SHA完全相同87ba06aa43e59b4c69cfcbc8c999625fa3c7326cd8b3b75660cbe3c204950efa，shape6000×52；原provider upper选择0:5与8:10列，七维整体范围约[-0.022725,0.881407]且全部有限。au45=false、smooth_eye=false。三个捕获batch的eye逐元素精确等于原provider的bs[img_id]映射；首density事件真实eye精确匹配bs第5882行。occupancy完整state附带batch是上一步，已明确排除作事件输入。

已有state的eye_att权重L2从ep4的11.5841增至首scaled梯度异常14.2377、首density Inf15.2533、首scale<1的15.4609；sigma_net L2为18.3504→18.7030→18.8239→18.8878。这证明变化出现在学习到的条件映射中，未发现这些捕获输入的新增缩放/行映射错误；L2增长本身不是完整梯度驱动的因果证明，不重新认证bs源提取算法。

原renderer网格更新使用max(old_grid×0.95,new_grid)，已入网格的Inf在该规则下不会通过有限的新值自行衰减恢复；这是源码数值性质，不冒称每次skip由网格导致。原eye regularizer只约束渲染得到的下半脸区域attention，空间正则约束邻近点差，均不是全空间眼部attention或density logit硬上界。仍需区分哪些训练loss梯度/区域最先推动该权重轨迹。

独立F保存只读结果与入口source于readonly_supplement01，result SHA1a4de96f85b3e50fdb339bf69f0000dfde6d722d172fdd0878bda40ecde93bfe；verification.json passed仅指字节校验。没有新增TFG训练更新。

## Observations
- [result] 独立FP16保护通过双环境测试及原计数/trace对照，拦截已知非法更新而保留正常AMP。
- [result] 同一计算图干预将28个非有限参数梯度降为0，定位密度logit反向传播；固定点干预显示眼部条件与sigma权重参与首发exp溢出。
- [decision] 原版本、失败、精度与科学预算保留；保护资格不等于稳定完整训练资格。
- [blocker] 上游训练轨迹为何长大到无界密度区域仍未完全隔离；不能仅加guard即启动24模型重训。
- [next_step] 只读追查眼部输入适配及已有state中eye_att/sigma演化，设计可区分loss梯度驱动的后续最小对照。

## Relations
- implements [[044-synctalk-fp16-preupdate-guard-revision]]
- relates_to [[核查 SyncTalk FP16 适配与官方环境]]
- relates_to [[SyncTalk FP16 非有限训练失败的适配与官方环境核查]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]