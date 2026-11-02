---
title: native-axis-factor-pilot-20261013
type: note
permalink: tts-exp/tasks/native-axis-factor-pilot-20261013
status: paused
assigned_to: root
started: 2026-10-13
current_step: 23
parent_task: Develop cross-TFG natural-audio enhancement head
---

## 用户暂停（2026-10-15）

用户明确要求先不继续跨TFG音频增强头研究。本任务及native-axis后续停放；32参数已完成梯度/LP结果和有限6forward SOURCE_ONLY源码保留，当前无后续执行准入。恢复须用户再明确要求。独立处理链L1/L2预测审计有自己的Task/run，不受此研究线暂停影响。

# native-axis-factor-pilot-20261013

## Description

固定RMS、内容和原时钟，在TFG原生空间隔离分离度、局部稳定性、能量/谱形，并用独立口部运动核对SyncNet。ROOT负责理论/冻结小设计/BM及一次关键实际QA，代码和实验沿用原GPT-6 Luna/max。先有效性小筛查再扩大，长期增强头active。

## Current State

报告39–42/codec/前端parity、16参数6forward/6VJP及saved-J box-LP均关闭，不重跑。16局部线性STAB幅度未达门；32严格超集唯一6forward/6VJP也已真实rc0，ROOT一次六数组phi/J/FD/旧Jbridge QA dd9400b2…通过。原Angular saved-J primary3+固定旧尺度secondary3 LP已rc0，ROOT实际原/对偶QA3fa5b6fd…通过、三primary全部达raw门；条件有限6forward源码准备中，尚无actuation/TFG准入。保持原自然RMS/N/clock/内容和全部统计门，长期head active，可用原型仍model-aware/RMS fallback。

## Completed Steps

| 步骤 | 已关闭的科学/资格 | 权威入口 |
|---|---|---|
| 1–5 | native18端点0/9格双人；physical6端点0joint、meanΔC+0.001 | [[39-native-input-factor-mechanism-screen]]；ROOT9e03f9c5…/46fc8894… |
| 6 | natural9FIT构音属性、1896tokens/162anchors/1972margins；无共同三TFG属性门 | [[构音属性与口部功能坐标诊断]]；ROOT95c21c92… |
| 7–9 | 8MFA/144旧轨迹与完整17lag；参考p后六格Cp/Hp过门，旧head无共同收益 | [[40-functional-response-coordinate]]；ROOT4dd93b5e… |
| 10–13 | 6CPU native编码、18旋转/同featurehook视频/评分/口部；完整双向门1/6格、0模型双人 | [[41-native-articulatory-response]]；CPU QAe9735787…、responseQA72ef059b…、PCMclockQA3d4b46d7… |
| 14–15 | 同phone局部支持/odd-even及18token分量；Ditto6/6 signed odd正，Wav/Muse语境混合 | [[原生构音干预的局部支持与奇偶响应诊断]]；[[接收端局部构音响应坐标]] |
| 16–17 | 实际forward/observer作用域；Muse旧Cp直接暴露0/14，非trackerstate | 接口备忘4338750e…/temporal boundary6f69b668…，报告41解释修正 |
| 18 | output-owned六行完成、两句功能双向通过；编码传播/共享primitive CPU诊断完成 | [[42-output-owned-native-coordinate]]；ROOTf470d1aa…/44e4eb5e… |

## Step 19 — completed bounded codec diagnostic

冻结协议 `.codex/scratch/native_attribute_response_probe_20261015_codec_diagnostic_attempt01/codec_diagnostic_protocol_attempt01.json` SHA062d01aae34493ea82bb781672b7a6d6a18f7cf0fa3a2b549e8df82b3ab932ee；执行source01 SHA4befcb7705790fd732aab7dead62cadd2ebc597d95845e0a851e2cb290d9cf29。ROOT已审并条件准许唯一六行串行：同504PNG→512 FFV1/yuv420p→同224 Lanczos FFV1/yuv420p→严格原PCM→既定Stage-A SyncNet，复用full/guard20支持。FFV1仍有RGB/YUV转换与4:2:0抽样，属于CODEC_DIAGNOSTIC_ONLY，不替代正式H264分数。逐codec的plus/zero/minus差值和codec间差值全报，不后验选符号或lag。

唯一worker2578355/50189231于UTC2026-10-14T22:31:26–22:32:17Z rc0，6/6PASS/18命令rc0。postrun22:34:19Z launcher/supervisor/worker均GONE，GPU0%/5MiB/compute空。remote output manifest c24b245d…；compact54selected文件260964B，local manifest e81175b9…。ROOT12新矩阵QA f1135ddc…通过，official f32曲线舍入≤1.858e-6，full/guard20支持原样；不重评分/重launch。

FFV1 plus fullΔC S22−0.040/S24+0.024，但D0+0.040/+0.008，两句未过joint；四个±paired arm均full joint失败。guard20 plus C由H264+0.118/+0.398变FFV1−0.119/+0.060。原事件lag0仅从已有曲线定点提取，旧I plus仅S24增强，新I+4的PNG raw C0两句不动、H0下降。旧H264与新PNG观察方式分开，不改p4冻结门或推断物理延迟。完整结果见[[接收端局部构音响应坐标]]。

## Step 20 — completed actual forward parity

adapter01 df5dffc1…/driver01 3ff43d07…/manifest01 00bda125…；ROOT冻结protocol109d37ceb8bfa2d00208ca1ab742146d9f0c3c607bb43de23cb8e4376f5df393（mel2e-6、native2e-5/relative1e-6），实际来源/8 local body绑定通过。

唯一两CPU forward parity已真实完成：supervisor2584009/50324585、worker2584012/50324585，UTC22:54:00–22:54:06Z，实际terminal rc0、2forwards，已退；外层SSH一次EOF/rc2发生在已打印PID之后，未重试。原HF mel与旧CPU6 features两句逐值完全相等，ROOT原数组QA12f49127ae9e19754c1024613c578c284bd1ad07b380c647c817d0f36cc427d9：mel/native maxabs、relativeL2、RMSerror全部0，原f64 j25 clock exact。只有input/output requires_grad旗标，无backward/优化/TFG；不可重复forward。

前端exact parity建立了继续waveform研究的可执行基础，不能写成已验证梯度数值、波形可达或head效果。scratch source/输出路径及实际身份见HANDOFF，source/runtime与原CPU6 qualified Whisper root保持，无新权重下载/依赖。

## Step 21 — completed six-forward physical-coordinate phase1

Task21由原Muse Luna仅SOURCE_ONLY实现16参数自然波形basis：原400fft/160hop STFT，8平滑log频带×2原MFA core时间profile（plateau/早晚反对称），a(+)/i,y,u(−)固定类别符号，未知未支持0；delta频谱合成加回原x再globalRMS补偿，原N/indices/clock不动，theta0 identity、无clip/warp。仅模块与精确time绑定草稿，不执行DSP/encoder/Jacobian/优化/TFG/实现测试。ROOT下一设计真实frontend内分离/稳定/局部能量三坐标的Jacobian/nullspace小可控性探针；不盲目拟合p4+，无新执行预算。

ROOT已冻结阶段1数学协议realizable_native_coordinate_protocol_attempt01.json SHA0c631824423407bf0efc5b747538e47e69cfd8c6e8b87492c978db032151988f，当前FROZEN_MATH_PENDING_SOURCE_BINDINGS，非执行批准。固定original I/同S17，natural9FIT per-phone protos（去tone、speaker/utterance平衡、≥3其他speaker）定义SEP实际nearest margin、STAB负角离散度、ENV class-signed log native radius比；均f64并basephone等权。这只是三summary scalar，不是完整Gram/noncore/u纯度。head02 b68f656a…/contract02 7bbefcf9…仅修16k守卫，01未执行保留。原Angular准备calib/helper schema和线性控制reader，原Wav准备runner；source齐后ROOT另准一次CPU calibration及最多6CPU forward/6VJP（2theta0+4FD，h=.05dB），0GPU/训练/TFG。J每clip3×16并共享stack；记录raw norms/row-normalized SVD/PInv三共同targets残差，不强制rank6。所有门/阈值与scope见协议，不自行变更。

执行绑定补充 `realizable_coordinate_phase1_execution_binding_attempt01.json` SHAc76d0a22…：首次校准PID113840在fold JSON元数据speaker_id字段KeyError、rc1，无JSON/NPZ，原receipt/null字段与addendum保留；ROOT显式准入helper02仅一行元数据修复并第二次CPU聚合，总尝试cap2（1失败+1修复），不改科学定义或阈值。runner02修calmeta.fold_order接口、θ0身份/phi几何门和FD先波形后encoder；实际产物绑定/ROOT admission前仍不encoder。

波形basis的原相位仅指构造，OLA后phase/localpower collateral仍报告。未来dose/偏差cap与三native坐标定义未冻结；真实frontend共享约束、原事件同步及full自然音轨评分要分开，不后验翻号/选lag/扩大cohort。长期head目标active，当前仍model-aware route/RMS fallback。

### 真实波形三坐标可控性：校准完成，CPU phase1已准入

可微前端逐值parity后，采用同一个16参数局部频带波形basis，直接穿过真实Whisper frontend，测原audio core I的SEP/STAB/ENV三summary scalar。自然9FIT校准已完成：18 natural序列、466phone tokens、47basephones中40合格；六target own均合格（a/i/u各9speaker，y4speaker），实际nearest/second gaps为0.018460–0.039640。ROOT实际target数组/φ/m/r0/nearest与prototype unit/support核验最大2.22e-16；这是小可控性设计资格，不是口型/score/head收益。

冻结数学protocol0c631824…、head02 b68f656a…、calibration JSON ee45a753…/NPZ f289185b…、ROOT scoped QA a0238f84…。首次聚合在写JSON元数据时KeyError rc1、无输出；ROOT显式准入一行speaker_id修复后第二次rc0，两个实际attempt均保留，不隐瞒原单次预算例外。无科学公式/阈值变化。

ROOT单次phase1准入c4001b3f…绑定runner03 fab0eab6…/input03 58e6751d…，仅两句CPU、最多6encoder forward/6VJP（2θ0+每句1±FD），原N/RMS/j25、h=.05dB、0GPU/TFG/训练。θ0身份与基线φ门先于VJP、FD波形门先于encoder；共享stack三目标用固定SVD/PInv检查，不要求rank6。当前已交原Wav Luna staging/执行，尚无实际J/FD读数，不把SOURCE_ONLY方法或线性预测写成有限剂量单因素/泛化效果。权威恢复：Task21/HANDOFF及probe/realizable_coordinate_jacobian_attempt03/。

Phase1执行身份更新：03 worker2586601/50573256在clock gate rc1，0forward/0VJP（把continuous core端点误与native frame centers相等比较），原件已读回；ROOT仅准入source04修被比较对象，无时钟/数据/门阈值变更。04 ADMITTED e39c2b82…，runner2b404dcd…/input4fd39018…，worker2587456/50599920、supervisor2587453/50599905于UTC23:39:53.465977Z启动；唯一output `/dev/shm/cross_tfg_realizable_native_coordinate_jacobian_20261015_attempt04`，control加_control。尚待终局，不重launch03/04。

## Step 22 — completed saved-J bounded controllability

### 已执行：真实波形phase1，梯度合格、三目标未全部过剂量门

唯一修复attempt04 worker2587456/50599920于UTC2026-10-14T23:39:56.602701–23:40:09.396761Z真实完成，6CPU forward/6VJP、rc0；所有相关进程已退/GPU compute空，六完整NPZ与control已本地读回。先前03在错误的continuous区间端点与frame centers比较门前rc1、0forward/0VJP，保留；04只修比较对象，原N/RMS/clock/indices不动，不重复运行。

ROOT实际六数组QA b4230b2e…：两θ0 PCM/native逐位同既有基线，clock exact；φ独立公式误差≤9.99e-16，J与receipt误差0；FD最大绝对误差9.58e-7、相对误差0.001235（0.124%），全六坐标在冻结门内。FD物理RMS相对误差≤1.05e-7、波形relativeL2≤0.001201、gain≤0.085232dB。共享row-normalized 6×16 J满秩6、最小奇异值0.130988；局部线性可控方向已获得，尚未实际actuate三个独立目标。

固定min-L2解在coeff≤0.25dB的原剂量门：SEP预测(S22,S24)=(0.000099266,0.000317180)，前者略低于1e-4；STAB=(0.000050083,0.000021385)，两者未达；ENV=(0.000233122,0.000136534)，两者达到。三common targets residual约1e-15、其他scalar预测spill≤1.36e-19。不降低原阈值或把rank6当全部过门，不启动新TFG。ROOT独立PInv QA a5c9afa4…与原Angular reader03 c58db7f9…一致；reader02缺换行SyntaxError原件保留，03仅修换行、数据-only重分析，不encoder。

下一只用已保存J做有界LP：在同一Mθ=αb/coeff±0.25dB下求最大α，分清min-L2选解保守还是该16basis本身受限。该分析不放松common target/剂量/threshold，不再encoder/DSP/VJP。原Luna复用已有LAN CPU环境，若无SciPy则不安装/不扫描，仍不GPU/训练。实际原件在probe/realizable_coordinate_jacobian_attempt04/，readback manifest35b183be…/receipt da61a990…；线性包装NPZ5eba675e…。

## Step 23 — completed grouped32 CPU gradient qualification; bounded LP active

**16参数的稳定性幅度瓶颈已定位；松开类别滤波耦合的32参数实际梯度资格已通过，保存的J三目标有界可行性已通过。** 三坐标为SEP实际nearest-phone margin、STAB负角离散度、ENV class-signed log native radius，均original audio I/原S17/原N/RMS/j25。它们仅是三个summary scalar，不是完整Gram/noncore/u的单因素纯度。自然9FIT校准18序列/466tokens/47basephones中40合格，六target全部合格，直接复用不再聚合。

16实际6CPU encoder/6VJP的ROOT QA b4230b2e…通过，θ0 PCM/native exact，FD maxrelative0.001235。saved-J box-LP的ROOT原/对偶QA363961f6…通过；coeff≤.25dB下αmax/两句raw1e-4所需α为SEP.276350/.178062、ENV.436443/.210566通过，STAB.090980/.273843未过。因此SEP旧min-L2不足是选解保守，STAB在该局部线性共同目标/box中存在幅度瓶颈；不降低门，也不称非线性不可行或head无效。

32参数[2group,8band,2time]只松开a与i/y/u强制相反的滤波，其他支持/时钟/RMS/DSP/剂量/科学门不变，无speaker或单phone特异自由度。唯一worker2590233/50800792、supervisor2590232/50800788于UTC2026-10-15T00:13:38.538108Z真正rc0并退出，6forward/6VJP，六NPZ与control完整selected readback16/16匹配。local manifest77beec89…、receipt a3819828…。首dispatcher在长度门rc2/0spawn，旧partial control保留；修UTF8 byte传输后仅这一个科学worker成功，control_retry01，非重复模型试跑。

ROOT一次32实际六数组QA **dd9400b2347367432cbc9d1a8016e6c239597516c6b49ac0ef279dd07cbd7420**：θ0 PCM/native与旧基线逐位同，φ/m独立公式合格、原j25 clock保持；FD最大relative=0.000466070（约0.047%），RMSrelative≤1.13e-07、waveform relativeL2≤0.000739。严格超集bridge Jgroup0−Jgroup1对实际oldJ maxabs4.92e-11/relativeL2≤8.42e-8，normalized6×32 rank6/minsv.438568；未增加encoder验证bridge，未将rank当效果。

下一数据-only主分析仍用各32row norm归一化的三LP和原raw≥1e-4门。另冻结旧尺度比较 db61af5e…：J32/旧16row norm，不二次归一化，同共同b/coeff±.25，3个secondary LP验证新增自由度。不同family self-normalized α对应不同跨句raw比例，不能直接比较α推断改进。原Angular source04 3da1d19e…已ROOT阅读，仅saved-J共6LP、0encoder/DSP/VJP/TFG；未执行有限剂量或声称已胜RMS。

32 saved-J全部6个LP已单次rc0（PID2591556/50874223，UTC00:25:36.499–.796），result76c76f19…/实际receipt02 ec7819a7…；receipt01错误local source路径/null字段保留，02只修记录，未重跑。ROOT原/对偶QA **3fa5b6fde823c12d8ed4603327ebda4b9761a7fb649d938515551a6767a78e64** 从actual theta/J重建所有6个KKT/原对偶gap≤2.22e-16，三primary目标均达两句raw≥1e-4：SEP(.000376260,.000937314)，STAB(.000442617,.000243567)，ENV(.000440926,.000268562)；非目标线性spill≤4.07e-20。

固定旧16尺度比较中α32/α16：SEP.564611/.276350、STAB.580322/.090980、ENV.550549/.436443，三个均过原尺度raw门；稳定性最大共同幅度约6.38倍。这说明解除类别滤波强制耦合在该局部线性族中移除了原稳定性幅度瓶颈；未增加dose/改变RMS/clock/原三个scalar。有限剂量是否仍能隔离，以及口部/正式分数是否受益，需接下来的6forward小验证，不能由LP推断。

条件有限剂量协议400a4ce1…已冻结但**未执行准入**：仅当32actual QA及primary三目标全部过门，三个共享LP轴各两句共6新CPU encoder、0VJP/0baseline/0TFG。固定原RMS/N/clock，target raw≥1e-4、预测误差≤20%、归一化offtarget L2比≤10%，完整记录native collateral。small DSP/clock/RMS约束不等于词义或口型准确验证。若不通过，按具体瓶颈做合理小迭代，不扩大cohort/训练掩盖。

权威probe=`.codex/scratch/native_attribute_response_probe_20261015_attempt01/`。16六NPZ/QA/旧LP在realizable_coordinate_jacobian_attempt04/；32六NPZ/实际QA在realizable_grouped_waveform_jacobian_attempt02/，数学/模块/条件协议在realizable_grouped_waveform_coordinate_attempt01/。32 runner02 80bf0506…/input02 83be7c46…/admission0dbc8b3e…保持。已终局任务不得重launch；scratch/SHM机器局部，不宣称完整备份。

## Evidence and Recovery

- 报告42小science bundle `basic-memory/docs/experiments/assets/42-output-owned-native-coordinate/`，manifest SHA bd195da8f86f35912bc117d4af974acb4560fc6b6a34e84a8c33568a44467538，107文件，包含67compact readout/887654B及全精度CSV；不包含大媒体，不宣称完整备份。
- Probe `.codex/scratch/native_attribute_response_probe_20261015_attempt01/`；完整媒体 `output_owned_musetalk_render_readback_attempt01/output/` 543文件197713752B，504PNG/6H264/6PCM mux。
- compact读回必须使用 `output_owned_musetalk_readout_attempt02_readback/readout/`；manifest1b4f4544…；effect bb5fa06e…，ROOTresponse f470d1aa…，全PCMclock563e8172…，CPU0a77ad77…，same-H264比较39703dfe…。
- renderer controller2555964/49915805于UTC21:50:39Z rc0/6PASS；readout02 worker2560257/50011173于UTC22:03:25Z rc0/6PASS，均已退；不能重复launch。失败readout01 root.arms错误0processed rc1与consumer前置错误保留，不重跑成功effect。
- 用户取消关机；wjj@192.168.50.2:22运行/公钥SSH。旧训练/中心/Dev/Confirm/归档/cloud donor/visual teacher停放；秘密不文件/argv/BM/log，不运行实现tests/pytest/compileall或安装。
- 最新同chat checkpoint `tts-exp/codex/tts-exp/codex-checkpoint-2026-10-15-t00-21-11-z-bounded-waveform-control-and-grouped-basis-readback` 是不可变快照；当前Task/HANDOFF和真实receipt优先。

## Observations

- [result] Muse的原生构音功能控制已在正确输出所有者坐标下两句复现，但正式full Sync-C收益未出现。
- [decision] 共享primitive/条件所有者/输出事件应作为可复制坐标的一部分；不将任意独立prompt行旋转等同可合成音频。
- [next_step] 接有限剂量源码和实际LP候选绑定，ROOT一次新增接口检查后仅6forward小验证；不重跑已关闭梯度/LP或开TFG。

## Relations

- tracks [[42-output-owned-native-coordinate]]
- continues [[41-native-articulatory-response]]
- relates_to [[接收端局部构音响应坐标]]
- indexed_by [[研究交接总览]]
- contributes_to [[开发跨 TFG 的自然音频增强头]]
