---
title: 022-fit-only-cross-tfg-native-score-refinement
type: decision
permalink: tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement
status: open
project: tts-exp
decided: '2026-10-09T02:38:48Z'
tags:
- audio-enhancement
- fit-only
---

# 022-fit-only-cross-tfg-native-score-refinement

## Decision

下一轮把Wav2Lip、MuseTalk、Ditto的实际生成视频与原自然PCM SyncNet分数纳入Fit内共享参数选择和优化。复用现有9参数局部过渡STFT头，先固定抽取训练轨迹step30/60/90/120；在完整Fit40/20说话人、同一生成seed20261003上比较四个快照，再以三模型黑箱联合优化修正共享theta。增强音仍只驱动生成，最终恢复完整原自然PCM和时钟。未来Dev20及Confirm80/40/LatentSync保持已有冻结分界。

记录抽取支持已完成的legacy Fit120与未来DirectD0两种明确区分的training family；旧轨迹不得写成新D0结果，中间快照不得伪称terminal step120。现有terminal-only dispatcher保持原契约，新增独立Fit-only记录/应用/计划/容量门。

## Context and rationale

独立源码审查验证了12个实际源文件及其SHA：Fit120和DirectD0都只沿Wav2Lip generator+SyncNet反向传播。MuseTalk/Ditto只参与后续评估；encoder probe在inference mode或CPU ONNX/NumPy执行，不构成它们的同步训练信号。实际旧Dev中step120相对RMS的FULL Cbest均值W2L+0.068454、Muse+0.000900、Ditto−0.035908；D0正向改善分别+0.030824、−0.028086、−0.103724。由此优先处理训练目标覆盖不足；这是一项方法设计依据，后续仍测量各前端声学响应和视频效应。

W2L可微代理、旧Fit27-query投影诊断和最终编码视频官方SyncNet分数各有明确端点。新联合Fit消费者只接受实际三模型生成/编码视频、原自然音及其源时钟的评分证据；旧27-query W2L数组只可作已标记的代理或几何参考。不能为缺失的Muse/Ditto数据补虚拟模型行、track ID、时间戳或未来SHA。

## Selection contract

四快照和identity/RMS两对照共有6种方法，40片×3模型×单seed=720个独立生成/评分行；每个候选的三臂360行引用两份共享对照。Fit roster全40片/20人固定，不读取Dev作theta选择。每个clip/model/seed按所有比较方法的实际query keys/valid masks求同一FULL源时钟交集，至少1个有效query；不把不同快照各自的三臂支持直接混用来排名。某计划方法/context缺失或失败导致整个比较族证据不完整，返回MISSING_FIT_EVIDENCE/CONTROL_FAILED，保留完整分母且不给选择或初始化结果。

端点先片段/seed等权到说话人，再说话人等权到模型。Cbest/Canchor改善=candidate−reference；D0改善=reference−candidate。合格Fit候选须对identity及RMS两参考逐模型Cbest>0、D0与Canchor>=0，且等模型Cbest相对RMS>=0.050。合格者按等模型RMS Cbest降序、最差模型RMS Cbest降序、原训练step升序排名。Fit资格用于选方法进入下一阶段，正式Dev验收仍执行ADR021的FULL/GUARD20/content/visual和完整分母。

没有合格快照时仅允许对完整评分候选选优化初值：V=mean(model,reference)(relu(−D0改善)+relu(−Canchor改善)+relu(−Cbest改善))+relu(0.050−等模型RMS Cbest改善)。按V升序、失败门数升序、最差模型RMS Cbest降序、等模型RMS Cbest降序、step升序。严格Cbest=0计失败但hinge为0。这项初始化不代表准入，也不将RMS基线作为已完成增强方法。

## Joint refinement contract

采用SPSA的双侧同时扰动估计。它只需要目标函数测量，能够使用现有三模型原生推理接口；每次梯度估计用两次目标测量，每次测量在本实验中包含3模型×8片段，而不是两段视频。[Spall原始综述](https://www.jhuapl.edu/spsa/pdf-spsa/spall_an_overview.pdf)

本轮固定20次更新，4说话人×每人2片的cyclic批次遍历20人四次，RNG seed20261009，Rademacher九维方向。a_k=0.5/(k+5)^0.602，c_k=0.25/(k+1)^0.101；g_i=(Lplus−Lminus)/(2*c_k*direction_i)，每分量更新−a_k*g_i裁至±0.25dB，不裁扰动theta。沿用头的3dB平滑径向投影、正实STFT mask、实际RMS/N/源时钟契约，输出应逐行核验。

tau=0.050；L_C=mean_reference(tau*log(mean_model(exp((target_reference−delta_Cbest)/tau))))，target_RMS=0.050、target_identity=0；L_guard=2*mean_model,reference(tau*softplus(−delta_D0/tau)+tau*softplus(−delta_Canchor/tau))；L_theta=0.001*mean((theta−initial_theta)^2)。全部delta采用上述改善方向和说话人等权。使用稳定logsumexp/softplus。该目标权重与schedule是本轮前瞻选择，不称最优或保证收敛。

两扰动必须使用相同预定batch、源身份和生成seed，并在identity/RMS/plus/minus的四臂相同源时钟交集上评分；theta、source pins、receipt必须匹配ask产生的实际proposal。缺失/失败/非有限值返回明确NO_UPDATE，保留失败，不补分数、自动重跑或重启job。更新20次共960个新候选生成/评分行，另外k5/10/15/20四个center完整Fit评分480行；初始720行与二者共2160个独立物理行。每阶段运行前须实际容量/模型/私有启动门通过，归档与传输继续遵守现有可恢复契约。

只有四个预定center的完整Fit分数能参与最终选择；mini-batch分数不用于方法准入。若与原四快照一起选择，则最终八候选+两对照的10方法在同一FULL交集比较。选定后冻结theta及源码，执行新独立Dev20；失败则保留结果继续Fit内方法迭代，确认集仍封闭。

## Alternatives and consequences

继续只优化W2L可微D0可能改善该模型，但缺少其他原生生成器的优化信号。完整Muse/Ditto可微视频路线需另建模型和裁剪反向路径，当前实现没有；本轮选择可直接调用现有原生推理的Fit-only黑箱路线。只从旧Dev选择theta会重复使用已暴露结果；本轮参数选择在已有Fit身份内完成，新Dev承担独立开发验证。

新路线增加实际生成、原音评分和归档成本，也会遇到不完整支持或工程失败。当前尚无新的三模型Fit分数、SPSA视频结果或方法晋级；记录/控制器代码和synthetic测试不能充当实际实验。运行状态读取仍被host审批规则拦截，不借此重放已有DirectD0启动或绕过审批。

## Evidence and affected work

- `.codex/scratch/fit40_cross_tfg_training_scope_audit_attempt03/cross_tfg_fit120_directd0_model_scope_audit_attempt03.json`，12853B，SHA `39e553e8fda4927294f16f1eadd250b919995230043f898b2b07c2c63839aaff`；source pins与旧attempt02实际12项已ROOT核对。
- `.codex/scratch/independent_candidate180_support_recalc_attempt01/independent_candidate180_support_recalc_attempt01.json`，实际配对分数、距离方向及支持身份来源；完整NPZ/媒体联合读回仍待。
- legacy实际trajectory `.codex/scratch/fit120_final_rescore_packet_attempt09_revision05/root_independent_terminal_readback_attempt01/`；DirectD0尚无本地终局，必须late-bind实际文件/pins后抽取。
- 拟新增Fit snapshot record/apply/plan与选择/SPSA core由GPT-6 Luna/max实现；当前无实际remote dispatch或Fit native scored bundle。
- [Repository](https://github.com/dogLeft233/tts-exp)，branch `codex/cross-tfg-audio-head-20261003`，HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`为本地未推送，不构造commit URL。

## Observations

- [decision] 下一轮以完整Fit40三模型的实际原音视频评分联合选择/优化同一theta，固定快照30/60/90/120与20轮SPSA，旧Dev不作参数选择。
- [rationale] 已验证当前Fit120/DirectD0训练信号只来自W2L，而Muse/Ditto出现原时钟和Cbest退化；把它们纳入真实目标可直接处理此覆盖缺口。
- [consequence] 新增honest Fit-only接口、族公共源时钟支持、完整失败分母和容量门；不修改旧terminal-only dispatcher、ADR021或已保存失败。
- [verification] ROOT已核对模型范围审查实际源码pins；三模型Fit/SPSA实际效果尚未运行，后续资格只接收实际终局证据。

## Relations

- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]
- relates_to [[021-future-dev20-natural-audio-admission-policy]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]