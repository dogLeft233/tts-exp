---
title: Patrick同AVE逐录音RMS归一化三档粗训练与27格评测
type: experiment
permalink: tts-exp/experiments/patrick-同-ave-逐录音-rms-归一化三档粗训练与-27-格评测
status: concluded
tags:
- synctalk
- rms
- patrick
execution_status: NUMERICAL_FAILURE_PRESERVED
scientific_status: INCONCLUSIVE
conclusion: 冻结执行版本已数值终止；原coarse10/new27目标未完成，未产生新的评分或感知结论。
---





# Patrick同AVE逐录音RMS归一化三档粗训练与27格评测

## 问题和授权
2026-10-19用户明确要求加入音量归一化的音频编码器运行Patrick粗训练，沿用三档音量并在完成后评测；随后明确选择“同一AVE加RMS归一化”。这是独立新实验，不恢复已暂停的Patrick fine14或Chris/Cathy。原数据、失败、模型及coarse27结果全部保留。

## 冻结设计
同官方AVE encoder和原权重，先施加−6/0/+6dB的float64标量gain，再按每条完整main/test录音计算float64 RMS并乘固定0.1/RMS，转float32进入原AVE mel/时钟。固定目标RMS0.1（−20dBFS），不减均值、不加epsilon、不limiter/clip、不改采样率/时长/文本；拒绝零RMS和非有限输入。三档分别真实重算AVE并做完整重复前向，不直接重用0dB缓存，不强制特征或训练state相等。记录纯gain抵消后的残差，理想比例抵消不等于数值逐位一致。

PatrickLeahy原准入数据、seed42共同完整随机初始化（非旧coarse/fine warmstart），每gain coarse1–10各6000attempts/共60000，三档上限180000。FP16/defaultGradScaler、原loss/AdamW/EMA、65536rays/maxsteps16、随机采样和验证政策不变；逐行实际成功/AMPskip/finite loss、跨gain采样和LR核验，终末/失联封存停止，不重试/重放/补步/调scaler。实际更新不等保留CONTROL_FAILED，不把新训练与旧失败预算拼接。

完成三档coarse10后固定EMA60次和原六test，先9固定帧重复FP16/FFV1PCM像素资格，再9新视频×3评分gain=27格/162矩阵。driver在AVE前做同一RMS处理；评分端保持原0PCM16/±6float64原PCM单位、SyncNetFP32 TF32off、原时钟、source-owned ROI guard20主/full次；H264仅预览。独立矩阵/六clip等权/媒体像素与持久F/cloud全读回后交付并与旧coarse27并列，不能以Sync-C称感知改善。

## 当前执行和证据

本新版本已于0档coarse4/global18993触发真实数值终止：density_grid有52个+Inf、mean_density为Inf，guard在该步前向/反向/optimizer前拒绝；其余model state、optimizer及EMA在CPU终末检查中有限。三档coarse1–3共9完整epoch已封存独立全验，最新完整累计0/−6/+6成功17985/18000/18000，actual updates不等；coarse4的992完整trace为971新成功/21普通skip，末行global18992/累计18956，另1次global18993被拒绝不在trace。全trial54992已记录attempts/54956成功/36普通skip，另1拒绝attempt，不能冒称被拒绝步loss已知。

终末training archive32成员+job5成员fresh全SHA、rawfailure/trace和三最新coarse3完整cp全验；terminal_state.pth SHA21ef29fcb575ff9ad03c1d5af1381ef615fbf0d4b0f59b7cac02c8699b3f4f3f，训练archiveSHAb2e08eac6e513e0f624dfc11bd4b82015f94c1fb333a87666ed1e3e5df58fc0e。现场包含model/optimizer/scheduler/scaler/EMA/counters/RNG/gradients，仅作诊断，不是合格继续训练state。CPU terminal加载明确已执行，0模型构造/GPUforward/optimizer/重放；密度缓存刷新附近的数值现场不等于完整上游根因证明。入口六次成功资格不能证明完整稳定性。

权威F持久证据：F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_v1/terminal_review_v1/report.md、independent_review.json、terminal_cpu_inspection_source_exact.json、numerical_failure_source_exact.json及delivery_manifest.json。原合同SHA5cc4d22b57a32e5b6c3672d1e782dd40914b33850da5ae9b1e819b34ebb17859/14冻结源fresh精确。此前Windows反斜杠缓存退休故障与revision02审计修复另保留，不混淆本次数值失败。

本版本全部训练/评分/交付host已退出，−6/+6 coarse4未启动；coarse10和新27格未完成，0新RMS视频/评分，不写成归一化科学阴性、总体gain因果或感知结论。禁止本版本新job/换gain/重放/自动改scaler或精度，旧fine14/Chris/Cathy及原24full队列继续暂停，所有原断点和Brown/旧coarse CONTROL_FAILED保持。后续数值方向等待人类新指令。

存储可用性：用户已明确授权旧资源云端归档后删除，已有资源通过逐片和整项Ubuntu独立云端SHA读回，再经Windows独占handle复核后移除E盘原副本，空间已足够发布本故障记录。原路径旁cloud_restore_locator.json和F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/storage_reclamation_20261019/恢复locator保存，未来需显式按序拼接分片并核原SHA后使用；最新迁移范围以该目录receipt.json为准。原科学结果和失败现场未删除。

## 密度刷新复现与归一化排查

用户2026-10-19新授权仅复现密度刷新和排查归一化关联，暂不恢复训练。独立诊断已完成：原terminal RAW冻结权重、occupancy step18992/seed42、local frame449/source499及原上下文/7维眼部条件下，全量2097152点查询重复产生9个Inf，经dilation成为52个，位置与原终末mask逐位一致。首个非有限运算明确为official/nerf_triplane/network.py:386的torch.exp(h[...,0])；所有上游观测模块输出有限，sigma logit最大124.25，高于FP32 exp log上限88.722839。仅诊断FP32整段前向仍有9→52 Inf，不支持把本次简单归为FP16 exp范围问题或偶发硬件错误。

失败权重换回旧未归一化特征仍同9→52 Inf（logit最大126）；旧未归一化coarse10 RAW换用RMS特征，在同一坐标/眼部条件下仍有限（最大logit10.265625）。本RMS试验最近完整coarse3 RAW在两种特征下同样有限（最大9.546875）。眼部向量置零诊断使失败权重当前查询有限（最大64.6875），只说明条件交互参与，不是许可修改数据或训练修复。RMS前端本身未产生非有限值；当前特征替换排除了“必须RMS当前输入才会溢出”的简单解释，但不能排除RMS通过此前优化轨迹间接促成权重不稳定。旧coarse10与失败权重预算不同，旧coarse3 cp已合规轮转，未重放恢复；训练因果效应NOT_IDENTIFIED。

本次16次独立只读GPU密度刷新共33554432查询点，0Trainer、0backward、0optimizer，所有参数逐case hash未变。原前次CPU终末审计和本次GPU诊断范围分开。fresh density query/dilation得到上述结果，完整刷新前旧网格未捕获，不冒称完整缓存重建或自动修复；原污染缓存不因新query有限就恢复。14冻结训练源、contract和terminal SHA再次保持，实际job仍10、无训练进程，旧fine/Chris/Cathy暂停。4项CPU集成检查passed；前两个导入准备失败均模型构造前0forward/0optimizer，原source版本和错误另保留。没有新视频/评分，原coarse10/new27未完成，任务不done，自动巡检不重建。

证据根F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_v1/diagnostics/density_refresh_v1/：report.md、comparison.csv、independent_review.json、delivery_manifest.json、remote_results、post_diagnostic_readback.json。reportSHA171d2426175b95de517730151a5e6a0f51d3b350d3a613e21c72577b7524e436，reviewSHA81d6475ea4657f47ae3fdaf19123ae53810cf12f2d85b571c54503e1a157d967。原terminal_review_v1失败报告和原始执行归档保持，不覆盖科学结果。

## 强制恢复独立短试跑

用户进一步明确确认“强制恢复，先独立短试跑观察影响”，已实际执行独立patrick_rms_density_recovery_probe_v1，未恢复原失败版本或其他gain。新冻结合同SHA36234ebef65e02f6bc7071e21daad86ec6b1fa0c2e5666cd8f54b7a1c2fcecb9，计划最多256attempts。保持原terminal RAW权重/optimizer/scaler/scheduler/EMA并逐组件tree SHA exact；只以最近完整coarse3有限grid/bitfield/mean_density作为初始缓存，之后每次refresh事务式检查，坏缓存回退到前一有限缓存，参数/loss/梯度guard全部保留，不改精度、clip或scaler。首个采样frame/source/pixel/rays/lip/rect与原拒绝入口完全相同。3项CPU缓存事务测试passed。

实际仅完成116条完整trace（global18993–19108），116普通AMPskip、0成功更新/0optimizer入口，另global19109第117次进入后被拒绝；8次密度refresh全部非有限并回退8/8。记录loss仍有限（0.0002164436155–0.001150034834），但终末28个梯度tensor含NaN。scale从2^-12连续减半116次至2^-128≈2.938735877e-39，其倒数转FP32为Inf，guard在该拒绝步backward后、unscale/optimizer之前终止nonfinite_inverse_scale_before_unscale，不能把额外拒绝步loss说成有完整trace。全部模型权重逐tensor未变，仅density_grid/density_bitfield/step_counter缓存buffer变化；optimizer moments/step tensors未变，完整optimizer state hash差异仅为原scheduler在AMPskip后继续推进LR，不冒称发生参数更新。

开始前8固定验证帧RAW MSE均值0.00032046022352；失败后未跑validation_after，0有效更新不能作训练质量/SyncNet或感知对比。结论：仅丢弃坏密度刷新、保留有限缓存无法使该失败权重恢复有效FP16训练，并非偶发一次坏cache后恢复。新state只是新失败诊断现场，不是合格继续训练断点。该结果不证明RMS通过优化轨迹的因果，原16次零更新密度定位结果仍保留。

独立F证据F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_rms_density_recovery_probe_v1/：report.md、independent_review.json、116trace/8refresh/原failure JSON、baseline、firstskip与新terminal state，remote_manifest_source_exact.json全部14文件freshSHA核验。新terminalSHA99555eb8ee61790070b99027a0bdc8db92772f98e26f997500d51afdb151337e，原terminalSHA21ef29fcb575ff9ad03c1d5af1381ef615fbf0d4b0f59b7cac02c8699b3f4f3f不变；原trial job仍10、14源与contract精确。短试跑已退出，无训练GPU进程，不自动重启/补足剩余140预算，旧fine/Chris/Cathy继续暂停，coarse10/new27和原24full未完成。

## Observations
- [status] concluded
- [decision] 用户明确采用同AVE逐录音RMS，三档gain在归一化前施加；新版本独立训练，旧fine/Chris/Cathy保持暂停。
- [verification] 3CPU测试通过：纯gain抵消和波形比例保留、无clip/无去均值、静音/nonfinite/多通道拒绝；0新分数。
- [result] 本冻结执行版本以NUMERICAL_FAILURE终止，coarse10/new27未完成；这不是归一化科学阴性。
- [next_step] 密度刷新定位已完成；训练轨迹层面的RMS因果与后续修复需另立受控方案，当前不恢复训练、不重放当前版本。

## Relations
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- compares_with [[SyncTalk Patrick coarse10 seed42 27格描述性预览]]
- relates_to [[暂停 Chris 并精调 Patrick 三档及27格]]
