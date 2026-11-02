---
title: 027-durable-spsa-journal-and-final-center-selection
type: decision
permalink: tts-exp/docs/adr/027-durable-spsa-journal-and-final-center-selection
status: open
decided: '2026-10-09T13:55:07Z'
project: tts-exp
---

# 027-durable-spsa-journal-and-final-center-selection

## 决策

跨TFG联合SPSA采用创建后不覆盖、可重放的ask/evaluation/tell记录。完整初始720真实评分及实际选择回执通过后才初始化；每次ask产生的正负提案和完整随机数状态先落盘，再应用音频和生成视频。恢复优先读回已有pending pair与真实终局，接续完成，不能重新ask同一轮或把未知child终局当失败重跑。

最终中心选择独立于初始选择：identity/rms真实240参考行加k5/10/15/20四中心480行构成另一完整720支持集。中心theta必须匹配已重放的真实tell记录，并产生冻结selector06的实际SnapshotSelection；portable应用theta与选择资格回执分别保存。

## 背景与理由

原目标是适配多个TFG、未见说话人超过固定RMS的自然音频增强头。优化包含20轮两侧提案的960评分和四中心480评分，运行跨多个模型、原生Python环境、容量归档和上下文压缩。只保存最后theta会丢失pending提案、随机数序列与哪批视频已经完成；重复ask或重新生成会改变实验轨迹。明确持久化边界可恢复同一实验，并让最终权重的资格追溯到实际完整输入。

ADR022/023/024/025/026的评分端点、完整计划分母、共同FULL、等说话人/模型权重、原PCM/时钟、容量与共享波形规则保持。科学准入阈值保持，Fit资格不等于Dev/Confirm资格。

## 实现契约

Initial selector receipt使用fit40_initial_family_selector_receipt_v1，状态INITIAL_720_SELECTOR_COMPLETE，记录真实serialized SnapshotSelection、canonical torch_cpu、完整consumer03 ingest index及初始family的raw/self pins。

Journal保存20个按序hashchain entry及可恢复状态：theta、iteration/history、PCG64完整整数状态、pending pair、halt、源head/constructor/inputs、roster/reference pins、actual plus/minus point IDs和参数、真实子进程rc、完整raw NPZ evaluation身份和tell返回状态。文件创建后不覆盖；读回先校验raw/self/previous、输入和逐轮确定性replay，再使用恢复状态。PowerShell不能把PCG64大整数转为浮点后再序列化，journal序列化使用Python。

旧240行block runner硬要求单model、240行和每六行同sample六methods，新SPSA每model8/40行使用fresh thin point runner调用冻结renderer/scorer原语。派生execution plan/rows是共享24/120 point plan的真实有序子序列，不增加虚构行；terminal envelope携带真实row IDs、process rc和全产物pins，由consumer核验。

Final-center consumer验证完整初始选择、20轮actual journal和四中心480行，绑定实际tell k5/10/15/20 theta，使用原identity/rms240参考和四中心产生actual SnapshotSelection(point_kind=spsa_center)。初值选择和final选择各有独立输入/回执；仅有seed选择或manual诊断application不能封为final候选。

## 备选与后果

仅保存theta不能确定恢复随机数与pending pair；在恢复时重新ask会改变真实实验；把初始FamilySelection或seed资格用于最终theta会混淆支持集。这三种方案均未采用。完整journal、点执行回执和最终中心选择增加小量元数据，但避免重复生成和错误资格继承，并允许之后封闭Dev/Confirm验证可用方法。

当前是工程路线决策：journal/point runner/final consumer正在实现，尚无实际科学初始化或20步终局；不将该文档当运行证明。

## 影响

冻结fit_only_spsa_selection_attempt06.py和consume_real_fit_stagea_evidence_attempt03.py保持；新journal/dispatcher/point runner/final-center consumer和portable03承担恢复与资格绑定。实施入口是HANDOFF.md、增强头Task与Experiment，实际source/receipt以对应run pins为准。

## Observations

- [decision] pending ask pair先于application/render持久化，完整PCG64/θ/history与actual evaluation/tell链可确定性恢复。
- [rationale] 多模型长实验会跨压缩和容量回收，真实pending与终局记录可避免重复优化或伪造状态。
- [alternative] 只保存theta、恢复重新ask、以初始seed资格代替final-center资格均未采用。
- [consequence] 新point runner使用真实8/40 permodel行，不padding240；最终另一720支持集产生实际SnapshotSelection。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[026-shared-waveform-and-model-projection-receipts]]