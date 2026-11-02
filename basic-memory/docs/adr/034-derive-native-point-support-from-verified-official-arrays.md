---
title: 034-derive-native-point-support-from-verified-official-arrays
type: decision
permalink: tts-exp/docs/adr/034-derive-native-point-support-from-verified-official-arrays
status: open
project: tts-exp
decided: '2026-10-10T06:03:36Z'
tags:
- audio-enhancement
- native-support
- immutable-provenance
---

# 034-derive-native-point-support-from-verified-official-arrays

## 上下文

ADR033的完整pair恢复driver02真实执行一次后，在fresh dispatch intent前rc2：passing Stage-A row lacks its producer full-support summary。旧session07仍六事件，fresh08仍只有init/ask两个事件，两个replay receipt目标均未创建；日志在.codex/scratch/fit40_completed_pair_replay_attempt02_local_capture/，旧产物与失败事实保留。

ROOT/Dev/ASR/Signal从真实48行ledger及封包源码核对：Wav2Lip16行有support_by_support、metrics_by_support、lag_curve_by_support，MuseTalk16与Ditto16均缺少这三个字段。其官方七数组/原PCM/时钟/frame-map已ROOT独立全资格，并非评分缺失。W2L adapter调用冻结core._finish_supports，Muse/Ditto逐行_render_and_score后直接公开序列化而未调用它。selector06官方适配器沿initial-ingest03进入final-ingest05时严格要求producer full query_count和anchor字段，因此在正式tell前拒绝。

初始720的多方法producer summary与SPSA单方法point的支持集意义有区别。冻结core按每context的方法集合取共同valid query-key集合，排序后full；guard20在full至少41项时取full_keys[20:-20]，trim后的集合至少41项才eligible。它按排序集合序数裁剪，并不是时间窗口或source frame减20。单方法native point的full等于该行valid keys；真正的SPSA评价仍在selector06中重新取identity/RMS/plus/minus四方法共同原时钟支持集。

## 决策

在新版本中增加显式native-point转换入口，仅在完整计划、raw NPZ全文件/七数组哈希、官方source frame-map、原PCM与scope所有既有门通过后，从真实数组及已绑定时钟派生full/guard20支持和anchor15。返回独立的readback derivation receipt，明确是consumer从原证据推导；不伪称旧producer已输出汇总，不改旧ledger/manifest/point/session/receipt/source pins，不使用执行器scalar替代原矩阵。

初始六方法family继续使用现有严格producer summary验证；已有producer summary仍严格检查，半缺/畸形不能作为全缺的native兼容情况放过。新selector/final-ingest/durable session与所有调用闭包必须用新路径和真实pin，冻结SPSAController数学与初值选择不变。为减少每批兼容分歧，未来Muse/Ditto native producer在内部数组尚未删除时按同一冻结core规则补完整汇总；这是下一版真实producer输出，旧48条不修改。

恢复仍遵循ADR033：新的真实initial receipt/session必须对同720诚实重放，逐项theta/RNG/ask等价，完整旧48只读导入正常API；旧失败08也保留，未发生tell不得写成NO_UPDATE。任何新预检失败先检查持久事件/receipt状态，再做版本化修复，不能盲重试或放宽科学门槛。

## 理由与替代方案

官方数组与时钟足以精确重建缺失的确定性元数据。重复模型评分不能改善消费契约，编辑旧结果会损坏事实链；给所有初始行无条件取消summary验证则扩大兼容范围。独立native入口将兼容范围限定到已完成官方资格的单方法point，同时让四方法真实共同支持仍由冻结控制器决定。

## 当前验证与影响

ROOT已独立只读复算首pair的96实际行（48固定自然/RMS控制+48候选）、672数组、24contexts与2502四方法共同keys，Torch CPU float32/单线程。supportSHA0c1c0d10ac938cbfccb6b11e9fbf65cd20c4617acf634d47f4e186f380fe338a。Lplus=.11204426507887727，Lminus=.15492162381717417；按冻结公式预期更新幅度.016272267769119907，尚未写任何session/theta。QC .codex/scratch/root_first_pair_objective_actual_qc_attempt01/root_actual_qc.json，selfe879d08136590038b3c2d4726c3a5cf3d7283d4aaa4ab40497e76fc92c751090；48控制实际132files/17206942B读回在.codex/scratch/root_first_pair_control_actual_readback_attempt01/，05:58:44Z fresh仍2events。这是独立数学/数据核验与后续tell对照，不是候选准入或未见说话人收益。

科学路线initial720/20jointpairs/四full120中心/final720、2160唯一上下文、±3dB/同RMS/原音轨时钟、FULL/GUARD20、31lags/anchor15和封闭Dev/Confirm不变。新支持转换源码正在实现；目前0tell，没有测试、新renderer或评分调用。

## Observations

- [decision] 增加只接收完整已验证官方native point的支持派生入口与独立readback receipt，初始family仍严格要求producer汇总，旧产物不修改。
- [rationale] 缺项是Muse/Ditto序列化路径遗漏确定性汇总，七数组与官方frame-map足以重建；四方法共同支持和冻结Torch归约照常执行。
- [alternative] 不采用改旧ledger补伪producer字段、全局跳过summary校验、重新渲染相同48结果或执行器scalar分数。
- [consequence] 新来源闭包和fresh初始receipt/session诚实绑定；未来native producer补实际汇总，旧48以独立readback派生凭证导入。

## Relations

- refines [[033-complete-pair-consumer-replay-after-parent-link-repair]]
- relates_to [[003-immutable-execution-repair-provenance]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]