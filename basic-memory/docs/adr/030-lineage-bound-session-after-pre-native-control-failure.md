---
title: 030-lineage-bound-session-after-pre-native-control-failure
type: decision
permalink: tts-exp/docs/adr/030-lineage-bound-session-after-pre-native-control-failure
status: open
project: tts-exp
decided: '2026-10-09T21:32:34Z'
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
---

# 030-lineage-bound-session-after-pre-native-control-failure

## Choice and context

对已经保存POINT_DISPATCH_INTENT、却在任何原生模型/adapter构造或评分之前因工程schema检查失败的首对，保留旧会话/事件/失败协调器及所有产物。使用显式来源关联的新会话完成同一冻结20轮；复用同一initial720/family/selector初值、PCG64种子和科学参数，独立核对首对theta/batch/direction/a/c及后续轨迹。修复工程源码和必须的package/loader pins，数据、模型、原音时钟、评价门槛、planned分母及optimizer公式不变。当前为已选定的执行恢复路线，新的会话/package尚未真正创建或启动。

用户已授权实验重跑和持续自主推进。旧coordinator05三值解包错误在任何派发前失败；coordinator07恢复时无条件拒绝已存在pending ask，08只修该条件后实际launch一次。08首对child666452/t6608589退出2，parent666409/t6607624退出，旧会话iteration0/pendingpair仍在，没有tell；新event2仅POINT_DISPATCH_INTENT。child stderr为 execution package runtime pin differs from the frozen family runtime binding，dispatchtree无文件。

具体原因是 point_execution04 validate_point_scope 将family runtime_ref.get('bytes')与package bytes无条件比较。family Wav2Lip/Ditto refs只有path/raw_sha256，Muse有bytes；三个实际文件/path/SHA和package完全一致。实际来源 .codex/scratch/fit40_spsa_recovery_launch_attempt02/runtime_pin_comparison_precise_attempt01.json：W 2884B/a6638095c8ca05ffc6c48fe3d1017e9fd2f3d5da01b3279c01ca97b3e0a67d58，D22530B/275c8ac529653ab4d066320b65a25dd3c3fb181ca89f09f6928d092013acfc2b，M4438B/ee474826c0cea8b2f039f4b7dac986d7da98fa88fa1f0dcff85620c12a3f18de。源码 execute_point_block在validate_point_scope之后才构造adapter_factory，因此该失败可按pre-native control failure判定；不是模型效果阴性。

## Rationale and alternatives

旧耐久会话已对plus记录派发意图，现有at-most-once合同拒绝blindretry。删除event2、修改旧manifest、把UNKNOWN改成成功都会破坏来源；为所有未知状态放开重复派发更不适用。给旧journal增加新abort事件需要改变解析器和整个重放合同，当前没有必要。选择新会话并关联已证明零原生输出的旧会话，能够保持旧证据完整，恢复科学等价的确定性优化轨迹。

新point_execution05只兼容已经存在的family schema：path和rawSHA必须完全相等；如果family声明bytes则也必须相等；package中的完整正整数bytes及实际文件整SHA/size仍验证。Muse仍严格声明字节数。科学参数没有调整，不把缺字段解释为runtime漂移。新工程pins必须真正沿bootstrap/nativefactory/harness/package闭包加载，不能仅加一个未调用的修复文件。

## Consequences

新session须保存旧session ID/路径、events0/1/2整SHA、旧pendingpair、0tell/0模型输出、失败source及stderr/退出过程证据，并说明仅控制失败后的继续。新初始化seed/theta/optimizer/fitroster从同一资格输入产生；首ask的数值/有序样本/随机方向须独立比对。若实际发现已有视频/评分输出或运行进程，停止该零执行恢复路线并按原journal读回，不重复执行未知状态。

ADR027的耐久journal、闭合中心选择及每个session内at-most-once原则保留。旧session仍不可直接重试；新session不是把旧失败计算成一个完成pair。控制失败attempt不合并进科学score分母，全部实际模型调用及完整失败行仍保留。一旦开始新的真实模型执行，沿正常实际终态/全量读回推进20轮、四中心和final720，再决定下一代局部特征。

## Observations

- [decision] 在确认旧首对0原生输出/0tell后，以来源关联的新会话和相同确定性科学轨迹继续，保留旧POINT_DISPATCH_INTENT不修改。
- [rationale] 当前失败为family缺可选bytes字段与consumer错误强制检查冲突；实际runtime path/SHA/size没有漂移。
- [alternative] 不删除旧事件、不重写UNKNOWN、不放开所有未知派发；当前也不新增复杂abort事件解析器。
- [consequence] 新工程closure和实际首ask等价需完整核验；科学端点/门槛/分母和模型不变；若发现旧原生输出需另按journal读回。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[003-immutable-execution-repair-provenance]]
- relates_to [[027-durable-spsa-journal-and-final-center-selection]]