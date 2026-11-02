---
title: 023-decoded-video-window-support-for-native-syncnet
type: decision
permalink: tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet
status: open
project: tts-exp
repository: dogLeft233/tts-exp
decided: '2026-10-09T04:31:36Z'
---

# 023-decoded-video-window-support-for-native-syncnet

## Context

ADR022要求原自然PCM与源时钟、三个TFG真实视频和官方SyncNet矩阵。新V100工程smoke02完成四条identity/RMS重复后，实际render均155 decoded frames，但官方run_pipeline.py按较长的6.312秒完整自然音轨、`-async 1 -r 25`转换为157帧AVI。frame map已如实标记155–156为nominal_clock_outside_render_decode。core06却将152个query的valid_query_mask全部设True。SyncNet每query使用五帧视频，query151使用151–155，因此包含了非生成视频帧。

这项问题由源代码、真实frame-map/PTS和完整原音长度定位，在任何新Fit候选效果比较前处理。ROOT独立核对35终局文件/657966B，打开四raw evaluator NPZ并核验矩阵byte SHA、dtype/shape/finite和同方法bitwise重复；读取source ZIP内完整PCM验证100992原采样与PCM16评分窗口。收据`.codex/scratch/root_verify_native_smoke_terminal_attempt03/readback_receipt.json`self6442064a821741d0da7252ec8232321b65a7b5ceee442e21a116916fbf80b299。七数组pack正在独立下载以直接核对query/mask，而不是只依赖元数据。

## Decision

三个TFG统一使用机械的真实视频窗口适格规则：query只有在官方SyncNet所消费的五个crop帧均可映射到实际decoded render帧、源帧索引合法且有一致有理PTS时才有效。原始官方距离矩阵、全部query、lag offsets和旧产物原样保留；新版本的valid_query_mask标明该规则。不能把因较长音轨产生的补帧当作增强方法生成的口型。

六方法同一clip仍按ADR022取FULL共同有效query key交集；全部40 Fit contexts/20说话人及720物理计划行保留，context最小支持要求不改。GUARD/固定anchor/距离方向与其余指标公式不改。该规则在三模型候选效果读取前冻结，不根据分数、方法名或提升大小选窗。所有模型采用相同五帧规则；不同实际有效长度会通过明确的共同支持处理。

对于此工程smoke，raw152行的query0–150五帧均为decoded render，query151触及补帧155，因此真实帧有效query为151。旧四行PASS的含义仍是工程链可运行与重复一致，旧mask不满足正式native准入，不能在720 family中借用。新W2L正式child同时记录原生mel/生成prewrite帧与PTS，正式身份不通过engineering alias跳过native-frame检查。

## Rationale

用户要提高用自然音频驱动的视频同步。评分窗口必须包含生成器真实生成的视频帧；仅按音轨长度或pipeline crop长度认定有效会混入后处理的复制帧。明确mask保留官方数值与原PCM，允许查清边界处理，并避免改变源时钟、丢弃整条语句或缩小计划分母。

## Alternatives

- 保留all-True mask：会把至少一个含补帧query作为真实口型证据，拒绝。
- 对成片截短原音以匹配视频：违反完整自然PCM契约，拒绝。
- 补齐生成视频并将复制帧称为native生成：混淆生成与后处理，拒绝。
- 修改官方run_pipeline/evaluator或按效果重新裁切：引入额外模型/评估变更，当前不采用；保留原评估raw数组并用可复核frame support更直接。
- 删短片/context或只保留高分窗口：改变完整分母并产生效果选择，拒绝。

## Consequences

core06与正式row producer需要新窄版本/overlay及对应source tests，旧版本与工程产物不覆盖。四七数组pack独立读回会报告旧mask问题，不能修改旧NPZ。W2L240只作为720 family一个实际slice，在其他模型完整前禁止选快照/优化初值。每行容量与runtime资格包括真实render/prewrite、crop、原完整音轨与query mask身份；媒体暂存SHM与有界归档，完整raw评分和failure证据持久保存。

## Observations

- [decision] 三TFG统一五帧全部来自实际decoded render且源PTS合法才将query标为有效，保留raw official矩阵与原完整PCM。
- [rationale] 音轨引起的尾帧复制属于评分预处理，不是生成器产生的口型；实际窗口支持避免将其混入增强效果。
- [consequence] 旧工程四行只支持可运行/重复，all-True mask不作正式准入；新native版本需明确frame-map/mask并保持40context/720计划分母。
- [verification] ROOT实际四raw evaluator NPZ/sourcePCM/35终局pins复核PASS；七数组pack和正式native资格仍待执行，未计算候选效果。

## Relations

- affects [[022-fit-only-cross-tfg-native-score-refinement]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]