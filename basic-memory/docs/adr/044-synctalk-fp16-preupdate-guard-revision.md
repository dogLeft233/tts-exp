---
title: 044-synctalk-fp16-preupdate-guard-revision
type: decision
permalink: tts-exp/docs/adr/044-synctalk-fp16-preupdate-guard-revision
status: accepted
decided: '2026-10-18T02:33:22.041141+00:00'
project: tts-exp
---

# 044-synctalk-fp16-preupdate-guard-revision

## Decision
接受用户明确授权，保持FP16，以独立fp16_preupdate_revision_v1修复更新前数值保护、继续上游因果核查。原实验版本与所有失败不可覆盖；完成保护资格和统一稳定性资格后才允许新的完整重训，不能把诊断或重复更新合并进旧attempt。

## Context and rationale
旧官方依赖也复现非有限loss；原非有限loss断言位于更新后。两环境CUDA核在scale2^-128时可将有限scaled梯度反缩放成NaN/Inf而found_inf仍0，导致非法更新。loss有限检查不足以保护模型。独立更新前state/逆scale/unscaled梯度/实际optimizer入口检查可在不变更正常FP16数学与AMP跳步的前提下阻止该已知路径。密度exp上游溢出另行定位，不能用保护通过宣称训练稳定。

## Alternatives
- FP32训练：用户明确排除，本版本不采用。
- 删除断言、重启旧attempt、重放并合并：会掩盖失效和重复更新，拒绝。
- scaler夹限、耦合skip、梯度clip或改density公式：改变数值政策，未由本保护版本实施；后续若需要应独立冻结统一合同和资格。

## Consequences and affected work
新增fp16_preupdate_guard_v1.py、controlled_training_fp16_v1.py及对应CUDA资格。precheck拒绝后终止，不能就地恢复同一对象；postcheck不承诺回滚。正常AMP skip和真实optimizer调用计数仍保留，但调用数不等于数值有效证明。原spec EXECUTION_FAILED及数据passed保留。当前修复结果与证据边界见 [[SyncTalk FP16 更新前保护资格与密度溢出因果核查]]，正式epoch/supervisor未切换。

## Observations
- [decision] 用户已接受独立FP16保护修订和继续上游核查，原来的等待范围答复已失效。
- [constraint] 先资格，后新版本重训；原版本不改pass，原失败不删除。

## Relations
- affects [[核查 SyncTalk FP16 适配与官方环境]]
- affects [[SyncTalk 四新身份官方完整训练电平实验 Spec]]