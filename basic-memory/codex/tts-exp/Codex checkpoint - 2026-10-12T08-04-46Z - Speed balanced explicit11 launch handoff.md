---
title: Codex checkpoint - 2026-10-12T08-04-46Z - Speed balanced explicit11 launch
  handoff
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t08-04-46-z-speed-balanced-explicit11-launch-handoff
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 08:04:46+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-12T08-04-46Z - Speed balanced explicit11 launch handoff

## Summary

旧 9 参数路线的最终四中心与 final720 已确认全部未准入；在用户要求减少审计、控制时间成本后，研究切换为一次性推进显式 11 参数谱形/局部功率路线，并准备 fresh initial360 实验。

## Story

本项目的目标仍是找到可跨 TFG 使用的自然音频增强头，并用原自然 PCM 与原时钟完成真实 Sync-C 比较。此前工作已经完成旧 9 参数的完整 20-pair、四中心与 final720，但所有四中心均未通过既定准入门，继续围绕旧封存产物做更多来源审计的边际收益很低。用户明确要求在可靠性与速度之间重新平衡，因此本轮只保留能影响科学结论的硬约束：固定三模型、40 样本、原自然评分输入、既定统计门和可恢复的实际运行来源；把停放的归档/关机支线与重复审计留在原处。

ROOT 已从已有严格 240 条实际 PASS ledger 记录派生出 240 行显式 11 参数参考 roster，并确认每行包含 sample/model/method 身份、真实 result_record、RMS renderer 输入与原始 score NPZ pin。远程 Luna 代理已固定可直接绑定的 live resource helper、bridge/runner/source map pins；另一个代理负责用这些真实输入装配新 packet 并在一次 live capacity check 后启动 fresh initial360。既有 CPU480 显式 11 参数输出只复用，不重跑。这样保留了复现所需的最小来源闭包，同时把主要时间花在获得新科学结果上。

## Working State

Durable:
- 旧 9 参数 finalizer：`.codex/scratch/root_finalizer06_actual_transport_and_gate_qa_attempt02.json`，状态 `PASS_COMPLETE_CPU_FINAL720_NOT_ADMITTED_ALL4_CENTERS_FAILED9_GATES`。
- 旧 9 参数选择回执：`.codex/scratch/fit40_final_selection_capture_attempt06/finalizer_capture_attempt06/selector_receipt.bin`，selection `NOT_ADMITTED`。
- 显式 11 参数参考 roster：`.codex/scratch/fit40_native11_reference_roster_derived_attempt01/reference_roster_attempt01.json`，240 行，SHA `0c5a97373b340f99496bd699f4610c6fd7ca69e0ae020c672072437c169d264b`。
- 远程 helper binding 回执：`.codex/scratch/fit40_native11_resource_helper_binding_attempt01.json`，SHA `75e1cb5ff9dc10ae70ce90ac131d9ba0e1e33de69b82f812bfba8e2e970070fd`；直接 helper pin 为 `406f95e4a262fb62bb02bc67249f3e6a7bf211b803a014622e042329d772fdf7`。
- 显式 11 参数 CPU480 已生成并通过既有独立 QA，不得重复运行；fresh initial360 是下一次 GPU 科学执行。
- 服务器保持运行；不要执行旧关机、fresh07 归档准入或 final04 支线。

Fragile/current:
- 执行应继续使用原 QUEUE `/dev/shm/fit40_queue_remote_package_attempt02/project`、package12 实际 package11 runner08、既有 source pins；旧 staging map 不是新 packet 的最终来源。
- 当前活动代理：`/root/native11_roster_bind` 做最小 roster QA；`/root/native11_packet_launch` 装配 packet 并准备一次 initial360 launch。它们都不应展开成递归审计。

## Changed Files

本轮只改写了仓库 `HANDOFF.md` 的当前执行游标，使其明确记录旧 9 参数已终局未准入、显式 11 参数为下一阶段、以及用户的速度约束；大量临时实验文件仍是工作树历史/未跟踪状态，不应误当作新的正式代码变更。

## Verification

- 只读核对 git branch、root、HEAD、当前用户/主机和 BM 配置。
- 只读确认旧 finalizer、selector、roster、helper binding 的实际路径与摘要；没有重复训练、没有启动新 GPU 进程，也没有运行实现测试。
- 旧 9 参数四中心结果与未准入状态来自实际 finalizer/selector 产物，不是根据旧代理状态推断。

## Observations

- [result] 旧 9 参数路线已完成且四中心全部未准入；`selected_method_id` 为空。
- [decision] 以一次 fresh initial360 显式 11 参数实验替代继续堆叠来源审计；只保留影响科学效度的最小绑定和一次 live capacity check。
- [decision] 保留原自然 PCM、原时钟、三模型×40 样本、既定统计门及 Confirm 封存边界；不复用旧 360 或提前读取 Confirm。
- [blocker] 新 packet 尚未由代理装配/发布，GPU initial360 尚未启动；若真实来源缺失，只补建必要 derived binding，不回到递归审计。
- [next_step] 等 `/root/native11_packet_launch` 完成最小 packet 绑定并确认无旧进程后，启动一次 fresh initial360 并读取实际终局。

## Relations

- continues [[Codex checkpoint - 2026-10-12T07-42-59Z - Corrected finalizer boundary and speed balanced explicit11 continuation]]