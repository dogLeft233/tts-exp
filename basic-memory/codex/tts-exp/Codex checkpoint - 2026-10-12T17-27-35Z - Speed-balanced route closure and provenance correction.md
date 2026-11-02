---
title: Codex checkpoint - 2026-10-12T17-27-35Z - Speed-balanced route closure and
  provenance correction
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t17-27-35-z-speed-balanced-route-closure-and-provenance-correction
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 17:27:35+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-12T17-27-35Z - Speed-balanced route closure and provenance correction

## Summary

本轮按用户“平衡可靠性与推进速度”的要求收敛审计：模型感知路由已可执行，主 holdout 与 Ditto 修复 plan 的 provenance 已纠正，并复用了一份真实的生成→无损换轨→SyncNet 工程 smoke，不追加 GPU。

## Story

目标仍是交付一个可跨 TFG 使用、失败时安全回退的自然音频增强头。当前最可靠的证据是同一 10 个未见说话人的三模型 holdout：MuseTalk 候选保留，Wav2Lip 与 Ditto 回退固定 RMS，三模型等权相对 RMS 约 full +0.004688、guard20 +0.008456。为让接口可用，工作树加入 `model_routing.py` 与 CLI 路由参数；provider/source pin、样本数、采样率和原始 PCM 评分策略都写入 receipt，候选不可用时 fail closed 到 `rms_fixed`。用户明确要求降低时间成本后，没有重开旧 9 参数/四中心或递归审计。

远端已有工程 smoke 覆盖 Wav2Lip 实际生成、原自然 PCM 无损换轨和官方 SyncNet，manifest 标记 engineering-only；它证明工程链路可复用，但不替代 unseen holdout 效果门，也没有 PID/startticks。主 holdout plan SHA `007495aa3b4cf1fc9b20f506c33593da8bb95e54ea1e0914ac56f3d9a21cad3b` 与 Ditto 修复 plan SHA `c71e47e7acc2eb2ec92062b6fc0d4c882c96309af2ff770ef40db7a5144213a5` 现已分别绑定到不同路径，避免把 Ditto plan 冒充主 plan。

## Working State

- Durable: 模型感知路由配置 SHA `886f75c03977ab7d8ac7a5caecd568421c9a46cbf455d6b9cef87a9bf24fb00`；本地 JSON-only provenance bundle 在 `.codex/scratch/holdout_repro_bundle_20261013/`。
- Durable: 旧 9 参数、旧四中心、旧关机/归档支线保持停放；当前安全路由是交付原型，不宣称单一候选对所有 TFG 胜出 RMS。
- Fragile: 远端 GPU 保持运行但本轮未启动新训练；大量既有 dirty/untracked scratch/科学产物仍在工作树中。

## Changed Files

- `scripts/audio_enhancement/model_routing.py`
- `scripts/audio_enhancement/cli.py`
- `scripts/audio_enhancement/__init__.py`
- `scripts/audio_enhancement/README.md`
- `scripts/audio_enhancement/configs/model_aware_routing_v1.json`
- `scripts/audio_enhancement/configs/model_aware_routing_v1.md`
- `HANDOFF.md`
- `CONTEXT.md`
- Basic Memory task and routing ADR notes

## Verification

- Bundled-Python CPU smoke on one 16 kHz/56320-sample WAV: Wav2Lip alias, MuseTalk safe fallback, and unknown-model RMS all produced finite outputs with preserved length/rate.
- Route JSON static parse and source/shape pin checks passed; route contract repair separated requested method from actually applied method and made RMS fallback explicit.
- Parsed both local holdout plans and verified their file hashes, run IDs, and distinct 60-row/20-row scopes.
- No implementation tests or new GPU runs were performed. Existing remote engineering smoke is referenced as an already completed artifact, not rerun here.

## Observations

- [result] A low-risk model-aware interface is now usable even when the candidate runtime is absent.
- [decision] Preserve the real renderer smoke as engineering evidence and stop spending GPU time to add PID/ticks to it.
- [blocker] The new route files were not synchronized to the remote renderer tree; strict route-to-renderer invocation remains an optional future one-line integration step.
- [next_step] Finish the minimal method usage/reporting surface around the frozen route, then wait for a concrete scientific question before scheduling any new GPU execution.