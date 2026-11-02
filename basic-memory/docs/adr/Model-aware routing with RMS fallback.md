---
title: Model-aware routing with RMS fallback
type: decision
permalink: tts-exp/docs/adr/model-aware-routing-with-rms-fallback
status: open
project: tts-exp
decided: 2026-10-13 00:00:00+00:00
tags:
- decision
- explicit11
- routing
---

# Model-aware routing with RMS fallback

## Decision

Freeze a model-aware natural-audio enhancement interface: MuseTalk receives the explicit11 anti-symmetric local-power/spectral envelope candidate; Wav2Lip and Ditto use fixed RMS normalization; unknown TFG models default to RMS.

## Context

Two complete explicit11 initial360 runs on the development cohort showed a cross-model signal but no universal candidate win. An independent holdout of ten unseen speakers produced valid rows for all three TFGs after one bounded Ditto provenance repair. Wav2Lip candidate minus RMS was -0.008916 full and -0.005989 guard20; MuseTalk was +0.014064 and +0.025367; Ditto candidate and RMS were identical with zero delta in both summaries.

## Rationale

The route uses the candidate only where the independent holdout supports a positive effect and preserves RMS as a safe fallback elsewhere. Equal-model routed contribution is approximately +0.004688 full and +0.008456 guard20 on the ten-speaker holdout. This is a practical prototype with explicit failure-safe behavior and lower time cost than another broad parameter sweep.

## Alternatives considered

- One universal explicit11 candidate for every TFG: rejected because Wav2Lip is negative on the independent holdout and Ditto shows no gain.
- Reopen the old nine-parameter/four-center route: rejected because it is terminally NOT_ADMITTED and would repeat high-cost work.
- Continue recursive source audits before measuring: rejected under the user's time-budget instruction.

## Consequences

The model profile and fallback rule must be explicit in any future generation wrapper. Scores from the holdout support a usable prototype, not a universal scientific claim. A future independent confirmation set may test generalization; it should not silently change the routing rule after seeing scores.

## Affected work

- HANDOFF.md
- Basic Memory task: TFG natural audio enhancement head
- Holdout outputs under the remote holdout_speed_balanced_20261012_attempt01 tree

## Executable integration (2026-10-13)

The route is now implemented in `scripts/audio_enhancement/model_routing.py` and exposed by `scripts/audio_enhancement/cli.py` as `--model-id` plus `--routing-config`. Exact IDs are checked before case-folded aliases; the explicit11 provider verifies its three source pins and falls back to RMS on missing PyTorch, pin drift, malformed output, or non-finite/shape errors. The route receipt records the config hash, match kind, fallback reason, sample counts, and the policy to retain untouched natural PCM for scoring and final muxing. A bundled-Python CPU smoke passed for Wav2Lip, MuseTalk (expected RMS fallback locally because torch is unavailable), and an unknown model on one 56320-sample 16 kHz WAV. No implementation tests or GPU runs were added in this integration step.

## 成本平衡后的执行边界（2026-10-13）

已确认一份可复用的真实 Wav2Lip 工程 smoke（生成视频、原自然 PCM 无损换轨、官方 SyncNet，四行 rc0，manifest SHA `61399ad57c082727ba756b9ceff6300c6b3e45202f5854f4262664b796492d9b`）。它只证明工程链路，不作为 unseen holdout 效果门；由于没有 PID/startticks，不为补审计而追加 GPU。主 holdout 与 Ditto 修复 plan 的 SHA 已分别绑定到 `007495aa...cad3b` 与 `c71e47e7...4213a5`，不再共用路径。

[decision] 以三模型 holdout 的模型感知安全路由作为当前可用原型，并把一次性 renderer smoke 视为工程证据；后续优先写方法说明和用户可调用接口，只有新的科学问题才启动有预算的实验。