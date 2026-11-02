---
title: 037-qwen-realtime-donor-assets-after-http-model-unavailable
type: note
permalink: tts-exp/docs/adr/037-qwen-realtime-donor-assets-after-http-model-unavailable
status: open
project: tts-exp
decided: 2026-10-14
scope: Two experimental donor assets; no production default or renderer change
tags:
- qwen
- donor
---

# 037-qwen-realtime-donor-assets-after-http-model-unavailable

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

## Decision

Use the documented Qwen3 clone model `qwen3-tts-vc-realtime-2026-01-15` with the official realtime SDK for the two donor assets. Keep the existing natural targets and independent same-speaker references, and preserve raw generated PCM/WAV. This is a chosen engineering route, not a claim that the new API has executed successfully. User approval for Qwen TTS/ASR and data downloads already exists; no new permission gate is introduced.

## Context

The repository HTTP provider defaults to `qwen3-tts-vc-2026-01-22`. The first actual donor call registered S0044 successfully, then synthesis returned HTTP404 `InvalidParameter: Model not exist`. S0052 was never called and no TTS audio exists. Preserve this current-endpoint result rather than infer global model retirement or a scientific negative. The old output root is create-only and must not be rerun.

## Rationale and alternatives

The realtime Qwen3 snapshot supports cloned voices and keeps the intended model family. The official SDK avoids a custom WebSocket protocol. Current HTTP Qwen-Audio-TTS would additionally require a workspace endpoint and public reference-audio URLs, while changing model family; that is more engineering for two assets. Repeating the failed old HTTP model would not resolve a deterministic model-availability error. A full local TTS deployment is disproportionate at this stage.

## Consequences

- Install only the small pinned official SDK dependencies into `${USER_HOME}/.codex/scratch/qwen-realtime-runtime-20261014/deps`; keep global environments and repository code unchanged. Stop before compiling large packages or exceeding30MiB downloads.
- A new model requires two new voice registrations; do not reuse a voice bound to the unavailable model. New budget: at most2enrollment POSTs and2WebSocket synthesis sessions, no automatic retry. The prior failed1registration/1HTTP synthesis remain separately counted.
- SDK worker source/version/input hashes are frozen before execution. API key only enters a ROOT-scoped private no-echo prompt; no key in argv, files, BM, logs or agent messages. Agent execution handles are not writable from ROOT; ROOT owns credential plumbing only.
- Both references are independent same-speaker utterances: S0044 W0161 at6.971s and S0052 W0495 at6.312s. Do not label this selfclone. Raw outputs receive no resampling, RMS normalization or retiming in this phase.
- No frontend tensor or renderer is admitted here. Any later model-native donor mapping must retain original natural scoring PCM/clock, and cannot substitute for the remaining isolated-factor or physical spectral-envelope scope.

## Affected work and evidence

- `.codex/scratch/qwen_independent_donors_20261014_attempt01/input_manifest.json`, SHA `1668c859d03febd4296ae1671815a2ad7dfcccc9b5731298f38e76229e6d94f3`.
- `cloud_attempt01/BAC009S0044W0266/synthesis_response.body`: actual unavailable-model response; old worker terminal rc1, no secret echo.
- `scripts/tts/dashscope_vc.py` remains the historical HTTP protocol source and has not been edited by this decision.
- [Official realtime Python SDK](https://help.aliyun.com/en/model-studio/qwen-tts-realtime-python-sdk).
- [Official voice cloning HTTP API](https://help.aliyun.com/zh/model-studio/voice-clone-design-http-api).

## Observations

- [result] Actual default HTTP synthesis is unavailable under the current endpoint; voice enrollment success alone does not prove synthesis model availability.
- [decision] Spend only a small isolated SDK setup and two synthesis sessions to obtain donor assets; avoid full local-model deployment or custom WS implementation.
- [next_step] Review the pinned realtime SDK worker and source/input bindings before private runtime execution.

## Relations

- affects [[tts-exp/tasks/native-axis-factor-pilot-20261013]]
- relates_to [[tts-exp/experiments/native-axis-factor-pilot-20261013]]
- relates_to [[研究交接总览]]