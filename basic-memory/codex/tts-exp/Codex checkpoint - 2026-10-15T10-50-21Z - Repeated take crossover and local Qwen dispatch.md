---
title: Codex checkpoint - 2026-10-15T10-50-21Z - Repeated take crossover and local
  Qwen dispatch
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-15-t10-50-21-z-repeated-take-crossover-and-local-qwen-dispatch
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-15T10:50:21+00:00'
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

# Codex checkpoint - 2026-10-15T10-50-21Z - Repeated take crossover and local Qwen dispatch

## Summary

The user authorized a bounded independent natural/TTS repeated-take crossover, eight real natural pairs are prepared, and the existing local Qwen model is dispatched after the old cloud synthesis snapshot returned an actual 404.

## Story

The long-term project asks why TTS changes talking-face synchronization scores. The user has paused the cross-TFG enhancement head and now explicitly authorizes the proposed N1/N2/T1/T2 experiment. The prior natural/TTS four-cell interaction did not exclude ordinary mismatch between different realizations of the same text. This study adds independently recorded natural pairs and independently sampled TTS pairs, preserving each recording's actual clock and comparing matched, within-class and cross-class scores. It does not turn a residual contrast into a pure source-preference or generator-quality effect.

ROOT froze the small design before outputs: one existing Wav2Lip, first four speakers/16 videos, extend to eight/32 only on engineering completion and cost, sentence RMS fixed, direct source PCM scoring, one common fully valid query support for all sixteen cells and all31lags. GPT-6 Luna/max agents handle assets/code/experiments; ROOT owns interpretation, protocol, BM and one essential actual-array QA. The natural corpus supplies distinct same-speaker/same-text utterance IDs and PCM, but no formal retake/session labels and no certification of pretraining cleanliness. LibriSpeech remains excluded from inference and the completed report43 audit remains closed.

The previously supplied key was used only through no-echo private runtime input. Registration first failed400 due to name length, then succeeded200 after a bounded repair; synthesis of qwen3-tts-vc-2026-01-22 returned404 Model not exist. A single compatible-model GET returned200. No cloud TTS waveform was produced; this is a particular model availability failure, not proof of global retirement. Before any TTS/video/score, ROOT froze a provider amendment to the already installed local Qwen3-TTS-12Hz-0.6B-Base on the LAN V100. No installation/download or open-ended cloud debugging is authorized.

## Working State

Authoritative active task is `tts-exp/codex/tts-exp/repeated-take-crossover-20261015`, status in-progress; spec is `Experiments/独立自然发声与Qwen重复合成的交叉失配实验 Spec 2026-10-15.md`. Machine-readable execution root R=`runs/repeated_take_crossover_20261015_attempt01/`. Latest protocol `R/protocol_v2_local_qwen_addendum.json` SHA256 `2c7454722ae3acb839dd6a2b04546270f11aee75e058d974a6c5b533bd197152` preserves v1 and fixes T1 seed2026101501+100*numeric speaker ID, T2 seed2026101502+100*ID, full ICL independent reference+actual transcript, do_sample/temp0.9/top_k50/top_p1, offline FP16+SDPA and separate45min TTS ceiling. Do not label local results as the old cloud provider.

Actual asset manifest `R/assets_discovery/aishell_repeated_realizations/manifest_v3.json` SHA256 `ae2711bea33339de911e15cee0832488af0b411ffc224d81721c88a892bc12ae`; use v3, because v2 had the wrong short S0006 clone reference. S0006 now uses explicit two-reference concatenation plus0.12s silence and the actual combined transcript. Speakers S0003/S0006/S0007/S0008 pilot, S0014/S0015/S0016/S0020 extension. Local and LAN sources exist; sixteen target natural WAVs were actually prepared CPU-only at16k PCM16/RMS0.0376838172675746, no trim/pad/warp. Remote audio_manifest SHA256 `cd4f247152d1731f83ddad21f1d34407452270fd62cecda9bca271896a852a87`. Maximum lagged natural-pair correlations0.0839–0.1551, not near copies.

LAN root `/home/wjj/tts-audio/tts-exp`, SSH wjj@192.168.50.2:22. Source/model/runtime pins and files are machine-local, dirty/untracked, not pushed or complete backups. Keep host running; no shutdown/old training/center/Dev/Confirm/backup branch. No API key/password belongs in files/argv/BM/log.

Local TTS source `scripts/experiments/repeated_take_crossover_20261015/local_qwen_takes.py` SHA256 `5c7bc77e66d29eb23d26eb677dc05dbdf68b47681bbf75b1289354ee3d23965e` was ROOT-reviewed and staged matching. `/home/wjj/.venvs/qwen3/bin/python`, Python3.12.3/torch2.6.0+cu124/qwen-tts0.1.1/transformers4.57.3, model `/home/wjj/tts-audio/tts-exp/models/Qwen3-TTS-12Hz-0.6B-Base`. Main weights SHA256180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6; tokenizer836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258. ROOT dispatched AgentA `/root/paper_evidence_gap` to run S0003 T1/T2 then, on actual distinct valid output and acceptable runtime, remaining seven incrementally with identical protocol. Actual S0003 pilot PID2643278/startticks54621755 ran10:50:11.821–10:50:31.568UTC, OS rc1. T1 failed CUDA multinomial probability inf/nan/negative assertion in FP16 code predictor; T2 and remaining seven were not run, and no WAV exists. Frozen input plan SHA9f78754dd3256ad6ac018f5ad1b6befe7ffdbe9b2b512fcba24da26bdcd58cf9. Actual launch and failure logs are in R/local_qwen_attempt01_launch_s0003/. Agent reports V100 idle. Preserve01 and implement minimal FP32 repair in separate local_qwen_attempt02_fp32; ROOT source review and new provider numerical amendment precede any retry. Same seeds/text/reference/sampling, no logit clamp. Do not duplicate an unknown process. Receipts should be in `R/local_qwen_attempt01/{input_manifest.frozen.json,runtime.json,results.json,events.jsonl,provider_native_24k/}`.

AgentB `/root/crossover_interpretation_check` owns matrix runner. Current `run_matrix.py` SHA256 `9a35c45306f48271493d357796e2e23e63e7cbc6dbb543847b27e203352fa219`, including ROOT-requested nested-script sys.path fix; staged hash matches. ROOT reviewed source PCM identity, shared support, reduction and source bootstrap. Freeze final full8 config byte-identically before pilot render, binding v2 and actual TTS hashes, then reuse for extension. Wav2Lip Python `/home/wjj/.venvs/wav2lip/bin/python`; scoring Python `/home/wjj/.venvs/syncnet/bin/python` has python_speech_features. GPU rendering/scoring not dispatched yet. Fixed face image3 SHA f425463ba279fea88ac807d02bf048fd2f891d4efae1c8903eae3afc22d216c9; ROI[33,18,462,447]. No visual/audio embeddings or scores exist yet.

AgentC `/root/paper_landscape` owns timing_diagnostics.py SHA157f99305c36c397eeedcd76d03a93d6426ac68e59a22c9798d51af1f9f6743d. Await final canonical32 WAVs then one Mandarin MFA pass in existing mfa2.2.17 environment; natural alignment has not run. Unknown phones count against coverage. Add recognized-phone coverage of actual query support as a diagnostic, without changing scoring thresholds. Do not align natural-only now then redo it. MFA event estimates diagnose clock differences, not human ground truth.

Primary analysis uses equal source weights, ordinary2000-source bootstrap seed20261015. Mean-over-query distance curve precedes median/min: C=median(curve)-min(curve); retain B,Dmin,D0,offset/full arrays. Within-minus-cross is descriptive, not a universal mismatch correction. Raw-clock N/T timing remains a confound, even with equal RMS and same text. Any future warped/realigned endpoint needs a separately labeled processing/sham design.

## Changed Files

Dedicated cloud client/one-shot availability helper, local Qwen runner, matrix runner and timing helper under `scripts/experiments/repeated_take_crossover_20261015/`; bounded run protocols, assets and failed cloud receipts under R. Living map, existing paper-writing topic, experiment spec, active Task and short HANDOFF were revised. Source/BM/run artifacts are still machine-local and uncommitted. Repository had9tracked changes and1005untracked entries, mostly inherited; do not stage everything.

## Verification

Startup Router, map, experiment instructions, bm-writing/checkpoint and repository skills were loaded. Actual eight-speaker assets and source pins were checked; agent natural preparation verified all16 canonical hashes/RMS. ROOT performed necessary source review; no implementation tests/pytest/compileall. API registration200 and modelsGET200 show the key authenticated; actual synthesis404 is preserved. No cloud waveform or TFG output existed before local-provider amendment. Local Qwen help/source staging did not load CUDA. The actual FP16 pilot ran19.747s and failed numerical sampling before a waveform. FP32 source repair is delegated, not yet accepted or executed.

Read-only Git confirms root/branch/HEAD. GitHub REST confirms repository and zero exact-head PR; commit422 means HEAD is local/unverified remotely. Same-chat search returned384coding/codex session matches with no further pages and0legacy; newest title and direct-read exact agent/session/started verified the09-30-52Z predecessor. The04-41-45Z checkpoint is another chat and must not be adopted. CLI implementation explicitly passes overwrite=False and output_format=json to write_note when --overwrite is absent; only action created is acceptable.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Local/unverified remotely HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`.
- `R/cloud_tts_attempt01/client_audit.json`, `R/cloud_tts_attempt02/client_audit.json`, `R/cloud_model_availability.json` preserve bounded actual failures/authentication without secrets.

## Observations

- [result] Eight independent natural pairs are available and sixteen fixed-RMS canonical WAVs actually prepared; no new crossover scientific result yet.
- [decision] Execute only the authorized small repeated-take crossover; enhancement-head and closed predictive audits stay paused.
- [decision] After actual old-cloud synthesis404, freeze existing local Qwen provider before any output and avoid unbounded API troubleshooting.
- [blocker] FP16 local Qwen pilot failed a CUDA probability assertion before any audio; FP32 source repair/explicit amendment and valid waveform pairs are pending.
- [next_step] Review the minimal separate FP32 Qwen repair, freeze its numerical amendment and retry the same S0003 seeds once; after valid bounded generation freeze full quartet config and dispatch serial Wav2Lip/SyncNet plus one MFA pass.

## Relations

- continues [[Codex checkpoint - 2026-10-15T09-30-52Z - LibriSpeech exclusion and paper contribution recommendation]]
- relates_to [[repeated-take-crossover-20261015]]
- relates_to [[独立自然发声与Qwen重复合成的交叉失配实验 Spec 2026-10-15]]
- relates_to [[研究主线与证据地图]]
- relates_to [[论文实验数据整理与写作主线]]