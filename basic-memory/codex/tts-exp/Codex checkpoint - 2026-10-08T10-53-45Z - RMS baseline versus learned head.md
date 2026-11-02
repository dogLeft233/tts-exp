---
title: Codex checkpoint - 2026-10-08T10-53-45Z - RMS baseline versus learned head
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t10-53-45-z-rms-baseline-versus-learned-head
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c93e2a20655775230861f22f69abd34738b6d060
started: '2026-10-08 10:53:45Z'
username: LAPTOP-E133GSLE\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f7fc-0e98-7d91-9a49-5f1d3e77081d
trigger: compact
model: gpt-6-luna
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T10-53-45Z - RMS baseline versus learned head

## Summary

Fixed-RMS generator input has replicated SyncNet gains, while the learned natural-audio head has not passed its held-out cross-TFG Dev comparison.

## Story

The project is developing a usable cross-TFG natural-audio enhancement head while preserving the original natural speech in the final mux. The latest user question was whether the learned model is worse than simple normalization, and what that normalization does. I reviewed the current Basic Memory experiment records and the live autonomous task. The normalization applies one utterance-level gain to set the natural generator-input waveform to target RMS 0.0376838172675746; the generated video is then muxed with untouched original natural PCM on the original clock. This simple control raised Wav2Lip generator Sync-C by +0.192 (male 80 clips/40 speakers, 99% CI [+0.111,+0.276]); an independent female set also had a positive generator response (+0.250, 99% CI [+0.117,+0.417]). These are SyncNet endpoint results, not proof of human-perceived improvement.

The frozen learned STFT head showed a Fit-only differentiable Wav2Lip proxy increment over the same RMS control (Cbest +0.058; anchor +0.148; far-face +0.141; D0 -0.047). That does not establish whether it is better or worse on held-out data. Fit diagnostics show Cbest close to static-only and gate-shifted results, with much of the increase arising from background-distance growth; it should not be read alone as better lip alignment. The frozen candidate's cross-TFG Dev run remains active, with no held-out candidate-versus-RMS result yet. The latest direct thread snapshot says the ASR client misread an inner execution command as the outer launch command and is being corrected; a downloaded archive was locally concatenated but its member verification remains unresolved.

## Working State

Durable project records:
- RMS mel-coordinate report: basic-memory/Experiments/句级音量的生成响应经过哪些 mel 坐标.md
- Fit-only head gate/spectral analysis: basic-memory/Experiments/冻结增强头的门控时序与频谱贡献.md
- Head training and source-restoration report: basic-memory/Experiments/跨 TFG 自然音频增强头的训练与原音换轨验证.md
- Active task: basic-memory/codex/tts-exp/开发跨 TFG 自然音频增强头.md

Machine-local/fragile state: the autonomous remote Dev run and worker/observer processes remain active. The latest exact count recorded in the active task note is 221/360 control rows passed at 10:42 UTC; the live thread snapshot at 10:50 UTC still reports unresolved ASR invocation and archive-member checks. The local workspace is already dirty with many experiment scripts, scratch evidence, and untracked reproduction files. This turn changed no repository files. Current checkout is branch codex/cross-tfg-audio-head-20261003 at c93e2a20655775230861f22f69abd34738b6d060. No pull request is attached.

## Verification

Read the cited Basic Memory notes and checked the live task snapshot. Read-only git inspection confirmed branch, HEAD, and dirty working tree. No tests or experiment commands were run in this turn.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Current scientific pointers are the experiment and task notes listed above.

## Observations

- [result] Fixed-RMS normalization is a one-scalar gain on the full generator-input utterance; the final audio track is restored from untouched natural PCM.
- [result] Fixed RMS has independent positive SyncNet generator responses in male and female subsets; scope remains the SyncNet endpoint.
- [result] Learned-head evidence is still Fit-only proxy performance; held-out cross-TFG performance is unknown.
- [decision] Do not label the learned head worse or better than RMS until the frozen Dev candidate is compared with paired RMS controls using the official end-to-end endpoint; proxy Cbest gain alone is insufficient.
- [blocker] The autonomous Dev control/candidate run is incomplete; ASR launch validation and archive-member verification need resolution.
- [next_step] Continue the frozen Dev workflow through ASR and archive repairs, complete paired RMS-control and candidate scoring, and report the held-out cross-TFG comparison with failure counts and audio-restoration checks.

## Relations

- relates_to [[开发跨 TFG 的自然音频增强头]]