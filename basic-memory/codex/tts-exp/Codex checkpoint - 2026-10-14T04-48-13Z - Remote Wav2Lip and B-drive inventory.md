---
title: Codex checkpoint - 2026-10-14T04-48-13Z - Remote Wav2Lip and B-drive inventory
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t04-48-13-z-remote-wav2-lip-and-b-drive-inventory
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 04:48:13+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a132d7-ec88-7b91-97b7-31d2625bf033
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-14T04-48-13Z - Remote Wav2Lip and B-drive inventory

## Summary
Confirmed by read-only SSH inventory that the LAN V100 machine at `192.168.50.2` contains the original `tts-exp` checkout, many paired TFG outputs, and SyncNet score artifacts; the exact highest Wav2Lip sample and the B-drive archive contents still need a focused scan.

## Story
The user asked whether the older Wav2Lip high-improvement samples are still recoverable from `192.168.50.2`, and noted that some data is on the B drive. I first read the project Startup Router, research map, and server-access note, then inspected the remote checkout without changing it. The remote project `/home/wjj/tts-audio/tts-exp` contains multiple historical run trees, `natural_raw`/`tts_raw` pairs, score JSONs, and about 11,200 MP4 files, so the old evidence is very likely recoverable there. Project memory also records the old Wav2Lip headline (`Delta Sync-C` about +1.53 in an n=13 batch) and a later unified-PCM qualification (+0.935, 13/13 speakers positive), while warning that one old sample's unusually high score was partly an audio-chain/sample-rate/AAC artifact. The remaining work is to identify the exact Wav2Lip run and top paired sample, then inventory B-drive candidates and check their archive provenance before treating them as authoritative.

## Working State
Remote access uses `wjj@192.168.50.2` over SSH. The likely source checkout is `/home/wjj/tts-audio/tts-exp`; the remote Wav2Lip runtime is `/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python`. Read-only inventory found historical trees including `runs/20260707T020414Z`, `runs/two_stage_hubert_aishell1_20260810`, `runs/multiset_pipeline`, and model result directories, plus `syncnet.json` and `scores_250.json` artifacts. No remote files were modified or copied yet.

B-drive evidence is still unexamined in this turn. Existing project memory says `B:/tts-exp-project/archives` and `B:/experiment-archive` are candidate local mirrors, but a prior checksum inconsistency means B should be treated as an inventory source until independently reverified against a trusted C/F copy or manifest.

## Verification
- Read the project Startup Router, research map, and server-access note before remote inspection.
- SSH batch-mode connection to `wjj@192.168.50.2` succeeded with strict host-key checking and read-only commands.
- Confirmed remote project paths, paired `natural_raw`/`tts_raw` trees, score files, and approximately 11,200 MP4 files.
- Local Git status, branch, repository root, and HEAD were read. Worktree is heavily dirty with pre-existing changes and untracked experiment artifacts; this checkpoint does not attribute those changes to the inventory task.
- No automated tests were run; no files were changed by the inventory work.

## Observations
- [result] `192.168.50.2` is a viable recovery source for historical Wav2Lip/TFG media and SyncNet outputs.
- [decision] Preserve the distinction between the old headline Wav2Lip score and the later unified-PCM qualification; do not present an unqualified old high score as pure visual evidence.
- [blocker] The exact highest Wav2Lip paired sample and B-drive artifact integrity are not yet established.
- [next_step] Run a read-only score-pair scan on the remote checkout, then inventory B-drive archives and compare candidate manifests or hashes before selecting a recoverable sample.

## Relations
- continues [[Codex checkpoint - 2026-10-13T01-54-44Z - Research map and startup routing]]
- relates_to [[Startup Router]]