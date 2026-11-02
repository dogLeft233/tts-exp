---
title: Codex checkpoint - 2026-10-16T10-15-49Z - Amplitude experiment launched and
  clock baseline clarified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t10-15-49-z-amplitude-experiment-launched-and-clock-baseline-clarified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c6c1e7bd005abeaa4a2cc52991391ef2aaec5223
started: 2026-10-16 10:15:49+00:00
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

# Codex checkpoint - 2026-10-16T10-15-49Z - Amplitude experiment launched and clock baseline clarified

## Summary

The frozen independent 20-speaker amplitude study has produced its first eight technically qualified videos and scores; old amplitude evidence is reconstructed, while report45 clock recovery is explicitly separated from enhancement over natural self scoring.

## Story

The project originally sought the cause of native TTS score advantages and a transferable natural-audio head. The user paused head research, excluded LibriSpeech from independent conclusions, and requested bounded mechanisms and cost control. Following the completed repeated-take and four-source local-clock studies, the proposal delegation prioritized six amplitude links: pure scalar waveform controls, independent RAW/FIXED replication, old encoder and mel mechanisms, complete lag-curve decomposition, old native TTS level gaps, and a new fixed dose curve. P2 pause repair stays an independent pending task; no new head training, language/model scan or MFA is authorized by this amplitude scope.

ROOT froze the design before new scores in [[幅度六环与新说话人增益曲线 SPEC 2026-10-16]] (raw SHA fb84fd8d0622c189130cb0a0867fc6f32957e30b3ae503ad2bd6fca2f3bc069b). The study uses 20 new project-independent speakers (10M/10F), one sentence each, RAW/FIXED twice plus four nonzero scalar doses. The five dose points are -6/-3/0/+3/+6 dB, with RAW_R1 reused at zero; 160 videos are planned. Original natural PCM evaluates every generated condition. Primary FIXED effects average repeats within speaker; the four dose contrasts use simultaneous 95% intervals. FIXED has an actual per-sentence gain and is not one relative-dose point. Support, crop, model, seeds and analysis stay fixed; continuation is based on technical gates, not signs.

The user's latest question challenges both baseline and responder coverage. Report45's +1.770 to +5.495 compares local-clock corrected evaluation audio with an endpoint-affine mismatched instance on the same fixed video. It is not gain over natural self-drive/self-evaluation. Five of24 pairs qualified (20.8%), from only two people; all five improving is a conditional result, with complete-source coverage0/4. These counts cannot establish broadly effective generation enhancement. The current amplitude study directly uses the requested natural-self baseline instead.

## Working State

The authoritative task is [[proposal-validation-p0-p1-p2-20261016]]. Its local Markdown is newer than the BM CLI indexed body: reconcile by explicit project note writes/sync rather than overwriting from stale read-note content. The evidence map and HANDOFF also need their current execution sections refreshed. Preserve proposal `basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md` (last read raw SHA d968dd7f6b9ed9371f43c8c24d6d9ed609721840c4e33b459729a3d77945fdb7), including its newer section3.1 and appendixE. Final update is targeted, not a replacement from an older snapshot.

LAN wjj@192.168.50.2:22 remains running, public-key SSH, repo `/home/wjj/tts-audio/tts-exp`. New work is volatile SHM `/dev/shm/p1_gain_dose_response_20261016_attempt02`, packet `input/runner_packet.json` raw SHA cc3b293f1d04c1fb343a88bf3fd3aace1283ab3a20ddeca2e4c074b604a1239e. Packet pilot source SHA416fb8842680c0eb01d96cbf774ca07f4dab470f6e50a5905b6e8f965f18779f; analyzer5ecd6c92…, seed wrapperca40d20f…. Renderer parent68511/startticks2719302 finished S0102 eight cells without failure. Encoder parent72044/2746497 finished120 actual encodes. Score parent72495/2750910 finished eight S0102 scores; shared primary101q20–120/full113q15–127, score JSON81b46de93122db4007a39985d15b3eef81d346ac01404f8191a3feda5628b3be. Agent is authorized to continue remaining152 without another admission loop. Read actual current processes/receipts before any launch; never duplicate a worker.

Attempt01 log-parent failure happened before any renderer child (0scientific forwards). Attempt02 initial encoder import failed before model load (0forwards): three exact existing local helper modules were recovered and bound in `run/provenance/encoder_dependency_recovery_attempt01.json`, SHAfd178358f1efa73e086a0686af82dd2e0a6cfd6db22a460fc829aa14f5b5d086. Only encoder retried; eight videos and original packet/audio remained unchanged. No new installation or model substitution.

Original Luna/max agents own implementation/execution: `/root/local_clock_execution` runs the GPU pipeline; `/root/local_clock_controls_review` owns independent mouth overlays; `/root/proposal_pause_recovery` completed old level evidence. The mouth first-speaker gate passed eight conditions,512x512/25fps/start0,145/145 common-valid frames and144 original-adjacent edges; absolute opening speed matches abs(delta)*25. Its local readback is `runs/p1_gain_dose_response_20261016_attempt02/run/mouth_postprocess/s0102_measurement_attempt01/`, manifest656401cd389bcfbac2788d4e22f218d4c777d30f021e05c6a20b8dea94cb2cb9. Mouth failure never removes usable SyncNet scores. Measurement code source762362c6… and MediaPipe task64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff are bound. Full160 mouth processing is additive after receipt-derived inputs arrive.

The original no-budget goal still has its original P0/P1/P2 objective and externally paused status; tools cannot resume/change it. Latest user-authorized compatible work continues, without claiming that incomplete P2 or the original goal is complete. Do not create a duplicate goal. Do not message the proposal chat without explicit human authorization.

## Changed Files

New untracked code lives under `scripts/experiments/p1_gain_dose_response_20261016/`, with corresponding four tests under `tests/experiments/`; old evidence reconstruction source is `scripts/experiments/reconstruct_amplitude_evidence_20261016.py`. ROOT evidence readers and immutable failures are under `.codex/scratch/proposal_amplitude_20261016/` and the derived run folders. No new commit or push this phase. Nine inherited tracked dirty files remain: CONTEXT, audio_enhancement README/init/cli, three portable_source_axis files, rebind_world source, and its portable validator test. Preserve unrelated dirty/untracked work.

## Verification

ROOT independently decoded all20 source WAVs and checked identity/text/RMS/scalar safety: receipt `.codex/scratch/proposal_amplitude_20261016/root_source_pcm_qa_attempt01.json`, SHA036e3fa0e3841c4e7f6782239e4521fc3a70a338b8a7f7fb8a25808285adfa87. Source manifest `F:/tts-exp-p1-gain-dose-response-20261016-source-attempt02/source_manifest.json` raw SHAf3cc98a1598a22f0bd84af77d3e0bf802c5ad62ef3c76c5461e091e46b1535f7 differs from its canonical internal fingerprint9e44024f…; do not conflate them. Existing publisher-verified AISHELL archive was reused, not redownloaded or rehashed15GB. Peak at+6dB remains below1.

Old40 actual reconstruction QA401a37a0aac5c30bf6097aec184331b1d7666606cba257dc87928dffc0af63be passed2560RMS/2560MEL/320motion rows with0copy/math residual. RAW→FIXED direction C+0.161, radius+0.020, joint+0.190; mel interior+0.171/boundary-0.009. `source_axis_consumer08` is a different manipulation, not the real RMS consumer07. Root viewed the final mechanism figure; original MEL full lag payload remains missing and explicitly labelled.

Old15 actual native level QA c30304c32b4cadc5579e2683a7f14b0d4d8493b5ef5d96255816e3b9bdc9aeba reconstructed all71paired clips/15speaker formulas exactly from original curves, with three guard exclusions retained. Native gap q00+0.438 becomes FIXEDq11+0.254; G-only gap change-0.101. LEVEL midpoint is separate. Controls360 existing ROOT numeric QA b3299d28… reused, not repeated; equal-model RMS C+0.152 mainly accompanies B+0.155, not comparable D0 gain. Old148 evaluation videos were intentionally not retained; do not claim recoverable video bodies. Legacy final figure `derived_attempt09/native_gap_control_attempt02.png` still needs one ROOT visual inspection.

Execution agent reports final17/17 focused source/runner/analyzer tests pass; reviewer reports8/8 mouth fixtures pass after status adapter and absolute-speed fixes. These are agent-run outcomes, not independent ROOT reruns. ROOT has not yet independently reconstructed the new120 saved scalar WAVs/eight raw score arrays; do that once while the authorized remaining render batch proceeds. No new scientific amplitude conclusion yet.

## References

- [Repository](https://github.com/dogLeft233/tts-exp), canonical URL verified by read-only GitHub API; exact branch PR endpoint returned[].
- Current local commit `c6c1e7bd005abeaa4a2cc52991391ef2aaec5223`; no remote commit link claimed.
- Old40 final evidence: `runs/recovered_20260930/independent_validation/amplitude_evidence_reconstruction_20261016_attempt03/`.
- Old15 final evidence: `.codex/scratch/native_level_gap_audit_attempt01/derived_attempt09/`.

## Observations

- [decision] Keep natural-self baseline, mismatched-instance clock recovery and native TTS paired score gap distinct; report all planned denominators and independent speakers.
- [decision] Continue the frozen160 amplitude cells after technical smoke success without sign-based selection, new qualification loops or automatic restoration of the paused head.
- [result] First8 renders/scores and120encoder inputs are technically complete; independent mouth first8 has100% common support. No final amplitude effect claimed.
- [blocker] BM CLI indexed body lagged newer local notes; refresh from authoritative current Markdown. Remote SHM remains volatile; local readback is not a cloud/full backup.
- [next_step] Read back and independently check new scalar WAVs/eight saved raw score arrays once while the original agent completes remaining152; then analyze160 with frozen statistics and update report/proposal/map.

## Relations

- continues [[Codex checkpoint - 2026-10-16T04-17-47Z - Bounded local clock result and coverage limit]]
- relates_to [[proposal-validation-p0-p1-p2-20261016]]
- relates_to [[幅度六环与新说话人增益曲线 SPEC 2026-10-16]]
- relates_to [[研究主线与证据地图]]
- relates_to [[45-local-clock-crossover]]