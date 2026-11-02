---
title: Codex checkpoint - 2026-10-14T13-11-23Z - Canonical factors and isolated angular
  SEP closed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t13-11-23-z-canonical-factors-and-isolated-angular-sep-closed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 13:11:23+00:00
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

# Codex checkpoint - 2026-10-14T13-11-23Z - Canonical factors and isolated angular SEP closed

## Summary

All available canonical operator endpoints and a four-row norm-preserving angular SEP probe are complete; neither screen produces cross-speaker enhancement, and only CPU stability candidates are being prepared.

## Story

The user wants fixed RMS, unchanged content and natural clock, isolated separation/stability/envelope manipulations in each TFG native input space, and mouth aperture/velocity alongside SyncNet. ROOT owns theory, protocol and BM; GPT-6 Luna/max agents implement and execute, with small gates before scale.

The canonical ownership repair completed Ditto/MuseTalk and then Wav2Lip. Coverage v4 records 22/24 cells (six no-op baselines, 16 factor endpoints); Wav2Lip ENV is unavailable under its frozen calibration gate. Five factor rows pass the response screen but no factor passes both speakers in a model. Earlier round-mask rows remain engineering-only; old Wav2Lip STAB is pre-calibration and SEP_RD is a closed separate variant.

Old SEP/STAB operators have native-norm collateral. A spherical coordinate decomposes x into norm and direction. The executed SEP_ANG_ACTIVE05 uses a common orthogonal rotation per token, selecting a dose solely from actual nearest-rival geometry, retaining norms and within-token Gram. All 68 operated CPU tokens increase actual margin; ROOT read the source and four float32 arrays and verified hash, source, finite, shape, unselected identity and norm errors. Its four GPU endpoints still fail replication: Wav2Lip S0044/S0052 Cbest deltas +0.006329/-0.170022, Ditto +0.006279/-0.080050. Both S0044 rows pass, both S0052 rows fail. MuseTalk expansion is closed.

The apparent common FIT separation direction depended on the estimand. On shared LOSO natural anchors and paired support, fixed-rival means are positive in all three models, whereas actual token-nearest-other margin is Wav2Lip -0.000249 (4/9 positive speakers), MuseTalk +0.003932 (6/9), Ditto +0.003140 (9/9). A tone-stripped base-phone analysis also leaves Wav2Lip negative. These label coordinates are not viseme truth. Existing 12-row motion localization is weak/variable and does not turn score changes into articulation accuracy.

The next small question is the user's stability isolation. Luna/max angular_radial_cpu_feasibility is preparing CPU-only STAB_ANG_FIT for Wav2Lip/MuseTalk at preset FIT-scale mean-angle reductions 0.004/0.008 rad, retaining native norms and unit-centroid actual margin. Ditto's observed stability direction is opposite. No renderer is authorized until ROOT reviews source/float32 invariants and freezes at most four rows.

## Working State

Authoritative local/remote scratch is fragile and not a Git backup:

- Coverage: .codex/scratch/native_axis_factor_pilot_coverage_manifest_20261014_attempt04.json, SHA a1a4b082c999648e4b84fe4a9c827e52a139ee777f793cc3e75297deb20b5ec3.
- Canonical Wav2Lip final manifest: .codex/scratch/native_axis_wav2lip_canonical_cpu_20261014_attempt01/local_readback_attempt01/native_axis_wav2lip_final_manifest.json, SHA e90224a6f6d9547c474722281489ec6091061cc18c9dae52675b79afa5029303.
- Active angular source/CPU receipt: .codex/scratch/native_axis_angular_radial_cpu_20261014_attempt01/sep_ang_active05/, source SHA 2b6bce34a26830ece5d4d304bfe8667e04483d2c6bb2ccf0a8360389a401bbf0, outputs/receipt.json SHA a860afb9ca8c0125f1f487ada40f1f36d4caad93d0310a6bda951f68f58db2a6.
- ROOT four-row summary: .../sep_ang_active05/root_four_row_summary.json, SHA 357b987e1ab83380aaf0a556d4b4ecba2be475b98f9d38e78bdcc0c448bb7235.
- Wav angular compact readback: .../sep_ang_active05/readback_wav2lip_attempt01/compact_readback_manifest.json, SHA d20309a6e8d1d8831fb045ae609eb22cb98beaace48f246ba59088d6cc25b36d.
- Ditto angular compact readback: .codex/scratch/native_axis_sep_ang_active05_ditto_readback_20261014_attempt01/, manifest SHA dd5858fbafd28c5fca47d1c7927d514abbc9417ee7cef09c5b92a8bbdddaee05.
- Actual-nearest FIT: .codex/scratch/native_axis_actual_nearest_loso_20261014_attempt02/receipt.json, SHA c889116baa257d6f7779838c7e625f2599596ef6d7bab58b613ef2a3566b34d0. Attempt01 count display is superseded.
- Base-phone FIT: .codex/scratch/native_axis_base_phone_loso_20261014_attempt01/receipt.json, SHA 0389cc386b2b1b582944eb4df488421a8160584e7859b906aa375a19423d5336.
- Motion localization: .codex/scratch/native_axis_motion_localization_20261014_attempt02/receipt.json, SHA ed96c4c05b6fe56a76b4fa78eaa1ecbe2308d1f61b8cc1c7249d14a7d3fff019; use summary_rho_corrected.md. Its replay source is explicitly reconstructed/unexecuted, not the original inline source.
- Expected pending STAB CPU directory: .codex/scratch/native_axis_stab_ang_fit_20261014_attempt01/. At snapshot time only source/coordinate preparation was reported, no completed receipt or GPU launch.
- wjj@192.168.50.2:22 remains running. Wav angular workers 2464248/46673334 and 2464994/46675744 exited rc0. Ditto renderer/scorer processes are gone; its exact OS rc and second renderer PID were not persisted and remain null. Current GPU snapshot at UTC12:58:17 showed 5MiB/util0/no compute apps. Never relaunch these completed groups.
- The remote Ditto dynamic runner snapshot SHA is cf84ac09b2a8bc3c9442867c47a12abeb6eb4c3d9fa75d4dd2f68bb9854e304e. The local old same-name runner fixes 104 frames and must not be reused.
- Correction to the preceding checkpoint: its transcribed Ditto score-manifest SHA was truncated. Actual unchanged manifest SHA is 83b784e574b5893e4e2697dc3065e18fe924d0f6eed41b60b92c5392fe3b30a5.

## Changed Files

Research map, native-axis experiment/Task, Task Board, frozen spec result pointers and HANDOFF were maintained in basic-memory/. Scratch contains new CPU sources/candidates, readbacks and summaries. No production source was changed by ROOT in this continuation. Git still has nine pre-existing tracked modifications and many untracked research sources/artifacts.

## Verification

ROOT inspected canonical and angular readbacks, actual Ditto injection source and source/PCM/clock/tracking checks. Four angular candidate arrays passed independent scientific readback QA, with maximum norm error about 2.03e-7 and exact unselected identity; the source records <=1.50e-9 within-token Gram error. Wav actual forward captures match candidate arrays. Complete Ditto scientific artifacts support the screen despite job-control metadata gaps; no repeat render was ordered for those gaps. No pytest or implementation tests ran. F:/Apps/MiniConda Python lacked NumPy; scientific array QA used the bundled runtime successfully.

Repository URL was verified through the GitHub public API. Current Git SHA is local/unpushed or otherwise unresolved by that API; no commit URL is claimed. GitHub CLI is unavailable, so PR state was not established.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Current local/unpushed SHA: 5688784dd5c6c1272074f0c4b6c5ff79f117c907.

## Observations

- [result] Available canonical arms and clean angular SEP both fail cross-speaker enhancement.
- [result] Actual nearest-rival geometry differs from the fixed-rival average direction; dropping tone labels does not create a common three-model positive margin.
- [decision] Close SEP_ANG_ACTIVE05/MuseTalk expansion under the frozen gate; no new training, four-center, Dev or Confirm.
- [decision] Preserve Ditto unknown OS rc/PID values as null; use complete scientific outputs without rerendering solely for missing process metadata.
- [decision] Keep aperture/velocity descriptive and separate from articulation accuracy.
- [blocker] Pure STAB CPU source/arrays/constraint receipt is still pending; no new GPU launch may precede ROOT review.
- [next_step] Read Luna's STAB_ANG_FIT CPU artifacts when ready, validate native norm and unit-centroid margin invariants, then freeze at most four causal rows if qualified.

## Relations

- continues [[Codex checkpoint - 2026-10-14T12-05-50Z - Canonical two-model factor screen and Wav2Lip boundary]]
- relates_to [[tts-exp/tasks/native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]