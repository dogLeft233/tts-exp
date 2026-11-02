---
title: Codex checkpoint - 2026-10-15T09-30-52Z - LibriSpeech exclusion and paper contribution
  recommendation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-15-t09-30-52-z-libri-speech-exclusion-and-paper-contribution-recommendation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-15T09:30:52+00:00'
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

# Codex checkpoint - 2026-10-15T09-30-52Z - LibriSpeech exclusion and paper contribution recommendation

## Summary
The predictive audit is complete, LibriSpeech is excluded from independent inference, and the recommended publication direction is a controlled study of driver-versus-evaluator effects with validated synchronization endpoints.

## Story
The original research asked why resynthesized speech raises talking-face synchronization scores and whether that direction transfers to a natural-audio enhancement head. The human paused the enhancement-head research, then commissioned an existing-data predictive audit. That audit is finished; no required scientific execution remains. After the human reported LibriSpeech leakage, ROOT amended report43, the evidence map, the completed Task and the experiment index to exclude every LibriSpeech positive and negative result from main conclusions, English mechanism/efficacy, independent confirmation, generalization and proposal support. The exact exposed upstream component/sample was not independently established and must not be invented. Raw calculations remain diagnostics.

The latest question is which contribution is worth publishing given the current field and which experiments are missing. ROOT reread the current canonical report, map, scoped codec intervention and 40-source crossover evidence, and delegated bounded read-only literature and evidence reviews to GPT-6 Luna/max. Primary sources show that THEval already reports encoding sensitivity and human-preference evaluation, AV-HuBERT expert work already proposes alternate sync metrics, and the 2026-10-06 PVSync preprint already separates timing and articulation. Generic claims that SyncNet is biased or that timing and articulation differ are insufficient novelty.

The recommendation is a source-conditioned controlled evaluation study: separate the driving audio, scoring audio, processing chain and support policy; retain their interactions; establish what changes a score versus actual synchronization; deliver a practical evaluation protocol whose nuisance robustness and true-error sensitivity are both measured. This is advice, not a user-approved experimental launch or a proved universal mechanism. Existing codec high-frequency rescue and scoring support effects are stronger core evidence than further speculative enhancement training. The study should include independent synchronization judgments and a small fresh-source confirmation instead of expanding the failed head search.

## Working State
Durable scientific record: `basic-memory/docs/experiments/43-processing-chain-predictive-audit.md`, permalink `tts-exp/docs/experiments/43-processing-chain-predictive-audit`, SHA256 `f32a52e2b25428293036e5a39bfbdc55fa932827491b8e84ab60bbdea9768c4b`. Delivery interpretation v2: `runs/processing_chain_predictive_audit_20261015_attempt01/delivery_manifest_final_v2.json`, SHA256 `79befec52b661615c9b9f133a58bfc6028af6448ec181e6c41a3bb62b9af54c4`. Amendment snapshots, eligibility and actual four-note readback live in `amendment_librispeech_20261015_attempt01/`. Original v1/raw statistics and frozen protocols are preserved. The Task `processing-chain-predictive-audit-20261015` is done and its goal complete. Do not restart it from the older checkpoint's unfinished extension steps.

Main source-PCM Wav2Lip/Qwen74pairs/13speakers: acoustic N/T AUC0.972 but seven-family grouped OOF gain R2−0.140, MAE0.850 versus mean0.743. Same74 legacy scorer R2+0.095, relative+0.175, block-null p0.010; wide group intervals, no stable single-family increment. Source ΔC+0.935 with D0 interval crossing zero; legacy ΔC+1.032 with D0 worsening+1.016. These are separate historical contracts, not a causal codec contrast or visual truth.

The codec intervention already isolates an evaluator-chain pathway: fixed video and same sound, bandwidth/high-frequency damage and restoration move C. Its reported85.7–96.1% recovery concerns that same-sound encoding loss only. The 40-person crossover has strong G×E interaction and support-dependent conclusions, but equal total duration does not equal local phone timing; raw TTS/natural crossed scores cannot identify an independent generator-quality effect. A time-aligned TTS derivative requires explicit processing controls and remains a derived arm.

Publication advice is maintained in the existing `basic-memory/Research/论文实验数据整理与写作主线.md`, permalink `tts-exp/research/论文实验数据整理与写作主线`. It now has a 2026-10-15 current-literature comparison, three proposed contributions and prioritized missing evidence. New-source pilot suggestion is at most8sources×1utterance×3TFGs×2audioarms=48generation units, not a power justification. Formal confirmation size must follow group variance and a meaningful effect threshold, with pilot/confirm separated. Independent human sync judgments, calibrated mouth/phone events and an alternate AV evaluator should accompany known offset/articulation positive controls and fixed-video audio-processing controls. Simple mouth opening/speed is descriptive, not accuracy. No new metric training, TTS/TFG/SyncNet forward, head/GPU launch, tests or downloads occurred for this advice.

Machine-local state: dirty branch with9tracked changes and1004untracked entries, mostly inherited scientific/development work. They are not attributed to this request. LAN wjj@192.168.50.2 remains the existing environment, but no live process claim was made or remote job inspected in this turn. Cross-TFG enhancement research stays paused. Do not recreate the missing tfg automation, restart old sessions/centers/Dev/Confirm, or treat this recommendation as launch authorization.

## Changed Files
The preceding completed amendment changed report43, the evidence map, completed audit Task and experiment index, plus versioned interpretation/eligibility artifacts under the audit run. This turn revised the existing paper-writing topic and added a current recommendation link to `basic-memory/Research/研究主线与证据地图.md`. No scientific outputs or code changed in this turn. This immutable checkpoint and local helper/receipt are handoff artifacts, not a new experiment report.

## Verification
Read exact project Startup Router, map, experiment instructions, bm-checkpoint/bm-writing and doc-system. Actual report43 and delivery-v2 hashes match the prior completed amendment. Basic Memory reindex observed1170notes and indexed the2revised living notes; their readback checked proposed status, paused head, current sources,48unit ceiling and map linkage. Prior original-array/statistical QA was completed and remains authoritative; no duplicate QA or implementation tests were run here.

Git read-only evidence confirms root, branch and SHA. GitHub REST confirms canonical repository and no exact-head PR; commit lookupHTTP422 means HEAD is local/unverified remotely. gh was absent, so public REST was used successfully. Same-chat search returned383coding_session notes, no more pages, zero codex_session or legacy results; titles/current direct-read identity select the 2026-10-15T02-36-18Z checkpoint as the latest valid predecessor. The newer04-41-45Z proposal checkpoint belongs to a different session_id and is not this chat's lineage.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- Local/unverified remotely HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`.
- [THEval v4, Appendix B](https://arxiv.org/html/2511.04520v4)
- [PVSync preprint2026-10-06](https://arxiv.org/html/2610.09223v1)
- [AV-HuBERT expert, CVPRW2024](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Yaman_Audio-Visual_Speech_Representation_Expert_for_Enhanced_Talking_Face_Video_Generation_CVPRW_2024_paper.html)
- [Who is a Better Talker, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Who_is_a_Better_Talker_Subjective_and_Objective_Quality_Assessment_ICCV_2025_paper.html)

## Observations
- [result] Predictive audit and LibriSpeech eligibility amendment are complete; diagnostics do not establish contamination-free English evidence.
- [decision] Preserve the human pause of cross-TFG enhancement research; publication recommendations do not resume experiments.
- [recommendation] Favor driver/evaluator/processing-support controlled evidence and a validated practical synchronization protocol over a universal enhancement-head contribution.
- [blocker] Independent human/articulation validation, fresh-source confirmation and measured protocol benefit are still missing for the proposed publication claim.
- [next_step] Agree the paper claim and bounded confirmation scope with the human, then freeze the small experiment design before any new execution.

## Relations
- continues [[Codex checkpoint - 2026-10-15T02-36-18Z - Primary predictive audit complete and extensions bounded]]
- relates_to [[43-processing-chain-predictive-audit]]
- relates_to [[processing-chain-predictive-audit-20261015]]
- relates_to [[研究主线与证据地图]]
- relates_to [[论文实验数据整理与写作主线]]
- relates_to [[SyncNet 编码偏差的码率频谱与表征机制拆解 2026-09-26]]
- relates_to [[独立40人同长交叉与评分边界探索 2026-09-29]]