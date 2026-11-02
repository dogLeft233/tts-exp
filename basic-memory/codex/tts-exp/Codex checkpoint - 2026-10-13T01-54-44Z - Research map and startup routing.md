---
title: Codex checkpoint - 2026-10-13T01-54-44Z - Research map and startup routing
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-13-t01-54-44-z-research-map-and-startup-routing
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-13 01:54:44+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a132d7-ec88-7b91-97b7-31d2625bf033
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-13T01-54-44Z - Research map and startup routing

## Summary
Created a durable research mainline and evidence map for the TTS-TFG project, and changed the project Startup Router so future agents enter through that map before searching formal experiments and later corrections.

## Story
The user wanted the historical TTS/TFG work made globally legible because isolated searches can miss shared factors, ablations, negative results, and later boundary corrections. The approach was a pointer-first synthesis rather than another progress log. The map separates raw/native TTS advantage, natural-audio replacement enhancement, and human/visual perception; indexes numbered experiments 01-38 plus later thematic routes; records current evidence status and superseded claims; and gives agents query families and follow-up paths. The canonical Startup Router now requires reading the map immediately after the Router, then loading experiment instructions, searching both `Experiments/` and `docs/experiments/`, and following corrections and evidence-type boundaries.

The map is a navigation and synthesis layer; formal reports, run artifacts, Git, and CONTEXT remain quantitative authority. The worktree already contained extensive unrelated dirty and untracked experiment files; this checkpoint attributes only the two documentation files changed for this request.

## Working State
The map records the integrated judgment: raw TTS/native gains are robust in several Chinese and selected cross-dataset/model settings but not universal; energy/scale, representation stability, local spectral/recipient compatibility, and front-end sensitivity are shared clues rather than one proven causal factor; RMS is the strongest simple baseline; the generic trainable enhancement head has no confirmed cross-TFG win; model-aware routing is research-only with a positive MuseTalk route and fail-closed RMS fallback; and several mechanisms remain observational, proxy-only, fit-only, or blocked by interface/dependency constraints.

The numbered index in `basic-memory/docs/experiments/README.md` remains the chronological report index. The map is organized by research question and evidence status, so future updates should revise it when a formal experiment, correction, withdrawal, or route stop changes the mainline.

## Changed Files
- `basic-memory/Research/research-map.md` - new research navigation and evidence map covering historical numbered experiments, thematic routes, evidence glossary, search protocol, open questions, and relations.
- `basic-memory/instructions/Startup Router.md` - canonical startup sequence now reads the map first and requires map-guided full-text search and correction tracking.

## Verification
- Read back both changed files after writing.
- Confirmed the map exists and is 17,952 bytes; confirmed the Router contains the map-first sequence and navigation-maintenance section.
- Searched the map and Router for the canonical title, `01-38`, evidence-status guidance, and maintenance rules.
- No automated tests were run because this change only adds and updates Basic Memory documentation.
- Repository evidence: branch `codex/cross-tfg-audio-head-20261003`, local HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`; worktree dirty with many pre-existing changes and untracked artifacts.

## References
- `CONTEXT.md` - project scope and pipeline entry point.
- `HANDOFF.md` - current engineering snapshot.
- `basic-memory/docs/experiments/README.md` - numbered formal-report index.
- `basic-memory/Research/RESEARCH-MAP.md` - new canonical research navigation layer.
- `basic-memory/instructions/Startup Router.md` - updated agent entry protocol.

## Observations
- [decision] Treat the research map as an index and current synthesis, never as a replacement for formal experiment reports or reproducible run evidence.
- [decision] Keep three estimands separate: raw/native TTS advantage, natural-audio replacement enhancement, and human/visual perception.
- [result] The map preserves positive, negative, inconclusive, proxy-only, fit-only, dependency-blocked, superseded, and research-only outcomes so future search does not collapse them into a single success narrative.
- [changed_file] Startup Router now makes map-first orientation a required session step and requires searching both experiment trees with correction follow-up.
- [next_step] On the next research question, start with the Router and map, then validate the relevant map entry against its canonical report and any listed correction before drawing a conclusion.

## Relations
- relates_to [[Startup Router]]