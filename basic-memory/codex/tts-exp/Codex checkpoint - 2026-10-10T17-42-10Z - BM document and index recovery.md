---
title: Codex checkpoint - 2026-10-10T17-42-10Z - BM document and index recovery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-10-t17-42-10-z-bm-document-and-index-recovery
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-10T17:42:10Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a12633-df41-7dc3-87cd-a4404475713c
trigger: compact
model: gpt-6-luna
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-10T17-42-10Z - BM document and index recovery

## Summary

Rebuilt the tts-exp full-text and vector indexes after a full-text scan surfaced a large set of previously unindexed project notes; the final project status showed 790 entities and 28,416 of 28,416 chunks embedded.

## Story

The task began by checking whether the project had gained a batch of Basic Memory documents and reconciling the knowledge base with the BM skills. The user then explicitly asked to rebuild the index. Comparing `docs/public-memory/` with the local Basic Memory source found 329 Markdown files in the mirror, 326 overlapping paths, 276 byte-identical files, 50 differing files, and three public-only files. The mirror was left untouched because those differences were not safe to resolve from the available evidence.

The graph cleanup removed one malformed empty checkpoint, corrected two misspelled relation targets, and added one canonical relation. A full-text reindex then reported 789 observed items, 465 indexed, zero deleted, and zero stale search rows purged. Project entities rose from 324 to 790. This proves that indexing coverage expanded; it does not establish that all 466 entities were newly created files.

Vector indexing first hit duplicate vector indexes with the default LiteLLM batch size. Reindexing with command-local provider batch size 1 and sync batch size 32 completed without embedding errors. After the full-text scan expanded the corpus, additional chunks appeared and required another incremental pass. The last reindex process exited with code 1 and no diagnostic, so completion was accepted only after `bm project info` confirmed 28,416 total chunks, 28,416 embeddings, zero orphaned chunks, and `reindex_recommended=false`.

## Working State

The project memory is `tts-exp`, stored under `basic-memory/` and captured in `codex/tts-exp/`. No persistent BM settings were changed; the batch overrides applied only to the reindex commands. The default automatic-sync batch settings remain unchanged and may still hit the duplicate-index behavior.

The repository is `dogLeft233/tts-exp`, branch `codex/cross-tfg-audio-head-20261003`, local HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`. There was no PR for the branch. At inspection the worktree had 999 dirty or untracked entries; `git status --short -- basic-memory` showed zero entries, and no repository source files were changed by the indexing work.

## Verification

- `bm reindex --search --project tts-exp`: 789 observed, 465 indexed, zero deleted, five batches, and zero stale search rows purged.
- `bm reindex --embeddings --project tts-exp`: run with `BASIC_MEMORY_SEMANTIC_EMBEDDING_BATCH_SIZE=1` and `BASIC_MEMORY_SEMANTIC_EMBEDDING_SYNC_BATCH_SIZE=32`; final `bm project info tts-exp --json --local` reported 790 entities, 28,416 chunks, 28,416 embeddings, zero orphaned chunks, and `reindex_recommended=false`.
- No tests were run; this was a knowledge-base and index-maintenance task.

## References

- [tts-exp repository](https://github.com/dogLeft233/tts-exp)
- Local mirror audit: `docs/public-memory/`

## Observations

- [result] The full-text rebuild increased indexed project entities from 324 to 790; these are newly indexed entities, not necessarily newly created files.
- [result] The public mirror audit found 329 Markdown files; 276 of 326 shared paths were byte-identical, 50 differed, and three existed only in the mirror. No mirror files were changed.
- [result] Graph cleanup removed a malformed empty checkpoint, corrected two misspelled relation targets, and added a canonical relation.
- [decision] Keep indexing scoped to `tts-exp`; preserve the public mirror differences for a later evidence-based reconciliation.
- [decision] Use command-local embedding batch overrides for the rebuild and leave persistent settings untouched.
- [verification] Final project status was 28,416 chunks, 28,416 embeddings, zero orphaned chunks, and `reindex_recommended=false`; the last command exit code was 1 without diagnostic text.
- [blocker] The configured embedding credential value appeared in a tool output during configuration inspection. Its value is intentionally omitted here; rotate or revoke that credential.
- [next_step] Rotate or revoke the exposed embedding credential, then verify provider connectivity before the next embedding operation.