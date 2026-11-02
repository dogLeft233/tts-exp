---
title: Experiment
type: schema
permalink: tts-exp/schemas/experiment
entity: Experiment
schema:
  status(enum): '[planned, running, concluded, abandoned], current state'
  hypothesis?: string, what the experiment tests
  result?: string, headline result (e.g. ΔSync-C)
  conclusion?: string, what was learned
  report?: string, repo path of the full report (docs/experiments/...)
  inputs?: string, input data/audio used (paths or descriptions)
  outputs?: string, output artifacts produced (runs/... or results/... paths)
  model?: string, models used (HF ids, frozen checkpoints, revisions)
  code_paths?(array): string, scripts/tests locations
settings:
  validation: warn
version: 2
---

# Experiment

Schema for experiment notes — the numbered experiment reports of this project.
Live in Experiments/, titled `{NN} - {kebab-slug}` matching the repo's
docs/experiments numbering where applicable, or a descriptive title for
newer downstream evaluations. Status reflects where the experiment stands.

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |
| v2: added inputs/outputs/model/code_paths | August 14, 2026 | user |

## Observations
- [convention] Every experiment carries status as an observation
- [convention] Concluded experiments get a conclusion observation with the key result