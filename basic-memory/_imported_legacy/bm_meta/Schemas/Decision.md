---
title: Decision
type: schema
permalink: tts-exp/schemas/decision-1
entity: Decision
schema:
  date: string, decision date (Month D, YYYY)
  status(enum): '[proposed, accepted, rejected, superseded], current state'
  rationale?: string, why this decision
  alternatives?: string, options considered and rejected
settings:
  validation: warn
version: 1
---

# Decision

Schema for decision notes (ADRs). Live in Decisions/, titled
`ADR-{NN} - {slug}` aligning with the repo's docs/adr numbering where the
decision is also recorded there, or a descriptive title for lighter decisions.
Status reflects whether the decision stands.

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |

## Observations
- [convention] Every decision carries date and status observations
- [convention] Superseded decisions link the replacing decision via `replaces`/`replaced_by`