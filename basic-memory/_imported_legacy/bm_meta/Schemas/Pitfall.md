---
title: Pitfall
type: schema
permalink: tts-exp/schemas/pitfall
entity: Pitfall
schema:
  area: string, domain (e.g. syncnet, ditto, env)
  symptom: string, how you recognize this trap
  cause?: string, root cause if known
  fix?: string, the solution
settings:
  validation: warn
version: 1
---

# Pitfall

Schema for pitfall notes — mistakes, traps, and hard-won debugging lessons.
Live in Pitfalls/, titled with a short description of the trap. The goal is
searchable survival memory: next time the symptom appears, the fix is one
search away.

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |

## Observations
- [convention] Every pitfall carries area and symptom observations
- [convention] Fixed pitfalls carry a fix observation; unresolved ones stay symptom-only