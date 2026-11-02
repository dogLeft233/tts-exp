---
title: ResearchTopic
type: schema
permalink: tts-exp/schemas/research-topic
entity: ResearchTopic
schema:
  status(enum): '[active, parked, concluded], current state'
  question: string, the research question being pursued
  decision?: string, current working decision for this thread
settings:
  validation: warn
version: 1
---

# ResearchTopic

Schema for research topic hubs — one note per open research thread, holding the
question, current understanding, and links to its sources and step reports.
Live in Research/. The hub's "current understanding" is a living synthesis,
rewritten (with a changelog row) as understanding evolves.

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |

## Observations
- [convention] Every research topic carries status and question observations
- [convention] Step reports and sources link back via `part_of [[Topic]]`