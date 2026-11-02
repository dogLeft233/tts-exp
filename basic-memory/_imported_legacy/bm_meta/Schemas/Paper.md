---
title: Paper
type: schema
permalink: tts-exp/schemas/paper
entity: Paper
schema:
  status(enum): '[to-read, reading, read, cited], reading state'
  authors?: string, author list
  url?: string, paper URL
  key_insights?: string, takeaways for this project
  topic?: ResearchTopic, related research topic
settings:
  validation: warn
version: 1
---

# Paper

Schema for paper/reading notes. Live in Papers/, titled by the paper's title.
Every paper links its research topic via `part_of [[Topic]]`.

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |

## Observations
- [convention] Every paper carries status as an observation
- [convention] Read papers get key_insights observations (one per line)