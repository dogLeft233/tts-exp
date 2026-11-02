---
title: ToolNote
type: schema
permalink: tts-exp/schemas/tool-note
entity: ToolNote
schema:
  tool: string, tool or topic name
  status?(enum): '[active, deprecated], tool status'
  gotchas?: string, traps and quirks
  install?: string, how to install/set up
settings:
  validation: warn
version: 1
---

# ToolNote

Schema for tool/usage notes — how a tool works, gotchas, workflow notes.
Live in ToolNotes/, titled by tool or topic name. Covers deployment
knowledge, environment quirks, and reusable procedures that aren't experiment
conclusions.

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |

## Observations
- [convention] Every tool note carries the tool name as an observation
- [convention] Deprecated tooling gets a status observation: [active, deprecated]