---
title: Task
type: schema
permalink: tts-exp/schemas/task-1
entity: Task
schema:
  close_date?: string, date completed (Month D, YYYY)
  due?: string, due date (Month D, YYYY)
  openspec?: string, openspec/changes/<change-name> path when the task executes an
    OpenSpec change
  priority?(enum): '[low, medium, high], urgency'
  project?: Experiment, parent experiment or project note
  status(enum): '[open, in-progress, blocked, done], current state'
settings:
  validation: warn
version: 2
---

# Task

Schema for task notes. Tasks live in Tasks/, titled with a short imperative
description. Status moves open → in-progress → done; blocked is a holding state.
Tasks link their parent experiment/project via the `project` relation field.

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |
| OpenSpec requirement format: Requirements/Scenario section, openspec field, single-status rule | August 27, 2026 | user |

## Observations
- [convention] Every task carries exactly one status observation; vocabulary is open/in-progress/blocked/done (never "active" or other invented values)
- [convention] Task body contains a Requirements section in OpenSpec spec format: each Requirement is one verifiable SHALL statement with at least one WHEN/THEN Scenario
- [convention] A task is done only when every Scenario's THEN has been verified; any unmet THEN becomes a [blocker] observation and status moves to blocked
- [convention] Tasks executing an OpenSpec change record an [openspec] observation with the openspec/changes/<name> path and keep BM status in sync with the change's tasks.md checkboxes
- [convention] Each task note has exactly one Observations section; updates go through edit or full rewrite, never a second appended Observations section
- [convention] Done tasks get a close-date observation before dropping off the Task Board