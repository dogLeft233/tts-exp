---
title: Task Template
type: template
permalink: tts-exp/templates/task-template
---

# {Task Title}

{一句话: 做什么、为什么、当前阻塞。}

## Requirements

<!-- OpenSpec spec 格式。每条 Requirement = 一个可验证的 SHALL 语句; 每个 Requirement 至少一个 Scenario。done 判定 = 所有 Scenario 的 THEN 验证通过。 -->

### Requirement: {SHALL 语句, 如 "系统 SHALL 在 X 条件下产出 Y 且 gate Z 通过"}
{可选: 补充上下文/边界}

#### Scenario: {主场景}
- **WHEN** {条件}
- **THEN** {可验证的预期}

#### Scenario: {异常/边界场景}
- **WHEN** {条件}
- **THEN** {预期}

## Observations
- [status] open
- [due] {Month D, YYYY}
- [priority] {low|medium|high}
- [project] {Experiment it belongs to}
- [openspec] {openspec/changes/<change-name> — 仅当任务执行一个 OpenSpec change}

## Relations
- project [[{Parent Experiment}]]