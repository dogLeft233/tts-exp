---
title: Basic Memory 技能与连接验证
type: note
permalink: tts-exp/tool-notes/basic-memory-技能与连接验证
status: active
tags:
- tool-note
- basic-memory
- memory
---

# Basic Memory 技能与连接验证

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |
| Restructured to ToolNotes/ with schema observations | August 12, 2026 | user |

## Context

本线程确认了当前环境中的 Basic Memory 是否可用,并梳理了已安装的 Basic Memory 技能及其适用场景。

## 结论

Basic Memory 在本项目中工作正常。`tts-exp` 项目可以被正确发现,并已完成一次写入、读取和删除测试,测试记录已清理,没有留下临时笔记。

当前已安装的技能覆盖以下用途:

- `/memory-capture`:把当前工作线程的决定、洞察和背景整理成一条 coherent note,并在同一线程再次捕获时覆盖更新。
- `/memory-continue`:从 Basic Memory 恢复之前的工作上下文。
- `/memory-reflect`:回顾近期对话和记录,提炼长期有价值的记忆。
- `/memory-notes`:创建和改进结构化 Markdown 笔记。
- `/memory-curate`:整理知识图谱、发现孤立笔记、合并重复内容并建立关系。
- `/memory-defrag`:重组过大的、重复的或过时的记忆文件。
- `/memory-lifecycle`:管理实体状态、归档已完成内容并移动笔记。
- `/memory-schema`:推断、创建、校验和维护 Picoschema。
- `/memory-metadata-search`:按 frontmatter 字段、标签、状态和日期搜索。
- `/memory-tasks`:创建和跟踪可跨会话恢复的结构化任务。
- `/memory-ingest`:把会议记录、聊天记录或文档摘录提取为实体、观察、关系和行动项。
- `/memory-research`:进行外部调研并把结果沉淀为结构化实体。
- `/memory-literary-analysis`:对完整文学作品建立章节、人物、主题和事件知识图谱。
- `/memory-onboarding`:从零设计 Basic Memory 的目录、schema、标签和使用方式。
- `/memory-ci-capture`:在 CI 中收集 GitHub 交付上下文并生成项目更新。

## 使用约定

本项目同时存在 `main` 和 `tts-exp` 两个 Basic Memory 项目。与当前代码库相关的记忆应明确写入 `tts-exp`,避免误写默认的 `main` 项目;使用项目 ID `ccca11cd-ff3f-433a-8a0f-48ef6c5ab2b1` 可以稳定地指向该项目。

## Observations
- [tool] Basic Memory
- [status] active
- [install] 通过 opencode.json / ~/.claude.json 的 MCP basic-memory 配置接入;嵌入模型 openai/qwen3.7-text-embedding (DashScope)
- [gotchas] ALL_PROXY=socks:// 会导致 MCP 启动崩溃,须覆盖为 http://;默认项目是 main,tts-exp 内容要显式传 project

## Relations
- relates_to [[tts-exp]]
- relates_to [[研究交接总览]]
