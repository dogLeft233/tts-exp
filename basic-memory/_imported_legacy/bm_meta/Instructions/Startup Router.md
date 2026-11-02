---
title: Startup Router
type: note
permalink: tts-exp/instructions/startup-router-1
---

# Startup Router

> **Read this note at the start of every session, before any knowledge-base work.**

这是 tts-exp 项目记忆系统的唯一入口。任何 Basic Memory 操作前必须先读本笔记并遵循下面的分发表。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial Document | August 12, 2026 | user |
| 目录写入规则强化: directory 参数白名单 + permalink 小写陷阱 + 写后自检 | August 27, 2026 | user |

## 不可违反的规则

1. 每次 Basic Memory 调用(read/search/write/list)都显式传 `project="tts-exp"` — 写错项目是隐形破坏
2. 建任何笔记前先 `search_notes`(2–3 个查询变体);实体存在就用 `edit_note`,不新建
3. 覆盖任何已有笔记前先全文读取,保留 changelog 表并追加行,绝不重建
4. 目录名严格区分大小写。`write_note`/`move_note` 的 `directory` 参数只允许以下取值(逐字照抄,首字母大写): `Tasks` `Experiments` `Research` `Decisions` `Pitfalls` `Papers` `ToolNotes` `Instructions` `Schemas` `Templates`。**permalink 永远是全小写** (`tts-exp/tasks/...`),它不是文件路径,严禁照抄进 directory 参数。写入后核对返回的 file_path: 以小写 `tasks/`、`experiments/` 等开头 = 建了平行目录,立即 `move_note` 修正
5. 写入结果 permalink 若以 `-1`/`-2` 结尾 = 刚建了重复笔记,删除重写
6. 不确定归属时问用户,不猜

## 分发表

| 任务类型 | 按序加载 |
|:--|:--|
| 实验笔记 / 实验结果 | 1. `memory://instructions/实验指令` |
| 任务 / 项目工作 | 1. `memory://instructions/任务指令` · 2. `[[Task Board]]` |
| 决策 / ADR / 踩坑 | 1. `memory://instructions/决策指令` |
| 论文 / 文献 | 1. `memory://instructions/文献指令` |
| 工具部署 / 环境备忘 | 1. `memory://instructions/工具指令` |
| 回答过往工作/决策问题 | 1. `search_notes(query=..., project="tts-exp")` · 2. 无结果再搜 `project="main"` |
| 其他 / 不确定 | 加载全部指令笔记 |

## Known failure modes

- **跳过 search 直接创建 → 重复笔记**: 建前必搜;人名/专名用标题搜索,语义搜索对名字不可靠
- **目录大小写错 → 平行目录树**: 根因 = permalink 全小写 (`tts-exp/tasks/...`),agent 照抄进 directory 参数;Linux 下 `tasks/` ≠ `Tasks/`。directory 参数只取规则 4 的白名单;写入后核对 file_path,小写开头立即 `move_note`
- **append 出第二个 Observations 节 → status 冲突**: 笔记只保留一个 Observations 节,更新走 find_replace 或整篇重写
- **片段编辑索引 → 表格损坏**: Task Board 一律整表重写
- **覆盖不读 → 第二个 changelog 表**: 覆盖前必读,changelog 行追加不编辑
- **忘记 project 参数 → 笔记写进 main**: 每次调用显式传 project="tts-exp"

## 项目枢纽

- [[tts-exp]] — 项目总览