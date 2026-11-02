---
title: 远端遗留笔记归档（_imported_legacy）
type: reference
permalink: tts-exp/reference/imported-legacy-remote-notes
project: tts-exp
tags: [import, legacy, remote]
---

# 远端遗留笔记归档

从训练机 `wjj@192.168.50.2` 的 `tts-exp` 检出中以只读方式导出、本地原本不存在的笔记。
**这不是现行知识图谱的一部分**：BM 元层（入口/指令/schema/模板）、旧版根文档与个人库
都放在这里，避免与现行 Startup Router、Instructions 与项目约定冲突。

| 子目录 | 内容 |
|---|---|
| `bm_meta/` | 远端 BM 的元层：AGENTS/CONTEXT/HANDOFF/tts-exp 入口笔记、Instructions、Schemas、Templates |
| `personal_global/` | 远端 `~/basic-memory`（个人/全局库）的笔记，凭据文件已剔除 |
| `repo_root/` | 远端仓库根文档（含 2026-10-08 版 HANDOFF.md）与 tmp 手记 |
| `conflicts/` | 与本地同名但内容不同的笔记（未覆盖本地版本） |

来源清单与哈希见 `.codex/scratch/notes_import_staging_attempt01/import_manifest.tsv`。
