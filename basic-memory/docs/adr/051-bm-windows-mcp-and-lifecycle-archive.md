---
title: 051-bm-windows-mcp-and-lifecycle-archive
type: note
permalink: tts-exp/docs/adr/051-bm-windows-mcp-and-lifecycle-archive
status: open
decided: 2026-10-22 10:33:00+08:00
project: tts-exp
---

# ADR-051: BM Windows MCP 编码与生命周期归档

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

## 上下文

2026-10-22 排查发现，Basic Memory 0.23.2 的 CLI 读取和检索可用，但 Codex Desktop 启动 MCP 时反复握手断开。Codex 的本地日志明确报告 `Failed to read MCP server stderr (uvx): stream did not contain valid UTF-8`。服务端 FastMCP 启动横幅按 Windows 当前编码输出，未满足 Codex 的 UTF-8 管道要求。

同时，共享生命周期收件箱有 3123 个待处理 envelope，归档数为 0，last flush 为 never。BM 0.23.2 的 SessionStart/PreCompact 只捕获事件；`hook flush` 需要单独触发。它只归档本地操作轨迹，不会生成 checkpoint 或其它图谱笔记。

## 选择与理由

在用户级 Codex 配置中，给同名 `basic-memory` 注册显式 stdio 启动配置，使用本机已安装的 `${USER_HOME}\.local\bin\basic-memory.exe mcp`，只给这个服务设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。用户配置覆盖插件自带的同名 MCP；保留插件技能和原有受信任 hooks，避免编辑升级时会被替换的插件缓存。

用 Windows 当前用户的计划任务 `Basic Memory - Archive lifecycle inbox` 在登录时及每 15 分钟执行 BM 原生 `hook flush`，采用原生 30 天 processed 保留期和并发锁。维护脚本位于 `${USER_HOME}\.basic-memory\maintenance\flush-hooks.ps1`，使用隐藏窗口，不要求账号密码。

Codex 为 MSIX 应用，其进程看到的 `AppData\Roaming\uv` 是虚拟目录。计划任务运行在应用容器之外，必须使用物理路径 `${USER_HOME}\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Roaming\uv\tools\basic-memory\Scripts\python.exe`，通过 `from basic_memory.cli.main import app; app()` 调用 CLI。直接运行用户 bin 下的 uv trampoline 在计划任务中会失败。

## 替代方案

修改插件 `.mcp.json` 会在升级后丢失；修改全局 Python 编码会影响无关程序；只手动 flush 不能解决长期积压。没有升级 BM 或 FastMCP，因为已定位的问题可以通过独立启动环境修复。

## 验证与边界

首次 flush：3123 swept，2937 archived，186 duplicates，0 pending，0 invalid，0 pruned。全部事件仍在 processed 中；flush 前已创建并校验 ZIP 备份。

修复后的 MCP 完成 initialize、列出 21 个工具，并通过 `read_note` 读取项目 Startup Router。独立启动与当前 Desktop 相同版本的 Codex app-server，仅调用 `mcpServerStatus/list`，成功发现 21 个 BM 工具；没有创建聊天或调用模型。当前已运行聊天的工具清单仍需客户端重新加载，未声称在这个聊天内已热加载成功。

Windows 计划任务实际执行返回 LastTaskResult=0。将一个已归档 envelope 原样重放到 inbox 后，由计划任务执行得到 `1 swept, 0 archived, 1 duplicate, 0 pending, 0 invalid, 0 pruned`，验证了后台路径和幂等归档。

## 后果与恢复

这些修复是本机用户级配置，适用于该用户的多个项目，不改变研究结论。计划任务依赖当前 BM 安装环境；迁移或卸载 Codex/BM 时应重新核对物理 Python 路径。

备份位于 `${USER_HOME}\.basic-memory\maintenance\backups\`：`codex-config-20261022-102533.toml` 和 `hook-inbox-20261022-102533.zip`。维护日志为 `flush-hooks.log`，超过 1 MiB 轮换一份。移除计划任务并撤销用户配置中的 `[mcp_servers.basic-memory]` 即恢复插件默认启动方式；原研究笔记没有被清理或重写。

## Observations

- [decision] 为 BM 的 MCP 子进程单独固定 UTF-8 编码，使用用户配置覆盖插件同名服务。
- [decision] 生命周期事件用原生 flush 和本机计划任务维护，保持其与作者编写的知识笔记分离。
- [result] 初始 3123 个事件全部归档，0 invalid、0 pruned；后台幂等重放验证通过。
- [result] Codex MCP 工具发现接口返回 21 个 BM 工具，项目笔记读取通过。
- [constraint] MSIX 外部维护任务必须使用物理 uv 环境路径；不能假设应用内 AppData 路径可被计划任务访问。
- [limitation] 当前聊天内的工具热加载未验证，需要重新打开聊天后核对工具清单。


## 2026-10-23 修订：跨客户端共用独立用户运行环境

当天复发时，CLI 和 MCP 报 `uv trampoline failed to canonicalize script path`；共享归档任务仍返回 0，笔记能经原环境的物理 Python 路径读取。日志记录 VS Code 客户端也使用用户配置中的 `.local\bin\basic-memory.exe`。应用容器外不存在它内嵌指向的 `AppData\Roaming\uv\tools\basic-memory` 路径。10 月 22 日的修复只覆盖了计划任务的路径，未消除 MCP/CLI 对 MSIX 虚拟目录的依赖。

本次将运行环境重新安装到 `${USER_HOME}\.basic-memory\tools\basic-memory`。导出旧环境的全部 169 个包版本并原样固定安装，BM 仍为 0.23.2，FastMCP 仍为 4.0.0b1；没有升级依赖。锁文件在 `${USER_HOME}\.basic-memory\maintenance\bm-runtime-0.23.2.lock.txt`。uv 安装时指定 `UV_TOOL_DIR=${USER_HOME}\.basic-memory\tools`、`UV_TOOL_BIN_DIR=${USER_HOME}\.local\bin`；以后维护这个安装也应显式传相同目录，避免又写入客户端各自的 AppData 视图。

CLI 的 basic-memory/bm 入口由 uv 重新生成，指向独立用户环境。MCP 直接使用该环境 `Scripts\python.exe -X utf8 -c "from basic_memory.cli.main import app; app()" mcp`，继续固定子进程 UTF-8 环境，跳过 trampoline。归档脚本也指向同一个独立环境；本次不修改插件缓存或 hook 信任配置。

验证：basic-memory 和 bm 均报告 0.23.2；Desktop 及 VS Code 的实际 Codex app-server 分别发现 21 个 BM 工具，无需创建聊天或调用模型。计划任务切换后 LastTaskResult=0，共享收件箱 pending=0、processed=3127。MCP 读、搜及本附录的 append/readback 由验证脚本实际执行；该附录以回读一致为通过条件。当前已运行聊天的工具清单是否重载，仍以客户端实际暴露为准。

配置、原 CLI 入口和归档脚本备份在 `${USER_HOME}\.basic-memory\maintenance\backups\20261023-111046\`。旧 MSIX 环境保留供回退；知识库目录、数据库和研究笔记不迁移。本节修订前文关于“归档任务必须使用 MSIX 物理环境”的执行选择：现在所有客户端和任务统一使用独立用户环境。

### 本次修订的 Observations

- [decision] CLI、MCP 与归档任务统一使用 AppData/MSIX 之外的用户运行环境，MCP 直接启动 Python。
- [constraint] Windows 应用内安装的 uv trampoline 不能当作所有客户端共享的持久入口。
- [result] Desktop 与 VS Code 各自的 Codex 工具发现接口均返回 21 个 BM 工具；定期任务返回 0。

### Changelog

| Date | Change | Author |
|:--|:--|:--|
| 2026-10-23 | 补齐 MSIX 外客户端的启动路径修复，统一独立用户运行环境并验证 MCP 读搜写 | Codex |
