# doc-system

项目文档系统的结构、命名规范、写作约定和维护规则。Agent 在创建/修改文档前应加载此 skill。

## 目录布局

```
CONTEXT.md                  ← 入口：项目目标、在用/弃用、技术栈（≤100行）
AGENTS.md                   ← agent 配置（本文档）
HANDOFF.md                  ← 当前状态快照，供 agent 交接
│
docs/
├── adr/                    ← 架构决策记录（ADRs）
│   └── NNN-<slug>.md       ← 编号，一个决策一个文件
│
├── experiments/            ← 最终实验报告
│   ├── NN-<slug>.md        ← 编号报告（01–05 是核心；06+ 是新的）
│   └── <topic>.md          ← 补充详细报告（保留已有，从编号引用）
│
├── deployment/             ← 每个模型的部署记录
│   ├── DEPLOY.md           ← 总览：服务器表、模型矩阵、常见问题
│   └── <model>.md          ← 每个模型一个文件（环境、命令、坑）
│
├── results/                ← 可复现评分表
│   └── SYNC-C-SCORES.md    ← 全排行含条件（唯一文件，直接更新）
│
├── reference/              ← 参考指南和服务器信息（不是实验结论）
│   ├── pipeline.md         ← 实验流水线步骤
│   ├── server-access.md    ← 服务器地址和连接信息
│   ├── ditto.md            ← Ditto 部署指南
│   ├── chattts.md          ← ChatTTS 部署指南
│   └── indextts2.md         ← IndexTTS2 部署指南
│
├── agents/                 ← agent 工作流约定（domain, issue-tracker, triage-labels）
│
├── superpowers/            ← Superpowers skill 产物（handoff/plans/specs）
│
└── original-spec.md        ← 原始项目规格（仅保留，不再使用）
```

## 内容放哪里

| 你有… | 放在… |
|---|---|
| 新实验结论 | `docs/experiments/NN-<slug>.md` |
| 模型部署指南 | `docs/deployment/<model>.md` |
| SyncNet 分数 | `docs/results/SYNC-C-SCORES.md`（更新已有） |
| 服务器 IP/密码变更 | `docs/reference/server-access.md` |
| 流水线命令或工作流变更 | `docs/reference/pipeline.md` |
| 基础项目决策 | `docs/adr/NNN-<slug>.md` |
| 新模型的 MP4 输出文件 | `results/<model>/{natural_raw,tts_raw}/` |

## 命名规范

- **编号实验**: `NN-<kebab-case>.md`（例如 `01-tts-tfg-baseline.md`）。核心 01–05 已占用；新实验从 06 开始。
- **ADR**: `NNN-<kebab-case>.md`（例如 `001-tts-as-tfg-audio.md`）
- **部署**: `<model>.md`（例如 `echomimic-v2.md`、`imtalker.md`）——小写，用连字符
- **results/ 中的结果文件**: `{1..13}.mp4`，按样本 ID 命名——不要用描述性文件名
- **Superpowers 计划**: `YYYY-MM-DD-<topic>.md`，在 `docs/superpowers/plans/` 下

## 写作约定

### 必做

1. **标记在用/弃用状态** — 每个模型、服务器、TTS 引擎用 ✅/🔄/❌
2. **Sync-C 分数用 3 位小数** — 与 SyncNet 输出格式一致（例如 `5.670`）
3. **密码绝不写入文件** — 运行时用 `sshpass` 加环境变量
4. **CONTEXT.md 保持在 100 行以内** — 超过就拆分到具体文档

### 禁止

5. **不写进度日志**（`*_progress_*.md`） — 用 git log
6. **不写超过 50KB 的单文件计划** — 拆成任务卡
7. **不在 `docs/experiments/` 放进度更新** — 只放最终报告
8. **不在 `docs/reference/` 放实验结论** — 那属于 `docs/experiments/`

### 元数据与格式

- 文件开头用 `# <Title>`，与文件名 slug 匹配
- 实验报告可选地以状态字段结尾：`✅ 已完成` / `🔄 进行中`
- 排名表用 Markdown 表格——不要用图片
- 服务器和密码用 `docs/reference/server-access.md`

## 文件由谁拥有，谁可以编辑

| 文件 | 常用写者 | 编辑频率 |
|---|---|---|
| `CONTEXT.md` | 开发者 + Agent | 每部署/实验一次 |
| `HANDOFF.md` | Agent | 每次会话后 |
| `docs/adr/` | 开发者 | 按需 |
| `docs/experiments/` | Agent | 每实验一次 |
| `docs/deployment/DEPLOY.md` | Agent | 每模型一次 |
| `docs/deployment/<model>.md` | Agent | 部署时 |
| `docs/results/SYNC-C-SCORES.md` | Agent | 每评估批次后 |
| `docs/reference/server-access.md` | 开发者 | 服务器变更时 |
| `docs/reference/pipeline.md` | 开发者 + Agent | 流水线变更时 |

## 常用操作

### 新增模型

1. 创建 `docs/deployment/<model>.md`
2. 更新 `docs/deployment/DEPLOY.md` 中的模型表和服务器矩阵
3. 运行推理 → 得到结果 → 把 MP4 放在 `results/<model>/{natural_raw,tts_raw}/`
4. 运行 SyncNet → 更新 `docs/results/SYNC-C-SCORES.md`
5. 若模型未成功，加到 `CONTEXT.md` 的弃用表

### 新增实验报告

1. 确定下一个编号（当前最大是 05）
2. 创建 `docs/experiments/NN-<slug>.md`，遵循模板：
   ```markdown
   # NN — <Title>

   ## 问题
   <一句话>

   ## 方法
   <2-3 句总结>

   ## 关键结果
   <表格或要点>

   ## 发现
   <要点列表>

   ## 状态
   ✅/🔄 — <一句话>
   ```
3. 除非需要补充细节，否则不要创建补充 `<topic>.md`
4. 如果与 ADR 矛盾，明确标注

### 新增 ADR

1. 确定下一个编号（从 `docs/adr/` 中查看）
2. 创建 `docs/adr/NNN-<slug>.md`，内容：
   ```markdown
   # ADR-NNN: <Title>
   ## 上下文
   ## 决策
   ## 替代方案
   ## 后果
   ```
3. 引用编号化的 ADR

### 升级评估排名

- 直接编辑 `docs/results/SYNC-C-SCORES.md`
- 将新模型作为行插入排名表，排序方式：Natural Sync-C 降序
- 更新顶部注释的时间戳

### 切换上下文给下一个 Agent

- 在会话结束时更新 `HANDOFF.md`
- 包含：当前状态、最后完成的工作、下一步步骤
- 保持不超过 60 行

## 历史：删过什么

删掉的文件和原因：

- `docs/tfg_models/` — 合并到 `docs/deployment/`（按模型区分）
- `*_progress_*.md` 进度日志 — 用 git log
- 旧部署计划（>50KB） — 拆成任务卡
- `runs/` 中的旧 Ditto 输出 — 已完成，只保留在受控存档中

## 反模式

| 反面模式 | 正确做法 |
|---|---|
| 创建 `docs/experiments/my-new-finding.md` 作为新发现 | 用下一个编号 `06-my-new-finding.md` |
| 把部署命令写在实验报告里 | 放进 `docs/deployment/<model>.md` |
| 把分数截图贴进来 | 放进 `docs/results/SYNC-C-SCORES.md` 表格 |
| 在实验文件夹里写 `progress_2026-07-25.md` | 不写——用 git log |
| 把 SSH 密码写在 reference 文件里 | 永远不——用 `sshpass` 加 env 变量 |
| 创建单文件 50KB+ 计划 | 拆成多个文件 + 任务卡 |
| 恢复旧的 `docs/tfg_models/` 目录 | 用 `docs/deployment/<model>.md` |
| 在 CONTEXT.md 里写执行细节 | 链接到对应的 reference/experiments 文件 |
