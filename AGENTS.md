# TTS-TFG 实验项目

**核心问题**: TTS 生成的语音是否比自然语音产生更好的唇形同步 (TFG)?

AISHELL-1 中文语音 → ASR → 自克隆 TTS (`faster_qwen3` 0.6B ICL) → Ditto/9 个 TFG 模型生成口型视频 → SyncNet 评分。核心结论: 中文成立 (Ditto ΔSync-C=+1.25, p=0.0005), 英文不成立; 效应无法由单一声学特征解释。

入口: `CONTEXT.md` (项目状态/数据/已弃用, ≤100 行), `HANDOFF.md` (当前快照), `docs/`。大部分文档为中文。

## Commands

```bash
# 全流水线 (步骤 00→05), 10 样本; 每步可独立重跑
./scripts/run_all.sh
./scripts/run_all.sh --smoke          # 1 样本 (smoke_id 见 config)
RUN_ID=custom_id ./scripts/run_all.sh # 显式 run id (默认 UTC 时间戳)

# 单步执行 (共享 runs/<run_id>/)
python scripts/02_tts.py --run_id <run_id> [--smoke]

# 测试 — 每个脚本一个 pytest 文件
pytest tests/                         # 或: pytest tests/test_mandarin_alignment.py
```

- 步骤 00–03, 05 在 ditto conda 环境运行 (`/root/autodl-tmp/envs/ditto/bin/python`); `04_eval.py` 内部 subprocess 调用独立的 syncnet 环境。
- 云 TTS/ASR 需要 `DASHSCOPE_API_KEY` 环境变量。国内模型下载需 `HF_ENDPOINT=https://hf-mirror.com` (run_all.sh 已设置)。
- 服务器: `sshpass -e ssh -p <port> root@connect.westc.seetacloud.com` — 密码只走环境变量, 永不入文件。见 `docs/reference/server-access.md` 和 `remote-gpu-deploy` skill。

## Architecture

### 编号流水线脚本 (`scripts/`) — 核心流程

| 步骤 | 作用 |
|---|---|
| `00_datacheck` | 只读数据体检 (阈值告警不阻断) |
| `01_asr` | Qwen ASR Flash (DashScope) → 转写 |
| `02_tts` | 自克隆 TTS (text = ASR(ref), ref_audio = 自然音频) |
| `03_ditto` | Ditto TRT 口型生成 (natural_raw / tts_raw 双臂) |
| `04_eval` | SyncNet 评分 — 正则解析 stdout → Sync-C (越高越好) / Sync-D (越低越好) / AV offset |
| `05_report` | 配对统计 + 图 (方向判定, Cohen's d, p 值) |

编号 `06+` 为 Wav2Sem 式深度分析阶段 (特征可分性, 因果干预, SyncNet 剂量-响应, 英/中 MFA 对齐, oracle Fs, t-SNE)。共享代码: `tfg_feature_common.py` (dataclass `AudioItem`/`TokenSpan`/`EmbeddingRecord`, 常量, 统计) 与 `utils.py` (配置加载 `load_config` 深合并, `detect_sample_ids`)。

### Runs 与可复现性

- 产物在 `runs/<run_id>/<step>/`; `run_all.sh` 把 `config.yaml` + git commit 快照进每个 run。
- `scripts/config.yaml` 是单一真相源 (seeds, paths, providers, conditions, eval 字段映射)。脚本经 `utils.load_config(repo_root)` 读取, 支持按 run 的 override 文件。
- 样本 ID 来自 `samples.ids` (或 `audio_dir` 里的 `<id>.wav`); `--smoke` 收敛到单样本。

### TTS provider 工厂 (`scripts/tts/`)

- `base.py`: `TTSProvider` ABC + `TTSResult` dataclass。契约: provider 只返回 **mono numpy audio + sample_rate**, 磁盘 I/O 与元数据由 `02_tts.py` 处理; provider 不得全局重设 torch/np 种子。
- `factory.py`: `get_tts_provider(tts_cfg, run_id, repo_root, env)` 注册表 — `faster_qwen3` (本地), `dashscope_vc` (云端), `fish_audio`, `f5_tts`, `higgs_tts`, `siliconflow`。新增后端 = 实现 ABC + 注册进 `_REGISTRY`。
- 跨模型对比配置在 `scripts/configs/` (如 `r1_faster_qwen3.yaml`)。

### 评估

- SyncNet V2 (`third_party/syncnet_python`), 权重 `data/syncnet_v2.model`; `04_eval.py` 用正则解析 `demo_syncnet.py` stdout。
- Sync-C 一律 3 位小数。排名表在 `docs/results/` (如 `SYNC-C-SCORES.md`)。

### 约定

- 文档: 最终报告 → `docs/experiments/NN-<slug>.md` (01–05 为核心, 新报告从 06 起), 模型部署 → `docs/deployment/<model>.md`, 分数 → `docs/results/`, 决策 → `docs/adr/NNN-<slug>.md`。**不写进度日志** (`*_progress_*.md`) — 用 git log。`CONTEXT.md` ≤100 行。完整规则见下方文档系统章节与 `doc-system` skill。
- 模型 MP4 输出 → `results/<model>/{natural_raw,tts_raw}/<sample_id>/`。
- 测试: 改动某脚本时同步更新对应 `tests/test_<script>.py`。

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Basic Memory (跨会话记忆)

持久记忆在 Basic Memory 的知识图谱 (MCP tools)。两个项目:
- **项目记忆** = `tts-exp` (本仓库的 `basic-memory/` 目录)
- **全局记忆** = `main` (`~/basic-memory/`)

**启动时必读**: 会话开始、任何知识库操作之前,先 `read_note(identifier="Startup Router", project="tts-exp")`(memory://instructions/startup-router,搜不到就按标题搜),按其中的分发表加载对应指令笔记。**在所有读/写操作前加载完,再动手写。**

**搜索顺序固定**:回答关于过往工作/决策的问题时,先搜项目记忆,再搜全局记忆:
1. `search_notes(query=..., project="tts-exp")` — 项目内搜索
2. 无结果(或想找全局结论)再 `search_notes(query=..., project="main")`
3. 命中后用 `build_context(url="memory://...", project=...)` 或 `read_note` 取全文

写入约定:项目相关笔记一律 `write_note(..., project="tts-exp")`;涉及跨项目/个人通用知识才写 `main`。避免在 `main` 里重复存项目内容。目录规范、命名、changelog、索引规则都在 Startup Router 与各指令笔记中。相关 skill: `memory-notes` (笔记格式), `memory-capture` (捕获工作状态), `memory-continue` (恢复上下文), `memory-tasks` (任务跟踪)。

### Domain docs

Entry point: `CONTEXT.md` — project overview, active/deprecated components, tech stack.

**项目文档由 Basic Memory 管理**(`basic-memory/` 目录,经 MCP 索引)。仓库根不再维护独立 `docs/` — 文档写 `basic-memory/docs/...`,索引路径见 Startup Router 分发表。Architecture decisions: `docs/adr/`(bm 内)。Experiment reports: `docs/experiments/`(bm 内)。Results: `docs/results/`(bm 内)。

### Documentation rules

1. `CONTEXT.md` is the entry point — keep it under 100 lines
2. `docs/experiments/` (bm 内) is for final reports only — no progress logs
3. `docs/deployment/` (bm 内) is for per-model deployment records
4. `docs/results/` (bm 内) is for reproducible score tables
5. `docs/reference/` (bm 内) is for deployment guides and server info
6. Don't write progress logs (`*_progress_*.md`) — use git log
7. Don't write >50KB plan files — break into task cards

## Documentation system

### Directory layout

```
CONTEXT.md                  ← entry point: project goal, active/deprecated, tech stack (≤100 lines)
AGENTS.md                   ← agent configuration (this file)
HANDOFF.md                  ← current state snapshot for agent handoff
│
basic-memory/docs/          ← 项目文档 (由 Basic Memory 索引管理)
├── adr/                    ← architectural decision records
│   └── NNN-<slug>.md       ← numbered, one decision per file
│
├── experiments/            ← final experiment reports only
│   ├── NN-<slug>.md        ← numbered reports (01–05 are core; 06+ are new)
│   └── <topic>.md          ← supplementary detailed reports (keep existing, reference from numbered)
│
├── deployment/             ← per-model deployment records
│   ├── DEPLOY.md           ← overview: server table, model matrix, common issues
│   └── <model>.md          ← one per model (env, command, gotchas)
│
├── results/                ← reproducible score tables
│   └── <METRIC>.md         ← e.g. SYNC-C-SCORES.md (full ranking with conditions)
│
├── reference/              ← guides and server info (not experiment conclusions)
│   ├── pipeline.md         ← experiment pipeline steps
│   ├── server-access.md    ← server addresses and connection info
│   └── *.md                ← per-tool deployment guides (ditto, chattts, indextts2…)
│
└── agents/                 ← agent workflow conventions (unchanged)
```

### What goes where

| If you have… | Put it in… |
|---|---|
| A new experiment conclusion | `basic-memory/docs/experiments/NN-<slug>.md` |
| A model deployment guide | `basic-memory/docs/deployment/<model>.md` |
| A SyncNet score table | `basic-memory/docs/results/SYNC-C-SCORES.md` (update existing) |
| A server IP/password change | `basic-memory/docs/reference/server-access.md` |
| A pipeline command or workflow change | `basic-memory/docs/reference/pipeline.md` |
| A fundamental project decision | `basic-memory/docs/adr/NNN-<slug>.md` |
| New model MP4 output files | `results/<model>/{natural_raw,tts_raw}/` |

### Naming conventions

- **Numbered experiments**: `NN-<kebab-case>.md` (e.g. `01-tts-tfg-baseline.md`). Core 01–05 are taken; new experiments start at 06.
- **ADRs**: `NNN-<kebab-case>.md` (e.g. `001-tts-as-tfg-audio.md`)
- **Deployment**: `<model>.md` (e.g. `echomimic-v2.md`, `imtalker.md`)
- **Results files in results/**: `{1..13}.mp4` for per-sample videos, named by sample ID

### Writing conventions

- **Mark active/deprecated status clearly** — every model, server, and TTS provider should say ✅/🔄/❌
- **Sync-C scores use 3 decimal places** — consistent with SyncNet output format
- **Server passwords never in files** — use `sshpass` with env vars at runtime
- **Deleted files** — removed `docs/tfg_models/` (consolidated into `docs/deployment/`), `*_progress_*.md` logs (use git), stale deployment plans (completed work)

### Project skills

- **remote-gpu-deploy** (`.claude/skills/remote-gpu-deploy/SKILL.md`) — GPU 服务器部署工作流：权重下载、conda 环境管理、常见代码修复、多模型兼容性矩阵
- **doc-system** (`.claude/skills/doc-system/SKILL.md`) — 项目文档系统：目录布局、命名规范、写作约定、内容分配、常见操作和反模式
