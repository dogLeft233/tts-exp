# TTS 音频资产目录设计

## 状态

- 状态：已通过讨论，待实施
- 范围：仅整理本地仓库
- 日期：2026-07-27

## 背景

当前项目中的 TTS 音频分布在多个位置：

- 流水线运行目录：`runs/<run_id>/02_tts/`
- 手工整理目录：`data/data/audio_en_qwen3_tts/`
- TFG 结果目录：`results/<model>/tts_raw/`
- 特征分析目录：`data/wav2sem_analysis*/`

`scripts/02_tts.py` 默认将音频写入运行目录，`runs/` 又被 Git 忽略。历史分析文件还会直接引用具体的 `run_id`，因此无法通过 provider、模型、语言或数据集稳定地查找音频。

需要区分两种职责：

- `runs/` 保存一次实验的原始、可复现运行产物。
- `data/tts/` 保存本地可检索、可复用的 TTS 音频资产。

当前的 `data/data/audio_en_qwen3_tts/manifest.json` 明确标记 provider 为 `dashscope_vc`。它不能因为目录名含有 `qwen3` 就归类为 `faster_qwen3`。

## 目标

1. 通过 provider、模型、语言、数据集和实验条件找到 TTS 音频。
2. 每份归档音频都能追溯到原始 run、输入音频、文本和配置。
3. 不破坏现有 `runs/` 流水线、历史分析和已完成结果。
4. 下载远端运行结果后，用一个本地命令完成归档。
5. 让 `faster_qwen3` 和 `dashscope_vc` 在目录和文档中明确分离。

## 非目标

- 本次不整理远端服务器目录。
- 本次不迁移或重命名 `data/data/` 下的原始数据。
- 本次不重写所有历史分析文件中的旧路径。
- 本次不改变 `02_tts.py` 的默认运行输出路径。
- 本次不删除旧音频或历史结果。

## 目录设计

新增本地 TTS 资产库：

```text
data/tts/
├── README.md
├── catalog.json
├── faster_qwen3/
│   └── qwen3-tts-12hz-0.6b-base/
│       └── zh/
│           └── aishell1/
│               └── self_clone/
│                   └── r1_faster_qwen3_20260707T104138Z/
│                       ├── audio/
│                       │   ├── 1.wav
│                       │   └── ...
│                       ├── manifest.json
│                       └── config.yaml
└── dashscope_vc/
    └── qwen3-tts-vc-2026-01-22/
        └── en/
            └── librispeech/
                └── self_clone/
                    └── <run_id>/
                        ├── audio/
                        ├── manifest.json
                        └── config.yaml
```

目录字段规则：

| 字段 | 规则 | 示例 |
|---|---|---|
| provider | 使用代码中的稳定 slug | `faster_qwen3` |
| model | 使用规范化模型名 | `qwen3-tts-12hz-0.6b-base` |
| language | 使用 ISO 语言码 | `zh`、`en` |
| dataset | 使用项目内统一数据集 slug | `aishell1`、`librispeech` |
| condition | 描述参考音频与目标音频关系 | `self_clone`、`held_out_reference` |
| run_id | 保留原始运行 ID | `r1_faster_qwen3_20260707T104138Z` |

样本文件保留现有的数字命名，例如 `1.wav`，避免破坏下游按 sample ID 查找的逻辑。

## 资产 Manifest

每个 artifact 目录包含一个 `manifest.json`。路径字段优先使用相对于仓库根目录或 artifact 目录的相对路径，禁止写入机器特定的绝对路径。

最小结构如下：

```json
{
  "schema_version": 1,
  "artifact_type": "tts_audio",
  "artifact_id": "faster_qwen3__qwen3-tts-12hz-0.6b-base__zh__aishell1__self_clone__r1_faster_qwen3_20260707T104138Z",
  "run_id": "r1_faster_qwen3_20260707T104138Z",
  "created_at_utc": "2026-07-07T10:41:38Z",
  "source": {
    "dataset": "AISHELL-1",
    "language": "zh",
    "source_manifest": "data/data/audio/manifest.json",
    "source_audio_dir": "data/data/audio"
  },
  "tts": {
    "provider": "faster_qwen3",
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "clone_mode": "icl",
    "x_vector_only": false,
    "append_silence": true,
    "seed": 42
  },
  "provenance": {
    "repo_commit": null,
    "source_run_dir": "runs/r1_faster_qwen3_20260707T104138Z",
    "config_snapshot": "config.yaml"
  },
  "files": [
    {
      "sample_id": 1,
      "path": "audio/1.wav",
      "reference_audio": "data/data/audio/1.wav",
      "text": "示例转录文本",
      "status": "ok",
      "duration_s": 5.2,
      "sample_rate_hz": 24000,
      "channels": 1,
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  ]
}
```

Manifest 不能包含 API key、密码或其他服务器凭据。旧的 `tts_meta.json` 保留在 `runs/` 中，导入器可以读取它并生成标准 manifest。

## Catalog

`data/tts/catalog.json` 是所有本地 artifact 的索引。每个条目至少包含：

- `artifact_id`
- artifact 相对于仓库根目录的路径
- provider、model、language、dataset、condition
- 样本数量和状态
- 是否为该筛选组合的推荐版本

同一筛选组合最多有一个 `preferred: true` 条目。推荐版本通过 catalog 管理，不使用跨平台兼容性较差的 `latest` 符号链接。

## 本地 CLI

新增 `scripts/tts_catalog.py`，其接口负责隐藏目录拼接规则，避免其他脚本重复硬编码路径。

### 列出资产

```bash
python scripts/tts_catalog.py list --provider faster_qwen3
python scripts/tts_catalog.py list --language zh --dataset aishell1
```

### 查找资产路径

```bash
python scripts/tts_catalog.py locate \
  --provider faster_qwen3 \
  --dataset aishell1 \
  --language zh
```

### 导入运行结果

```bash
python scripts/tts_catalog.py import \
  --source runs/r1_faster_qwen3_20260707T104138Z/02_tts \
  --run-id r1_faster_qwen3_20260707T104138Z \
  --provider faster_qwen3 \
  --model qwen3-tts-12hz-0.6b-base \
  --language zh \
  --dataset aishell1 \
  --condition self_clone
```

导入行为：

- 复制 WAV 到新的 artifact 目录。
- 不删除或修改原始 `runs/` 文件。
- 读取已有的 `tts_meta.json` 和配置，补全 manifest。
- 计算每个文件的 SHA-256。
- 检查 sample ID、文件格式和必需元数据。
- 目标 artifact 已存在时拒绝覆盖，除非显式传入覆盖选项。
- 支持从已有的固定目录导入，并要求调用方明确提供缺失的 provider、模型或数据集信息。

已有固定目录没有原始 `run_id` 时，使用 `legacy_<目录名>` 作为 artifact 标识，并将 `run_id` 记为 `null`；不得从目录名虚构运行时间或实验编号。

### 校验资产

```bash
python scripts/tts_catalog.py verify --artifact <artifact_id>
```

校验包括文件存在性、SHA-256、sample ID 和 manifest/catalog 一致性。

## 与现有流水线的关系

第一阶段保留现有运行方式：

```text
02_tts.py -> runs/<run_id>/02_tts/*.wav
03_ditto.py -> runs/<run_id>/03_ditto/...
```

`data/tts/` 是发布到本地的副本，不替换 `runs/`。远端运行结果下载到本地后，通过 catalog CLI 归档。

`03_ditto.py` 已支持 `paths.tts_audio_dir`。后续新实验可以把该配置指向选定 artifact 的 `audio/` 目录；历史实验仍可继续读取自己的 `runs/<run_id>/02_tts/`。

已有的固定路径暂时保持不变：

- `data/data/audio/`
- `data/data/audio_en/`
- `data/data/audio_en_qwen3_tts/`
- `runs/<run_id>/02_tts/`
- 分析结果中的历史 `runs/` 引用

## 迁移顺序

1. 新增 `data/tts/`、manifest schema、catalog 和说明文档，不移动现有文件。
2. 导入现有的 `data/data/audio_en_qwen3_tts/`，按 manifest 中的 `dashscope_vc` 归档。
3. 导入能下载到本地的 faster Qwen 运行结果，使用真实 run metadata，不根据目录名猜 provider。
4. 为后续 TTS 下载流程补充一个固定的“下载后导入”步骤。
5. 将新的 TFG/分析实验优先改为引用 catalog 中的 artifact。
6. 搜索并确认旧路径不再被活跃流程依赖后，再考虑清理旧副本。

迁移过程中不删除文件，不重写历史 manifest，不覆盖现有结果。

## 文档调整

新增：

- `data/tts/README.md`：日常查找和导入的快速说明。
- `docs/reference/tts-artifacts.md`：完整目录规则、manifest 字段和 CLI 用法。

更新：

- `docs/reference/pipeline.md`：补充运行产物到本地资产库的流程。
- `CONTEXT.md`：补充 `data/tts/` 的职责和 faster Qwen 入口。

## Git 和存储策略

Git 跟踪：

- `data/tts/catalog.json`
- 每个 artifact 的 `manifest.json`
- `config.yaml` 快照
- `README.md` 和参考文档

默认忽略生成音频：

```text
data/tts/**/audio/*.wav
```

音频保存在本地磁盘，manifest 提供可复现索引。确实需要发布的少量音频可以单独显式加入 Git 或外部存储。

## 测试和验收

需要覆盖：

- 目录 slug 生成和 artifact 路径生成。
- 从 `tts_meta.json` 导入并生成标准 manifest。
- 从已有 `manifest.json` 导入 DashScope 音频时保持 provider 正确。
- 缺少文件、重复 artifact、重复 sample ID 和非法元数据时失败。
- `verify` 能发现文件删除、内容修改和 manifest 不一致。
- `list` 和 `locate` 能按 provider、模型、语言、数据集筛选。
- 现有 TTS、Ditto 和分析测试不受影响。

验收标准：

1. 执行 `list --provider faster_qwen3` 能列出所有本地 faster Qwen 资产。
2. 执行 `locate` 能直接返回选定 artifact 的 `audio/` 路径。
3. 任意归档 WAV 都能追溯到原始 run、输入音频、文本和配置。
4. `faster_qwen3` 与 `dashscope_vc` 不会因为模型名相似而混在一起。
5. 不运行服务器、不移动历史目录也能完成本地整理。

## 风险与取舍

- 复制 WAV 会增加本地磁盘占用，但换取独立于远端 `runs/` 的可用性。
- 暂时保留旧路径会产生少量重复，但能避免破坏历史分析。
- manifest 依赖调用方提供准确的 dataset 和 condition；导入器必须拒绝静默猜测关键字段。
- 默认忽略 WAV 意味着新 clone 不会自动拥有音频，需要通过 manifest 和外部资产恢复。

## 决策

采用“运行目录保持不变 + 本地 TTS 资产库 + manifest/catalog 索引”的方案。先完成可检索和可追溯，再逐步迁移活跃实验；不进行一次性的大规模目录重构。
