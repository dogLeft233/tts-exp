# P 阻塞修复：下游执行入口

状态：已实施；2026-09-11。目标：先得到通过独立验收的 12 正式 + 2 smoke 新来源队列，再交回原 P candidate/freeze。当前本地演练因完整历史排除得到 0 个可用新组，科学队列仍 blocked；面向 Luna，按本页顺序执行，任务清单见 [tasks.md](tasks.md)。

先读本页及[原设计第 2、3、8 节](../probe-fresh-source-cross-generator/design.md)。实现边界的验收情景见 [spec.md](specs/fresh-source-input-unblock/spec.md)。无需通读旧实验历史。

## 1. 修什么，何时算完成

旧 run 的 86 历史组、6 fresh、3 时长候选、0 fully-screened 是该次快照。新执行必须重扫，不能把这些数写成常量。阻塞有两个独立来源：候选池不足，以及没有可验证的视觉审计。只补 `--visual-audit` 不保证达到 14 组。

本修复分两个验收点：

| 节点 | 必须满足 | 允许的下一步 |
|---|---|---|
| `COHORT_READY` | 12+2 不重合新组、完整输入审查、cohort-only validator `integrity=GO` | 构造原协议 direct_v1；此时 inputs 仍可为 `PENDING_CANDIDATE` |
| `INPUTS_FROZEN` | 14 条 N/C/R/ref 齐全、原 P 完整 validator GO、freeze 绑定通过 | 按原协议启动 A/B/C；D 继续受原测量门槛约束 |

本修复的必要交付止于 `COHORT_READY`，同时给出可执行的 candidate/freeze 交接命令。若下游任务还包含继续完整实验，则执行第二节点与原 A/B/C/D；不能把本 spec 的完成称为全部实验完成。

## 2. 新 run 与历史排除

使用新根 `runs/fresh_source_cross_generator_20260911_unblock_r1/`；已存在且身份不一致就递增 r2，不覆盖旧 run。准备产物统一放 `<run-root>/acquisition/`，P 仍使用现有根文件及 `<run-root>/run/shared/` 镜像，避免引入第三套 shared 路径。

先复用 `scan_history(..., ignore_run_root=当前根)`。保留全历史及封存组排除、覆盖歧义阻断与文件 SHA。准备和 P 使用同一 run-root，避免准备清单在另一 run 中被当成历史使用。旧 run 的候选记录可能被保守扫描纳入排除；如实列出，不通过忽略旧目录或批量豁免找回样本。

获取前写 `acquisition/plan.json`，包括历史快照 SHA、来源清单、固定排序、预算和输出路径；不写最终 cohort.json。P prepare 只在来源池及视觉审计完成后执行一次。中途资产缺失写准备终态；不要为探路反复生成 blocked cohort 再用 `--resume` 加样本。

## 3. U1：有界来源补充（CPU，一个 worker）

复用已有来源，依次检查本地未用 pretrain 文件、未提取归档成员，再检查项目已记录且当前可访问的 LRS3 pretrain 下载位置。路径固定为 `data/dataset_samples/lrs3/pretrain/<source_group>/<clip>.mp4` 和配套官方 `.txt`，沿用现有 P 路径校验；不引入任意 data-root 开关。

1. 固定目标为最多 24 个 fresh source groups，按原视频 ID 字典序；每组只看按文件名排序的前 8 个 clip。先按元数据排除历史组，再读取媒体。24 是超额备选预算，不是最终样本量。
2. 来源 URL/本地归档、split=pretrain、访问依据、成员清单、下载字节上限和 SHA 写入 plan/manifest。不得猜测存在的下载包或用无官方转写的网络视频替代。若尚无可访问来源，写 `BLOCKED_SOURCE_ACCESS` 和具体缺失的 URI/路径/访问条件。
3. 新下载默认总上限 10 GiB、总墙钟 45 分钟，未知长度按流式计数限制；空间至少保留 5 GiB。按来源固定顺序取包，在后续包加入前仍沿用原预算。到限写 `BLOCKED_ACQUISITION_BUDGET`，不隐式无限扩池。只下载所需 pretrain 资产，不读取 trainval/test 封存媒体。
4. 解包只取必要 mp4/txt，拒绝绝对路径、`..`、链接及越界成员；同路径已存在仅在 SHA 相同情况下复用，冲突写错误。保留原始文件，不改写旧数据。
5. 输出 `acquisition/source_manifest.json` 和 `pool.json`：每条含 group、clip ID、原来源/成员、mp4/txt 路径与 SHA；池固定后不按视觉通过率重新排序。所有被检查片段和拒绝原因保留在 ledger。

复用 `inspect_clip` 和 `transcript_for` 检查完整 6–10 秒、N 解码后 96000–160000 samples、源 AV 起点差 ≤20ms、完整官方英文 Text；不裁句、补时长、合并短片或拿新文件名冒充新 group。已知 speaker 与历史重合要排除；未知保留 `speaker_independence=unverified`。

## 4. U2：实际视觉审计（CPU，与 U1 实现并行）

新增薄入口 `scripts/experiments/fresh_source_inputs/visual_audit.py`。复用 `scripts/experiments/lrs3_tts_visual_advantage/video_features.py` 的 `create_landmarker` / `extract_video_features`；若缺少原尺寸眼距或首帧框，仅增加所需原始 landmark 导出，不另造视觉框架。

先确定可用 MediaPipe 版本、模型资产文件及 SHA、运行选项和 landmark 索引，写入 plan 并冻结。不得仅填一个看似有效的 SHA；模型缺失且无法在 U1 总预算内获取时写 `BLOCKED_VISUAL_ASSET`。不得用其他检测器静默替代。

逐组按固定 clip 顺序，在通过时长/PTS/转写检查的片段上执行：

- 在完整 R 的零 PTS、25fps 时间轴取前 140 帧；保留帧到源 PTS 的映射、valid 数组、原始 landmarks 和首帧人脸框。valid fraction 的分母固定为 140，要求至少 133 帧有效，首帧有脸，首帧原分辨率眼距 ≥40px。
- 人脸框采用原图坐标 `[x,y,width,height]`；复用原协议 1.5 倍正方形固定裁剪、越界反射填充和 512² 规范化。记录插值，检查固定框覆盖；不凭生成效果换首帧或框。
- 为完整 6–10 秒输入保存可播放预览和检查记录，确认单个讲话人、无切镜/配音、嘴无遮挡、转写完整。输入审查可由能实际查看媒体的 agent 完成，注明 reviewer 和证据；不能仅凭“检测到 landmarks”写 mouth_visible=true。无法判断留 pending 并拒绝入组。此审查不代替 B 的两名真实人类盲评。
- 每组取最先通过全部检查的 clip；前八条都失败则拒绝该组。不看 SyncNet、动态幅度或生成视频。`clip_audit.jsonl` 保存逐片段 PASS/FAIL/ERROR、真实测量和拒绝原因；不为凑数伪造数值。

产出 `acquisition/visual_audit.json`，兼容当前 `validate_visual_audit` 的 schema_version=1、method=`mediapipe_face_mesh_v1`、model_path/model_sha256、thresholds 和 groups 映射。成功项包含实际选中 clip_sha256、valid_fraction、frames_covered、eye_distance_px、first_frame_face、mouth_visible、face_box；额外绑定原数组和输入审查 SHA。没有 PASS 的组可在 groups 中省略，其完整失败记录留在 clip ledger。

修复 `validate_visual_audit`：合法 FAIL 的低有效率/无首帧脸是数据结果，不能导致整份审计解析失败；FAIL 永不参与接纳。PASS 必须保留全部门槛及有限数/几何合法性校验。缺失文件、SHA 不符、格式错误仍是工程错误。组映射中的 PASS 必须绑定同一片段 SHA，禁止把一个 clip 的结果借给同组另一 clip。

## 5. U3：接回 P 与独立验收（主 agent）

U1/U2 合并后再运行 P prepare，显式传 `--visual-audit`。本修复 run 存在 acquisition/plan.json 时，prepare 与 validator 必须加载其绑定的 pool.json，仅审查池内 group/clip，且验证这些成员确为每组原始成员排序的前八条；不得继续隐式遍历全部本地来源而突破 24 组预算。旧 run 的读取兼容性保留。修正 prepare 中写死的 visual_screening 文本，使其与实际“未运行/已运行但不足/通过”相符。将本补充设计/spec、来源 plan/manifest/pool、视觉 producer 代码、模型身份、原数组及审查文件身份纳入 execution_contract；不能只绑定旧 change。新增代码和资产必须固定后才运行 prepare。

给现有 `validate.py` 增加 `--phase cohort|inputs`，默认 inputs，保留原完整验收。cohort 模式不要求 C 或 freeze，独立重扫历史、重读媒体/转写/视觉数组，重算有效率、首帧眼距和排序，核对 12+2、SHA、root/shared 与执行身份；不能只信 PASS/self-hash。可共享 I/O，不调用 producer 接纳函数重放答案。

cohort 模式写 `cohort_validation.json`，`status=COHORT_READY` 仅在 cohort.status=GO 且所有独立检查通过时成立。样本不足仍可 `integrity=GO`，但 status 必须保留 BLOCKED；工程不一致为 `integrity=NO_GO`。`BLOCKED_HISTORY_COVERAGE` 即使候选 ≥14 也合法，不能因数量足够误报工程损坏。写 `acquisition/final.json` 绑定验收 SHA，并列出每层计数：发现/历史排除/媒体合格/视觉已审查/接纳/正式/smoke。

以下第一条是本 spec 要新增的入口，后两条是现有入口的补充接口；在代码实现和测试通过前不能声称命令已可用。来源获取由 U1 小脚本完成并输出固定 pool，不要求构建通用下载 CLI。

```bash
python -m scripts.experiments.fresh_source_inputs.visual_audit --run-root runs/fresh_source_cross_generator_20260911_unblock_r1 --pool runs/fresh_source_cross_generator_20260911_unblock_r1/acquisition/pool.json
python -m scripts.experiments.fresh_source_inputs.runner --run-root runs/fresh_source_cross_generator_20260911_unblock_r1 --stage prepare --visual-audit runs/fresh_source_cross_generator_20260911_unblock_r1/acquisition/visual_audit.json
python -m scripts.experiments.fresh_source_inputs.validate --run-root runs/fresh_source_cross_generator_20260911_unblock_r1 --phase cohort
```

视觉入口从 acquisition/plan.json 读取冻结模型路径/参数，不自行挑模型。若 pool 尚未完成，不生成空 PASS 审计。实际 run ID 变化时，所有命令一致替换。

`COHORT_READY` 后的交接命令，按顺序逐条检查成功才继续：

```bash
python -m scripts.experiments.fresh_source_inputs.runner --run-root runs/fresh_source_cross_generator_20260911_unblock_r1 --stage candidate
python -m scripts.experiments.fresh_source_inputs.validate --run-root runs/fresh_source_cross_generator_20260911_unblock_r1 --phase inputs
python -m scripts.experiments.fresh_source_inputs.runner --run-root runs/fresh_source_cross_generator_20260911_unblock_r1 --stage freeze
python -m scripts.experiments.fresh_source_inputs.validate --run-root runs/fresh_source_cross_generator_20260911_unblock_r1 --phase inputs
```

inputs 验收支持 freeze 前检查全部输入、freeze 后额外核对冻结绑定；只有最后一步通过才报告 INPUTS_FROZEN。candidate 失败保持队列并报工程阻塞，不换组。`--resume` 身份检查不放宽；资产/代码/spec 改变用新 run ID。

## 2026-09-11 r4 执行状态

`runs/fresh_source_cross_generator_20260911_unblock_r4/` 已完成本地 CPU 阶段。来源为新的 LRS3 pretrain 归档；24 个候选组中 14 组通过实际 MediaPipe 与输入审查，固定为 12 formal + 2 smoke。`acquisition/final.json` 报告 `COHORT_READY`、`integrity=GO`；candidate 14/14 完成，freeze 为 `FROZEN`，独立输入校验为 `status=GO`、`integrity=GO`、`errors=[]`。本次仅修复校验器缺陷（PCM16 哈希读取和 pool 上下文解包），未修改冻结输入或科学处理。

GPU 阶段尚未开始：AutoDL 实例当前 SSH 不可达。实例恢复后，按原跨生成器协议同步冻结的 `run/shared/media/`，先跑 Ditto 及另一已有生成器的 natural/candidate 双臂，再跑原自然音轨 SyncNet 评分、重复生成控制和报告；全部产物回传并审查后关闭实例。此处不把 CPU 门禁结果写成 replacement 科学结论。

## 6. 资源与交付

U1/U2 使用本地 CPU，每 worker ≤2 线程；主 agent 在各阶段审查测试及产物。不为下载/输入审查开启付费 GPU。若继续执行原实验，需要 GPU 时再沿用原设计第 8 节的实例核查、两小时预算、回传 SHA 和关机收尾；不得调用 shutdown 的帮助参数。等待来源访问或人工盲评不占用运行中的 GPU。

最终交付代码、测试结果、source/clip ledger、visual_audit、cohort_validation 和 acquisition/final。BM 更新原“新来源跨生成器与视觉验证 2026-09-10”实体，写实际计数、终态、产物路径及下一步；任务未通过不得勾成完成。只完成 spec 时记“修复协议已设计，待执行”。
