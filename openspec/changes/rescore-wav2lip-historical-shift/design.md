# 历史 SHIFT_200 同媒体重评分：下游执行合同

按 proposal → 本文 → specs/wav2lip-historical-shift-rescore/spec.md → tasks 阅读。只实现本诊断；本文是数值与媒体合同的唯一真相源。

## 1. 问题与关键区别

比较同一历史 `(V_SHIFT_200,N)` 相对 `(V_N,N)` 的收益，在 `LEGACY_TRACKED` 与 `FULL_FRAME_V4` 两个评分流程下如何变化。前者读取旧矩阵；后者对旧生成视频做新的 CPU forward。不是重跑已经完成的 FULL/I 缓存对账。

**v4 的 ROI boxes 用于 Wav2Lip 生成，不用于 SyncNet 再裁剪。** 其 `SyncNetScorer` 读取完整 224×224 输出帧。本轮 `FULL_FRAME_V4` 直接评分历史整帧，不用新 cohort 的 ROI、不新增嘴部 crop，不调用生成 worker。旧视频仍保留自己的 full-frame-fallback 生成方式；统一评分无法统一生成方式或两轮 cohort。

两路径差值仅支持“这些固定媒体上的评分流程敏感性”，包含 crop、编码/frontend 和旧 track 上下文的联合影响；不能分解为纯 crop、纯 padding、纯 mouth leakage 或真实运动质量。12 条 v4 科学结果只作背景，不能与历史 23 条合并或按 sample 顺序配对。

## 2. 固定输入与只读预检

仓库相对路径：

- `H = runs/lrs3_phase_preserving_replacement_envelope_20260904`
- `G = runs/wav2lip_global_shift_response_20260908_v4`

以下为**文件字节 SHA-256**，不是 JSON 内部 `artifact_sha256`：

| 输入 | SHA-256 |
|---|---|
| H/00_protocol/cohort.json | 850c224856bc7acb95aa709a18f4c0b3369ca724d8759603d1c4bf3aac74d72a |
| H/01_candidates/audio_manifest.json | 628e9e16ec4708421c9db57f636ad2871123bb0fddd5db3e33f611c109ad00cb |
| H/03_videos/videos_manifest.json | 0008acd3da7a7ec0066ce181b9c7204d34569d27e0aa4f30a92326b4561ac2bc |
| H/04_scores/scores_manifest.json | da7bac4f19903954573752abf029a65265029fe5462c9434a21659112abcf788 |
| G/history.json | 0108fe32b7aadb748708e14e4b21483eb8658c9c7c84a1f3fb0332efba6fd61c |
| G/protocol.json | 4c06a58f88942415b9cdb5f0ff54024eb1d403c132d19169c16d126b8919e140 |
| G/final.json | 19de6d86d3411c7da35673edd6620d417924d4fce2dec2bc291dcde97144ea6e |
| G/scores/manifest.json | 19dc5e4ee7644a08e34e23d0b8bf16d2f7b8c13b8477208fb8e1dde20791dbad |

固定使用 H/cohort 的全部 23 条、23 个 source groups，按 `(source_group,sample_id)` 排序。按 key join，拒绝重复/缺失，不能换样或按分数过滤。新 run 的 protocol 冻结上述输入、实际使用的资产/源码/模型/工具版本与 proposal/design/spec hash；tasks 不绑定。

- 视频：`H/videos_manifest.rows[].arms.{N,SHIFT_200}.{output,output_sha256}`；每条两臂 hash 必须匹配，来源 face/checkpoint/geometry 绑定一致。
- 音频：`H/audio_manifest.rows[].arms` **是列表**；按 `arm` 找 N 和 SHIFT_200。使用 `output/output_sha256`，自行保存 decoded PCM hash。评分的 N 是历史已绑定的 canonical N，另核对它与 cohort natural_audio 的 decoded PCM 一致，不拿容器 hash 当 PCM hash。
- SHIFT 仅作输入审计：mono/16k/PCM16，和 N 等长，精确 `shift[:3200]=0; shift[3200:]=N[:-3200]`；无需重写候选。
- 旧矩阵：由 `G/history.rows[].cells` 定位 `V_N/A_N`、`V_SHIFT_200/A_N` 共 46 个 tracks/activesd 文件；先验证 history 中的文件 hash 再加载可信本地 pickle。history 的 `common_absolute_support=[start,stop]` 为**半开区间端点**，不是帧列表。
- 要求旧 cached matrix 为单 track、finite float32 `[T,31]`，track.frame 连续；保留实际起点 f0。每 cell 的生成来源、旧 mux 来源及其 decoded N 绑定必须能串回上述资产。

设计时已只读核对 46 个旧视频的文件 hash、224×224/25fps、配对帧数一致（72–555 帧），并检查 23 条第 4 节 J 均非空（7–490 行）。这是可实施性预检，不替代执行时解码、PTS、PCM、窗口和来源验证。两条历史 track 有非零起点：`6ZiN9ZJT294_00005` 为32，`73rUjrow5pI_00005` 的 N 为0、SHIFT 为20；不得默认为0或因此删除记录。

## 3. 执行顺序与预算

### A. prepare + 历史 parity（CPU，无 forward）

只读资产审计；复算 46 个旧 FULL 的 C/D/offset，与 H 原日志/manifest 对齐（C/D abs≤0.000501，offset 完全相同，沿用旧 torch float32 reduction）。由历史三位小数及 seed=20260904、10000 draws 复算 C benefit mean=0.298、CI=[0.13147717391304337,0.4706554347826087]，D benefit mean=0.16382608695652173、CI=[-0.0056097826086957455,0.33039999999999997]，abs≤1e-6。

G/history 已存的 I_H endpoint 可用于核对缓存复算（abs≤1e-6），不能直接当作本轮 J 统计。先完成全部记录的媒体结构/共同窗口计划，再允许评分；所有工程失败写 discrepancy，停止，不把部分记录当新 cohort。

### B. 两个评分器 parity cells

固定使用 G 中 sample `lrs3_6ORDQFh0Byw_00008`、face_mode=`DYNAMIC` 的 `(V_N,N)` 与 `(V_DELAY_200,N)`。按四字段定位 G/scores/manifest，直接使用它们已经冻结的 mux media 与 N；在新目录 fresh forward，不重新编码这两个 parity 输入。

复用 `scripts.experiments.wav2lip_roi_peak_recheck.worker.SyncNetScorer`，`device=cpu, batch_size=20, threads=4`，解释器 `/home/wjj/.venvs/syncnet/bin/python`。模型 `third_party/syncnet_python/data/syncnet_v2.model` 的 SHA 为 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`。

保存新 embeddings/matrix。和 G 对应矩阵形状相同，最大绝对差≤1e-4；相同支持下 C/D/anchor 误差≤1e-6，offset 完全相同。任一不通过，停止46-cell重评分，不调容差。两个 parity cell 不进入23组统计，不算新增实验样本。

### C. 46 个历史媒体重评分 cells

每条旧 N/SHIFT 视频均从原始 `output` 解码全部 BGR uint8 帧，使用同一 FFV1/Matroska 无损路径编码；帧数/顺序/224×224 与解码像素逐项保持。时间从0开始、25fps，先验证源 PTS，编码后也验证。派生视频不产生新口型；禁止 drop/duplicate/pad/interpolate、resize、跟踪或独立选 crop。

每个派生视频 mux 整条 untouched N，视频 stream-copy、音频 PCM16；评分前后 decoded PCM 字节完全相同。旧 SHIFT 视频内置的 driver 音轨不是 replacement 音轨，必须明确替换。音轨不截短、不移动，不用 `-shortest`。复用 `wav2lip_global_shift_response.media` 的 decode/encode/mux helper，先读其与 parent helper 的真实合同。

用 B 中同一个 CPU scorer 评分；每 cell 保存 JPEG/MFCC 提取证据、embeddings、完整距离矩阵和实际调用记录。保持 v4 的 BGR 0..255、JPEG、MFCC 默认参数、5视频帧/20 MFCC帧、stride=4、vshift=15、epsilon=1e-6。新矩阵不能用历史 embeddings 或距离矩阵冒充。

总预算：46 个复用旧生成视频、46 个无损派生视频、46 个新 mux、46 个目标评分 jobs + 2 个 parity jobs；TFG forward=0、CUDA jobs=0、training=0。一个 cell-level job 包含多次网络 batch，报告时不要把48当成底层 forward 函数调用次数。顺序 CPU 执行，至少预留5 GiB，按帧数估算不足则报所需空间，不清理用户资产。

## 4. 同一绝对时间支持与指标

新 full-frame 行 r 对应原视频帧 g=r。设完整视频 F 帧，完整 N 为 L samples：`T_new=min(F,L//640)-5`，各新矩阵必须匹配这个形状，不能假设总有 F−5 行。

每记录先冻结：

```text
I_H = range(*G.history[record].common_absolute_support)
I_new = range(30, min(T_new_N,T_new_SHIFT)-30)
J = I_H ∩ I_new
旧 cell 取行 r=g-f0_cell，新 cell 取行 r=g，g∈J
```

所有四个矩阵在同一个 J 上计算主比较；J 在看到新分数前冻结。逐项核验历史 I_H 到 tracks/matrix 的映射、对应旧局部窗口与原 PCM 支持，新 J 的5帧视频/MFCC/±15搜索支持也有效，且排除±200ms的补零/末尾支持。不能为保留全23条而外推行号。J 非空即可，最短7行如实报告，仍按23个 source groups 推断；不按帧数加权。旧 track 裁切产生的上下文差异仍属评分流程差异，J 不能消除它。

每矩阵先求 `z=mean(matrix[rows,:],axis=0,dtype=float64)`，再求：

```text
D=min(z); M=median(z); C=M-D
offset=15-argmin(z)  # tie取最小列
```

对 J 固定**共享锚点** `o0=offset(LEGACY_TRACKED,N,J)`，四个矩阵都用 `D_anchor=z[15-o0]`。这是旧自然基线的评分坐标，不是物理嘴部运动真值。保留每路径 N 自己的 free offset，不能为 SHIFT 重选 anchor。

每路径配对收益（正值均为好）：

```text
benefit_C = C_SHIFT-C_N
benefit_D = D_N-D_SHIFT
benefit_anchor = D_anchor_N-D_anchor_SHIFT
delta_M = M_SHIFT-M_N
offset_delta = offset_SHIFT-offset_N
benefit_C == delta_M+benefit_D
```

主输出为 J 下两路径的上述收益，以及每记录 `frontend_change_X=benefit_X_FULL_FRAME_V4-benefit_X_LEGACY_TRACKED`（X=C/D/anchor）。offset 差也逐记录列出；相对 +5 帧的比例仅描述，当前数据没有新增 same-video shift/生成控制，不再发 generated_shift_follows 门禁。

两路径各自 FULL 的 C/D/offset 与收益另列为描述性附表，支持不相同，不能用于主评分流程差值。另附旧原始 +0.298、旧 I_H 的 +0.191 以防混淆。新 J 数字不得覆盖这些历史口径。

## 5. 统计与停止条件

23 source groups 等权；PCG64 seed=20260908，10000 次配对 group bootstrap，保存 `[10000,23]` draws，同一 draws 用于所有 J 收益和流程差值；percentile 95% CI，linear quantile。报告均值、CI、正向数、C/D joint wins、offset差≤1帧的数量。全部新区间为回顾性探索、未做多比较校正。

每路径只输出两个描述性标志：

- `positive_score_signal`：C mean>0.05 且 C CI下界>0。
- `anchored_joint_signal`：上述成立，且 D、anchor benefit CI下界均>0，且23条中至少20条 `abs(offset_delta)≤1`。

此外 `frontend_C_change` 按其**配对差值 CI**输出 `POSITIVE/NEGATIVE/UNRESOLVED`。不能用“一条显著另一条不显著”证明流程差异，也不能把任一标志false当作零效应或等效性证明。保留全部原始量，单个标志不替代解释。

| 条件 | 必须输出的终态 |
|---|---|
| 任一绑定/parity/媒体/独立验收失败 | engineering=BLOCKED，diagnostic=INCOMPLETE，scientific=not_available；保存原因与实际计数 |
| 全部完成且独立验收valid | engineering=GO，diagnostic=COMPLETE，scientific=NOT_A_CONFIRMATION；next_action=CLOSE_SHIFT_DIAGNOSTIC |

所有分支 training_authorized/generalization_established/historical_gate_repaired 均false。科学结果不触发重试、扫描参数或额外生成。完成后的报告回答：旧分数是否复现；相同J下收益是否依赖评分流程；共享自然坐标是否改善；哪些旧/新差异仍未解释。即使残留正向信号也只能记为未确认观察。本分支收口，后续“完整natural + 正确/错配/无内容辅助”由独立 spec 决定。

## 6. 最小实现、验收与交付

新增包建议 `config.py / runner.py / analysis.py / validate.py`，按需一份 media helper。复用明确的底层 worker，不导入旧 runner 执行生成或其门禁。先写关键合同测试，再实现CPU流水线。

待实现 CLI（本 spec 交付时不执行实验）：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_historical_shift_rescore.runner --run-id <id> --stage prepare
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_historical_shift_rescore.runner --run-id <id> --stage all --resume
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_historical_shift_rescore.validate --run-root runs/wav2lip_historical_shift_rescore_<id>
```

顺序 prepare→parity→rescore→analysis→独立validate→final。新 run 目录写入；resume 仅复用输入/源码/输出hash均匹配的已完整cell。源码变化用新run-id；终态失败目录保持原样，工程修复重试不得改候选/阈值。

最低产物：`protocol.json, input_audit.json, parity.json, media_manifest.json, scores_manifest.json, endpoints.json, bootstrap_indices.npy, analysis.json, review.json, validation.json, final.json, result.md`；每cell有worker/embeddings/matrix及来源。工程失败另存 `discrepancy.json`。

独立 validator 从绑定资产和 embeddings 重建48个新矩阵（float32 abs≤1e-4），独立加载46个旧矩阵，独立复算J、endpoint、bootstrap/flags（float64 abs≤1e-6）；不能调用producer的指标/判定函数。验证exact PCM、派生视频decoded pixel identity、PTS、46+2分母、模型/源码hash和零生成声明。两个 parity cell 对照G的缓存，而不是自己新输出互相比较。验收无需第二轮网络forward。

最低有意义的测试：乱序key join及缺/重复cell拒绝；非零/不同f0与半开区间的J映射；`min(F,L//640)-5`及短窗口边界；先均值曲线后median的反例与D-benefit符号；共享anchor不随SHIFT移动；配对bootstrap流程差值；PCM/pixel/hash篡改拒绝；parity失败不启动46个目标jobs。自审如实署名，不把自审称为独立agent审查。

BM：先读Startup Router、实验指令，按2–3个查询变体搜索并读当前 `Wav2Lip historical shift rescore 2026-09-08`；planned→running→concluded或blocked，保存数字、解释边界和产物指针，保留changelog、写后读回。只修改该实验笔记，不改写旧阴性终态。完成实现与验收后才勾选tasks中的执行项。
