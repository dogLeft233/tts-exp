## 下游入口与唯一问题

阅读顺序：proposal → 本文 → specs/wav2lip-replacement-endpoint-reconciliation/spec.md → tasks。加载 BM Startup Router、实验指令，读取 `Wav2Lip spectral structure replacement 2026-09-08` 与 `LRS3 natural-to-TTS bridge confirmation result`。

问题：在共同 22 条已见记录上，历史微小正向描述与最新负向描述的差异，分别有多少对应评分支持变化、多少仍属于音频/生成/裁剪链的混合差异？本轮是 retrospective diagnostic，不是 independent confirmation。+0.078 的 23 条 discovery 只引用历史背景，不能与本轮按记录配对或合并 bootstrap。

预算固定：0 GPU、0 新媒体、0 网络、0 模型 forward；仅解码已有音频、读取缓存和统计。设计时只做了资产结构/支持检查：旧选定 66 cells 均恰好一条从 frame 0 开始的连续 track，合计 176 个矩阵均支持 U；最短 U=19 行、共同内部支持=84 行。这不代替下游验收，也不是本轮结果。

## 1. 冻结入口

路径均相对仓库根，SHA 为文件字节哈希，不能与 JSON 的 `artifact_sha256` 混用。

`H = runs/lrs3_natural_to_tts_bridge_confirmation_20260904`

`S = runs/wav2lip_spectral_structure_replacement_20260908_cpu_prepare_v3`

| 路径 | SHA-256 |
|---|---|
| H/00_protocol/cohort.json | b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b |
| H/01_audio/audio_manifest.json | 2c279dc5a74dbeb53ae547235347870c6664fd9caf656a594b48a8310d9bd288 |
| H/02_videos/videos_manifest.json | 680053f3ca851cea6a2572654a228671bbe616f369b00d84a552b22b305543c2 |
| H/03_scores/scores_manifest.json | 0022c07d68f859a0189833c58fd7f8daf639787d8d4ffc99e48e6ccaa1a99e5b |
| H/04_final/analysis.json | 0841a0e95efb8747516f866427bafa601c1fa5270ca4bcaaf6cfd396fc77556f |
| H/04_final/final.json | df0ca9767e70ccc384c86c1da23243c53fa609be12abd1dda20f9075b6732c6e |
| S/protocol.json | 284dea1b07fbccc07a309512b164265d215075af41b8009b38d163362d502cfa |
| S/audio/manifest.json | 2086e55df0e116e60656577f0ce255aad32c87a52ae02e279c2d8e0b96206ae3 |
| S/videos/manifest.json | 92b091c6bc77a1ad0487af3d701173d7fd969c62db940f3c930c02e69981aeba |
| S/scores/manifest.json | a35c09c0a1d5ed2ae9f6ea6a363fa1e0a6e190ff3d69dcb380b186d2ecec1270 |
| S/analysis.json | 957c8cb83d59f9fc5bb4ec3ec7c321449fa87a6a582c8d8fe4e0b9f7769e2316 |
| S/validation.json | 82a85878f9dbe50d7a406f1ab32457b30b2f4e94cd1953e7836e32ee895e0bd0 |
| S/final.json | 60454d9d5a70494db42f8f5b8bf5e0106db2cc92aee70dbe6c4f0057653ff701 |

取 H/cohort.records 的原顺序，按 `sample_id` join S/protocol.records，核对 source_group；必须恰好 22/22，没有重复、丢失或额外记录。ordered-ID hash 沿用父算法及值 `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`。不得 zip 或按得分选择。

Stage audit 先写 protocol：绑定以上入口、所有选中缓存/sidecar/日志/track/音频/相关媒体、本文/proposal/spec 及本轮全部执行代码的路径和 SHA；tasks 是可变 checklist，不绑定。已存在父资产 hash 必须匹配。旧 activesd/track 若未被父清单哈希绑定，记 `provenance=legacy_cache_locked_at_audit`，在加载数值前计算哈希；不能声称具有不存在的历史哈希或原始 embeddings 证明。新 run 不覆盖父文件。

## 2. 固定缓存矩阵及字段

所有 cell 的评分音轨均为 N；candidate 自身音轨 cell 不进入本轮。

| 来源 | video arms（固定顺序） | 距离矩阵数 |
|---|---|---:|
| H | N, N_REPEAT, BRIDGE_075 | 66 |
| S | N, N_REPEAT, RT, MAG, ENV | 110 |

H：`03_scores/scores_manifest.json.scores`，以 `(sample_id,cell)` 唯一定位 `V_N/A_N`、`V_N_REPEAT/A_N`、`V_BRIDGE_075/A_N`。用其 reference 定位 `03_scores/syncnet/<ref>/pywork/<ref>/{tracks,activesd}.pckl`。activesd 是每 track 一个 `[T,31]` float32 距离矩阵的 list，不是 embeddings；track 帧索引位于 `tracks[0]['track']['frame']`。

仅载入已经绑定哈希的这些本地 pickle。要求每 cell 恰好 1 track/1 matrix，track.frame 为从 0 起的连续整数，关联 pycrop 的 `00000.avi` 存在且来源可追溯；不能静默选择多 track 的最好或第一条。历史 parser 只取第一条日志数字，这次审计必须显式检查整份 score_log 只有一组 C/D/offset，与 manifest 一致。pycrop 的裁剪时长/音轨是历史评分处理的一部分，不能假称它是完整 N；严格 replacement mux 的 N PCM 另核对。

S：`scores/manifest.json.scores`，按 `(sample_id,video_arm,audio_arm=N)` 定位；直接使用 `matrix`、`visual`、`audio_embedding`、`worker_result` 及其 `*_sha256` 字段，禁止猜文件名或把 worker 当成分数。要求所有矩阵 finite、float32、`[T,31]`。共同 SyncNet 权重 SHA 为 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`；只核对绑定，不加载网络。

任何缺失/歧义/哈希错误都记录为工程 BLOCKED。不得自动重评分、重新检测或删除样本。旧矩阵没有 embeddings，验收只能通过哈希、track/媒体来源与历史输出复算建立缓存一致性；报告明确这一证据边界。

## 3. 音频与处理链对账

逐条读取原 N、M 和已生成 WAV；验证 mono/16k/PCM16/长度，分别计算容器 hash 与解码 PCM hash。H 的 `audio_manifest.rows[].arms` 是 list，以 arm 查找 `output`；S 的 `audio/manifest.rows[].arms` 是 dict。注意 H 的 `format.pcm_sha256` 实际由 `file_sha256(path)` 填入，不能把字段名字当解码 PCM 证据。

输出 22 行配对审计：两轮 N/M 身份及长度；H BRIDGE_075 与 S MAG 的 PCM equality、不同采样数、max_abs_diff_lsb、RMS_diff（int32/float64 计算，避免 int16 溢出）；N/N_REPEAT 同一性；S RT 对 N 差异；两轮已存 STFT/缩放元数据。只比较已有音频，不重新合成，也不因微小量化差异调 α。

N/M/source_group/source face 身份不匹配意味着配对失效，BLOCKED。BRIDGE/MAG PCM 不同是待解释结果而非自动工程失败；保留差异，不能将残差归因于视觉处理链。容器不同而 PCM 相同应明确区分。

另输出一张 H/S 处理链表，证据来自 frozen manifest/command/worker/track，当前源码仅供解释：生成框（H constant_full_frame_fallback；S 逐帧 ROI）、batch、checkpoint、回贴/裁切、codec、帧数/PTS、SyncNet 检测裁剪、JPEG/MFCC 前处理、支持行及均值精度。无法核实的项写 unknown。不得从“多个设置同时变化”推断某一个设置是原因。

## 4. 三种固定 endpoint

每条记录的 8 个矩阵共用 U 与 I；FULL 保留每个矩阵自己的历史全行范围。矩阵行 r 是 25fps 上五帧窗口的起点；列 k 的 offset 为 `15-k`，音频 embedding 行为 `r+k-15`。

| 名称 | 行定义 | 解释 |
|---|---|---|
| FULL | `range(T_cell)` | 全部历史矩阵行，保留边界零 embedding 距离；S 的 FULL 只叫全行重汇总，不能叫重新运行的官方整段 pipeline |
| I（COMMON_INTERIOR） | `range(15, min(T_8cells)-15)` | 8 cells 的共同真实 embedding 支持，无 offset 零 embedding；仍沿用各自既有 MFCC/裁剪处理 |
| U | S/protocol 的 `plus_rows + minus_rows` | 冻结目标时间局部窗口，逐项保持顺序，不能改成 Q/source rows |

检查 U 无重复且为 I 子集、I/U 非空、track 起点同为 0、媒体 FPS=25 且未发生时钟重映射；不支持即 BLOCKED，不取新的交集或删行。I 由全部 8 个矩阵长度决定，不能由得分决定。

对每个矩阵和行集合 R：先转 float64，`z=mean(distance[R,:], axis=0)`，然后 `D=min(z)`、`C=median(z)-min(z)`、`offset=15-argmin(z)`（并列最小列号）。禁止逐行算 C 再平均；不得根据 offset shift 视频/音频。保留 31 点曲线、行 hash、C/D/offset，输出 176×3=528 行，唯一 key `(sample_id,origin,video_arm,audio_arm,endpoint)`。

先做两个 parity，全部通过才写跨口径结论：

1. H 全部 66 矩阵使用历史 torch float32 全行均值复现日志 C/D（abs≤0.000501），offset 完全相同；再用 H manifest 的三位小数分数与原 bootstrap 配置复算原报告 repeat/bridge（abs≤1e-6）。新 float64 FULL 与历史 float32 汇总也记录差异，不强行逐位相同。
2. S 全部 110 矩阵按 U/float64 复算逐 cell 与逐记录差值；复算原报告涉及这些 cells 的 repeat、RT、MAG/ENV 比较和原 95%/97.5% CI（父 seed/draw/排序，abs≤1e-6）。只读 parent analysis，不能用报告数字反填缓存。本轮没读 N/P，因此不重新声称验证 Stage A 的错配控制。

parity 失败保留 discrepancy.json 和已完成审计，工程 BLOCKED；禁止调整容差或选择匹配的子集。

## 5. 固定配对统计与解释

统一 `benefit_C=C_candidate-C_baseline`，`benefit_D=D_baseline-D_candidate`，两者正值为好。raw D 和 benefit_D 必须分列。每 endpoint 均报告：H BRIDGE−N、H N_REPEAT−N；S MAG−N、ENV−N、MAG−RT、ENV−RT、N_REPEAT−N、RT−N。保留 C、D 均值、CI、正向记录数、joint win 数以及 offset 差≤1 的计数。

本轮描述性 CI 使用 sorted source_group、NumPy default_rng/PCG64 seed=20260908、10,000 draws；预生成并保存一个 `[10000,22]` group index 矩阵，所有差值/endpoint/对比共用；每 group 内先平均，再等权平均 group，percentile `[2.5,97.5]`、linear 方法。所有记录逐一配对，绝不把 H/S 或 FULL/U 当独立样本。95% CI 仅用于诊断，不替换父实验 97.5% 科学判据，不从多项结果挑一个宣布 gain。

对主候选令 `b_H^R = benefit(H BRIDGE,N,R)`，`b_S^R = benefit(S MAG,N,R)`，C/D 分别计算以下逐记录量及共用 draws 的 CI：

```text
total_gap = b_S^U - b_H^FULL
support_term = b_H^I - b_H^FULL
window_term = b_H^U - b_H^I
pipeline_term = b_S^U - b_H^U
identity_check: total_gap == support_term + window_term + pipeline_term
interaction = (b_S^U-b_S^I) - (b_H^U-b_H^I)
common_pipeline_gap = b_S^I-b_H^I
```

这是指定路径上的算术分解，不是因果/方差贡献分解。support_term 同时包含边界排除、共同长度与支持变化，不能称“零 padding 导致的份额”；pipeline_term 包含生成、裁剪、编码及任何已发现的 candidate 差异。不要报百分比贡献，尤其 total 接近零时。interaction 用于显示窗口影响是否依赖处理链，不能假设两项独立可加的物理效应。

结果必须回答：音频是否相同；原数字能否复现；H/S 各自的方向是否随 FULL→I→U 改变；在共同 I/U 上两轮是否仍有差异；N_REPEAT/RT 噪声是否与微小收益同量级；哪些原因仍无法区分。将“均值正负变化”与“CI 支持的变化”分开陈述，不显著不等于等效。

工程完整且独立验收 valid：`diagnostic_decision=RECONCILED`，`scientific_decision=NOT_A_CONFIRMATION`，`next_action=STOP_CURRENT_SPECTRAL_CONSTRUCTION`。即便某个重新汇总的数值为正，也保留历史 CONTROL_FAILED 和最新 NO_USEFUL_GAIN_ESTABLISHED；研究新路线须另写 spec，本轮没有自动后续实验。工程不完整：`engineering_decision=BLOCKED`、`diagnostic_decision=INCOMPLETE`、`scientific_decision=not_available`，`next_action=REVIEW_INPUT_OR_CACHE_DISCREPANCY`，不得写科学阴性。

两类终态始终 `training_authorized=false`、`generalization_established=false`、`historical_gate_repaired=false`、`new_media_count=0`、`model_forward_count=0`。不由本轮宣判所有 TFG 或所有生成头不可行。

## 6. 最小实现、验收和交付

建议小包 runner.py（入口/输入清单/审计）、analysis.py（纯统计）、validate.py（独立复算），按需加 config.py。复用父纯文件 I/O；不要导入有训练/生成副作用的父 runner 或直接调用其 gate。明确可参考的实现：`wav2lip_roi_peak_recheck/worker.py:pairwise_distance`、`wav2lip_spectral_structure_replacement/analysis.py`、`lrs3_natural_to_tts_bridge_confirmation/stats.py`，先 CodeGraph 定位。写代码前完成一轮 contract 自审，记录代码/spec SHA；实现者若使用独立 reviewer，仍以实际审阅内容为准，不依赖特定模型名。

输出到新 `runs/wav2lip_replacement_reconciliation_<id>/`：

```text
protocol.json              # 固定矩阵清单、输入/代码/spec hashes、行支持、预算
input_audit.json           # join/PCM/来源/处理链
parity.json                # 原结果复算差异；失败时另有 discrepancy.json
endpoints.json             # 528行及31点曲线
bootstrap_indices.npy
analysis.json              # 固定对比、分解、解释性限制
review.json                # 实现自审及绑定
validation.json            # 独立验收
final.json                 # 绑定 analysis/validation，终态与边界
result.md                  # 简短对账表、结论与下一步
```

validator 不调用 producer 的聚合/bootstrap/分解函数；独立从原缓存复算 FULL/I/U、parity、所有 CI、分解恒等式（abs≤1e-6）、counts、终态和 hashes。S 的 110 个矩阵从已有 embeddings 独立复建：audio 上下各 pad 15 个零向量，`sqrt(sum((visual[r]-audio_padded[r+k]+1e-6)^2))`，float32 距离容差 abs≤1e-4；H 如实只有缓存一致性验收，禁止伪造 forward/embedding 验证。纯 I/O 可以共享。先 analysis→validation→final，避免循环 hash，再检查 final 与验收结论一致。

最低测试覆盖：打乱清单仍正确 join、重复/缺记录拒绝；容器 hash 与 PCM hash 分离及 int16 极值差；非零 track 起点和 U 支持错误拒绝；FULL 保留 padding、I 排除 offset padding、先均值后 median/min 的反例与并列 offset；C/D 符号及配对分解；固定 group draws 可复现；parity 或 hash 篡改必拒绝且不会触发模型。测试允许合成小矩阵，不读取真实样本得分作期望值。

CLI 为下游待实现接口：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_replacement_reconciliation.runner --run-id <id> --stage audit
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_replacement_reconciliation.runner --run-id <id> --stage all --resume
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_replacement_reconciliation.validate --run-root runs/wav2lip_replacement_reconciliation_<id>
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m pytest -q tests/experiments/wav2lip_replacement_reconciliation
openspec validate reconcile-wav2lip-replacement-endpoints --strict --no-interactive
```

audit 仅冻结输入/结构/PCM/支持，不输出候选得分。all 可从空目录开始或恢复相同绑定的完整 audit；失败/部分分析用新 run-id，不能覆盖旧产物。只需 CPU 与现有 syncnet 解释器，无需检测 GPU、申请云卡或重装环境。

BM 使用同一笔记 `Wav2Lip replacement endpoint reconciliation 2026-09-08`：planned→running→concluded；阻塞写 blocked 与具体 discrepancy。每次更新按 Startup Router 先搜/全文读，保留 changelog，写后读回；结论记数字、来源、报告指针与“仅诊断，不解锁训练”。完成后勾选本 change tasks，交付结果与验收路径。
