# 三条并行 replacement 探索：下游公共契约

## 任务分配

这是设计与交接，不是已执行结果。先读本页，再读所领 change 的 proposal → design → spec → tasks。每个 agent 只实现一条；三条都不依赖 `audit-lrs3-visual-teacher-missingness` 的结果，也不互相等待科学结果。

| agent | change | 唯一问题 | fresh TFG视频/评分上限 |
|---|---|---|---|
| A | `probe-wav2lip-natural-temporal-contrast` | 不用TTS，轻微平滑/增强natural mel的局部时间变化是否有用？ | 34 / 36 |
| B | `probe-wav2lip-reference-conditioning-interaction` | 固定音频干预的效果是否依赖静态参考帧？ | 36 / 54 |
| C | `audit-wav2lip-residual-local-response` | 已有残差在其实际影响的时间窗中是否留下正向响应？ | 0 / 0 |

优先级建议 A、C、B。总上限70个新TFG视频、90个fresh评分，训练/TTS/vocoder/新模型下载均0。这是预算上限而非运行耗时承诺。A是新候选机制，B/C是诊断；不应把三个实验包装成三个已知有效的生成头方案。

## 历史边界

BM先读 `Startup Router`、`实验指令`，再读 `Wav2Lip natural content residual continuation 2026-09-09`、`Wav2Lip historical shift rescore 2026-09-08`、`Wav2Lip spectral structure replacement 2026-09-08`。A另读 `Natural-conditioned local Wav2Lip control direction`：旧自然控制policy内部dev为NO_GO，不能宣称首次做natural-only控制。

最新全体CORRECT/N：ΔC=+0.008，95%CI跨0，`NO_INCREMENT_ESTABLISHED`；控制通过不等于replacement通过。历史shift自由搜索分数不是固定natural anchor增益；TTS-mel迁移、频谱迁移、旧natural-only policy已有阴性。A检验固定、无训练的时间调制算子，不复活旧policy；B/C只解释已有构造，不重调残差幅度。

全部为已见16条/8个source-group的探索，不是独立确认，不打开其他split或sealed媒体。设计时已见父总体结果，未计算新三条的候选结果、参考交互或局部关联。所有终态保持 `replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`、`historical_shift_gate_repaired=false`。旧父阴性和旧视觉教师BLOCKED均保留。

## 固定父资产

路径相对仓库根：

```text
P = runs/wav2lip_natural_content_residual_20260908_v2
Q = runs/wav2lip_natural_content_residual_continuation_20260909_v1
D = runs/lrs3_masked_tts_new_confirmation_20260902
```

以下为设计时核对的文件字节SHA-256。每条均核对本表，不只读取PASS布尔值。

| 文件 | SHA-256 |
|---|---|
| Q/protocol.json | 41a2f508a01b58001487ccbb20af712bac656458a119893bea722a72cd556996 |
| Q/final.json | 24550b33ec94110b8a8f0585081f964cbe910ac77671f27263172d9267701b2e |
| Q/validation.json | 97b0c6c03dca773de18bdb24b285f9ea234cca45fd037e47b27fac62de474169 |
| Q/control_validation.json | 5ac0bbf4623a08fb028f98bf19cb38abbcc0455f79d82ed0f437de8ccf4d8c93 |
| Q/analysis.json | 83a3c27798a31b8a8d2ee1350dc9850fe8a5710a421cf51bae4c18e88be084cc |
| Q/candidate_scores/manifest.json | 1ee531b15b78c21ae29c1482f12511ad403fa3fbdc9d8b9d86cd79dfd869c4de |
| P/drivers/manifest.json | 4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf |
| P/control_scores/manifest.json | 143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2 |
| P/static_faces/manifest.json | e57ed01f78f4870998f69ee5ee140e1831b0975bb8195bec2cbb1c7972a4f56e |
| D/00_cohort/manifest.json | 8ec7a2ec0d37c6abf51075d57f1467658fc0b3de9574799a1b8f08f9a2bf21b5 |
| D/03_data/natural_mels.npz | b48628ee3f09c0e46abc3d121d717bc72af10384cf7f20ea8caa2f5a7aaff44f |
| D/06_renders/box_manifest.json | 1a1304633e2eae0b525e5b7f4a06391cf8c28a52668d7ea6e1598fc1f68a17be |
| D/05_drivers/drivers.json | ad5bc9412833ac30a017f27e77499e4a60e1f38eb9c01c02f5db9c909bb92410 |

以 `(source_group,sample_id)` 排序，ID一对一连接，必须16条、8组每组2条。使用的间接数组/boxes/媒体/embedding路径从上述manifest追溯并逐一验证其既存hash，写入本run input_audit；不按glob取“最新”run。父manifest未提供的间接绑定须追溯其worker/protocol，缺失则BLOCKED，不用当前文件自hash冒充历史绑定。C不读取或hash媒体，仅沿既有审阅记录继承媒体来源限制。

父N来自P/control_scores，CORRECT/WRONG/SHUFFLE评分来自Q/candidate_scores；不能将N_REPLAY当作主基线。每条protocol在候选生成或新统计前冻结输入、代码、本页与本change各文件hash、样本、规则和预算；结束再核对父摘要。代码/spec变更必须新run，hash一致才可工程resume。

## 公共坐标和端点

继承 `changes/probe-wav2lip-natural-content-residual/design.md` 的“时间范围”“生成与评分接口”，及 `changes/resume-wav2lip-natural-content-residual/design.md` 第2节控制公式，须读全文相应设计。继承仅限资产/媒体/评分契约；候选、统计阈值、预算以本页及各新design为准，不运行父runner的all。

natural mel为 `[80,308]`，编码N前61440 samples（3.84s）；官方16列chunk、25fps，93帧静态224×224 FFV1视频，评分矩阵 `[88,31]`。mux完整原N、16k mono PCM16、零起点、保留音轨尾部；不得归一化、缩时、补视频、`-shortest`、换音轨。SyncNet评分完整224画面、不再裁ROI或检测脸。冻结 `U=range(30,58)`、自然/候选lag=-15..15；必须真实支持，不padding。

从矩阵在U上先求float64均值曲线z，`C=median(z)-min(z)`，`D=min(z)`，`k0=argmin(z_N)`，并列取首列。每条记录的k0只从原参考帧F0的N计算，A/B/C都不得用候选或局部窗口另选anchor。正向收益 `ΔC=C_X-C_N`、`ΔD=D_N-D_X`、`ΔA=z_N[k0]-z_X[k0]`；B四cell均用同一k0。

A/B每记录先计算量、组内两条均值、8组等权；C的局部量按其design在组内汇总支持后计算，仍8组等权。bootstrap采用 `PCG64(20260910)`、20000×8有放回组索引，所有contrast共享并保存索引，NumPy linear quantile。新探索决策使用双侧99%区间 `[q.005,q.995]`，95%可附录但不参与判定。四个预指定主问题（A两方向、B交互、C局部关联）不再扩增；99%是保守筛选约定，不声称小样本bootstrap提供严格家族错误率保证。不因并行实验先出阳性就停止其他分支或调参。

## A/B模型执行与门禁

复用已存在的 `masked_tts_tfg_probe.direct_mel.chunk_mels`、`wav2lip_face_roi_replacement.generation_worker` 和 `wav2lip_roi_peak_recheck.worker.SyncNetScorer` 纯接口；不要调用旧有损direct_mel.render。checkpoint沿父hash绑定，Wav2Lip为 `wav2lip_gan.pth`，不能换非GAN版。宿主 `/home/wjj/.venvs/wav2lip/bin/python` 先做真实CUDA kernel预检，batch=4、eval/no_grad、seed20260909、确定性开、TF32关。SyncNet `/home/wjj/.venvs/syncnet/bin/python`，CPU batch20、threads2。

生成worker持有同一个OS `flock` 文件 `/tmp/tts-exp-wav2lip-gpu0.lock`，锁内检查空闲≥5GiB；不得同时加载两个生成模型。锁忙为QUEUED，保留状态，由调度者重试，不是科学失败；不得杀其他agent进程或重装驱动。CPU各agent最多2线程。多agent可并行prepare/测试/分析，单卡generation串行。

A/B都独立CPU复现父控制：旧13/16、修订匹配域16/16；固定anchor损伤mean=1.162567632539，95%CI=[0.844646827451,1.482466255980]，8/8组正（**父复现使用父seed20260909、10000次、95%契约，不用新99%替代**）。重跑旧parity绑定两评分cell，前两条F0/N各fresh forward+评分一次，逐像素对齐P/N、矩阵误差≤1e-4、offset相同。不能只相信旧control_validation。B另有F1控制见其design。控制阶段独立validator PASS后，各候选入口/all/resume才可生成。

C只从数组重建父矩阵/端点、主三contrast/控制证据，无fresh scorer/模型；不能因此声称复验当前运行时GPU链。矩阵从embedding按float32 `sqrt(sum((V[r]-A[q]+1e-6)**2))` 重算；自然q=r+j-15。保存真实支持范围。矩阵容差1e-4，曲线/统计1e-6，ID、索引、计数严格相同。

## 最小交付、隔离与BM

三个包名分别为 `wav2lip_natural_temporal_contrast`、`wav2lip_reference_conditioning_interaction`、`wav2lip_residual_local_response`；仅写自己的 `scripts/experiments/<package>/`、`tests/experiments/<package>/`、`runs/<package>_<run-id>/`、本change/tasks。不修改共享worker、本公共契约、父run、CONTEXT/HANDOFF及其他agent文件。共享代码不满足接口时在本包做小adapter；需要改公共契约则停止并报告具体冲突，不私自放宽。

待实现CLI：`PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.<package>.runner --run-id 20260910_v1 --stage all`；validator为同包 `validate --run-root runs/<package>_20260910_v1`。A/B支持prepare/controls/candidates/analyze/all；C支持prepare/analyze/all。analyze只读完整产物，不隐式生成。实现仅需runner、独立validate及少量纯函数，无新通用框架。

最少产物：protocol/input_audit、drivers或support、media/scores manifests（C写reused manifest）、analysis、validation、review、final（JSON）和result.md。独立validator可共享I/O但不能调用producer的候选构造、核心统计或终态函数；从原数组重算。合成数据手算测试覆盖零效应、符号、乱序/缺cell、共享anchor、组等权、阈值边界、缓存篡改和resume。A/B还验PCM逐样本身份/PTS/帧数/模型确定性；C验证读取零模型预算。final最后绑定validation/review，不循环hash。

工程错误为BLOCKED/scientific=not_available；有效实现下控制未通过为CONTROL_FAILED；输入无辨识力用各分支标签，不删样凑结果。科学阴性是完成；验收失败非零退出。实际forward、fresh/reused cell和失败尝试都记账；表中为成功输出上限，失败重试另列，禁止追加候选或科学重复。每个阶段保存诚实的self-review，不冒充独立子agent审查。

每条维护各design给定的唯一BM实验实体：planned→running→concluded（工程未解决为blocked），先Router/实验指令、搜索、读旧笔记，保留changelog，写后读回。BM只写本实体，不并发改总览或父实验笔记。最终报告数字、控制/工程状态、限制和下一步建议；不自动进入新实验、训练、部署或关机。
