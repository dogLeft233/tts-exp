## ADDED Requirements

### Requirement: Freeze the diagnostic sample and immutable inputs

实验 SHALL 使用父 `runs/wav2lip_face_roi_replacement_20260906_host_fix5/`；五份入口的固定文件 hashes 继承 `diagnose-wav2lip-roi-control-failure/specs/wav2lip-roi-control-failure-diagnostic/spec.md`，不得只信任运行时计算的 hash。另绑定 `runs/wav2lip_roi_control_diagnostic_20260906_audit4/final.json` 文件 SHA-256 `f575515b90da58c348565fbbf2980cf2f7e355ef5b308c3d77ffccb9c7b12d05`、`diagnostics.json` SHA-256 `24f8c52440499f5f86985439ec87e8921f479fbeba13357463ae1e05405ed608`，验证其 validation 为 valid 及证据绑定。

SHALL 核验父 22 条顺序/cohort、选用媒体/PCM、权重和历史矩阵的身份。fix5 指向 fix1 的合法 symlink 可接受，保存逻辑和 resolved 路径。选样固定为以下 8 条历史 C 失败以及父顺序前两条通过记录；最终执行顺序按父 cohort 顺序：

```text
fail:
lrs3_6wk4dkYSrV0_00006
lrs3_6qqqVwM6bMM_00007
lrs3_73cTNHEQhkQ_00007
lrs3_796LfXwzIUk_00007
lrs3_7CIq4mtiamY_00007
lrs3_7DCofMA9eQA_00007
lrs3_6ydYeyNSQVY_00008
lrs3_79tRTivyMSM_00014
pass:
lrs3_6WeS1bXRBOk_00006
lrs3_6ul2TSvUDog_00007
```

每条 SHALL 仅评分 `(G_N,N,false)`、`(G_W,N,false)`；完整键为 `(sample_id,video_arm,audio_arm,repeat)`，恰好 20 个 cell。不读取 heldout、不按新结果替换样本。评分前写 protocol，绑定本 change proposal/design/spec（不含 tasks）、全部执行源码、官方模型源码/权重、选中输入 hashes、预处理参数和设备版本。

#### Scenario: Inputs do not match the frozen cohort

- **WHEN** 父入口、样本标签、媒体、PCM 或权重不匹配，或必需资产缺失
- **THEN** 输出 BLOCKED 和精确证据，不修改父资产、不换 run 来源或跳过记录

### Requirement: Produce genuinely new scores with an independent implementation

worker SHALL 只复用官方 `SyncNetModel.S` 的结构和权重，权重 SHA-256=`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`，strict load、eval、no_grad/inference_mode；不得调用历史 evaluate/calc_pdist/worker 或读取旧 embedding/matrix 作为新评分输入。

SHALL 在本轮新目录中按官方契约从原 muxed media 重新提取 `%06d.jpg`（ffmpeg `-threads 1 -f image2`，不 resize、不换 RGB、不改帧率），cv2 读 BGR、0..255 float32。音频按官方 `-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000` 提取，核对 decoded PCM 与冻结 N PCM 完全一致；`python_speech_features.mfcc` 使用默认参数、int16 PCM，不先归一化。

令 `T=min(F_video,floor(L_samples/640))-5`。SHALL 对 r=0..T−1，视觉窗为 `[r:r+5]`，音频 MFCC 窗为 `[4r:4r+20]`。保存每个 cell 的全新 `visual.npy`、`audio.npy`，均为有限 float32 `[T,1024]`，以及 `[T,31]` distance.npy。

距离 SHALL 独立实现官方 pairwise_distance 语义：列 k=0..30 对应音频行 `r+k−15`，越界用零 embedding，`dist[r,k]=sqrt(sum((visual[r]-audio_or_zero[r+k−15]+1e−6)^2))`。先按 float32 运算，保存后汇总以 float64 进行。注意 epsilon 加在各维差值上，不能改成 sqrt(sum(diff²)+eps)。同一媒体只作这一次新评分，不根据是否匹配追加重试。

#### Scenario: Offset direction or padding is implemented incorrectly

- **WHEN** 合成 embeddings 在已知音频位移处具有唯一最小距离
- **THEN** 测试验证列和 `offset=15−k` 的方向、边缘零 padding 和 epsilon；不能用调用同一 producer 函数构造 expected 值

### Requirement: Compare local evidence without redefining the control

SHALL 使用已审计的父 PLUS/MINUS 行集合（本轮可复用它们，上一诊断已独立审计），对新旧矩阵各自用 float64 列均值算 31 点曲线；`argmin` 并列取最小索引、offset=15−k、gap=次小值−最小值。清晰峰要求 gap>0.010 且 offset 非 ±15。

两段各自 SHALL 报告 G_N/N 与 G_W/N 的新旧曲线、offset、gap，`actual=off(G_W/N)−off(G_N/N)`、`expected=−mean(d)`（固定父 masks）、`residual=actual−expected`。本轮 C 局部条件仍要求双方两段清晰且两段 |residual|≤1；保留已审计的父基线条件，不修改 1 帧容限。报告 `historical_fail` 8 条与 `historical_pass` 2 条各自重现数量，不汇总成 22 条新通过率。

SHALL 报告新旧矩阵最大绝对差、两段曲线最大绝对差。数值一致要求这两项均 ≤0.001（设备数值差容限）；同时两段 offsets、clear flags、逐条 C 判定与历史精确一致。全部 20 cells/10 records 满足才可给出 PEAKS_REPRODUCED。完整计算中的任何超容限或判定差异为 SCORER_MISMATCH，保留具体位置和双方值，不调整容限。此状态描述差异，不等同于已证明历史实现错误。

#### Scenario: Identical peaks but changed distances

- **WHEN** 整数峰一致但任一矩阵或局部曲线最大误差 >0.001
- **THEN** 输出 SCORER_MISMATCH 并报告“峰一致、数值不一致”，不得报告完全复现

### Requirement: Validate artifacts and preserve the scientific boundary

SHALL 输出至新 `runs/wav2lip_roi_peak_recheck_<run_id>/`：protocol.json、audit.json、scores/manifest.json、每 cell embeddings/matrix、analysis.json、result.md、final.json、validation.json。final SHALL 绑定证据 hashes、预期/实际 10 records/20 cells、新生成视频数 0、历史 scientific decision CONTROL_FAILED、own_audio_retested=false、bridge_executed=false、training_authorized=false、generalization_established=false。

离线 validator SHALL 从锁定的 embeddings 独立以 float64 公式重算距离（与 float32 存储矩阵逐元素最大误差≤1e−4），独立汇总曲线/峰/残差/差异/终态，不能调用 producer 的距离/峰/判定函数；允许共享 I/O、hash 和父身份审计。SHALL 验证输入/代码/spec/输出绑定、20 个唯一 cell、形状/有限值/帧数、无缺失，拒绝伪造终态。测试至少覆盖方向/epsilon/padding、0.010 与 1 帧边界、缺失/错配 cell、篡改 matrix/analysis/final；至少一个 expected 手工可算。

合法完整且 validator valid 时 status=complete，decision 为 PEAKS_REPRODUCED 或 SCORER_MISMATCH，退出 0；输入缺失、计算失败或验收失败 status=blocked/decision=BLOCKED，退出 2，保留实际完成数和原因。不得把 SCORER_MISMATCH 当基础设施失败或 PASS。

报告 SHALL 说明这是“新推理 + 独立距离/峰实现”相对上轮“旧矩阵复算”的新增信息。复现不能区分 SyncNet 表征限制、warp 声学变化和生成器响应不足，own-audio 未重测且仍失败；不解锁 bridge、不改历史终态、不宣称 replacement 或泛化成功。建议只提出一个有证据支持的下一步，不自动实施。

执行 subagent SHALL 先读 BM Startup Router/实验指令，使用 memory-capture 规范，在 project=tts-exp 按 2–3 查询变体先搜，创建 planned 后更新同一份实验笔记（directory=Experiments）。真实结论含 result/conclusion/status/report 观察，并关联两份父实验；更新已有笔记前全文读取、保留已有 changelog，写后读回检查目录与 permalink 无重复后缀。未运行不能写成功结果，阻塞也须写原因和已完成证据。

#### Scenario: Independent scores reproduce the failures

- **WHEN** 20 个 cell 新旧数值在容限内且局部峰/判定一致，validator valid
- **THEN** 输出 PEAKS_REPRODUCED、8/8 失败和 2/2 通过复现、BM concluded；历史 CONTROL_FAILED 保留，完成后停止
