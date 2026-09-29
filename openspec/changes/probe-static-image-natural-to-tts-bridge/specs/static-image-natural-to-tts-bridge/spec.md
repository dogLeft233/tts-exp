## Purpose

检验固定单张人像、完全不输入原说话视频动作时，历史 natural-to-TTS bridge 波形是否改善生成画面与原自然音频的同步评分。将音频靠近 TTS、replacement 分数收益和 LOCAL_SWAP 时序响应分别测量，提供可复算的下游实验契约。

## ADDED Requirements

### Requirement: Freeze the historical cohort and waveform identity

实验 SHALL 使用 `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json` 的原顺序全部 22 records / 22 source groups，以及该 run `01_audio/audio_manifest.json` 绑定的 N、BRIDGE_075、LOCAL_SWAP。SHALL 在任何新评分前冻结 ID 顺序、源视频、容器 SHA256、实际 decoded PCM SHA256、长度、模型/代码/参数版本；不得将旧名为 `pcm_sha256` 的容器 hash 当作解码 PCM hash。SHALL 标记 `seen_fit=true`，不称新样本确认。

N/B/S SHALL 逐采样等于历史资产；B 的 alpha 固定 0.75，不重新合成候选。所有臂为 16 kHz、mono、PCM16、相同 L 个采样。历史 B 的说明性构造为 natural phase + 25% natural log magnitude + 75% MFA-linear TTS log magnitude，保留 natural 时间网格与长度。不得按历史或新分数换样、换图或调强度。

#### Scenario: Missing or changed historical input

- **WHEN** 任一记录、来源、音频身份或所需资产不符合冻结契约
- **THEN** 输出 `BLOCKED_INPUT` 和明确原因，保留 22 条完整清单；不得用剩余样本宣称完成。

### Requirement: Use exactly one static reference image per sample

实验 SHALL 将对应源视频第 0 个解码帧无损保存为 PNG，所有生成臂仅使用该 PNG。SHALL 在评分前以冻结的人脸检测器取该帧最高置信度人脸（并列按左、上坐标升序），固定检测阈值 0.9、生成框 bottom padding 10 像素、其他 padding 0 并截断到画面边界。无合格检测 SHALL 阻塞，禁止整帧框兜底、人工选嘴型或后续换帧。

SHALL 固定 25 fps，逐生成输入帧记录 `source_frame_index=0`，以 RGB 像素 hash 验证所有参考帧相同；生成嘴部允许变化。评分 crop SHALL 从该首次人脸框固定得到，全部 arm 和音轨配对共用；禁止针对输出臂重新检测或选择更好 track。实际 PNG、生成框、评分框与可视化框图 SHALL 交付。

#### Scenario: Static input is only nominal

- **WHEN** 生成器读取了第 1 帧及以后的源动作、使用不同参考图或候选专属评分 crop
- **THEN** 独立验收失败，不输出有效静态实验结论。

### Requirement: Include reconstruction and independent repeat controls

实验 SHALL 使用以下五个生成臂。N_REPEAT 必须独立执行生成，不能复制 N 的输出；所有臂冻结相同模型、seed=42、batch、精度和编码参数。

| 臂 | 驱动音频 | 用途 |
|---|---|---|
| N | 历史原自然 PCM | 完整自然基线 |
| N_REPEAT | 与 N 字节相同 | 独立生成重复性 |
| RT | natural 的 alpha=0 STFT 往返 | 排除仅重建/归一化造成的变化 |
| B | 历史 BRIDGE_075 PCM | 唯一 bridge 候选 |
| S | 历史 LOCAL_SWAP PCM | 时序传递诊断 |

RT SHALL 使用历史 B 的 STFT 契约：float64 CPU、n_fft=win_length=1024、hop=256、periodic Hann、center=true、reflect padding、magnitude floor=1e-7、natural phase、ISTFT 精确恢复 L；随后仅一次全局 RMS 匹配、必要时一次全局峰值衰减至 0.999、一次 PCM16 量化。SHALL 保存每个缩放因子及对 N 的误差，不强行要求 RT 与 N 字节相同。

S SHALL 验证为原 PCM 按边界 floor(L/4)、floor(L/2)、floor(3L/4) 排列 A-C-B-D，不平滑拼接、不改样本值。评分控制音频 ND SHALL 为 N 延迟 3200 采样（200 ms / 5 帧）、前补零、尾截断，长度 L；ND 不生成新视频。

#### Scenario: Reconstruction differs from natural

- **WHEN** RT 与 N 的 PCM 或评分不同
- **THEN** 如实报告并保留 RT；B 必须同时与 N 和 RT 比较，不能将重建效果归为 TTS 迁移增量。

### Requirement: Produce the registered media and score matrix

实验 SHALL 按以下两个阶段执行，分母始终为 22。一个 cell 指唯一的视频/音频配对；同一 cell 的多个统计窗口不算新 cell。

| 阶段 | 每条新增生成视频 | 每条评分配对 | 总预算 |
|---|---|---|---|
| A 测量控制 | V_N、V_N_REPEAT | V_N/N、V_N_REPEAT/N、V_N/ND | 44 视频、66 cells |
| B bridge 与局部交换 | V_RT、V_B、V_S | V_RT/N、V_B/N、V_B/B、V_N/S、V_S/N、V_S/S | 66 视频、132 cells |

SHALL 先完成 A 的独立验收和测量门槛，再运行 B；A 门槛失败时 B 为 0、`NOT_EVALUATED`。全部成功最多 110 个新生成视频、198 cells；不得新增动态输入 arm、候选强度、参考图或科学重复。纯工程错误允许最多 4 次同参数失败重试，单独计费并记录原因；不得重跑成功 cell 挑分数。

用于评分的媒体 SHALL 为零起点恒定 25 fps，视频帧不得插值或循环补齐；所有音轨对同一视频使用完全相同的视频帧。SHALL 直接读取冻结 PCM 和解码视频帧评分，审计媒体使用无损视频/PCM；播放用有损副本不参与评分。尾部不足的窗口按输入支持统一排除，不为某臂额外补内容。

#### Scenario: Stage A fails

- **WHEN** 重复性或延迟测量控制失败
- **THEN** 输出 `MEASUREMENT_CONTROL_FAILED`、A 全部数据和 B 未运行原因，不把它记作 bridge 阴性。

### Requirement: Fix scoring geometry time support and lag conventions

实验 SHALL 使用父确认轮同一 Wav2Lip GAN / SyncNet V2 checkpoint（按父 manifest 核对 SHA256）及官方音频/图像 frontend，保存逐窗口、逐 lag 距离矩阵。对所有主比较使用同一绝对帧集合 W：在 [-15,+15] 帧每个 lag 下，视频及音频完整感受野均有真实数据的共同内部窗口；ND 的零填充边界也 SHALL 排除。W 在读评分前由长度和 frontend 支持计算并冻结，每条至少 20 个窗口；缺失记录不得静默删除。

定义 `d(t,k)=||v(t)-a(t+k)||₂`，`curve(k)=mean(t∈W)d(t,k)`，`D=min_k curve(k)`，`C=median_k curve(k)-D`，`k*=argmin_k curve(k)`（并列取最小 k）。SHALL 先平均曲线后取最小值，禁止平均逐窗最佳分数。官方 offset 与此 k 的映射 SHALL 由单元测试和实际 +5 帧控制验证，不照抄符号。`k0=k*(V_N,N)` 在每条固定，所有候选另报 `D_anchor=curve(k0)`。

SHALL 保存 crop RGB hash、视频/音频绝对时间索引、感受野有效掩码、W、矩阵和 curve。独立验收 SHALL 从矩阵重算 C/D/k，容差 1e-6；与相同输入支持的官方计算 parity 容差 1e-5。显示 Sync-C 为三位小数，统计及胜负按未舍入值计算。

#### Scenario: Audio is delayed by five frames

- **WHEN** 相同 V_N 分别与 N、ND 评分
- **THEN** 按定义 ND 的最佳 k 应接近 k0+5；不得因全局搜索后的 C 未下降判评分器失效，须检查 lag 位移和固定 k0 的距离损伤。

### Requirement: Predeclare measurement gates and bridge benefit

统计 SHALL 以 source group 为单位配对 bootstrap，22 组有放回抽样 10000 次，seed=20260913，等组权重，百分位双侧 95% CI，分位数使用线性插值。所有门槛是本探索协议的操作性阈值，不是指标的普适感知阈值。

A 测量门槛 SHALL 同时满足：

1. N_REPEAT−N 的 C 和 D 差值之组均值 CI 全部落入 [-0.05,+0.05]；至少 20/22 条同时满足 abs(ΔC)≤0.10、abs(ΔD)≤0.10、abs(Δk)≤1。
2. 至少 20/22 条满足 abs(k*(V_N,ND)−k0−5)≤1 且两个最佳 lag 不在搜索边界；延迟的固定 k0 距离损伤 `D_anchor(V_N,ND)−D_anchor(V_N,N)` 之均值 CI 下界 >0.10，至少 18/22 条为正。

B 的两个必要比较 SHALL 为 X∈{N,RT}：`gain_C(X)=C(V_B,N)−C(V_X,N)`，`gain_D(X)=D(V_X,N)−D(V_B,N)`，`gain_anchor(X)=D_anchor(V_X,N)−D_anchor(V_B,N)`。所有量正值代表 B 更好。

`STATIC_BRIDGE_GAIN_OBSERVED` SHALL 仅在完整验收有效且对 N、RT 两个比较均同时满足：gain_C 均值 >0.05 且 CI 下界 >0；gain_D 和 gain_anchor 的 CI 下界均 >−0.10；至少 20/22 条 B 与 X 的最佳 lag 相差≤1。否则为 `NO_STATIC_BRIDGE_GAIN_ESTABLISHED`。这是必须同时通过的两个比较，不从二者择优。V_B/B 只作说明，不进入收益判定。

SHALL 对 B−N、B−RT 分别报告 C 提升、D 改善、C/D 同时改善、持平、下降的计数与比例（分母 22），以及均值、中位数、组 CI、逐样本结果。Mel movement 另外报告，不当作收益成功率。未通过收益门槛不等于证明效应为零。

#### Scenario: Mean is positive but confidence interval crosses zero

- **WHEN** B−N 的 mean gain_C 为正，但 CI 跨零或 B 未胜过 RT
- **THEN** 报告原始正均值和改善比例，同时标记未建立静态 bridge 收益。

### Requirement: Diagnose local swap without equating it to normal speech quality

LOCAL_SWAP SHALL 独立报告 `SWAP_TRANSFER_OBSERVED` 或 `SWAP_TRANSFER_UNRESOLVED`，不得要求 `C(V_S,S)` 与 `C(V_N,N)` 非劣性作为传递前提，也不得以 swap 未通过自动否定 bridge 假设。

对原始两个中间四分之一段 j=2,3，SHALL 使用同一固定 k0、同一组完整感受野不跨原/交换音频拼接边界的绝对窗口 Wj；每段至少 5 个窗口，V_N/V_S 与 N/S 四配对取共同支持。此处不要求所有搜索 lag 都可用，不通过重新找局部 lag 消除交换。SHALL 分段计算：

- `p_N,j = mean_Wj d(V_N,S,k0) − mean_Wj d(V_N,N,k0)`。
- `p_S,j = mean_Wj d(V_S,N,k0) − mean_Wj d(V_S,S,k0)`。

只有两个段的 p_N 和 p_S 共四个组均值 CI 下界均 >0.10，且至少 18/22 条的四项值全部为正，才标记 `SWAP_TRANSFER_OBSERVED`。SHALL 同时报 D0(k=0) 及完整局部曲线作为辅助；辅助曲线每个 lag 另记有效窗口数，无支持值记 null，不用于替换固定 k0 的主量。不得合并中间两段掩盖差异。支持不足时 swap_transfer 为 `SWAP_TRANSFER_UNRESOLVED`、reason 为 `UNRESOLVED_SUPPORT`，不筛掉样本重算阳性。

#### Scenario: Bridge gain and swap response disagree

- **WHEN** bridge 收益条件满足，但 swap 传递条件不满足
- **THEN** 保留 `STATIC_BRIDGE_GAIN_OBSERVED` 与 `SWAP_TRANSFER_UNRESOLVED` 两个字段，解释为静态配自然音频的分数收益而非已确认时序机制；禁止用单一 GO 掩盖分歧。

### Requirement: Deliver independently reviewable evidence and bounded conclusions

实验 SHALL 交付 inputs/protocol、audio/video/score manifests、矩阵及支持映射、逐样本 CSV、analysis、final、validation、报告和可离线打开的播放页。SHALL 提供全部 22 条 N/RT/B 配同一 N 音轨的对比以及 N/S 四配对；未经人类观看时标记 `NOT_HUMAN_REVIEWED`，不凭 SyncNet 宣称观感改善。

独立 validator SHALL 从冻结输入、PCM、实际参考帧、输出和距离矩阵重算全部计数/统计/门槛，不直接信任 runner 的 pass 字段。final SHALL 分开包含 integrity、measurement、bridge_gain、swap_transfer、human_review；缺失、失败和未运行明确区分。桥接结果无论正负，`training_authorized=false`、`generalization_established=false`、`mouth_leakage_proven=false`、`historical_gate_repaired=false`。

#### Scenario: Comparing the new result to historical dynamic scores

- **WHEN** 新静态结果与旧动态 +0.031 均值不同
- **THEN** 仅并列说明，注明生成 ROI、评分 crop 和支持均改变；不得把差值当作静态输入的因果效应或嘴型泄漏证明。
