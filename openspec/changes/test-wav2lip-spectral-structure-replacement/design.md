## Read first

阅读顺序：proposal → 本文 → specs/wav2lip-spectral-structure-replacement/spec.md → tasks。先加载 BM Startup Router、实验指令，读取 `LRS3 natural-to-TTS bridge confirmation result` 和 `Wav2Lip integer plateau control 2026-09-07`。

本轮唯一问题：在未修改的 N 音轨下，MAG 的收益是否需要逐时刻 TTS 谱变化？ENV 仍来自同一配对 TTS，因此不能证明“任何 EQ 都有效”或“配对内容特异性”。MAG/ENV 的扰动范数不强行匹配；差异可包含时变结构、扰动量和重建效应，不能宣称已经隔离纯音素机制。

## Frozen inputs and scope

以下均为文件字节 SHA-256，不是 JSON 内的 artifact_sha256。

| 来源（相对仓库根） | SHA-256 |
|---|---|
| `runs/wav2lip_integer_plateau_control_20260907_all_v2/final.json` | `2c3eb59f6a1ee25916b65b899a3b7ba462dba26b4b1d337e71a062d6e4dfd3fb` |
| 同目录 `protocol.json` | `483e89a5de249bccefdcfd9e376b7a612c92d3f7085f5a47168bdca77d9ad384` |
| 同目录 `generated_validation.json` | `0f91ffa3c5a0e1eb47061af239f23dbac168fda0e6db29fe7a054b0fb9fc443b` |
| 同目录 `audio/manifest.json` | `ed5a8d028bbbfff0c2beb6b67b081fe267fad5e186be76ee8d3a16b122507fc8` |
| `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/protocol.json` | `70580209b05d940ab2e73fae25a4d8a243146bc4f7ccb2f82e66a26d4ad7bef6` |
| 同目录 `cohort.json` | `b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b` |
| `runs/wav2lip_face_roi_replacement_20260906_host_fix1/protocol.json` | `835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4` |
| 同目录 `roi/manifest.json` | `4d005b26e3e108f534597d5cfe4832633e680e325a33332aa98fe99d7f7e6723` |

以 plateau protocol.records 的顺序取全部 22 条/22 source groups；ordered-ID hash 为 `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`。按 sample_id join confirmation cohort.records 的 natural_audio / mfa_linear_audio 与 ROI protocol.records 的 face_video / roi.boxes_path；核对 source_group、N PCM、长度和全部资产 hash。不得按列表位置 zip。ROI manifest 用 rows，plateau audio manifest 用 rows[sample_id].arms.P。

本轮是已反复观察过的 seen-fit 机制试验，既非独立 replication，也非 unseen-source-group 泛化。不读取其他 fit 记录或 sealed 数据。前述旧数据只供输入/掩码及历史边界；所有新视频与分数 fresh，旧分数不并入统计。

## Exact audio construction

N、N_REPEAT 为原 mono 16k PCM16 的逐字节复制。M 为 cohort 绑定的 exact-natural-length MFA-linear PCM；二者以 int16/32768 转 float64。其他三臂在 CPU 使用 torch.stft/istft：n_fft=win_length=1024，hop=256，periodic Hann，center=true，pad_mode=reflect，normalized=false，onesided=true；ISTFT length=L。不增加 VAD、频率平滑或静音筛选。

定义 `S_N=STFT(N)`，`S_M=STFT(M)`，`L_N=log(max(abs(S_N),1e-7))`，`L_M` 同理，`phi_N=S_N/max(abs(S_N),1e-7)`。时间均值包括所有 STFT 列（含边缘反射列），按每个频率 bin 独立计算。

| 臂 | ISTFT 输入 |
|---|---|
| RT | `exp(L_N) * phi_N`（α=0 的重建处理对照） |
| MAG | `exp(L_N + 0.75*(L_M-L_N)) * phi_N` |
| ENV | `exp(L_N + 0.75*mean_t(L_M-L_N)) * phi_N`，均值沿 t 广播 |

ENV 是 time-invariant log-spectral correction；不直接用平均 TTS 谱覆盖 N，也不均值化 N 的时间结构。频域构造保留 N 相位，不声称 ISTFT 后相位/音素时序严格不变。

RT/MAG/ENV 均只执行一次 global RMS match 到 N，随后若 peak>=0.999 再乘 0.999/peak；最后 `np.rint(x*32768).clip(-32768,32767).astype('<i2')`。保存前后 RMS、峰值、两个缩放系数、DC、PCM/container hashes；有限、长度一致、RMS>0、无饱和是工程要求。不得为强行匹配扰动量改变 α 或缩放规则。

P 复用 plateau 音频资产并校验 PCM：前半内部源索引 +3200、后半 −3200，边界规则和 U 支持由其已绑定 protocol 冻结；不重新选拼接点。P 只用于负评分控制，绝不生成 V_P。

Stage 00 冻结五臂 PCM 和官方 Wav2Lip mel 的 shape、MAE-to-N/M、progress、orthogonal_ratio、MAG-ENV mel RMS 差；progress 投影与原 discovery 定义一致。诊断均在本轮评分前完成，不作为样本筛选条件。MAG 能否移动由新数据报告，不把构造 α 当实际 mel 位移。

## Generation and scoring

复用 `wav2lip_face_roi_replacement/generation_worker.py` 的 --audio-json/--outputs-json 接口，参考 plateau runner._run_generation 的轻量调用；不要调用父 runner。每臂/记录独立工作目录和 subprocess，N_REPEAT 必须第二次 forward，不能复制 N 视频。固定每条 face、逐帧 boxes、25fps、224×224、回贴方式及 batch=4；原 face 已在评分坐标，不再次检测、裁剪或重新跟踪，不按音频移动 face。

生成权重 `third_party/Wav2Lip/checkpoints/wav2lip_gan.pth` SHA=`ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`。宿主 V100 16GB，解释器 `/home/wjj/.venvs/wav2lip/bin/python`；先实测 CUDA kernel 并核对 worker.device=cuda，不接受静默 CPU fallback。普通沙箱缺 /dev/nvidia 不是重装驱动理由。记录环境、worker 文件 hash、完整命令；N/N_REPEAT 同参数，不搜索 seed。

无损 FFV1/Matroska，维持原帧数/PTS；mux 用视频流 copy 和完整 mono 16k s16le PCM。音频尾部可长于视频，按父支持窗口评分；禁止 -shortest、截断、重采样或补帧。核对解码 PCM 与源逐字节相等、视频像素/流身份相等。

评分复用 `wav2lip_roi_peak_recheck.worker.SyncNetScorer`，`/home/wjj/.venvs/syncnet/bin/python`、CPU、batch=20、threads=4。权重 SHA=`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`。每 cell 保存 fresh embeddings、[T,31] 距离矩阵、媒体/模型/配置绑定。必须传该 cell 的真实 source_audio，N/P cell 传 P。

固定评分行 `U=plus_rows || minus_rows`，来自 plateau protocol；所有自然音轨臂使用完全相同 U（自然目标时间）。不得改用 q/source rows、候选最优片段或各臂交集重选。先审计 U 对各 cell 全部 31 offset 列的真实支持，无零填充/边界嵌入；不支持则 BLOCKED。

对矩阵先在所选行做 float64 均值，得到 31 点曲线 z；`D=min(z)`，`C=median(z)-min(z)`，`offset=15-argmin(z)`，并列取最小列号。PLUS、MINUS 同法单独汇总用于控制；主收益只用 U。不要先逐行求 C 再平均。这是局部窗口 SyncNet endpoint，不是历史整条视频 endpoint；只能描述比较历史数值，不能称原 +0.078 的原协议复现。

## Two stages and controls

Stage A：每条生成 N、N_REPEAT、RT，评分 N/N、N_REPEAT/N、RT/N、N/P（共 66 视频、88 cells）。固定顺序按 record 再 N/N_REPEAT/RT。全 22 条完成后独立验收 A；以下任一科学控制失败，完整写报告，Stage B 不执行。

对任意 X/Y（均参考 N），`delta_C=C_X-C_Y`，`delta_D=D_Y-D_X`，正值更好。

- Repeat 与 RT：各自相对 N 的 C、D 均值的双侧 95% CI 必须严格包含在 (−0.05,+0.05)，U offset 差<=1 帧至少20/22。这里使用双侧等效性而非单侧非劣性，防止生成/重建波动被当成微小收益；±0.05 是本轮预先固定的精度要求，不修改旧阈值。报告逐条视频像素差和 mel 差，不要求像素完全相同。
- 敏感性：N/N 的 PLUS/MINUS 峰都清晰（次小−最小>0.010，|offset|<15）且两段 offset 差<=1，至少20/22。N/P 相对 N/N 的 PLUS offset 应为 +5、MINUS 为 −5（误差<=1、两边峰清晰），两段都满足至少18/22。U 上 `damage_C=C(N/N)-C(N/P)`、`damage_D=D(N/P)-D(N/N)` 的95% CI 下界都>0.10，且两者都正至少18/22。

本轮负控制仅检验同一视频的音轨错配可检测；不证明生成器对错时音频等变，也不解除旧 bridge 的禁用状态。

Stage B：仅 A 科学通过且独立验收 valid 后生成 MAG、ENV，分别只评分 MAG/N、ENV/N（44 视频、44 cells）。无候选 own-audio 附加评分。总预算上限为110视频/132cells，不含因工程故障留下的无效临时产物；不追加科学重复。

## Statistics and terminal decision

固定22 groups，按 source_group 排序，对 group 均值有放回 cluster bootstrap，PCG64 seed=20260908，10000 draws，linear quantiles；所有对比使用相同抽样索引。缺失/非有限值不丢弃，工程 BLOCKED。原始全精度判定，报告 C 显示3位小数。

控制用95% CI（2.5/97.5百分位）。候选结论有两条可能的正面路线，统一用双侧97.5% CI（1.25/98.75百分位）作保守的两路线校正；每条路线内部所有条件为 conjunction。另附95% CI仅作描述，不混用判定。

定义 `gain(X)`：相对 N 和 RT **分别**满足 C 的97.5% CI下界>0且均值>0.05、D 的97.5% CI下界>−0.10，且两项对比的 U offset 差<=1 各至少20/22。不能只胜过 N_REPEAT 或只检验 MAG−ENV。

科学终态按以下顺序唯一确定（独立工程验收失败优先 BLOCKED）：

1. A 任一控制失败：`CONTROL_FAILED`，不运行 B。
2. `gain(MAG)` 且 MAG−ENV 的 C CI下界>0、D CI下界>−0.10，offset 差<=1 至少20/22：`TEMPORAL_SPECTRAL_INCREMENT_SUPPORTED`。
3. `gain(ENV)` 且 MAG−ENV 的 C、D 的97.5% CI均严格位于(−0.05,+0.05)，offset 差<=1 至少20/22：`AVERAGE_SPECTRUM_SUFFICIENT_IN_SCOPE`。两条同时成立则按第2条，并保留第3条布尔值。
4. 任一 gain 为真但2/3不满足：`GAIN_WITH_MECHANISM_UNRESOLVED`。
5. 其他：`NO_USEFUL_GAIN_ESTABLISHED`。意味着本轮没有确认，不证明总体无效。

没有显著差异不等于两臂等效。即使 MAG 胜 ENV，也只支持该固定构造的增量收益；缺少错配 TTS、扰动范数匹配和独立 source-group 对照，不声称纯内容特异性。

所有终态：`training_authorized=false`、`generalization_established=false`、`historical_gate_repaired=false`。第2/3类最多建议另写独立 source-group 确认 spec；本轮不自动执行。第4类只报告不确定性，不能自动加臂/扩样本；第1/5类停止当前声学构造。本轮新生成 MAG 不算执行旧 bridge runner，须分别记录新 candidate execution 和 `legacy_bridge_executed=false`。

## Small implementation and acceptance

建议一个小包：runner.py（协议/编排/断点）、audio.py（三公式）、analysis.py、validate.py，可按需加 config.py。不复制旧流水线。可复用音频 I/O 与 MAG 构造；旧 blend helper 不接受 α=0，RT 需本地实现，不改旧全局 strengths。允许共享纯 I/O，独立 validator 不调用 producer 的候选构造、聚合、门禁或 bootstrap 函数。

输出 `runs/wav2lip_spectral_structure_replacement_<id>/`：protocol.json、input_audit.json、audio/manifest.json、videos/manifest.json、scores/manifest.json、control_analysis.json、control_validation.json、analysis.json、validation.json、final.json、result.md；各阶段绑定 spec/代码/输入/产物 hash。A 验收文件冻结且独立存在，不被 B 验收覆盖。避免 final/validation 循环哈希（先分析→验收→final绑定验收；验收本身不绑定final）。最终另核对 final 与已验收结果一致。

validator 从原 N/M 独立重建 RT/MAG/ENV（float容差1e-9、PCM逐样本完全相同），从保存的 embeddings 独立重建矩阵（abs<=1e-4），复算曲线/统计（abs<=1e-6）、所有 gates、全部 counts/来源/终态；媒体解码核对由共享 I/O 完成。只在实际执行的阶段要求完整产物。

使用原子写入、唯一 cell key=(sample_id,video_arm,audio_arm)，仅当完整绑定匹配才复用本轮已成功 cell；不同 run 的分数不缓存进来。工程中断可 resume，科学 CONTROL_FAILED 不续跑 B。磁盘至少15GiB空闲才开始 GPU，每条前复查；不足则保存断点 BLOCKED，不自动删除旧 runs 或媒体。GPU工作遵循宿主权限，不请求重装环境。

下列为下游待实现命令，现阶段只提交 spec：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_spectral_structure_replacement.runner --run-id <id> --stage all
# 工程中断后，仅接受绑定一致的已完成 cells
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_spectral_structure_replacement.runner --run-id <id> --stage all --resume
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_spectral_structure_replacement.validate --run-root runs/wav2lip_spectral_structure_replacement_<id>
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m pytest -q tests/experiments/wav2lip_spectral_structure_replacement
openspec validate test-wav2lip-spectral-structure-replacement --strict --no-interactive
```

BM 同一实验笔记 planned→running→concluded，工程阻塞则写实际状态和可恢复断点。每次更新先搜再全文读，写后读回，记录结论/数字/产物指针/边界；不要把 spec 设计写成已经成功。交付前自审 PCM、样本 join、U/Q、C/D 符号、两种CI、等效性和旧门禁隔离。
