## Purpose

在同一批既有 LRS3 媒体上交叉比较真实视频、自然音频驱动视频和局部变速音频驱动视频，定位控制失败发生在音频扰动可检测性还是生成视频响应环节。结果用于决定后续诊断方向，不构成 TTS 收益或训练准入证据。

## ADDED Requirements

### Requirement: Freeze existing assets before scoring

实验 SHALL 使用以下只读输入，以文件 SHA-256 校验入口并沿 manifest 验证所用媒体、模型与裁脸轨迹；文件 hash 与 JSON 自哈希 SHALL 分开核验：

| 输入（相对仓库根） | 文件 SHA-256 |
|---|---|
| `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json` | `b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b` |
| `runs/lrs3_local_timing_control_calibration_20260905_v5/05_final/final.json` | `318054ecc86aab5e0ea9b19fa9d5a6e0bcd81181ec200ac9e23f6abca0c08d8b` |
| `runs/lrs3_local_timing_control_calibration_20260905_v5/02_audio/audio_manifest.json` | `0617cb895574516b77aa430c44f0480eefe4be0e0582b3e8fdec4addf46e68e1` |
| `runs/lrs3_local_timing_control_calibration_20260905_v5/03_videos/videos_manifest.json` | `5157148c2aed4f797beffa2c832784991ba474c58e86e6391b1f01040c38e941` |
| `runs/lrs3_real_video_local_timing_20260905_tail_v2/final.json` | `5df3615f7b2219e60ad62fb49792cb30eccd068441d1ab5b8006d0fb92aee24d` |

SHALL 原样保留 cohort 顺序、22 records / 22 source groups；ordered-ID hash 为 `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`。标记 `fit_only_diagnostic`。N 是 natural PCM，W 是历史 `LOCAL_WARP_120` PCM；R 是 cohort 的真实视频，G_N/G_W 是 calibration v5 的 N/LOCAL_WARP_120 生成视频。

审计 SHALL 从 N 独立复建 W 并核对解码 PCM：长度 L，`s[n]=n+1920*sin(2*pi*n/(L-1))`，端点强制 0 和 L−1，CPU float64 线性插值、nearest-even 舍入到 int16；`L-1>2*pi*1920`，s 严格递增。该复建只作验证，不替换历史 W。N/W 均为 16 kHz mono PCM16、长度相同；不能把旧 manifest 中名为 pcm_sha256 的字段当作已验证的 PCM hash。

新评分前 SHALL 审计全部 22 条，冻结资产、时间轴、共同窗口、裁脸轨迹、本 spec 文件 hash、执行源码/环境/模型 hashes 及所有参数。任一项缺失或不匹配为 BLOCKED，不自行替补或重建历史输入。

#### Scenario: Historical bindings do not match

- **WHEN** 媒体损坏、跨记录配对、W 复建不一致或依赖链无法核实
- **THEN** 输出逐条审计证据及 BLOCKED，禁止在剩余子集评分或选择另一个历史 run

### Requirement: Use one timeline and crop policy across the six cells

SHALL 保留每个输入的完整 PCM 和全部视频帧。视频均为 25 fps，PTS 严格递增且相对首帧与 i/25 的偏差 ≤1 ms，各输入时间零点差 ≤1 ms；验证生成命令以源视频首帧和指定音频首样本开始，禁止估计 offset 后对齐。

真实视频沿用 `bounded_audio_tail_v2`：整数 `-640 <= L-640*F_R <= 1280`。生成视频使用其已绑定 manifest 的精确帧数，实际解码数必须相等，且 `F_GN=F_GW<=F_R`、画布宽高与 R 一致。生成视频不套用真实视频的一帧/两帧尾差上限；逐条记录 `L-640*F_G`。不补帧、不截音频、不循环视频；只有统计窗口取共同支持。

SHALL 从 tail_v2 protocol 读取并验证 R 的 `crop.processed_track`，R 使用完整轨迹，G_N/G_W 使用同一轨迹的前 F_G 项。三种原视频分别按该轨迹裁到 224×224，crop_scale=0.40，沿用父实验的像素处理和无损编码设置；不对生成视频重新检测、重选人脸或替换裁脸策略。相同视频搭配 N/W 时 SHALL copy 同一裁后视频流，mux PCM s16le，解码 PCM 与指定 N/W 完全一致；禁止使用生成 MP4 自带的有损音轨。

每条 SHALL 产生如下 6 个新 mux / 新评分 cell，共 132 个；另外对 R/N、G_N/N 各在独立进程和临时目录重新评分一次，共 44 个重复评分 cell、176 次评分。重复评分可以读取同一 mux，不能复制矩阵或分数；本轮不检验生成器重复性。

| 视频 | 音频 N | 音频 W |
|---|---|---|
| R | 真实基线 | 固定真实视频上的音频扰动 |
| G_N | 生成基线 | 固定生成视频上的音频扰动 |
| G_W | 生成响应相对自然音频 | 扰动驱动视频的自身配对 |

#### Scenario: Generated videos end earlier than natural audio

- **WHEN** 已绑定 G_N/G_W 帧数相等且仅覆盖 R 的前缀，PCM 尾部更长
- **THEN** 保留全部媒体，采用后述共同有效行；不因超出真实视频尾差上限而阻塞，也不按短视频重新定义音频变换周期

### Requirement: Predict local offsets from the audio map

SHALL 冻结官方 SyncNet V2 模型 SHA-256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`、官方前向预处理、batch_size=20、vshift=15，保存每个 cell 的未舍入 `[window,31]` distance matrix、绝对窗口起始帧、原始日志和执行绑定。列 k 的 offset=`15-k`；全矩阵均值曲线重建的 C/D 与官方日志偏差 ≤0.001，offset 必须一致。新矩阵全部重算，不混入旧评分。

局部统计 SHALL 使用同一绝对行集合：`Q=min(F_R,F_GN,F_GW,floor(L/640))`，候选行为整数 `15 <= r < Q-20`。逐 cell 核对矩阵实际行数为 `min(F_video,floor(L/640))-5`，并验证所选行的五帧画面和全部 31 个音频 lag 窗口有真实支持；不使用边缘 padding。每个 cell 原生全局 C/D 另行保留，跨视频比较 SHALL 使用共同候选行均值曲线导出的 C/D，不能比较长短不等的全局窗口。

SHALL 将已冻结的离散 s 按分段线性方式延拓到实数坐标，逆映射为交换坐标后的分段线性插值。令窗口中心 `t_r=640*(r+2)`：

```text
d(r) = (s(t_r)-t_r)/640
a(r) = (t_r-s_inverse(t_r))/640
PLUS  = 候选行中 d(r)>=2.5 且 a(r)>=2.5 的行
MINUS = 候选行中 d(r)<=-2.5 且 a(r)<=-2.5 的行
```

两段每段至少 5 行，否则 BLOCKED。所有 6 个 cell 及重复评分共用这两个掩码，不能按分数择窗。每段曲线为对应矩阵行的均值；最小值与次小值之差严格 `>0.010`，且 argmin 不在 ±15，才是清晰峰。并列最小为不清晰。

SHALL 逐段使用以下相对 offset 预测（基线分别相减，保留自然固有 lag）：

| 检查 | 实测变化 | 预期变化 |
|---|---|---|
| A：真实域音频敏感性 | off(R/W)−off(R/N) | mean(a(r)) |
| B：生成域音频敏感性 | off(G_N/W)−off(G_N/N) | mean(a(r)) |
| C：生成视频响应 | off(G_W/N)−off(G_N/N) | −mean(d(r)) |
| O：自身局部对齐 | off(G_W/W)−off(G_N/N) | 0 |

mean 只对本段冻结行计算。SHALL 用同一张表解释两段的相反方向；不能将“音频 warp”直接套用上一轮“视频 warp”的符号。预测是固定诊断近似，1 帧容差不允许事后调整。

#### Scenario: Output audio reads later natural samples in PLUS

- **WHEN** PLUS 中 W 在时刻 t 读取较晚的自然音频，d 和 a 均约 +3
- **THEN** A/B 的预期 offset 变化约 +3，而跟随 W 的生成画面在 C 中预期约 −3；符号反转测试必须覆盖此情况与 MINUS

### Requirement: Apply one conservative decision tree

工程缺陷 SHALL 产生 `engineering_decision=BLOCKED`、`scientific_decision=null`。176 个有效评分完整后，工程决定为 GO，科学决定 SHALL 按以下优先级执行：

1. 两个基线各自与独立重复评分的完整矩阵最大绝对差 ≤0.001，全局及两段 offset 相同，22/22 均满足；否则 `REPEATABILITY_FAILED`。
2. 每个基线的两段峰均清晰且 PLUS/MINUS offset 相差 ≤1，记为该基线可解释。R/N、G_N/N 各至少 20/22 可解释；否则 `BASELINE_INCONCLUSIVE`。
3. 对 A/B/C/O 分别计数：相应基线可解释、被比较 cell 两段峰均清晰、两段实测变化与表中预期之差绝对值均 ≤1，则该条成功。所有计数分母固定 22；每项至少 18/22 才通过。
4. A 未通过 → `AUDIO_CONTROL_UNRESOLVED`；A 通过而 B 未通过 → `GENERATED_ENDPOINT_UNRESOLVED`；A/B 通过而 C 或 O 未通过 → `GENERATED_RESPONSE_UNRESOLVED`；A/B/C/O 全通过 → `LOCAL_RESPONSE_ESTABLISHED`。

SHALL 报告所有可计算的计数/误差，即使更早的科学 gate 失败，但早期 gate 失败时不得将下游低计数当成原因证明。C 失败只能说明在该固定生成/裁脸/评分条件下未建立预期响应，不能证明 Wav2Lip 完全不听音频。

SHALL 另外报告共同候选行上的四个配对差值：自身 `C(G_W/W)-C(G_N/N)`、`D(G_N/N)-D(G_W/W)`；替换 `C(G_W/W)-C(G_W/N)`、`D(G_W/N)-D(G_W/W)`。报告均值及 95% source-group bootstrap CI（22 个组排序、有放回抽 22 组、10,000 draws、NumPy default_rng/PCG64 seed=20260905；每项重置 seed，线性分位数 2.5/97.5）。这些仅描述，不参与本轮 gate，也不代替旧控制判定；Sync-C 显示三位小数。

#### Scenario: Timing follows but confidence still falls

- **WHEN** A/B/C/O 全通过，而自身配对共同窗口 C/D 仍退化
- **THEN** 输出 LOCAL_RESPONSE_ESTABLISHED，说明局部时间响应与全局分数损伤可以并存；不把剩余差异直接命名为纯 acoustic mismatch

#### Scenario: A boundary or unclear peak is observed

- **WHEN** 峰间距等于 0.010、峰在搜索边界或误差大于 1 帧
- **THEN** 相应记录不算成功，但保留在 22 条分母；达到 18 条则满足计数门槛

### Requirement: Deliver an auditable diagnosis and stop

SHALL 在新 `runs/lrs3_wav2lip_timing_transfer_<run_id>/` 交付 `input_audit.json`、`protocol.json`、媒体/评分 manifests、176 份矩阵/日志/执行证据、逐记录分析、`final.json`、简短 `result.md` 和离线 `validation.json`。final 绑定 spec、父终态、审计、协议及全部下游 manifests hashes，给出 132 个主 cell、44 个重复 cell 的实际/期望数量、gate 与原因；始终 `reference_conditioned_audio_head_spec_eligible=false`。

离线 validator SHALL 从实际媒体、时间映射、矩阵独立复算掩码、统计、计数和判定，不调用 producer 的 gate/统计函数或只相信 pass 标记。SHALL 检测缺失 cell、交换音轨、篡改 hash/PCM/PTS、错符号、分母缩小、篡改 final；工程失败终态也须验证其失败证据，不能伪装为完整矩阵。

所有旧 runs 保持只读。正常 resume 只接受输入/代码/配置/输出身份全部一致的完整 cell；部分或终态记录不得覆盖，工程修复另起 run 并绑定旧失败记录。科学失败不得改幅度、掩码、门槛、裁脸或重新挑 run。本轮不运行 Wav2Lip/TTS、训练、MFA/DTW、bridge 重测、heldout/sealed evaluation，不自动提交 git。

完成后 SHALL 按 Startup Router 的实验指令，在 BM `tts-exp` 搜索并记录同一份本实验笔记（已有则更新），绑定报告、结论、局限及后续方向，frontmatter 与唯一 status 观察一致。SHALL 说明历史 full-frame Wav2Lip 生成 geometry、插值带来的声学变化及固定裁脸可能影响结论；上一轮盲审 PENDING 不代表本轮已有主观证据。本轮无需新增盲审包，数值验证完成即可结束。

#### Scenario: All diagnostic gates pass

- **WHEN** LOCAL_RESPONSE_ESTABLISHED 且独立 validator 为 valid
- **THEN** 交付诊断后停止，只建议另立控制/bridge 确认 spec；历史 CONTROL_FAILED、bridge 增益未通过及训练不准入保持原结论
