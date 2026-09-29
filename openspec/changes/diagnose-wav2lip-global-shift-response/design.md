## 下游快速入口

按 proposal → 本文 → specs/wav2lip-global-shift-response/spec.md → tasks 阅读。实现并运行本诊断即可；不要顺手开始语义模型。本文为数值和执行合同，spec.md 为验收要求。允许科学结果为阴性或无法解释。

先读 BM Startup Router、实验指令，搜索并全文读 `Wav2Lip replacement endpoint reconciliation 2026-09-08`、`Wav2Lip spectral structure replacement 2026-09-08`。最新谱迁移的 NO_USEFUL_GAIN_ESTABLISHED 继续有效，不能外推为所有自然音频变换无效。

## 1. 输入与范围

路径相对仓库根；下列 SHA 为文件字节 SHA-256。

`H = runs/lrs3_phase_preserving_replacement_envelope_20260904`

`S = runs/wav2lip_spectral_structure_replacement_20260908_cpu_prepare_v3`

| 输入 | SHA-256 |
|---|---|
| H/00_protocol/cohort.json | 850c224856bc7acb95aa709a18f4c0b3369ca724d8759603d1c4bf3aac74d72a |
| H/01_candidates/audio_manifest.json | 628e9e16ec4708421c9db57f636ad2871123bb0fddd5db3e33f611c109ad00cb |
| H/03_videos/videos_manifest.json | 0008acd3da7a7ec0066ce181b9c7204d34569d27e0aa4f30a92326b4561ac2bc |
| H/04_scores/scores_manifest.json | da7bac4f19903954573752abf029a65265029fe5462c9434a21659112abcf788 |
| H/05_final/final.json | 9bab2a853edda0cdb16e79e2b9a8c6d32ce3e36d721e4a1c1140306d9ffac8c1 |
| S/protocol.json | 284dea1b07fbccc07a309512b164265d215075af41b8009b38d163362d502cfa |

H 固定全部 23 条，仅 CPU 读取三类 cell：`V_N/A_N`、`V_SHIFT_200/A_N`、`V_SHIFT_200/A_SHIFT_200`，共 69 矩阵。不读 MAG 成绩选样。

新生成从 S/protocol.records 按 `(source_group,sample_id)` 升序取前 12 条。应逐项匹配以下列表，每条来自不同 source_group：

```text
lrs3_6ORDQFh0Byw_00008
lrs3_6SdtkXAQq3k_00009
lrs3_6VnKV1sr5VQ_00008
lrs3_6WeS1bXRBOk_00006
lrs3_6qqqVwM6bMM_00007
lrs3_6tSlMoMNSlY_00009
lrs3_6ul2TSvUDog_00007
lrs3_6weGCM3sWKc_00014
lrs3_6wk4dkYSrV0_00006
lrs3_6xtmm0MnaS0_00010
lrs3_6xy5pWOgeBY_00010
lrs3_6yR5OUVb2gY_00008
```

只读取这些记录的 `natural_audio`、`face_video`、`roi`、`frame_count`、`sample_count` 及其引用资产；不需要 M、MAG、ENV 或 plateau 音频。当前最短视频 147 帧。全体都是已见 fit 数据；H 与新 cohort 不按记录配对，不称独立确认或 unseen-source 泛化。

prepare 先冻结源 PCM／容器、face、ROI、模型、执行源码和本文/proposal/spec 的路径及 SHA；tasks 不绑定。按 key join，禁止 zip。输入缺失、重复、hash 错误或时钟支持不足，明确 BLOCKED，不换样本。

## 2. 两个阶段，固定预算

### A：历史缓存审计（CPU，无新 forward）

通过 `scores[].(sample_id,cell)` 和 reference 定位：
`H/04_scores/syncnet/<reference>/pywork/<reference>/{tracks,activesd}.pckl`。

先 hash 再加载可信本地 pickle；没有父 hash 的缓存标记 `legacy_cache_locked_at_audit`。要求 1 track、1 个 finite float32 `[T,31]` 矩阵、track.frame 从其实际起点 f0 连续；矩阵行数可以少于 track 帧数。多 track 不取最好的一条。核对 mux video 来源和 mux audio PCM；不能要求生成视频原来内置的音轨等于 replacement 音轨。

核对历史 SHIFT PCM 确为第 3 节的 DELAY_200。69 个 FULL 使用原 torch float32 reduction 复现日志 C/D（abs≤0.000501）、offset 精确一致。用历史 manifest 三位小数及原 bootstrap 口径复算原 SHIFT 统计（abs≤1e-6）。历史矩阵行 r 对应原视频帧 g=f0+r，每记录三个 cell 共用绝对窗口 `I_H=range(max(f0_cells+30),min(f0_cells+T_cells-30))`，在每个 cell 用行 `r=g-f0_cell` 取值，计算第 4 节指标。FULL/I_H 各 69 个 endpoint。I_H 必须非空；验证原 pipeline 按同一 track 起点裁切音视频的时间映射，保存本地行/绝对帧双坐标，不能通过移动波形修正。

设计时只读结构预检已发现：`lrs3_6ZiN9ZJT294_00005` 三个 track 都从 frame32 开始；`lrs3_73rUjrow5pI_00005` 的 N track 从0开始，两个 SHIFT track 从20开始。其余结构由下游重新校验。必须报告这些裁剪差异；不能直接按同一局部行号配对，也不能因非零起点就删记录。共同帧支持消除不了 crop/frontend 差异，H 的 anchored/offset 仍是原 track 评分坐标的诊断，不能宣称物理时间链完全等价。

说明 FULL→I_H 包含共同支持与边界排除，不唯一证明 padding 原因；历史处理链和新 ROI 链不同，不将两阶段差异直接归因于静态／动态人脸。parity 或来源审计失败则停止 GPU 阶段，保留 discrepancy。

### B：人脸设置 × 偏移（GPU 生成，CPU 评分）

两个设置：

- `DYNAMIC`：原始 face 前 F 帧与原 ROI boxes 前 F 项。
- `STATIC`：重复原始 face 第 0 帧 F 次，ROI boxes 重复原第 0 项 F 次。不能把第一帧和动态 boxes 混用，不能按画质／嘴型挑帧。两种设置的输入均统一解码成 BGR，再用同一 FFV1 无损流程写入；STATIC 每帧 decoded pixel hash 必须等于其第 0 帧。DYNAMIC 与原前 F 帧逐像素一致。

F 沿用 S/protocol 的 frame_count。每设置 fresh 生成 N、N_REPEAT、DELAY_200、ADVANCE_200；N_REPEAT 用相同 N PCM 单独调用生成进程，不复制视频。不同设置各有自己的 N baseline，不直接以 STATIC 的绝对 C 减 DYNAMIC 的绝对 C 当干预收益。

每记录、每设置的唯一评分矩阵如下，cell key 必须包含 face_mode：

| 生成视频 | 评分音轨 |
|---|---|
| V_N | N、DELAY_200、ADVANCE_200 |
| V_N_REPEAT | N |
| V_DELAY_200 | N、DELAY_200 |
| V_ADVANCE_200 | N、ADVANCE_200 |

共 12×2×4=96 个生成视频，12×2×8=192 个新评分 cell。另有最多 24 个派生输入 face 流；mux 文件不算额外生成视频，但须计数。新矩阵 FULL/I 各一份，共 384 endpoint；A/B 不合并统计。

先完成全部 N/N_REPEAT 和 V_N 三音轨 cells（48 视频、96 cells），验证重复性及 same-video 音频偏移可检测性。随后无论这些科学诊断是否通过，都完成既定 shifted-driver cells（再 48 视频、96 cells），因为这次要观察生成响应而非修复旧 own gate；工程失败仍立即停止。不得增加新臂或按效果重跑。

## 3. 精确音频与运行合同

令 N 为 mono/16k/PCM16、长度 L，移位 d=3200 samples：

```text
N_REPEAT = N（相同 PCM）
DELAY_200[n]   = N[n-d]，越界取 0   # +5 视频帧延迟
ADVANCE_200[n] = N[n+d]，越界取 0   # -5 视频帧提前
```

使用整数数组切片；所有候选长度均为 L。不使用 np.roll、插值、重采样、淡入淡出、RMS 匹配或循环边界。shift 只改 driver 内容，不改 WAV 时间戳、video PTS、mux 起点。replacement 音轨始终为整条 untouched N，不能同步移动回来，也不能因 `-shortest` 截断 N。own-audio 与 same-video 错配 cells 按表精确绑定相应 PCM。

生成复用 `wav2lip_face_roi_replacement/generation_worker.py`，batch=4，冻结 `wav2lip_gan.pth`；checkpoint SHA=`ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`。复用 `wav2lip_roi_peak_recheck.worker.SyncNetScorer` 的 JPEG/MFCC/矩阵处理，CPU batch=20、threads=4；SyncNet SHA=`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`。帧必须为 224×224、25fps；不重新做人脸检测／tracking／最佳 crop 选择。这是固定裁剪 SyncNet V2 endpoint，不能声称与旧官方整段检测 pipeline 完全相同。

所有臂复用完全相同的 mel chunking、色彩、回贴、codec、线程和评分路径。检查每臂生成帧数=F、25fps PTS 从 0 连续、FFV1 mux video stream-copy、评分前后 decoded PCM exact match。每 cell 独立目录；缓存 embeddings 可以在同一绑定输入之间复用，但每个 cell 都要保留完整矩阵与来源，不重复启动无谓 forward。

Wav2Lip 用宿主命名空间 `/home/wjj/.venvs/wav2lip/bin/python`，先做真实 CUDA kernel smoke；普通沙箱看不到 GPU 不能被诊断为驱动坏了。CPU 主进程用 `/home/wjj/.venvs/syncnet/bin/python`。单 GPU 串行生成，不租云卡，不重装环境，不静默退回 CPU。运行前按实际 F 估算无损媒体及 JPEG 空间，至少保留 15 GiB 余量；不足就报告所需空间，不清理用户资产。

## 4. Endpoint 与时间方向

Scorer 返回 T=F−5 行（不是 F−4）；各 cell 音频长度相同且支持完整，预检确认 T 一致。新每记录 16 个矩阵共用 `I=range(30,T-30)`，不沿用 plateau 的 U/Q。30 帧保守余量排除 ±5 帧 shift、±15 offset、mel/MFCC 窗口的首尾支持影响；prepare 必须显式验证所有 I 行、31 列及生成 mel 窗口的原始 PCM 支持均在有效范围内。源样本映射为 DELAY 的 n−3200、ADVANCE 的 n+3200，不能计入移位补零或末端重复 mel chunk。不得运行后按分数修剪 I。

FULL 为每个矩阵全行，仅描述性；I 为主诊断窗口。对每个 R 先求 float64 `z=mean(matrix[R,:],axis=0)`，再求：

```text
D = min(z)
M = median(z)
C = M - D
offset = 15 - argmin(z)    # 并列取最小列号
```

不要逐行算 C 后平均。每 face_mode/R 用 V_N/N 的 offset=o0 固定自然基线坐标；所有该设置候选同时报告 `D_anchor=z[15-o0]` 和 `C_anchor=median(z)-D_anchor`，不为候选重新选择 anchor。

每个 shifted-driver 的主配对比较为 `(V_shift/N)-(V_N/N)`：`benefit_C=C_shift-C_N`、`benefit_D=D_N-D_shift`、`benefit_anchor=D_anchor_N-D_anchor_shift`；正值为好。另报告 `delta_M`，校验 `benefit_C=delta_M+benefit_D`。自动 offset 搜索后 D/C 上升而固定坐标没有改善，不能称为原时间轴同步改善。

设 driver shift δ=+5（DELAY）、−5（ADVANCE），用合成 embedding 序列验证下列符号：

| 比较（同 face mode 的 I） | 理想 offset 变化 |
|---|---:|
| V_N/A_shift 相对 V_N/N（只动音轨） | −δ |
| V_shift/N 相对 V_N/N（生成画面完全跟随） | +δ |
| V_shift/A_shift 相对 V_N/N（两侧跟随） | 0 |

峰清晰定义：second_min−min>0.010 且 argmin 不在 ±15 边界。误差容忍±1帧；不清晰计入失败/不确定总数，不丢样。own C/D 可以下降，不能用其下降证明模型没响应，也不设置旧 own-audio 非劣性门禁。

## 5. 一个轻量视觉响应辅助量

从无损生成帧按该设置的 ROI crop，OpenCV resize 到96×96，取下半部48×96 BGR 作为固定区域，不新增 landmark 模型。对两种 shift 分别计算 `E(l)=mean(abs(patch_shift[r]-patch_N[r-l]))`，float32 0..255，固定 l∈[-8,8] 整数、r 为 I 的起点帧。最小 E 的 l（并列取较小 l）、E(0)、E(δ)、V_N_REPEAT 对 V_N 的 E(0) 全部保存；参考序列变化量为0时 best lag=null。所有滞后都报告，不据此改变 driver 或重对齐评分。

这是像素响应描述，不是嘴部运动真值：动态输入同时改变姿态、表情、ROI；STATIC 也改变视觉条件分布。只有动态组出现正向／两组 offset 响应不同，只能说明 face-mode 依赖，不能唯一认定 mouth leakage。每组前两条固定 ID 输出拼排视频用于人工核查，不挑最好案例，不自动生成主观改善评分。

## 6. 统计、诊断标志与收口

新12组等权；sorted source_group、PCG64 seed=20260908、10000 次配对 group bootstrap，共用保存的 `[10000,12]` draws；percentile 95% CI、linear quantile。历史新派生 I_H 使用独立 `[10000,23]` draws；复现历史原报告时保持其原 seed／精度。所有 C/D 报告均值、CI、正向数、joint wins 和 offset 差，不以 frame 数充当样本量。

完整报告两种 face_mode×两种 shift、FULL/I 全部结果，另报告每个 shift 的 `(benefit_DYNAMIC-benefit_STATIC)` 和 `benefit_I-benefit_FULL` 的配对 CI。不能由“一边显著另一边不显著”推出 interaction 显著。所有新 CI 都是探索性、多比较未校正，不发布 confirmatory gain。

每 face_mode 的诊断标志固定如下，只使用 I：

- `repeat_stable`：N_REPEAT−N 的 C、D-benefit 的95% CI均完全在[-0.05,+0.05]内，12/12 offset相同。
- `audio_shift_detectable`：两个音轨偏移方向各至少10/12满足清晰峰和−δ±1预测；重复性失败时标记 uninterpretable。
- `generated_shift_follows`：两个生成偏移方向各至少10/12满足清晰峰和+δ±1预测；前述重复性／音轨可检测性不满足时标记 uninterpretable，不能直接标生成器无响应。
- 每方向 `exploratory_positive_signal`：repeat_stable 且 audio_shift_detectable；benefit_C均值>0.05，C、D、anchor benefit的95% CI下界均>0，且至少10/12候选与基线offset差≤1。只用于标记是否值得另立独立复验，不是成功门禁；不得从4个比较挑赢家当确认。

终态保持简单：工程／独立验收失败为 `engineering_decision=BLOCKED, diagnostic_decision=INCOMPLETE`；工程完整则 `engineering_decision=GO, diagnostic_decision=COMPLETE`，无论上述标志是否成立。始终 `scientific_decision=NOT_A_CONFIRMATION, training_authorized=false, generalization_established=false, historical_gate_repaired=false`。工程失败时 scientific_decision 改为 `not_available`。

结果必须回答：历史数字是否复现、内部窗口是否保留正向、正确时间坐标是否改善、两种人脸设置如何影响响应、生成画面是否跟随、哪些原因仍未区分。所有分支 `next_action=STOP_AND_REVIEW`；不自动扫描延迟、重试旧控制或训练。语义辅助是后续独立问题：保留完整 natural，比较正确内容／错配内容／无辅助，并必须超过完整 natural baseline，不能以 masked reconstruction 恢复为成功。

## 7. 最小实现与验收

建议仅 runner.py（阶段/媒体计划）、analysis.py（纯函数）、validate.py（独立复算）、config.py，按需加 helpers。复用上文明确的 worker 与 mux 函数，不导入旧 runner 自动执行旧科学门禁；父代码常量含22条或U窗口，不能直接用于新12组统计。

顺序：prepare→history→generate/score→analysis→validation→final。prepare仅CPU输入／候选构建，不跑网络模型；history审计通过才进入GPU。新目录 `runs/wav2lip_global_shift_response_<id>/`；可 --resume 复用输入/代码/输出hash都一致的完整cell。发生源码变更用新run-id；失败部分不伪称完成，不覆盖历史或用负结果触发重试。

最少产物：protocol.json、input_audit.json、history.json、audio/media/scores manifests、每cell embeddings和matrix、endpoints.json、visual_response.json、bootstrap_indices.npy、analysis.json、review.json、validation.json、final.json、result.md。计数须分别列实际生成、派生输入、mux、评分cell、网络forward和缓存复用，不能把预算当执行数量。review.json如实注明自审或独立reviewer，不伪称subagent审查。

validator独立从绑定缓存/embeddings复算矩阵及FULL/I、C/D/anchor、bootstrap、标志和计数；不调用producer统计或判定函数。float32距离重建abs≤1e-4，float64派生统计abs≤1e-6。验证PCM移位、STATIC逐帧/box恒定、DYNAMIC保真、N_REPEAT独立生成、全部mux绑定。历史缓存只能声明来源/复算验证，不能假称重新forward。

最低测试：整数shift方向/不环绕/长度/int16极值；static帧和box同步冻结；含face_mode的乱序join与重复/漏cell拒绝；历史非零/不同track起点的绝对帧交集与局部行映射；先均值后median的反例及anchor；合成embedding的±5符号；I排除全部shift/offset/前处理边界；重复性或检测失败传播为uninterpretable；hash篡改拒绝。无需为每个I/O包装器写测试。

待实现CLI：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_global_shift_response.runner --run-id <id> --stage prepare
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_global_shift_response.runner --run-id <id> --stage all --resume
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_global_shift_response.validate --run-root runs/wav2lip_global_shift_response_<id>
```

all包含宿主CUDA子进程，应以产品允许的宿主执行方式运行；没有宿主权限时明确报告，不再安装驱动。

BM笔记题名 `Wav2Lip global shift response diagnostic 2026-09-08`。执行前先搜2–3个变体、读现有全文，planned→running→concluded或blocked，保存结果数字、解释边界与产物指针，保留changelog，写后读回。完成后勾选tasks并交付简短结论。
