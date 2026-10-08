---
title: Ditto TTS 局部时间辨识的偏移与语速控制 Spec
type: research
permalink: tts-exp/research/ditto-tts-局部时间辨识的偏移与语速控制-spec
status: concluded
date: '2026-09-18'
protocol: ditto_timing_rate_v1
question: 整体偏移与整句播放时长控制后，Ditto TTS配对的局部时间辨识优势是否仍存在？
tags:
- tts
- ditto
- syncnet
- temporal-identifiability
- rate-control
- spec
---

# Ditto TTS 局部时间辨识的偏移与语速控制 Spec

## 1. Objective

为下游 GPT Luna Max 规定一个小规模、可复算的机制诊断。协议固定为 `ditto_timing_rate_v1`，使用已完成 Ditto-50 实验中来源合格的 47 对历史视频。检验：在校准整体音视频偏移、统一两侧整句播放时长后，TTS 配对的 SyncNet 局部时间辨识优势是否仍然存在。

本文档在 spec 阶段规定了实现与判读协议；实现、smoke、全量运行、独立复算和 BM 结果笔记现已完成。科学阴性或支持不足仍按预先规则记录，不能通过修改阈值追求阳性。

实验分两部分：
- A：原生时钟下，用不重叠的输入时间区间分别校准偏移、测试局部排名。
- B：保持已有发音动作序列，成对同时改变音频与画面的播放速度；分别对齐到共同短时长、共同长时长，再重复 A。
- 局部错配由重新配对已提取的窗口实现，不重新生成嘴型、不注入新的生成器驱动，不称为生成端因果干预。

这里的“时间辨识”是：同一画面窗口与估计匹配时刻的声音，比与前后 40/80/120 ms 的声音更接近的频率。“估计匹配时刻”由另一半视频估计，不是真实同步标注。

历史证据与新增点：
1. Ditto-50：47 对新评分 ΔSync-C=+1.068，44/47 为正；但 VSR 自然侧 strict top1=25/47，未过校准。因此选用该整批来源合格 cohort，不按单句增益筛选，也不把 VSR 当真值。
2. 旧时间辨识 A 已经用交叉时间块估计 lag，A_rank=+0.0555。不能宣称旧实验“完全没有控制整体偏移”。新增加的是完整输入窗口隔离、Ditto 正效应 cohort、以及真实波形/画面的共同速度控制。
3. 旧连续轨迹干预对自然和 TTS 都有效；它没有证明 TTS 特有机制。F0 及若干新 Wav2Lip cohort 原生优势未确认，也不能据阴性结果排除相应机制。

研究定位：已见单说话人历史数据上的探索性机制诊断，预先固定本轮分析；不是全新独立验证。检验的是“整体偏移与整句时长是否足以解释局部排名差异”，不检验所有局部韵律，也不证明嘴型更准确。

## 2. Repository Model

数据流：

`runs/vsr_ditto50_linkage_v1/manifest.json`
→ 来源合格的 47 对及同 run 的 `syncnet/{id}/{natural|tts}/score.json`
→ 已保存的 `crop.avi`、`track.json`、模型/视频 hash
→ 官方风格解码后的 BGR 帧与 16 kHz PCM16
→ 原生 / 共同短时长 / 共同长时长
→ 冻结 SyncNet 的音频、视觉 embedding
→ 完整窗口隔离的 lag 校准与局部错配排名
→ 逐话语配对统计、独立复算、报告。

父run没有保存逐窗口SyncNet embedding/距离矩阵；其 features/*.npz 是 VSR encoder/logp，不能复用为本轮特征，需要新SyncNet前向。

本轮不需要 Ditto 生成环境、TTS 服务、MFA、VSR 或新模型。已有 crop 冻结 ROI；禁止对变速版本重新做人脸检测。

冻结来源，以下 SHA 在书写时已核对：
- `runs/vsr_ditto50_linkage_v1/manifest.json`：`b388453fbfe378058e39ed5ea33ed53e6e36b363175869f764c7539777bfefd6`
- 同目录 `analysis.json`：`e92482d24da0bf37db4cd4b36a56693430bbde88f0943006246fbfc7234da86b`
- 同目录 `validation.json`：`5c6f6bef74903b2518d3017cafbb6d1189617771b6a1233e54389d7b8cf36926`
- `third_party/syncnet_python/data/syncnet_v2.model`：`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`
- 推理环境：`[redacted-local-path]`。

manifest 有 50 条来源台账；继承其 `eligibility == "eligible"` 的 47 条，不重新引入 ID 2/42/44，也不依据 VSR 是否 top1、Sync-C 是否提高删除记录。继承 OOV 排除是为了保持 cohort 身份，不表示 SyncNet 需要汉字词表。样本身份使用 `(cohort_id, paired_key, id, condition)`，不用单独数字 id 连接其他 run。

历史单侧原视频时长：自然 min/median/max=3.12/4.48/8.92 s，TTS=2.72/3.84/10.08 s；单对长短比最大约 1.605。短句可能无法满足隔离窗口支持，应保留真实分母。父评分 offset 取值为 0 或 1 帧。已保留crop实际帧数自然min/median/max=80/114/225、TTS=70/98/254，较原视频普遍多2帧；最终canonical长度必须以crop解码PCM共同覆盖为准。只读ffprobe元数据按第6.5节规则预测共同支持约40对，预计短句排除ID 3/5/8/11/22/33/40；这不是新的评分或最终J，解码/QC后必须重新计算。

## 3. Code Anchors

| path | symbol/component | current role | required change |
|---|---|---|---|
| `scripts/experiments/vsr_ditto50_linkage.py` | `audit_ditto50`（360行）、`_run_syncnet_cell`（594行） | 生成父 manifest、已冻结 crop 与 score.json | 只读其契约及产物；不重新执行 VSR 或改父实验 |
| `scripts/experiments/tts_native_gain_attribution/syncnet.py` | `SyncNetEngine.__init__` | 加载冻结 SyncNet、暴露 `network` | 在新 runner 显式传本轮 weight、batch_size=20、device |
| 同文件 | `_stream_mjpeg`、`_visual_batch`、`extract_visual` | 官方 JPEG 风格 BGR 解码、五帧窗口前向 | 复用原生解码；新 runner 写薄的数组前向 adapter，仿照该批处理，不修改旧类 |
| 同文件 | `_audio_windows`、`extract_audio` | 16 kHz PCM16 → 默认 MFCC → 每 4 列取 20 列 → 音频 embedding | 所有模式统一复用；保存实际配置和窗口坐标 |
| 同文件 | `distance_matrix` | float32、eps=1e-6、31 lag 的官方距离矩阵 | 只用于官方口径重放与 INTERIOR-C 描述；不得直接冒充新 float64 rank |
| `third_party/syncnet_python/SyncNetInstance.py` | `SyncNetInstance.evaluate`、`calc_pdist` | 官方 JPEG/PCM 解码、窗口计数和 C/D/offset | 只读；smoke 从该函数返回的原始矩阵验证新前向链 |
| `scripts/experiments/tts_time_instance.py` | `_calibrate_k`（542行）、`run_a`（572行）、`official_distance_matrix`（172行） | 历史局部辨识分析 | 参考公式；不复制旧 cohort、分折、门槛、检验家族 |
| 新 `scripts/experiments/ditto_timing_rate.py` | `audit_inputs`、`prepare_cell`、`retime_pair`、`extract_cell`、`main` | 无 | 实现薄 CLI、媒体与前向管理 |
| 新 `scripts/experiments/ditto_timing_rate_metrics.py` | `build_support`、`calibrate_lag`、`rank_rows`、`summarize_pairs` | 无 | 纯 NumPy 统计；导入不能加载 torch/模型 |
| 新 `scripts/experiments/check_ditto_timing_rate.py` | `validate_run` | 无 | 独立从 embedding/receipt 重建支持、lag、rank 和统计 |
| 新 `tests/experiments/test_ditto_timing_rate.py` | CPU 协议测试 | 无 | 覆盖时间轴、统计、错误状态与独立复算 |

CodeGraph 对相同符号可能返回 `pre_repair_snapshot/`。实现必须使用当前 `scripts/` 源码，不能导入 snapshot；近期符号未索引时读取明确路径补核对。

## 4. Reference Pattern

最接近的科学模式为 `tts_time_instance.py` 的 A 阶段：先确定整段 lag，再对留出的行比较正配与附近错配；统计单位是话语。复用其严格相等计 0.5 的 rank 定义及“raw/unit 不混量纲”的做法。

最接近的资产模式为 `vsr_ditto50_linkage.py` 保存的 `score.json → crop/track/model/input SHA` 链。新实验直接读取已有 crop，避免重新检测的 ROI 变化。

不要照搬：
- 旧 A 按连续秒奇偶分折且未显式排除全部输入窗口交叠的支持规则；
- VSR 的 `matched` 仅重采样画面、不处理音轨的规则；
- 历史 rank、历史 C 或旧 bootstrap 结果填补本轮缺失；
- 把对 embedding 插值称为波形变速；本轮必须在音频波形和视频帧上处理后重新前向。

## 5. Invariants

1. 原始视频、音频、父 run 和第三方源码均只读；全部新产物进入新 run。
2. 原生、短、长三模式使用同一源 crop、同一已解码帧、同一音轨，身份、模型、空间 ROI 完全固定。音轨来自 crop 内嵌音频，不能悄悄替换为源目录的 clean WAV。
3. 不逐样本按得分选样、不按 lag 边界或排名好坏删样、不在测试行上找最优 lag。
4. 所有候选 lag、局部错位和两折均用预先确定的支持；不允许某个 lag 因可用行较少而获得不同支持。
5. 对每个条件，校准与测试所读取的视觉帧、MFCC 实际输入支持必须不相交。插入人工间隔不代表统计独立，bootstrap 仍只抽话语。
6. 统一总时长仅控制整句平均播放速度；没有对齐音素时长、停顿分布或发音事件，不得写成“彻底排除语速/韵律”。
7. TTS 减自然统一作为增益方向。rank 越高越好；D 的改善为 D_N−D_T。Sync-C 表格 3 位小数，计算不提前舍入。
8. 工程状态、测量可用性、原生优势、机制结果分别保存。无显著差异不等于等效，处理后效应变小不能自动归因于速度。
9. 缓存只凭输入/代码/配置/输出 hash 验证后复用；存在文件不等于完成。原始身份或协议变化必须新 run。
10. 不训练 head，不重新渲染 TFG，不下载模型，不做强度 sweep，不建立新的通用框架。

## 6. Implementation Plan

### 6.1 audit 与 CLI

新增 CLI：
`--stage audit|prepare|extract|analyze|validate|all --run-dir PATH [--smoke] [--device cuda|cpu] [--resume]`。
smoke 固定 ID 1/3/12；其中短句 ID 3 用来检查支持不足的合法路径，不能替换为更长句。正式遍历父 manifest 的全部 47 条。CLI 本身不依赖 shell 环境激活；GPU 推理使用已绑定 Python。

audit 校验冻结文件及每个 score receipt 的 input/model/crop/track SHA、COMPLETE 状态、condition、paired_key 和 manifest 一致；记录原 track 覆盖，不使用可能大于 1 的历史 coverage 比值作为本轮时长真值。直接探测 crop 的 PTS、实际帧数、音轨采样数、尺寸；要求 25 fps 单调等间隔视频、一个音轨。视频/音轨起始时刻差超过 1/16000 s 时不得静默归零，标 `UNSUPPORTED_CROP_CLOCK`，保留 A/B 缺失原因。

先仅由元数据预测支持分母，再开始新科学前向；不能看新分数后改 K、窗口数或最低 n。运行时记录 ffmpeg/torch/numpy/python_speech_features 版本、实际命令、code/weight hash、device。CPU 可运行同一模型，但一个正式 run 不混用设备；设备改变新 run。

### 6.2 canonical 原生输入 O

每个 crop：
1. 用 `SyncNetEngine._stream_mjpeg` 解码，得到 BGR uint8 `frames[n,224,224,3]`。核对实际 crop 尺寸；若不是 224×224，报告输入契约不符，不加一个未经验证的 resize。
2. 按官方 evaluate 的同一命令抽取 mono 16 kHz PCM16：ffmpeg 的 `-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000`。保存完整命令与解码后的 PCM hash。
3. `L=min(n_frames, floor(n_samples/640))`；canonical 帧为前 L 帧，PCM 为前 640L 个采样，duration=L/25。仅截掉尾部非共同部分，不移动起点。记录两侧被截长度；任一尾部截除超过 80 ms 标 `EXCESS_TAIL_TRIM`，该 cell 不可用于机制分析，不自动扩大阈值。
4. 保存 canonical WAV；逐个 cell 解码和处理，不在内存同时堆 94 个视频。可保存压缩 uint8 帧或只保存帧 hash 与可重放命令；正式验收必须能由原 crop 重建。
5. 对原生特征统一只保留 F=L−5 行。视觉 adapter 生成 v[i:i+5]，音频使用现有 extract_audio 并截取前 F 行。这与官方 evaluate 的 lastframe 一致；不要使用视觉额外的最后一行。

smoke 中直接调用官方 evaluate 获得 crop 的原始 31-lag 距离矩阵，与 canonical O 重建矩阵比较 max_abs≤1e-4，C/D 原精度差≤1e-4、offset 相同。不能只比较三位小数或借用旧日志。若不一致，修复解码/计数适配后再跑正式；不改模型、不自动放宽容差。

### 6.3 两个共同播放时长 S/L

每对由 canonical `L_N,L_T` 定义：
- SHORT：M_S=min(L_N,L_T)；
- LONG：M_L=max(L_N,L_T)。

每个来源 s 变换到目标 M，速率 r=L_s/M，输出严格 M 帧、640M 个采样。r>1 加速，r<1 减速。所有 r 由长度计算，不按评分优化。

视频处理：
- 第 j 个目标帧的源坐标 `u=j*r`，j=0…M−1；
- `lo=floor(u)`、`hi=min(lo+1,L_s−1)`；
- BGR 像素 `round_to_even((1−w)*frame[lo]+w*frame[hi])`，w=u−lo，clip 至 [0,255] 后 uint8；
- 最后一个源帧外的不足一帧范围做 endpoint hold；记录数量。禁止 align_corners 式 (L−1)/(M−1) 映射，因为它与音频 r=L/M 的时钟不一致；
- 流式/分块送入同一个 visual adapter，处理后重新前向，不保存有损 MP4 再评分。

音频处理：
- r 恰为 1 时逐采样复制，禁止不必要的编解码；
- 其他 r 用本地 ffmpeg `atempo=<r的17位有效数字>`，mono/16k/PCM16，不改音高、不调响度。r 超出 [0.5,2.0] 则标 `RATE_OUT_OF_PROTOCOL`；本地已核对长短比≤1.606，正常不应触发；
- atempo 输出与 640M 的差≤1280 samples（80ms）时，只在尾部截断/补零至精确长度；超过则 `RATE_LENGTH_QC_FAILED`。记录原始输出长度、修正量、削波比例；
- 不能在头部补零、用重采样代替保音高变速、或用 `-shortest` 隐藏长度差；
- 尾部修正的实际时间区间不得进入任何校准/测试窗口；第 6.5 节支持检查必须显式再验证，不能只假定 guard 足够。

rate=1 的变换必须逐像素、逐采样等于 O。两个目标中各有一侧 rate=1，可使用验证过的 O 特征别名，但为每个逻辑 cell 写独立 receipt 和 alias 源 SHA。不能因未重复前向将其计为缺失，也不能谎报真实前向次数。

这两种目标让加速、减速都被检查；没有保证算法伪影相同。变速导致优势消失时，必须保留“处理影响”作为竞争解释。

### 6.4 前向、数值与曲线

新 `extract_visual_frames` adapter 只模仿 `SyncNetEngine.extract_visual` 的 batching，调用同一 `network.forward_lip`；输入 BGR 0…255 float32，形状 [batch,3,5,224,224]，不归一化到 [0,1]，不 RGB 转换。音频经现有 extract_audio。model.eval、inference_mode、batch_size=20；记录实际 feature shape [F,1024]。

为每个逻辑 cell 保存 a.npy/v.npy、媒体及特征 SHA、F、L、速率、时间映射、decode/transform receipt。

并行保留两种距离：
- 官方风格矩阵：float32、eps=1e-6、lag −15…15；用于 native parity 与次要 C/D。
- 主 rank：原始 float32 embedding 转 float64 后算严格欧氏距离，不加 eps、不平方、不做事后标准化。

次要协议分数 `C_interior`：官方风格矩阵仅用行 i=15…F−16，按时间平均得到31点曲线，B=median(curve)，D=min(curve)，C=B−D。标为本协议 INTERIOR 分数，不与父 full-window C 混用。N/T 原生优势 gate 使用本轮 O 的 C_interior 差值。

### 6.5 无输入交叠的支持、共同进度与 lag 校准

固定 K=8，校准候选 k∈[−8,8]；局部错位 δ∈{−3,−2,−1,+1,+2,+3}，分别对应 120/80/40ms。历史 offset 仅0/1，K=8保留余量。距离坐标始终 `d(v[i],a[i+k])`，official_offset=−k。

窗口实际名义支持：
- 视觉：[0.04i,0.04i+0.20)；
- 音频 MFCC：[0.04j,0.04j+0.215)，另保守计入 preemphasis 向前1个sample；
- 校准及 rank 的所有候选音频索引最远 i±11。
为避免实现边界出错，统一用更保守的帧单位覆盖 `[i−12, i+17)`；它覆盖视觉、全部音频候选和 MFCC 前置采样。

每个 cell：
- `b=floor(F/2)`；半区 H0=[0,b)，H1=[b,F)；
- 半区 [lo,hi) 的安全行 Q={i整数：i−12≥lo 且 i+17≤hi}；
- 每半区必须至少10个不同安全行；不足只标支持不足，不用另一种分折补救；
- rank 测试 H0 时只用 H1 的 Q 校准 k0，测试 H1 时只用 H0 校准；
- 校准目标为 mean_i ||v[i]−a[i+k]||，所有 k 使用同一 Q；
- 取最小值；精确平局依次按 |k|、k 升序。不得看本半区结果调 lag、逐行重新最优化、或人为加减80ms。

为了比较模式时不同时改变采样的句子位置，正式主集合进一步采用共同进度支持：
- 预先定义200个 phase：p_j=(j+0.5)/200，j=0…199，前100个属于H0，后100个属于H1；
- 对每个 N/T×O/SHORT/LONG cell，phase 映射 i=floor(p_j*F_cell)；
- 仅保留该 phase 在全部6个 cell 中都属于指定半区 Q、且输入未触及尾部修正区的点；
- 任一 cell 任一半区在这些 phase 下少于10个不同 i，则整对不进入主集合 J；
- 同一个 i 被邻近 phase 重复命中是允许的；200点代表等权进度采样，不是200个独立样本。保存映射和重复计数；
- 校准仍用另一半全部 Q，不能用测试 rank 反向决定 phase；
- 这只是匹配句子相对进度，不是匹配音素；禁止称为同发音事件的逐帧因果比较。

若某 cell 的校准 k0 达到±8，保留该 cell 和 pair，不按分数排除；保存边界标记。分别对六种 arm×mode 汇总，任一层的边界折比例>10%，标 `CALIBRATION_RANGE_LIMITED`，不输出正式机制支持状态。

保存每个 fold 的校准行、测试 phase/i、候选 k 目标值、k0、物理支持区间。独立验证器必须证明同一 cell 的校准/测试输入区间无交集。

### 6.6 主指标、辅助量与统计

每个测试 phase 在其 fold 的 k0 下：
- d0=||v[i]−a[i+k0]||；
- dδ=||v[i]−a[i+k0+δ]||；
- wδ=1 当 dδ>d0；0 当 dδ<d0；严格相等时0.5。不得加固定距离容差改变 rank；
- 先对±方向与三个 |δ| 等权平均，再在每个半区的共同 phase 内平均，最后两个半区等权，得到 R_{s,m}；
- 同一支持上另算 k0=0 的 R_zero。它只是未校准对照。

每对两个主量：
1. H_native=R_{T,O}−R_{N,O}；
2. H_rate=0.5*[(R_{T,SHORT}−R_{N,SHORT})+(R_{T,LONG}−R_{N,LONG})]。

唯一正式主集合 J：来源合格、6个 cell 及支持完整、变换QC通过的配对话语。两个主量、原生参照、native-vs-rate差值都在同一个 J 上计算。保留47条的完整流转账；O 可用但 B 失败的 native-only 结果另列 J_A，不混入两个主统计。

最低 n_J=30 是可解释性门槛，不是功效证明。不足仍保存点估计、逐对数据和描述性95%区间，机制状态 INSUFFICIENT_PAIRS。没有个体正增益筛选。

统计：
- 按 numeric id 排序 J，话语成对有放回抽样；PCG64(20260918)，20000 draws，共享索引；
- 每项输出 n、均值、中位数、正/零/负数量、95% percentile CI；
- 两个主量固定 Bonferroni 97.5% CI，即1.25/98.75百分位，quantile method=linear；缺一项仍保留两个主检验家族；
- 不把窗口、phase、fold、两种目标时长当独立样本；
- 单说话人只说明该说话人的话语差异，不能做跨说话人推断。

辅助量均明确 descriptive，不另选显著结果替代主量：
- SHORT/LONG 的单独 H、H_rate−H_native、校准前 H_zero；
- 每个 |δ| 的正向/负向胜率与两者差，检查原先±1不对称；
- 所有版本 k0 的分布、两折差、边界比例、行/phase数量；
- 原始范数与单位范数 embedding 的 rank；单位化继续使用原始距离校准的 k0，不重新寻优；
- native 与两种 matched 的 C_interior/D/B 曲线；C上升不能代替rank；
- 自然/TTS时长比及被排除短句分布；
- J_A 原生结果及父47对 C，仅作不同支持/协议的对照。

### 6.7 控制与工程 gate

1. **身份与重放**：smoke 的六个 O cell 均与官方 evaluate 原始矩阵在1e-4内一致；同设备重复前向 embedding max_abs≤1e-5。失败先修实现。
2. **identity transform**：所有 rate=1 条件像素/PCM exact equality；smoke 至少 N/T 各一个独立重复前向。不能只用特征 alias 证明前向重复性。
3. **已知延迟**：smoke ID1 的 N/T，各构造音频±3200 samples 的等长头尾移动，视频不变；在共同有效内部支持上，校准 k 相对 O 增/减5帧，容许1帧。人工补零区域不参与比较。所有校准候选/局部比较的音频窗口均不得触及新增零区，先按该规则求O和shift共同支持，再拟合两者lag。共4个控制；失败时不称真实数据 timing 已校准。
4. **变速时间轴**：CPU像素时间标记例验证 j*r、rate=1、短/长输出长度；独立音频控制为4.8s（120帧，所列速率下目标帧数均为整数）、220Hz载波，包络为中心0.8/1.6/2.4/3.2s、σ=0.06s的四个高斯脉冲之和，峰值0.2。以 r=0.75/1/1.5 经同一 atempo 流程，峰中心应在输入中心/r的40ms内，主频偏差≤2%，不以最佳互相关平移补偿。峰检测在每个期望中心±0.15/r秒内取RMS包络最大值（160-sample窗、16-sample hop，时间标为窗中心；主频以输出全段FFT最大幅度的正频率bin估计，排除DC）；若区间重叠/峰不唯一须报告失败。不在新数据上搜索更容易通过的控制。
5. **尺度不变性**：所有待比较距离做3d+7，rank完全不变；复制N为T时两个H必须为0。
6. **控制失败分支**：speed控制失败只使B=RATE_CONTROL_FAILED，仍交付J_A上的A独立结果（标A_ONLY，不能作为同J联合机制结论）；延迟/原生parity失败则measurement=TECHNICAL_INVALID，不宣称TTS机制。

音频控制验证宏观变速和保音高，不证明真实语音没有局部WSOLA伪影。后者是科学解释限制，不能靠控制通过就删掉。

### 6.8 科学判读

analysis.json 分开保存：
- engineering_status=PASS/FAIL；
- support_status=SUFFICIENT/INSUFFICIENT_PAIRS；
- calibration_status=PASS/CALIBRATION_RANGE_LIMITED/DELAY_CONTROL_FAILED；
- rate_control_status=PASS/RATE_CONTROL_FAILED；
- native_gain_status：J上 O 的 ΔC_interior 95%CI下界>0才是CONFIRMED，否则UNCONFIRMED；
- H_native_status/H_rate_status：校正CI全正/全负/跨0对应POSITIVE/NEGATIVE/INCONCLUSIVE；
- temporal_validity：六个arm×mode的平均R及各自95%CI，只有各自下界>0.5才记ALL_ABOVE_CHANCE，否则INCONCLUSIVE。
- mechanism_status 依下列规则，不覆盖原始数值。

判读顺序：
1. 工程/延迟校准失败 → TECHNICAL_INVALID；n不足 → INSUFFICIENT_PAIRS；校准范围受限 → CALIBRATION_RANGE_LIMITED。
2. 本轮同J原生C优势未确认 → NATIVE_GAIN_UNCONFIRMED；可讨论本协议响应，不能称解释了已复现TTS增益。
3. H_native未判正 → NO_CONFIRMED_NATIVE_LOCAL_ADVANTAGE；不能断言原生C增益没有时间成分。
4. speed控制失败 → RATE_CONTROL_FAILED，A单独报告。
5. temporal_validity非ALL_ABOVE_CHANCE → TEMPORAL_MEASUREMENT_INCONCLUSIVE。
6. H_native/H_rate都判正，且SHORT/LONG各自H均值>0 → LOCAL_ADVANTAGE_SURVIVES_GLOBAL_CONTROLS。
7. H_native判正而H_rate未判正 → RATE_TEST_INCONCLUSIVE；不能写“语速解释了增益”。
8. H_rate判正但SHORT/LONG方向相反或有一个均值≤0 → TARGET_RATE_DEPENDENT，不概括两种速度下都成立。

支持时允许的结论：
“在这批Ditto历史配对上，整体偏移校准与整句共同播放时长控制后，TTS配对仍具有更高的局部时间辨识分数；单一整体offset或总时长差不足以解释该测量差异。”

不允许的结论：
- TTS已被证明减少声学相位抖动；
- TTS嘴型更真实/更准确、VSR已确认、音质更好；
- 增益全来自评价器、或某个百分比由语速中介；
- 已排除音素局部时长/停顿/韵律/声学域偏差；
- 可启动replace增强头或将控制后的分数当训练监督。

若效应在变速后缩小，可写“对共同播放速度处理敏感”；必须并列算法伪影、时间窗口内容改变和真实速度贡献，不能在本轮中选定其中一个原因。

### 6.9 独立复算、结果和运行预算

建议输出：
`inputs.json, protocol.json, media.json, receipts/, features/, support.json, row_metrics.npz, pair_metrics.json, analysis.json, validation.json, report.md, timing_curves.png`。

一幅图即可：native/SHORT/LONG 的正负错位胜率曲线及T−N差，标n与同J支持。无需网页或dashboard。

独立checker不能导入新metrics模块的support/lag/rank/bootstrap函数。它应从保存的a/v、媒体长度、receipts重新：
- 构建6-cell共同phase与两折支持，检查无重叠、无越界、未触及补零；
- 遍历17个lag找k0，复算全部d0/dδ、w、R与H；
- 用独立循环和同一冻结bootstrap索引重算CI、状态、集合与计数；
- 检查输入/输出hash、alias合法性和每个预期cell的状态。
feature原值转float64的距离误差≤1e-8，R/H/CI≤1e-12；计数、集合、lag与状态必须完全相同；exact tie不得由epsilon模糊化。

逻辑科学cell上限=47×2×3=282（每cell一组音频/视觉特征），不等于282次独立新媒体处理或282个生成视频。rate=1允许alias，实际前向与逻辑完成数分别计数。smoke六个O官方重放、独立重复、4个延迟控制另列，不混科学分母。

串行处理媒体，batch20；不保留逐帧JPEG目录；只清理本run明确创建的scratch。先用smoke实测磁盘/时长再外推；可用空间覆盖估计持久产物+单cell scratch+1GB余量。GPU不可用可显式新开CPU run，不安装其他生成模型。

## 7. Expected Change Surface

### Must change

下游实现时新增：
- `scripts/experiments/ditto_timing_rate.py`
- `scripts/experiments/ditto_timing_rate_metrics.py`
- `scripts/experiments/check_ditto_timing_rate.py`
- `tests/experiments/test_ditto_timing_rate.py`
- `runs/ditto_timing_rate_smoke/` 与 `runs/ditto_timing_rate_v1/`
- 完成后新 BM Experiments 结果笔记、本文状态、HANDOFF 当前结论指针。

本轮只新增此 spec；不创建伪结果run。

### May change

- 仅发现可复用helper的真实兼容缺陷时，最小修正 `tts_native_gain_attribution/syncnet.py`，保持旧公开签名/默认行为并补旧测试。优先把本轮adapter放新runner。
- 实际canonical crop时钟与假设冲突时，先记录明确事实并修订spec；不在执行中静默换来源/扩大阈值。

### Should not change

原始音视频、父manifest/score/crop、已有实验、第三方模型/权重、全局config、VSR规则、TTS provider。无清理旧数据、无依赖升级、无commit/push、无无关重构。

## 8. Validation Plan

下游依序运行：

```bash
[redacted-local-path] -m py_compile scripts/experiments/ditto_timing_rate.py scripts/experiments/ditto_timing_rate_metrics.py scripts/experiments/check_ditto_timing_rate.py
[redacted-local-path] -m pytest tests/experiments/test_ditto_timing_rate.py -q
[redacted-local-path] scripts/experiments/ditto_timing_rate.py --stage all --run-dir runs/ditto_timing_rate_smoke --smoke --device cuda
[redacted-local-path] scripts/experiments/ditto_timing_rate.py --stage all --run-dir runs/ditto_timing_rate_v1 --device cuda
[redacted-local-path] scripts/experiments/check_ditto_timing_rate.py --run-dir runs/ditto_timing_rate_v1
```

CPU测试必须验证：
1. identity时间轴逐样本/像素一致，短/长映射使用L/M；可手算的r=0.5/1/2帧标记例；PCM修正只在尾部。
2. 存在特征窗口但不足guard支持的短句返回合法缺失；不得偷偷用重叠折或减少候选lag补齐。
3. 合成有已知位移的非周期独热/随机时间序列正确恢复k；正延迟导致k增加，official_offset符号相反。
4. 完整window支持逐区间检查，不仅比较i是否不同；若删除guard制造交叠，checker必须FAIL。
5. 极短N/长T、六cellF不同、phase重复、一个mode缺失的主集合正确；两个H、baseline和差值必须同J。
6. 严格tie=0.5，3d+7不改rank，N=T时H=0，unit处理零范数显式失败/null而非填零。
7. 两折/正负方向/|δ|的等权公式，可用手工toy精确计算；句子不是按长度加权。
8. n=29不足，n=30可估；CI跨0不能标等效；H_rate正但短长方向相反不能标survives；smoke始终SMOKE_TECHNICAL_ONLY。
9. hash改变/错误paired_key/缺一侧/非法alias拒绝复用；故意篡改pair_metrics、lag、支持或analysis，独立checker发现。
10. 同bootstrap索引生产与独立复算一致；JSON无NaN/Inf，缺失用null+reason。

smoke不根据科学分数决定改协议。短句缺失是预期路径；系统性解码/模型/时间轴错误才修实现。正式全量后自审和修复实现缺陷，再复算；不因科学不显著改变分析。

## 9. Risks and Edge Cases

- 旧实验已交叉估lag；本轮是更严格隔离与新控制，不应将旧阳性重新包装成首次发现。
- official lag只是距离最优位置，没有真实同步标签；留出校准可避免直接用测试噪声找峰，但不能保证估计位置等于人的同步判断。
- 语速统一改变SyncNet一个固定窗口中覆盖的内容；这是问题的一部分，同时也意味着不能把结果直接移植到原速生成视频。
- atempo的局部时间/音色伪影与帧插值模糊无法由identity完全排除；尤其阴性结果要保留竞争解释。
- 同步变速后统一的是整句长度，不是逐音素位置。两臂同phase可能对应不同音素。
- 两折的窗口不交叠不等于独立说话样本；只有一个speaker，不使用帧级置信区间。
- 当前父crop的coverage比值可能>1；不能用它推导真实帧时间。探测PTS/采样数并冻结本轮canonical时钟。
- 源clean WAV与MP4/crop音轨经过不同codec；换音轨会改变科学对象。本轮绑定crop音轨。
- 过短句会减少J。必须在新前向前根据长度输出预计分母，不按结果替补或把阈值调到恰好通过。
- 查看过历史cohort，全部结论限探索性；本轮的预先规则不能将历史数据变成未见测试集。

## 10. Assumptions / Unknowns

- VERIFIED: 父run的47对来源合格记录、94个SyncNet score/crop receipt存在；历史Sync-C优势已记录。
- VERIFIED: 当前SyncNetEngine有音频/视觉前向及官方风格distance_matrix；官方evaluate可返回原始距离矩阵。
- VERIFIED: 旧时间辨识A已有交叉lag校准；当前需要增强输入支持隔离。
- VERIFIED: 父原视频长短比在atempo单段[0.5,2]范围内；父offset为0/1帧。
- LIKELY: crop实际帧/PCM时钟和官方抽取可直接复现；必须由audit/smoke证明，不将原video探测值当crop真值。
- UNKNOWN: 更严格guard及共同phase后J是否≥30、同J的原生C优势是否仍成立。
- UNKNOWN: 共同短/长时长后TTS的局部排名差是否保留；不保证阳性。
- UNKNOWN: atempo控制能否在本地版本通过；失败只阻断B，不抹掉A。
- UNKNOWN: 未来运行的资源可用性；下游实测，不依据文档猜剩余磁盘。
- UNKNOWN: 即使本轮阳性，具体声学线索、生成端贡献、视觉准确度以及跨说话人/模型泛化仍未确定。

## 11. Handoff Contract

严格按anchors实现薄runner、纯统计模块、独立checker和一个测试文件；保持第5节不变量，复用冻结SyncNet，不建立通用实验框架。顺序为来源审计→CPU测试→smoke→全量→独立复算→自审修复→BM结果与HANDOFF。

最终报告必须回答：
1. 47条中有多少具备完整隔离支持，排除了哪些及为什么？
2. 同J、同本轮口径下原生Sync-C优势是否存在？
3. 整体lag校准后H_native是否为正？
4. 共同短/长时长后H_rate是否仍为正，两种目标是否同向？
5. 该结果排除了哪些简单解释，还不能说明什么？

不得把完成模型加载、工程PASS或某一个漂亮均值当成机制成立。实际来源/接口/时钟与spec冲突时停止依赖该假设的阶段，报告具体字段差异；可以继续不依赖它的工作。不要换模型、改阈值、重挑句子或补历史分数。

### Observations

- [status] concluded；实现、smoke、全量运行与独立复算均完成，结果见 [[Ditto TTS 局部时间辨识与共同语速控制结果 2026-09-20]]。
- [question] 整体偏移与整句播放时长控制后，Ditto TTS配对的局部时间辨识优势是否保留？
- [decision] 复用整批47对历史Ditto资产；不新增生成器、不训练head。
- [boundary] 本实验衡量冻结SyncNet的时间匹配响应，不提供独立嘴型真值。

### Execution Result

- [run] 完成时间：September 20, 2026；全量运行目录为 `runs/ditto_timing_rate_v1/`，smoke 目录为 `runs/ditto_timing_rate_smoke/`。
- [result] 47 条来源合格配对中 40 条进入 J，排除 ID 3、5、8、11、22、33、40（共同 phase 完整支持不足）；282 个逻辑 cell 全部完成。
- [result] H_native = +0.06013，Bonferroni 98.75% CI [0.02572, 0.09507]；H_rate = +0.01978，Bonferroni 98.75% CI [-0.00960, 0.05244]。
- [conclusion] 原生 C_interior 增益和 lag 校准后的局部差异仍被测到；共同短/长时长控制后的 H_rate 未确认，机制状态为 RATE_TEST_INCONCLUSIVE。

### Relations

- follows [[Ditto-50 VSR 与 Sync-C 增益关联验证结果 2026-09-18]]
- extends [[TTS 时间辨识与同文本实例交叉实验 Spec]]
- relates_to [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]
- relates_to [[LRS3 局部时间控制校准实验结果]]

### Changelog
| Date | Change |
|---|---|
| September 18, 2026 | 按implementation-architect编写下游spec；限定完整窗口隔离、共同短/长时长、同J统计与独立复算，尚未运行实验。 |
| September 18, 2026 | 自审修正SyncNet维度为1024，补齐精确函数锚点、crop长度与约40对支持预测；将控制信号改为4.8秒以保证目标帧数为整数。 |
| September 20, 2026 | 完成实现、smoke、47对全量和独立复算；J=40，H_native为正，H_rate跨0；spec状态更新为concluded。 |