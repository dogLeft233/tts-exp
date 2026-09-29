# 固定执行协议 v1

本文中的数值是预注册参数。实现不能根据 Sync-C、可用缓存或人工偏好改变它们。方法调整必须保留旧 run，在新 run 冻结修订版，并报告变更原因。

## P0. 输入与来源审计（CPU，先交付）

1. 校验 `input-bindings.json` 的每个已绑定文件，复核 v15 的 final/validation/24 个矩阵与媒体身份。
2. 固定 ID `151..162`，从 v15 cohort 读取 `source_group`，并与原 manifest 的视频父目录交叉验证。`speaker_key=lrs3` 不是 speaker 分组；12 个来源也不等于已核实的 12 个说话人。
3. A：绑定 24 个历史 LeapTalk crop、对应原 MP4、12 个真实原视频。历史 crop 不重新选 track；真实视频按官方 pipeline、最长连续 track→最早开始帧→文件名字典序选一个 track，不能按分数选人脸。
4. 每个视频独立记录 PTS、帧数、crop 来源、音频采样点数与 codec 延迟，验证音频和视频来自同一 track 时间段。真实视频若选到了错误说话者，记录 `INPUT_INVALID`，不能静默替换 ID。
5. A 的 baseline PCM 从各自 crop 解码为 16 kHz mono int16，再转 float64 做处理。历史 baseline 的原始 PCM hash 必须匹配 v15 worker 的 `input_pcm_sha256`；如 ffmpeg 版本差异导致 hash 不同，先解决或建立单独的完整重放，不能混用旧矩阵。
6. B 使用 A 的自然/TTS baseline PCM，而不是从别处取同名 WAV。新视频以同 ID 的真实视频第一帧为固定肖像，提取规则、frame index、图像 hash 在任何生成前冻结。所有 source/transform/seed 共用这一张图。人脸处理失败就是缺失，不按输出效果更换图。
7. TTS provenance：原 multiset BM 记载 faster_qwen3 0.6B/seed42，但运行 metadata 才能验证。登记原文、ASR 文本来源、TTS provider/model/seed、自然参考和 codec 链；缺失字段标 `UNKNOWN`，不把旧 bridge 的云端目标误贴到此处。无需新合成来补 provenance。
8. B 部署审计写 `model_provenance.json`：LeapTalk repo commit、base/LoRA/audio-encoder/VAE/TAE 权重 hash、模式（lite/full）、采样配置、dtype、软件与设备。官方论文与项目实际 checkpoint 的对应关系要有证据；完整训练集不可访问则标 `TRAINING_DISTRIBUTION_UNKNOWN`。

只读先产出 `protocol.json`、`assets.json`、`resource_plan.json`、`model_provenance.json`、`claim_registry.json`。冻结对象包括文档、代码快照、样本、参数、seed、支持选择规则和统计检验；冻结后不得改写已完成 cell。

## P1. 音频算法：等长度与有效操纵

### 共同规则

基准 `x` 为解码后 int16/32768，处理在 float64，最终 WAV 为单声道 PCM16；量化采用 `round-to-nearest ties-to-even` 后验证范围，禁止饱和裁剪。所有输出采样点数与 x 完全相同、起点相同，不移除静音、不 `-shortest`、不重采样去对齐 N/T。

设 STFT：16 kHz，窗长/FFT=512，hop=160，periodic Hann，center=True，左右 reflect padding 256。`X[k]=rFFT(window*frame_k)` 使用未归一化正向 FFT、逆向1/512归一化；不能混入库默认的spectrum幅度缩放。iSTFT 使用相同窗、overlap-add 窗平方归一化、去掉 padding 并恢复准确 L 个样本；尾部不足一 hop以reflect补齐只用于计算，最终截回 L。必须保存窗、算法库版本和恒等重建误差，不能让库默认值决定协议。

用 x 的 STFT 帧能量排序（相等时按帧index），取最低 `max(1,ceil(0.10*K))` 帧估计噪声功率谱 `Pnoise[f]=mean |X[k,f]|²`。这里只称“低能帧谱”，不假定它是真实纯噪声。`RMS_k` 取加Hann之前的512点帧RMS，`RMS_peak=max_k RMS_k`；活动帧 mask 为 `RMS_k > max(RMS_peak*0.1, 10^(-50/20))`。将活动窗覆盖的原时钟样本取并集得 mask M，越过原始[0,L)的padding不计入M，所有变体共用 M。活动样本不足 16000 或 RMS=0 则该 source `AUDIO_INVALID`。N/T/R 的 mask 分别来自自身 x，不能按处理后 VAD 换分析区间。

### NOISE20：固定合成有色噪声，活动区 SNR=20 dB

从每个 ID 的自然 N 的低能帧谱构造噪声颜色：`Pnoise+1e-6*mean(Pnoise)`，频点插值到长度 L 的 rFFT 网格。PCG64 生成长度 L 的标准高斯序列，以该谱平方根滤波，irFFT 后减均值。R/T 也使用本 ID 的自然噪声颜色，按自己的 L 生成；不把额外自然语音片段混入。

噪声 seed 为字符串 `native-gain-v1|<id>|<source>` 的 SHA-256 前 8 bytes 按 little-endian 解码为 uint64；source 为 `N/T/R`。所有生成 seed 共用同一条噪声 realization。

`beta = rms(x[M])/(10*rms(e[M]))`，`H_noise(x)=x+beta*e`。这是合成有色噪声测试，不能称现场背景复原。最终量化后按已知 x 与混入分量复算有效 SNR，容差 ±0.1 dB；变体和基线的公共缩放不改变 SNR。

### DENOISE：固定谱抑制，不选最优降噪器

`gain[k,f]=clip(sqrt(max(|X|²−Pnoise,0)/(|X|²+1e-12)),0.25,1.0)`。不做按分数选择的时频平滑、不使用相位重建模型，保留 x 的相位，iSTFT 得 y。对 y 施加一个常数，使 `rms(y[M])=rms(x[M])`；这个常数和实际背景/语音变化均保存。活动 RMS 太小不能强行放大，返回 AUDIO_INVALID。

该方法也会去掉语音能量；有效性检查是操纵检查，不是“质量认证”。必须报活动/非活动 RMS、谱通量、x−y 能量、包络相关、波形互相关峰及人工听感状态。如果非活动样本少于 1600，背景衰减指标为 `NOT_ESTIMABLE`，不能设成 0。

### 统一 headroom 与 GAIN_MINUS6

为避免 NOISE 的峰值造成 PCM 裁剪，先为同 ID 的 N/T/R 计算上述未量化 x/noise/denoise，然后取一个**三类 source 和全部变体共用**的常数：

`g_i=min(1,0.98 / max_abs(all unquantized x/noise/denoise of this ID))`。

基线 `A0=Q(g_i*x)`，噪声 `An=Q(g_i*H_noise(x))`，谱抑制 `Ad=Q(g_i*H_denoise(x))`，幅度控制 `Ag=Q(g_i*x*10^(-6/20))`，其中 Q 为上述 PCM16 量化。禁用逐臂 limiter/loudnorm/峰值归一化。输出 `g_i`、所有峰值、RMS 和量化误差。

**A 额外保留 ORIGINAL=Q(x) 评分**，测量统一 headroom 本身的影响并衔接 v15。g=1 时 ORIGINAL/A0 为两个逻辑 cell，但可以共享完全相同的特征/矩阵 artifact，登记 `reused_from`。不能删掉 ORIGINAL 因为其分数与 A0 接近。

所以 A 实际是 5 音轨条件：ORIGINAL、A0、Ag、An、Ad。B 的 baseline 使用 A0；所有 fresh native 和干预效应均相对 A0，报告中并列 ORIGINAL→A0 的自然/TTS 差异，避免将 headroom 混入“降噪收益”。

### 时间与操作检查

- 数字文件长度、采样率、起点必须完全相同；所有变体不能经过有损 audio codec。
- STFT 恒等路径与 x 的 float64 最大误差 <1e-8；量化后恒等路径 PCM hash 与 ORIGINAL 相同。
- x/y 在活动区的归一化包络（10 ms RMS）互相关最大峰偏移绝对值 ≤1 个 hop，记录整个 ±100 ms 曲线。这不是音素级时间不变证明；若超限，将该干预解释为包含潜在时间污染，主对应机制判定 `MANIPULATION_FAILED`，保留数据、不删 ID。
- GAIN 实测活动 RMS 比为 −6±0.01 dB；NOISE 实测 SNR 为20±0.1 dB。谱抑制的实际效果完整报告，不以达到某个 Sync-C 值验收。
- 听评用音频统一对比组播放缩放，不能逐 clip loudnorm 后说在评原始响度。算法原始输出始终保留。

## P2. A 阶段评分矩阵与特征

科学 cell：`12 IDs × 3 video_types(V_N,V_T,R) × 5 audio_conditions = 180`。其中 V_N 只配 N 系列、V_T 只配 T 系列、R 只配 R 系列。A 的视频解码帧 hash 在五音轨下必须完全一致；只换音轨的操作不能重复运行人脸检测。

评分 worker 优先接受分离的冻结视频帧/crop 与 PCM，不做有损 remux。若封装，无损视频 copy + PCM 音频；解码逐帧 hash、PCM hash 与直接路径一致后才能共享特征。保存 `video_features[T_v,d]`、`audio_features[T_a,d]`、原始完整 `[T,31]` 矩阵、MFCC/取窗参数与矩阵身份。

主支持 U：同 ID、同 video_type 五条件共有的有效行交集，再按官方 vshift=15 去掉两端各15行；必须是相同时间索引，不能各臂独立选高置信窗口。最少25行。FULL 为诊断。N/T 的原生差使用各自 INTERIOR 并另报旧版 EQUAL_COUNT；它不意味着音素对齐。

新增控制仅在 ID151/152 的三种 video_type 上运行，各3个：恒等文件重建、+200 ms、−200 ms音频平移，合计18。平移保持总长度、边缘补零。控制支持需再去掉受5帧平移影响的边界；基线与延迟控制使用相同缩减 U。使用实际 PCM 平移后重新计算 MFCC/音频 embedding，不能只挪已有矩阵列。

- 恒等控制：像素/PCM hash相同，距离矩阵最大绝对误差 ≤1e-5，offset一致。
- 延迟：官方 offset 相对基线分别应改变 −5/+5 帧，容差1帧，且基线最优列处距离上升。若原最优列加已知偏移越界，不能通过改搜索范围追认；记录 `CONTROL_UNINFORMATIVE`。
- 控制输出使用 `IDENTITY_PASS` / `DELAY_DETECTED` / `CONTROL_FAILED` / `CONTROL_UNINFORMATIVE`，不沿用 v15 中容易误读的 CONTROL_FAILED 代表延迟检测成功的旧名称。
- 任一控制失败/不信息化，不阻止其他技术可运行 cell 落盘，但该路径的时间机制推断必须标 `CONTROL_LIMITED`，自动完整验收不通过，不能当阴性科学结论。

### A 的错误内容检索（CPU，不新增模型调用）

每种 video_type 单独用12来源的 A0 特征池。对每个视频，与自身音频及其他11个来源音频比较；R 用 R 池、V_N 用 N 池、V_T 用 T 池，不能混用不同池产生伪域差。共 `3×12×11=396` 个错误内容特征配对。

取该类型12条音视频全部 INTERIOR 支持的最小行数 K，按各自支持的均匀分位索引取 K 行（floor linspace，断言无重复）。每个音视频配对的距离为逐行原始欧氏距离均值；自身也按同样分位规则、固定零 lag，不额外搜索，不能仅给正确对齐享受择优。错误音轨的分位映射只是负对照定义，不是有效音素对齐。

每来源统计正确配对距离小于各错配距离的比例（平局0.5）、正确距离在12个候选中的名次、correct−median(wrong) 距离间隔，raw/unit范数各一套。N/T配对比较只给组均值、95%描述区间，无确认性 p 值。source_group 并不保证文本不同，必须核对12条转写；相同文本的 donor 对标记排除并给出有效 donor 数，若转写无法核验则该诊断 `CONTENT_UNVERIFIED`，不得宣称内容辨认度已证明。

## P3. B 阶段生成与四格评分

144 个科学视频：12 IDs × N/T × 3 drivers(A0/An/Ad) × seeds(42,43)。每个 driver 的有效时长来自自己的源音频。生成期间记录实际读入 PCM、模型 frontend 特征、初态 seed、肖像和全部推理参数。

同 seed 使用同随机初态并不能覆盖所有随机性，须 reset 所有相关 RNG/流式状态；两 seed 只是生成稳定性检查，不能计作额外受试对象。不得从多个生成结果中挑选最优的两个。

基线视频与同 source 的处理视频使用相同评分 ROI 规则。优先从 V0 冻结官方 crop 轨迹，再原坐标应用到 Vn/Vd；如果嘴部出框或 ROI不可用则 `ROI_INVALID`。不能为候选单独检测并选“更好”的 crop，也不能直接把完整肖像当官方 SyncNet crop。

共同支持 U 在同 ID、source 的两 seed全部14个评分cell上冻结；来源长度较短时按共同有效时间区间，不补重复视频帧凑时长。ROI 可以按 seed 从对应 V0冻结，但 seed内三视频的空间变换必须相同。全部视频的音频实际生成时钟及评分音轨均从 P1导出，保存消费样本与生成帧映射；模型尾部补帧不得进入支持。

评分为每 ID/source/seed 七格：

```
(V0,A0)
(V0,An), (Vn,A0), (Vn,An)
(V0,Ad), (Vd,A0), (Vd,Ad)
```

所有推断比较按四格计算，不能拿 A 的历史 V_N/V_T 代替本次 q00。A 的 evaluation效应与B的evaluation效应分开报告，避免将不同视频生成配置混在一个分解里。

控制固定 ID151/152、N/T、seed42：4个额外独立 A0 重复生成视频，每个评 A0，共4评分cell；同4个基线 V0另评+200ms A0，共4 cell。重复生成像素差、C/D差及offset均需保存；冻结推理配置下要求 `|ΔC|,|ΔD|≤0.05`、offset差≤1帧。超限 `GENERATION_UNSTABLE`，不反复生成挑选通过实例。延迟规则同P2。

生成端若无法恢复完全相同历史 checkpoint，但能绑定明确官方 LeapTalk版本，可在**生成前**冻结为 `NEW_LEAPTALK_CONFIGURATION`。其结果只能解释本次配置，依赖 fresh native基线；不得声称精确重放历史。若不能绑定 LeapTalk家族身份，B停在 DEPENDENCY_BLOCKED。

## P4. 统计与固定主检验

所有效应先在 record/seed 内配对，B 再平均两 seed，再平均同 source_group 的记录，最后12组等权。点估计与区间使用同一组等权 estimand。禁止将 frame、donor、seed、N/T条件计作独立样本。

固定六个主对比，均使用 INTERIOR Sync-C：

| ID | 定义（正值代表预期方向） |
|---|---|
| A_N_DENOISE_E | A的 V_N：C(Ad)−C(A0) |
| A_T_NOISE_E_HARM | A的 V_T：C(A0)−C(An) |
| A_R_DENOISE_E | A的 R：C(Ad)−C(A0) |
| B_N_DENOISE_G | B自然source：q10_d−q00 |
| B_T_NOISE_G_HARM | B的TTS source：q00−q10_n |
| B_NATIVE_FRESH | B基线：C(V0_T,T0)−C(V0_N,N0) |

PCG64 seed=20260915，20000次来源组有放回bootstrap，保存同一套12列抽样索引，对六个对比复用。报告95%描述区间及99.166667% Bonferroni区间（每尾0.05/(2×6)），分位数线性插值。A先完成也仍按六项校正，B缺失不重新缩小检验族。

阳性证据：校正区间下界>0。实用幅度只另标 `mean≥0.200`（诊断阈值，不是人类最小可感知阈值）；不能把它变成“同步改善”的充分条件。无效/等效只有整个校正区间落在[−0.200,+0.200]才可描述为“小于预定幅度”，区间跨0而宽只是不确定。对反方向证据用校正区间上界<0。

C/B/D分解、−D方向、D0、offset、FULL/EQUAL_COUNT、单位范数、错误内容、交互、优势收缩和声学相关均为次要描述。报告95%区间及逐组数据，不用其中显著者替代主检验。不做自动特征选择、12样本多变量回归或因果中介百分比。

缺失：保留全部12组台账。任一必需cell缺失，相关主对比置 `INCOMPLETE`；可另提供完整配对子集的探索性估计并清楚列出分母，不能宣称完成固定12组。声学操作失败是干预解释受限，不伪装随机缺失后删除。

主结论要求对应控制有效。fresh native 未阳性时，不能用“历史曾阳性”绕过本次基线；B的干预响应照常报告，原生归因标 `NATIVE_NOT_REESTABLISHED`。

## P5. 节奏和听评

节奏：使用P1冻结的10ms活动mask，连续非活动长度≥200ms为停顿、连续活动≥100ms为活动段；报告比例、时长中位数/IQR、活动段CV（少于2段时NA）、10ms RMS包络的0.5–8Hz调制能量占0.5–20Hz比例，以及相邻活动STFT归一化幅度谱L2变化均值。算法是在音频上定义，不能用嘴部输出或Sync-C挑选段。来源N/T配对，12组均值/描述区间；相关仅限上述5类预定摘要与native delta的Spearman，不宣称中介。

人工包分开：

- 同步包：B的每 ID/source/seed，分别比较 V0与Vn、V0与Vd，双方**都播放同一个A0**，共96对。先问“嘴部动作与声音谁更同步”，再独立问“视频伪影谁更少”；允许平局。
- 音质包：A 的每 ID/source=N/T，A0与An、A0与Ad，共48对；只听声音，分别评清晰度与自然度；不得预告哪个是降噪或TTS。
- OPTIONAL既有原生包：可链接v15的12对，它不能代替本次相同音轨评估。

seed20260915生成匿名顺序、左右随机化和盲化映射；操作人员保留映射，评分界面不能显示condition/provider/文件名。每一包至少3位相同评审完成全部正式对，增加ceil(10%)隐藏重复题（同步10对、音质5对，界面总计106/53对），只用于一致性说明，不凭偏好排除评审。正式统计不重复计入隐藏题。

无真人评分：包、空模板、分析CLI、缺失说明仍必交，状态 `PERCEPTION_NOT_ASSESSED` / `QUALITY_NOT_ASSESSED`；不得由模型代填。返回评分时先将平局计0.5，先在来源内平均seed和共同评审，再做来源bootstrap；小型面板不能声称泛化到所有观众。阴性面板必须区间足够窄才能支持“无感知差异”。

## P6. 资源和缓存

- CPU先完成输入/参数/预算及音频操作；绝不在同一 GPU 上并发 TTS/TFG/SyncNet。默认零新增TTS、零训练、零云调用。
- 每GPU阶段和长run定期查 `nvidia-smi`。使用跨本仓库实验共用的文件锁/lease，进入前无外部compute PID、连续3次间隔5秒利用率≤5%，空闲显存大于估计峰值+1GiB；不能抢占或终止其他用户进程。
- 外部负载出现时不启动下一个cell，保存状态，释放本进程资源；已运行的cell可安全完成或中断为未提交。采样检查不能保证排他，报告实际监控与竞争情况。
- A可复用crop和单次视觉特征，视频流式解码，禁止把全部样本的PNG帧落盘。B每次仅一视频在途，渲染后清理**本次cell可再生临时帧**；保留正式MP4、音频、特征、矩阵、hash和日志。
- 运行前计算持久增量+单cell峰值临时空间+1GiB保留。每cell前复查剩余预算；缺空间 `RESOURCE_WAIT`，不删除历史结果、不自行清理用户数据。下载权重计入持久空间，当前1.9GiB不足以假定可部署6.5GB级模型。
- 设备/模型支持精度、分辨率、推理模式只能在生成前冻结；OOM不允许悄悄换小模型或改协议。batch缩小且数学不变可记录后重试同cell，仍失败就资源等待。
- stage可独立续跑。cache key覆盖音频PCM、解码视频像素及PTS、ROI/支持、模型/代码/frontend/环境、seed和参数。只文件存在不是命中。结果原子写入，失败/中断不留下complete标志。

## P7. 产物与独立验证

```
runs/tts_native_gain_attribution_<run_id>/
  protocol.json / claim_registry.json / inputs.json / model_provenance.json
  00_audit/                 # 12组、来源、资源、hash与参考肖像
  01_audio/                # ORIGINAL/A0/GAIN/NOISE/DENOISE、算法参数、操纵检查
  02_fixed_video/           # 180科学+18控制身份、特征、矩阵、控制结果
  03_generation/           # 144科学+4重复视频，消费输入证明，种子与初态
  04_crossed_scores/        # 336科学+8控制身份、特征、矩阵
  05_analysis/             # 6主对比、完整四格、raw/unit、396错配、节奏、图
  06_perception/           # 96同步+48音质对、匿名映射、空评分模板、分析
  report.md / final.json / validation.json
```

Cell manifest 必须具有 `stage,id,source_group,source,video_type,driver_condition,eval_condition,seed,video_hash,pcm_hash,roi_hash,support_hash,model_hash,code_hash,matrix_hash,reused_from,status`；失败明确error_type与reason，数值不存在用null、不用0。

独立validator从绑定输入、PCM、帧/PTS、矩阵和抽样索引重新计算六个主对比及全部结论，不导入runner/analysis里的判定函数，不信任final自报complete；基础I/O/hash可共享。矩阵身份/四格置换/缺失seed/来源重复/控制失败/变换偏移/报告状态都要验收。

必要测试：

1. 手写四格 `q00=2,q01=3,q10=4,q11=7` 得 G=2,E=1,I=2,total=5；−D方向另测。
2. 构造矩阵证明先mean再median/min与逐帧C平均不同，验证offset符号及±5帧实际音频控制。
3. 距离整体加常数C不变、正数缩放C同比变化，单位范数诊断与official字段分开。
4. 音频恒等重建、长度/hash/GAIN/SNR/联合headroom及无裁剪；合成已知噪声检验噪声分量恢复，不用Sync-C测试音频算法。
5. 同组重复记录不增大独立样本数、两seed先平均、六项校正固定。
6. 篡改PCM/视频身份、交换q01/q10、删一seed或一cell、修改support、伪造complete，validator必须失败。
7. 外部GPU活动/磁盘不足/中断续跑拒绝无效cache，不启新GPU任务。

计数验证与测试通过是工程完成；报告还必须逐条回答：已定位哪条路径、哪些操作产生方向性响应、是否复现fresh native、训练分布推断还有什么缺口、感知是否真的评过。报告Sync-C显示3位小数，机器数据保留精度。
