# LRS3 bridge 的 TTS 来源与质量比较

## 1. 问题、假设与解释范围

`N` 是原始自然音频；`T_p` 是来源 p 的原始 TTS；`M_p` 是把 TTS 的声学特征按音素放回自然时间网格后，经同一声码器得到的目标音频；`B_p` 是保留 N 相位、向 M_p 幅度谱移动的 bridge；`V(X)` 是用 X 驱动 Wav2Lip 得到的视频。p 只取 `LOCAL`、`CLOUD`，名称不预置质量高低。

本实验依次回答：

1. 在完全相同的 N、视频条件和评分条件下，CLOUD bridge 的 replacement Sync-C 增益是否比 LOCAL 更大？这是唯一主检验。
2. 每个 bridge 自身是否真的优于 N？这是两项预注册次要检验，不能用来源相对优势代替。
3. CLOUD 的原始 TTS、进入 bridge 的 M 是否确实听起来更好？逐条音质差是否与逐条 replacement 差同向？这是独立质量验证和探索性关联。

历史发现轮 `MAG_075` 的增益约 +0.078，确认轮约 +0.031，确认轮 95% CI 为 [-0.034, +0.105]。不能把它们当成确定效应、功效估计或本地/云端对比。本实验复用确认样本，标记 `historical_exposure=true`、`scope=fit_only_paired_pilot`。不声称 22 条足以检出 0.030 的差异；必须报告 CI 宽度，跨零写不确定，不写两者相同。

## 2. 固定输入与两阶段冻结

所有路径相对仓库根。`input-bindings.json` 记录本次设计核对的父文件及 SHA-256。

队列只取 `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json` 原顺序的全部 22 条、22 source groups。ordered-ID hash 沿用 `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`。禁止重抽、增补、只取历史获益样本或把 AISHELL/RAMC 混入。source group 不是已验证的 speaker identity。

### 2.1 在任何新音频生成前冻结 setup

Stage 00 写 `setup.json`：22 个 ID、source group、transcript、自然参考与 face video 的 file hash/PCM hash、原始父清单 hash、两 provider 身份、软件/权重绑定、MFA 配置、渲染配置、随机种子、音质问卷与分配规则、完整矩阵和统计规则。不得读取历史逐条分数选择记录。

本地固定 `faster_qwen3`、`Qwen/Qwen3-TTS-12Hz-0.6B-Base`、ICL，记录实际权重 revision/hash 与实际调用类。现有 provider 有 backend fallback；实验必须在预检时解析并冻结实际 backend，运行中不允许静默切换。云端固定历史 `dashscope_vc / qwen3-tts-vc-2026-01-22`，不根据“更高质量”改为别的模型，不推断云端参数量。

### 2.2 输入审计与补齐

云端原始清单 `runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json` 已核对包含这 22 个 ID。逐条将 `tts_transcript`、`reference_audio_sha256`、`canonical_audio_sha256` 与样本绑定；再回溯历史 candidate manifest 的 `tts_audio_sha256`，证明旧 M 确实来自这一来源，不能只依据路径中出现 cloud。

本地音频先按 ID + 文本 + reference PCM + 模型 + backend +生成配置查找已有 metadata。只有完全匹配本次 setup 的产物才能复用；路径相似、同名或来源未知不算匹配。缺失项在本 run 内生成，默认最多 22 次成功本地合成；每条只取一次注册输出。文本使用 cohort 的 transcript，`ref_text` 相同，`ref_audio=N`，`language=English`。局部 RNG seed 为 `int(sha256(sample_id UTF-8).hexdigest()[:8],16)`，记录实际 decoding kwargs（包括模型默认值）；调用外围隔离 RNG，不改 provider 的全局随机契约。

云端只复用现有产物，默认新云端调用预算为零；缺失或身份不符记录 INPUT_BLOCKED，不自动换日期模型。断点恢复只接受完整身份一致的成功文件。进程崩溃且尚无完整输出可在相同 seed/config 下做至多两次基础设施重试；输出已成功但听起来不好、长度不理想、MFA 失败或 Sync-C 低，均不能重抽。

保留 provider 原始波形；供实验的 T_p 统一 mono/16 kHz/PCM16。复用已绑定云端 canonical；新增本地用 float64 通道均值、`scipy.signal.resample_poly`（up/down 由 gcd 确定，window=('kaiser',5.0)，padtype='constant'），一次 PCM16 量化。不得自动删停顿、降噪、响度归一化或全局拉伸到 N 长度。非有限、零信号、越界、长度比不在 [0.5,1.5] 或超过 30 s 记输入失败，不筛掉后继续主检验。量化前 `rint(x*32768)` 必须在 int16 范围内，不靠 clip 修复。

### 2.3 在任何新 SyncNet 评分前冻结 analysis lock

音频、对齐、目标、mel、几何和共同评分支持全部完成后，写只读 `analysis_lock.json`，包含 setup hash 及全部输入/中间产物 hash。尚未产生的渲染输出 hash 不预填；之后用 stage manifests 链接。所有运行时路径、模型 revision、依赖版本必须实际解析，不能留 TODO 后启动正式推理。听评问卷/盲化映射此时冻结，人工评分可稍后导入；不得根据听评或 SyncNet 改输入。

## 3. 两种来源共用 MFA-linear 处理

MFA 3.4.1，词典和 acoustic model 均为 `english_mfa`，固定其实际文件 hash、命令、词典解析规则。对 N、T_LOCAL、T_CLOUD 分别对齐同一文本，N 只对齐一次供两边共享；共 66 个 TextGrid。在同一个冻结版本的处理代码下，对两种来源都重新生成 M；旧云端 M 仅作 provenance 和差异诊断，不在主矩阵中一边复用旧 M、一边使用新流程。

复用并冻结：

- `scripts/experiments/lrs3_mfa_linear_replacement/mfa_alignment.py` 的 token 解析、`frame_owners`、`build_frame_mapping`、尾部 token extension。
- 同目录 `candidate_audio.py` 的 `interpolate_conditioning`、`candidate_from_features` 和 canonical PCM 规则。
- `scripts/wavlm_knn_vc_adapter.py`：WavLM-Large layer 6，16 kHz、stride 320、1024 维，knn-vc revision `c616845c4e309e24d5927f15adbdf277a3d65358`，冻结 prematched HiFi-GAN；记录两个权重 hash。此处是特征插值后 vocode，不做新训练或 kNN 目标搜索。

映射语义必须相同：自然第 j 帧中心 `(j+0.5)*320/16000` 归属一个音素；相对位置映射到匹配 TTS 音素内，然后线性插值左右 WavLM 帧。不能把旧 DTW、波形重采样或全局 duration matching 当 MFA-linear。未匹配自然静音只能映到 TTS 静音帧，按全局相对位置最近、下标小者优先；未知音素/未匹配 speech 失败，不将 `spn` 当静音。没有可用静音也失败。最终 token 只允许延伸到最后一帧中心所需范围，最大延伸 0.020 s；更大的覆盖问题失败。

声码器原始长度必须为 `natural_feature_frames*320`；只允许既有流程的右端裁剪/补零恢复 N 的完整采样点数，记录调整量，绝对调整量 <320。不能把此尾部处理用于修复内部音素时序。M 不作 loudness normalization；按既有 `canonical_pcm_s16le` 的 `rint(x*32767)` 量化，明确它与下一节 bridge 的 /32768 解码属于不同环节，禁止混用比例常数。

每条两边输出 token/映射 trace、speech 匹配率、静音 fallback 比例、音素时长差、原始 TTS/N 时长比、尾部补齐量、RMS/peak/clipping。不同 TTS 的实际音素边界本来就可不同；不强制复制云端边界给本地。共同算法控制了处理方法，仍不能消除不同输入的 alignment error；须在质量解释里保留这一混杂。

任一条任一侧失败，保留全部失败原因和22条 denominator，主实验 `INPUT_BLOCKED`；可完成失败率与原始音质描述，但不得用成功交集冒充完整结果。新的容错方法需要新协议版本，不在本 run 自适应修补。

## 4. Bridge、重建对照与音频检查

四臂固定顺序 `N, B0, B_LOCAL, B_CLOUD`。N 保持原始完整 PCM 字节；B0 是同一 bridge 运算取 alpha=0 的重建对照，不是复制 N 文件。两个候选 alpha 均为 0.75，不能扫描、按 provider 单独调强度或挑更好的 alpha。

对自然与 M 的 int16 采样用 float64 `/32768` 解码，CPU float64 torch STFT：n_fft=win_length=1024，hop=256，periodic Hann，center=true，pad_mode=reflect，eps=1e-7。M 与 N 的样本数、STFT shape 必须相同。

```text
S_N = STFT(N); S_M = STFT(M_p)
a_N = max(abs(S_N), eps); a_M = max(abs(S_M), eps)
S_B = exp((1-alpha)*log(a_N) + alpha*log(a_M)) * S_N/a_N
b = ISTFT(S_B, length=len(N))
b *= RMS(N)/RMS(b)
if max(abs(b)) >= 0.999: b *= 0.999/max(abs(b))
PCM_B = rint(b*32768).astype(int16)  # 先断言范围合法，不以clip修复
```

B0 使用同样计算且 alpha=0，目标固定取 N，走相同 RMS/peak/量化。目标来源不能影响 B0。复用历史 `phase_preserving_blend` 数学，但实现可提取显式 alpha 参数，不修改旧实验冻结常量。保存缩放前后波形统计、RMS/peak 因子、file SHA-256 和真实 decoded PCM SHA-256。旧函数把 file hash 命名为 `pcm_sha256`，新代码必须区分这两个概念并在迁移读取时显式转换。

在正式评分前，以官方 Wav2Lip `audio.melspectrogram` 从落盘 PCM 重算（不是直接改 mel），按其实际加载器与参数生成 mel/chunks。各来源分别计算：

```text
d_p = mel(M_p)-mel(N); u_p = mel(B_p)-mel(N)
progress_p = dot(u_p,d_p)/max(dot(d_p,d_p),1e-12)
relative_distance_p = norm(mel(B_p)-mel(M_p))/max(norm(d_p),1e-12)
orthogonal_p = norm(u_p-progress_p*d_p)
```

同时报告 `norm(u_p)`、`norm(d_p)`、两来源目标间距离、waveform RMS/peak/缩放差。`norm(d_p)^2<=1e-12` 标记退化；不能把 0/0 处理后的数字解释为移动。每侧 movement gate：至少20/22 progress>=0.15，均值95% CI下界>0.15，无退化。失败仍完成完整矩阵，状态 `BRIDGE_MOVEMENT_FAILED`，不换目标重跑。相同 alpha 不保证相同实际 mel 位移；本实验报告该差异，不事后匹配位移强度。

## 5. 渲染、评分支持与固定矩阵

每条记录四个驱动臂，各做 r=0,1 两次独立渲染，输入音频相同但进程/缓存/workdir 独立；不能将同一视频复制成 repeat。r 的渲染 seed 固定为 20260913、20260914，在同一 r 内对所有臂相同。按固定 PCG64(20260913) 对 `(sample_id,r,arm)` 作一次调度排列，避免所有云端都在同一批机器状态下运行；保存调度表。不挑 repeat，不把重复当成44个独立样本。

绑定历史 Wav2Lip GAN 权重 SHA-256 `ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`、SyncNet V2 权重 SHA-256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`。渲染参考确认实验 `render.py`，新 runner 只复用算法/底层函数，不继承旧固定 run 路径、禁止新 TTS 的 scope 或终态规则。

每条使用 cohort 的同一原 face video，25fps。Wav2Lip生成端明确沿用历史full-frame box：`--box 0 H 0 W --nosmooth --face_det_batch_size 16 --wav2lip_batch_size 16`，H/W取绑定视频实际尺寸；不是逐臂重新检测人脸，禁止遇到困难时改ROI。crop/resize及Wav2Lip mel-chunk调度在评分前冻结并共享所有臂/repeat。视频驱动使用完整音频，不允许换成首帧、静态人脸或另一个face后仍称同一实验。为让官方 mel chunker 覆盖冻结支持的最后帧，每个驱动音频在原始 bridge 文件末尾追加固定 1,920 个零采样点（120 ms）；该副本只供 Wav2Lip 生成右侧上下文，保留原始文件 hash，评分仍读取原始音频的精确支持前缀，零填充不进入评分。记录 full-frame box 和原视频嘴部可能提供时间先验的限制，不能将实验结果外推到其他 ROI、静态脸或其他 TFG。

评分共同支持在渲染前由 N 和原 face video 确定：`F=min(source_decoded_frames, floor(len(N)/640))`，起点0，只评分前F帧与前F*640采样点。禁止 loop、变速、从每个候选评分成功的 track 反推支持或用 `-shortest` 隐式裁尾。完整 N 文件不改写；评分 N 是其共同支持上的精确前缀。每臂必须有至少F帧；缺失不是靠缩短全组支持补救。支持至少50帧、同一确定轨迹覆盖全部支持；若不能满足则 `INPUT_BLOCKED`。

为了排除预处理差异，每个生成视频在 r 内只准备一份冻结 SyncNet 裁剪，再给这个视频的不同评分音轨复用。若官方流水线仍逐 cell 检测轨迹，validator 必须确认裁剪帧/时间支持相同。不同生成臂也必须共用原先冻结的几何/窗口支持；不能各挑最优 track。SyncNet轨迹/裁剪由原face video的官方预处理确定，在analysis lock写入逐帧坐标、resize参数与有效窗口，随后将同一坐标应用到全部生成臂；生成端full-frame box与评分端SyncNet人脸裁剪是两个不同环节。裁剪端的官方模型前向必须与同裁剪输入的官方实现 parity（C/D绝对差<=0.001，offset相同），所有cell均校验；不得把自定义D0或代理分数标为官方 Sync-D。parity是同cell的验证调用，单列计算开销，不增加新的科学cell。此共同裁剪支持契约比历史逐视频预处理更严格，本次不是旧+0.031的数值重放，不将新旧绝对分数直接相减。

每个 `(i,r)` 恰好七个 cell：

| Video | Scoring audio | 用途 |
|---|---|---|
| V_N | N | 自然基线 |
| V_B0 | N | STFT重建对照 |
| V_B_LOCAL | N | 本地 replacement |
| V_B_CLOUD | N | 云端 replacement |
| V_B_LOCAL | B_LOCAL | 仅自身配对诊断 |
| V_B_CLOUD | B_CLOUD | 仅自身配对诊断 |
| V_N | N_REV | 固定视频的错误音轨对照 |

`N_REV = reverse(N[0:F*640])`，逐 int16 采样翻转，不重采样、不交叉淡化；只用于评分，不新渲染。它验证此端点对严重错误音轨的响应，**不验证生成器对局部时间扰动的传递**。

总计22*4*2=176个新 Wav2Lip 视频，22*2*7=308个唯一评分cell。official cell key 为 `(sample_id,render_repeat,video_arm,score_audio_arm,protocol_hash)`。mux 用 copy video stream、16k mono PCM s16le（使用支持PCM的容器如MKV），核对视频 elementary-stream identity与完整评分PCM前缀，记录实际ffmpeg/ffprobe、执行代码、命令与权重hash。任何 mismatch、缺失、重复或交叉样本绑定，均不出科学结论。

## 6. 不依赖 SyncNet 的音质测量

必需输出匿名听评包、CSV模板、问卷和私有盲化映射。每条四个刺激 `T_LOCAL,T_CLOUD,M_LOCAL,M_CLOUD`，共88个；原始T与M分两场次，场次顺序在评审者间平衡。至少3名能理解英语、不了解provider标签/同步结果的独立人工评审者；每个记录、每个阶段的两个来源由相同评审者评，缺值不得填均值。评审者池与匿名ID在开始听评前冻结。本地/云端位置按 PCG64(20260913) 为每个评审者随机分配，保存种子和mapping，不在文件名/标签泄露provider。只提供参考文本，不播放同步分数或提示预期赢家。

主音质题为“整体听感质量”：1很差、2较差、3一般、4较好、5很好，综合自然度、清晰度、可闻失真。另独立标记漏读/重读/错读以及是否有明显伪影；不把这几个标签加权成新指标。不测speaker similarity，不能把其缺失解释为身份一致。

听评副本统一16k，逐刺激仅施加一次全局增益使RMS为-26dBFS，若峰值>=0.999再全局衰减，保存因子。无裁剪静音、时长修改、去噪；这些副本**永远不进入模型与SyncNet矩阵**。结论对应响度控制后的听感质量。完整原始/实验音频另保留供审计。

每条分别计算 `q_raw_i = mean_rater(Q(T_CLOUD)-Q(T_LOCAL))`，`q_target_i = mean_rater(Q(M_CLOUD)-Q(M_LOCAL))`。主报告每阶段均值和CI；质量比较的统计重采样同时有放回抽取22个记录和评审者，保持两来源/两阶段配对，10,000次PCG64(20260915)，percentile95%。不把66份评分当66个独立语音样本。匿名听评不足3个共同评审者的任何pair，则该阶段质量状态为NOT_ASSESSED；仍保存现有评分，不能宣称完整质量验证。

`q_raw`与`q_target`两阶段CI均下界>0才标记 `CLOUD_QUALITY_ADVANTAGE_OBSERVED`；raw有优势、target未建立优势标 `QUALITY_ADVANTAGE_UNCONFIRMED_AT_TARGET`，不把统计不确定写成质量优势已消失；两阶段都完成但不满足上述条件标 `QUALITY_ORDER_UNRESOLVED`。这几个状态用于解释，不用于选择音频。没有人工结果时完整自动实验仍可交付，质量状态 `QUALITY_NOT_ASSESSED`；模型不能假扮人类打分，也不能用文件名、参数量、Sync-C或未校准音质预测器代替。

## 7. 计算与统计

令 `C_i,p,r=SyncC(V_Bp,r,N)`，`C_i,N,r=SyncC(V_N,r,N)`，D类同理；B0单列。所有原始C/D保存官方输出精度，计算不提前舍入，报告Sync-C保留3位。

```text
gC_i,p,r = C_i,p,r - C_i,N,r
gD_i,p,r = D_i,N,r - D_i,p,r          # 两者正数都更好
gC_i,p = mean_r(gC_i,p,r); gD_i,p = mean_r(gD_i,p,r)
deltaC_i = gC_i,CLOUD - gC_i,LOCAL
         = mean_r(C_i,CLOUD,r - C_i,LOCAL,r)
deltaD_i = gD_i,CLOUD - gD_i,LOCAL
         = mean_r(D_i,LOCAL,r - D_i,CLOUD,r)
primary = mean_i(deltaC_i)
```

必须用同一份N基线相减，验证上面两种deltaC算法数值相等。例：某条云端gC=0.080、本地gC=0.020，则deltaC=0.060。若云端=-0.010、本地=-0.080，deltaC=0.070也不能称云端取得正replacement增益。

主 bootstrap 使用 PCG64(20260913)，10,000次有放回抽22个source groups，复用相同抽样索引计算所有端点；每组本协议只有一条。先在条内平均两次渲染，不重采样单独repeat/cell。percentile 双侧95% CI，线性quantile，所有判定严格比较未舍入值。主端点CI下界>0：`CLOUD_BRIDGE_STRONGER`；上界<0：`LOCAL_BRIDGE_STRONGER`；否则 `SOURCE_DIFFERENCE_UNRESOLVED`。deltaD为辅助端点，不能主C失败后换D宣布胜利。主结果不改成按speaker未知标签、帧或句长加权。

各来源相对N的绝对replacement增益另作两项次要检验：对gC_LOCAL/gC_CLOUD各报告双侧97.5% bootstrap CI（quantile .0125/.9875，Bonferroni控制两项C检验）。某来源的 `replacement_gain_observed=true` 需该CI下界>0，同时gD的95%CI下界>-0.100且至少20/22记录的两次offset均距对应N baseline<=1帧。再报告相对B0的gC/gD，不用它替代N主定义。

逐条关联是探索性：Spearman `(q_target_i,deltaC_i)` 为主要关联描述，raw同理另列。平均秩处理ties；对记录配对bootstrap 10,000次PCG64(20260916)，报告rho和95%CI；不将quality标签或重复渲染拆成独立样本。常量、有效记录不足22、或有效bootstrap<9500时rho/CI标undefined并说明，不填0。有效draw数量必须报告。不得按质量高低事后切分阈值或回归筛选特征。报告alignment fallback差、mel movement差和peak scaling差与deltaC的逐条并列表，保留混杂，不靠22条复杂回归宣称因果。

若质量优势已验证、主deltaC为正，仅可写“在这22条样本中，较高听评质量的云端来源有更强bridge replacement表现，符合质量关联假设”。rho的CI也>0时，可另外写逐条正相关。即使全部阳性，也无法区分provider、音色、韵律、对齐误差和质量的独立作用。若云端仅相对本地好而绝对gC未通过，必须明确没有建立正replacement增益。

## 8. 控制门槛与终态

控制在候选优劣解释之前计算，不能按控制结果调阈值：

- **重复性**：四臂各自r1-r0的C、D均值95%CI均完全包含于[-0.100,+0.100]开区间；每臂至少20/22 offset差<=1帧。另报deltaC在r0/r1的均值与差，禁止挑较好的repeat。
- **重建对照**：B0相对N、先平均r后的C和D差95%CI均完全位于(-0.100,+0.100)，至少20/22条两次offset都<=1帧。它检验STFT/RMS/量化本身是否带来可观端点偏移。
- **错误音轨响应**：每条先平均r，`damageC=C(V_N,N)-C(V_N,N_REV)`、`damageD=D(V_N,N_REV)-D(V_N,N)`；两者95%CI下界均>0.100，至少18/22条两者都>0。不对错误音轨要求offset一致，不声称已校准细微/局部时序敏感性。

自动实验完成后 `final.json` 必须分别保存 `engineering_status`、`controls_status`、两侧 `movement_status`、`source_comparison`、两侧 `replacement_gain_observed`、`quality_status`、`quality_association`、`scope` 和限制。互斥 scientific terminal 优先级：

1. 输入身份/格式/对齐缺失：`INPUT_BLOCKED`；渲染/评分/锁/矩阵/独立验证失败：`ENGINEERING_BLOCKED`。不得计算完整科学结论。
2. 完整矩阵但任一控制不通过：`CONTROL_FAILED`，来源差、绝对增益和质量数字仅描述，确认布尔值为false。
3. 控制通过但任一侧movement未通过：`BRIDGE_MOVEMENT_FAILED`，不称为合格双bridge比较。
4. 其余按主CI输出 `CLOUD_BRIDGE_STRONGER`、`LOCAL_BRIDGE_STRONGER` 或 `SOURCE_DIFFERENCE_UNRESOLVED`。即使听评未完成也输出这一自动结论，并将质量解释明确设为NOT_ASSESSED。

`quality_association`只在前三类阻断/失败不存在时解释；必须记录原始音质优势、目标音质优势和rho分别是否成立，不能压缩成“质量导致增益=true”。本协议永远 `causal_quality_effect_established=false`、`generalization_confirmed=false`。没有正结果不自动扩大样本/换模型/换评分器。

## 9. 阶段、产物、接口与资源

```text
runs/lrs3_bridge_tts_quality_<run_id>/
  00_protocol/      setup.json, input_audit.json, cohort.json, execution_order.json
  01_tts/           tts_manifest.json, local/, cloud_refs.json, failures.json
  02_targets/       alignment_manifest.json, targets_manifest.json, traces/, audio/
  03_bridge/        audio_manifest.json, diagnostics.json, geometry.json, analysis_lock.json
  04_quality/       listening_manifest.json, blind_mapping.json, ratings_template.csv,
                    ratings.csv (实际导入后), quality.json
  05_videos/        videos_manifest.json, 独立sample/arm/repeat目录
  06_scores/        matrix_manifest.json, scores.csv, raw_logs/, parity.json
  07_analysis/      paired.csv, controls.json, statistics.json, final.json, report.md
  validation.json
```

CLI要求 `python -m scripts.experiments.lrs3_bridge_tts_quality.runner --run-id <id> --stage <stage>`，stage取 `audit,tts,targets,bridge,quality-pack,render,score,analyze`；独立 `python -m scripts.experiments.lrs3_bridge_tts_quality.validate --root <path>`。`quality-pack`只制备听评材料，实际人工CSV通过 `--ratings <path>` 导入analyze；没有CSV时不阻止render/score/analyze。读入评分文件后只允许补充质量分析，自动评分结果、统计规则与hash不得改写；新增带父hash的analysis版本并保留旧final。

默认上限：22次成功本地TTS、0次新云端TTS、66个TextGrid、44个M目标、88个驱动音频（含N/B0）、176个Wav2Lip视频、308个评分cell、88个听评刺激*至少3人。不新增DTW、训练或大范围参数网格。两次render估计渲染不稳定性，**不能估计TTS随机生成方差**；结论限于本次绑定输出。

每个stage用 `protocol_hash + upstream hashes + cell key + runtime config` 做resume校验；成功cell不可按得分重试。失败有日志、完整身份和原因；主run不静默修改父资产/共用缓存。file hash、decoded PCM hash、video bitstream hash分别保存，JSON自hash排除自身hash字段并固定canonical序列化规则。

报告须给非项目读者说明“驱动音轨”和“评分音轨”区别，展示两条源流程、每臂C/D/offset、主差及CI、各侧绝对增益、B0/重复/错误音轨控制、raw→M质量差变化、逐条散点/数据表、失败与未知项。不得把尚未进行的听评、代码测试、推理或人工观看写成完成。
