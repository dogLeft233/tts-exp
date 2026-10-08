---
title: TTS 音高起伏双向交换的 Wav2Lip 因果诊断 Spec
type: research
permalink: tts-exp/research/tts-音高起伏双向交换的-wav2-lip-因果诊断-spec
status: concluded
execution_status: pilot_blocked
protocol: tts_f0_swap_v1
date: '2026-09-17'
question: 共同有声区的F0起伏是否能双向转移Wav2Lip的生成端SyncNet得分优势？
tags:
- tts
- wav2lip
- f0
- causal-intervention
- spec
result: tts-exp/experiments/tts-音高起伏双向交换-wav2-lip-pilot-结果-2026-09-17
---

# TTS 音高起伏双向交换的 Wav2Lip 因果诊断 Spec

## 0. 给下游的任务与范围

实现并执行一个小型、冻结参数的科学实验：检验“把 TTS 的音高起伏交给自然语音，是否改善 Wav2Lip 的生成结果；反向交换是否使 TTS 退步”。

仅使用本地 Wav2Lip + SyncNet V2。复用已有中文自然/TTS 音频及各自 MFA，不调用 Ditto、云服务，不训练模型，不重新生成 TTS。本 spec 已完成音频 pilot；正式阶段因预注册 QC gate 未进入 Wav2Lip/SyncNet。默认 4 对音频技术 pilot + 24 对正式样本；正式 8 条件/对，192 个视频，另最多 2 个重复控制视频。技术 pilot 不生成视频，不进入统计。

核心区别：主检验固定评分音轨，观察视频变化；“生成与评分同时换音频”的结果只做辅助。不能把 own-audio 涨分直接写成生成端改善。

实现保持简单：一个 runner、一个小型独立复算脚本、一个针对科学契约的测试文件即可；复用现有渲染/评分 worker。不搭新框架、不复制一套模型推理系统。

## 1. 前置证据与问题边界

- [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]：局部正确/错位的排名差异阳性；跨音素事件、实例交互、迁移均未确认；同轮新 Wav2Lip 队列原生 TTS 优势也未确认。
- [[18-tts-feature-control-validation]]：做过全局 median-F0 pitch shift 的音频检查，但其下游 GPU 结果表没有 F0 条件；不能称为完整 F0 机制实验。
- [[TTS 声学变化重组轨迹与剩余项干预：实现与结果]]：共享模板含音素信息，不代表功能干预已定位原因。

本轮只检验共同有声支持上的、时间映射后的 F0 起伏，不检验音素时长、停顿、清浊转换、完整韵律或所有微小抖动。F0 是连续量；不能称其为离散音素结构。中文音高起伏含声调及语调，阳性不能再细分为二者之一。

## 2. 输入审计与样本冻结

### 2.1 已确认存在的本地入口

主 manifest：
`runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json`

资源根：
`results/rhythm_style_500/aishell1_test_400/`

- `natural/{sid:04d}.wav`
- `tts/{sid:04d}.wav` 与同名 JSON/目录内 provenance
- `transcripts/{sid:04d}.txt`
- `mfa/natural/{sid:04d}.TextGrid`
- `mfa/tts/{sid:04d}.TextGrid`
- `audit/tts_meta.json`

manifest 内可能是旧 /mnt/e/... 路径。只允许用已确认资源根作明确的前缀重定位；必须检查 sample_id、paired_key、source_utterance_id、真实 speaker_id、transcript 和文件内容绑定。source_sha256 等字段先追溯其语义，禁止把“源自然音频 hash”误当作“TTS 输出 hash”。冻结实际使用文件的 SHA256，不盲信旧字段名。

不得用 `results/aishell100_phoneme/alignment/manifest.json` 的字符级/词内均分 token 替代真实 phone tier。读取各自 TextGrid 的 phones/words，不能给 TTS 复制自然 TextGrid。若旧对齐无法验证绑定，可对相同音频本地重跑 Mandarin MFA；保存原对齐与新对齐的区别，不生成假边界。

### 2.2 固定选择法：只看输入，不看分数

在任何新 SyncNet 评分前冻结候选清单和排序，禁止按历史涨分挑样本。

1. 配对原始 natural/raw 与 faster_qwen3 TTS/raw；同文本、自克隆 provenance 可查，排除派生 warp/VC/MVP 音频。实际 provider 不符则报告 INPUT_UNAVAILABLE，不默换 provider。
2. 两音频均完整、finite、单声道可转 16k，时长各 3–12 秒；不裁句、不变速。解码后 abs(sample)>=0.9999 的比例不超过 0.001。
3. 两端独立 MFA 的非静音 phone 序列须一致：只 strip 空白、Unicode NFC；保留声调标记。忽略集合固定为 {"", "sil", "sp", "<eps>"}；出现 spn/unk/<unk>、序列不一致、非单调/越界超过 10ms，排除整对，不用 LCS 猜配对。
4. 每位真实 speaker 内按 SHA256("f0_swap_v1|" + paired_key) 升序排列；speaker 按 ID 升序。全量输入审计列出排除理由。
5. pilot：前 4 位有候选 speaker 的第 1 对，共 4 对。
6. formal：删除 pilot 后，每 speaker 取排序最前 3 对；不足 3 对的 speaker 不入 formal。取前 8 位合格 speaker，目标 8×3=24。
7. 如只有 6–7 位合格 speaker，使用 18/21 对并明确标记 reduced cohort；少于 6 位或少于 18 对，只完成技术可行性，不作主统计结论。不改用单说话人或英文。
8. 正式名单一旦冻结，后续音频 QC、渲染或评分失败保留 missing，不补选。最终科学主分析至少 18 个完整 pairs、6 位 speaker，且每位纳入 speaker 至少 2 个完整 pairs；不满足即 INSUFFICIENT_SUPPORT。

这批资产有历史使用记录，本轮称“预先固定的探索实验”，不能称独立未见确认。旧 split 标签不改变，本实验不训练。

### 2.3 固定图像和运行时

复用 `runs/tts_time_instance_20260917_v1/inputs.json` 中 sample 151 的 portrait，并从该 run 的 `B/audio_manifest.json` 或 `B/video_manifest.json` 找出实际渲染 box。同一图像、同一 box 用于全部中文样本和全部条件；冻结图像、解码 RGB、box 和 checkpoint。若该路径不可用，先报告缺失，不从评分结果选择人脸。

本地模型入口：
- Wav2Lip: `third_party/Wav2Lip/checkpoints/wav2lip_gan.pth`
- Wav2Lip Python: `[redacted-local-path]`
- SyncNet: `third_party/syncnet_python/data/syncnet_v2.model`
- SyncNet Python: `[redacted-local-path]`
- render worker: `scripts/experiments/static_image_bridge/render_worker.py`
- 可参考 `scripts/experiments/tts_time_instance.py` 的静态渲染与特征评分，但不能继承其英文数据、样本 ID、统计或音素坐标方案。

冻结实际版本/hash。GPU 可用则顺序使用；CPU fallback 必须记录且同阶段条件一致。单图像限制需写进结论。

## 3. 八个条件与记号

每对样本的接收方 R∈{N,T}，供给方 D 是另一端。各接收方生成：

| suffix | 含义 |
|---|---|
| RAW | 原始音频，经统一解码/重采样及本节统一增益策略 |
| ID | WORLD 原样重合成，使用 R 自己的 F0、sp、ap |
| LEVEL | 在共同有声支持区，仅转移供给方的平均对数音高偏移 |
| CONTOUR | 在相同支持区，转移去平均音高后的起伏，保持接收方整体平均对数音高 |

八臂为 N_RAW/N_ID/N_LEVEL/N_CONTOUR/T_RAW/T_ID/T_LEVEL/T_CONTOUR。
LEVEL 是共同支持区的音高水平控制，非全句无条件 pitch shift。CONTOUR 使用半音域均值，非中位数；不要混写。

统一：16kHz mono、float64 分析、PCM16 落盘；重采样方法冻结。RAW/ID/LEVEL/CONTOUR 均不移动音频时间轴。

## 4. WORLD 与精确干预公式

采用 pyworld：harvest → stonemask 得 F0；cheaptrick 与 d4c 提取 sp/ap；synthesize 重合成。固定 frame_period=5ms、f0_floor=60Hz、f0_ceil=600Hz，其他默认参数和版本落盘。输入 contiguous float64。sp/ap 只从 R 原始音频提取一次，三种重合成共用相同数组，不能用修改后的 F0 重新分析 sp/ap。

实现依据：[PyWORLD 官方接口](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)。它提供 F0/sp/ap 分解及重合成；本 spec 的干预公式、门槛与推断规则是本实验自定。

默认 Python 当前未发现 pyworld；下游可在实验专用环境安装 CPU pyworld 并记录版本，不改坏现有 Wav2Lip/SyncNet 环境。安装或运行失败记 AUDIO_BACKEND_UNAVAILABLE，不用普通变速或全局 pitch_shift 冒充曲线交换。

### 4.1 对应时间和共同支持

同位置非静音 phone occurrence 一一对应，反向交换另行计算：

- R phone=[a,b)，D phone=[c,d)。
- 对 R 的 WORLD 帧时间 t，u=(t-a)/(b-a)，供给时间 q=c+u(d-c)。
- WORLD 时间用返回的 t 数组，不能套用 SSL 的帧中心规则。
- q 的左右供给 F0 帧必须在同一供给 phone 内、相邻、都 voiced(F0>0)，且各自距 q ≤10ms；用半音域线性插值。q 恰好落在帧上时可直接使用该帧。
- R 当前帧也必须 voiced 且在 phone 内。不跨无声间隙、phone 边界插值，不外推。
- 所有其余位置保留 R F0；R 的原始 F0==0 位置永远保持 0。
- 每 phone 内将有效支持拆成连续帧段；段长不足 5 帧不干预。每段 j=0..L-1：
  w_j=min(1, j/4, (L-1-j)/4)。
  段两端权重为0，内部20ms渐入/渐出。不跨 phone 连成一段。
- M 定义为 w>0；两种干预复用完全相同的 w/M。保存每帧供给位置、phone 索引、插值下标/权重、w 和无效原因。

### 4.2 半音域计算

对 voiced 帧 s_R(t)=12 log2(F0_R(t))；M 内供给插值为 s_Dmap(t)。
在整句共同支持上算加权均值：
μ_R=Σ(w s_R)/Σw，μ_D=Σ(w s_Dmap)/Σw。

ID: s_ID=s_R。

LEVEL:
s_LEVEL(t)=s_R(t)+w(t)(μ_D-μ_R)。

CONTOUR:
s_CONTOUR(t)=s_R(t)+w(t)[(s_Dmap(t)-μ_D)-(s_R(t)-μ_R)]。

M 外不改，F0=2^(s/12)；无声位置仍0。CONTOUR 的全句 voiced 半音均值与原 R 相同（ΣΔs=0），这是必须测试的恒等式。由于渐入/渐出权重会在恒等式检查中再次加权，实现会在应用 taper 后去除该加权残差；该修正只用于保持预注册的均值恒等式，不改变 M 外音频或干预方向。它不是每个 phone 均值都保持；phone 间相对音高也属于本轮起伏。保留供给方起伏幅度，不做方差归一化、不搜索混合强度。

禁止逐帧 clip F0。目标 F0 任一 voiced 帧超 [60,600]Hz，则该接收方干预不可行，整 pair 记 QC_FAIL，不缩放到刚好过门。

### 4.3 长度、增益和波形边界

WORLD 输出最多允许与原接收方相差一个5ms hop+1 sample=81 samples；只在末尾裁剪/补0使长度精确相同。更大差异报错，不循环/变速。记录实际长度修正。

对 ID/LEVEL/CONTOUR 分别仅做一次全句 RMS 匹配至本侧 RAW 的 RMS；再对该接收方四臂共同乘同一个安全系数 min(1,0.98/四臂最大峰值)，保存所有系数。这使四臂总体 RMS 可比，避免单臂 limiter。禁止局部包络修复、动态压缩、去噪或其他音频增强。

RAW 因共同安全系数可能有全局衰减，必须称“统一增益后的 raw anchor”，不是字节级原始波形。同时保存未处理原始文件 hash；本轮结论条件于这套响度控制。

不拼回静音波形：ID 是整个 WORLD 流程的处理对照。参数中未改无声位置不代表重合成波形那里逐样本不变。

## 5. 操作有效性与音频 QC（评分前冻结）

门槛为本轮工程性可解释条件，不是行业标准。pilot 允许修复 bug；若必须改公式/阈值，先修订 spec/版本，再从头冻结正式实验，不能看正式 SyncNet 后调门。

对每个接收方：
1. effective_coverage=Σw / R全部 voiced 帧数 ≥0.40；至少5个有 w>0 的 phone；voiced 总时长≥1秒。两方向均需满足。
2. finite、exact samples、无 clipping；sp/ap 与 ID 完全相同；输入 voiced mask 不变；CONTOUR 半音均值恒等式误差≤1e-8。
3. 另用 DIO→stonemask 重新提取输出 F0（同60–600Hz、5ms），不能把传给 synthesize 的目标 F0 直接当作测量成功。若该独立提取器无法可靠测 ID，不得对候选使用更宽阈值。
4. ID 相对原输入 harvest/stonemask：voiced mask 不一致比例≤0.10（全句帧），双方 voiced 上半音绝对误差中位数≤1、90分位≤3；覆盖不到原 voiced 的80%则失败。
5. LEVEL/CONTOUR 相对 ID：输出 voiced mask 不一致比例≤0.05；在 w≥0.5 且两输出均 voiced 的帧，比较“重新测量的干预−ID半音差”与“目标干预−原F0半音差”，误差中位数≤1、90分位≤3，测量覆盖≥80%。
6. 剂量诊断：w≥0.5 内目标差的 RMS 半音值；CONTOUR 的 RMS≥0.5半音才称该接收方有可辨识干预剂量。低剂量不按结果剔除，列出占比；少于75%的完整 pairs 在两方向都达到剂量门，则主结论为 LOW_MANIPULATION_DOSE，不能用阴性排除 F0。LEVEL 同样报告剂量，但不把低 LEVEL 剂量误写成“平均音高无作用”。
7. ID/候选整体 RMS 差≤0.1dB。辅助局部包络 QC：20ms窗、10ms hop，固定 ID RMS 高于其最大值−40dB的帧，计算候选/ID 的 abs dB差；中位数≤1dB且90分位≤3dB，否则记 ENVELOPE_CONFOUND。此类 pair 不进入“F0为主”的 complete-case，但保留原始结果和缺失理由，禁止再校正包络救样本。
8. 波形的谐波/频谱必然随 F0 改变。sp/ap 固定不等于音色/频谱/音质完全不变，禁止写“纯 F0 无混杂已证明”。保存长时频谱和包络差的描述性数值，不以候选 SyncNet 结果决定保留。
9. 生成 pilot 盲名试听包和对应表。若没有实际听检，写 QUALITY_NOT_ASSESSED，自动实验可继续；不得伪造听检或声调/可懂度保持的结论。人耳破音、声调保持与音质需要后续独立评价。

pilot 至少3/4 pairs在两方向均通过第1–5、7项数值门，才进入正式渲染；不满足时输出 MANIPULATION_NOT_VALIDATED，交付音频/曲线/失败诊断，不能把它解释为F0无效。pilot的剂量仍如实报告，不用于搜索干预强度。

任何数值 QC失败记录在 audio_qc.csv，整 pair 完整性按八臂共同定义。报告冻结24对、成功数、各原因失败数，不能只写幸存样本均值。

## 6. Wav2Lip 与评分矩阵

### 6.1 生成

固定25fps、seed42、静态图、box、模型、推理参数；音频用对应臂完整波形。N四臂总长度相同，T四臂总长度相同；N/T之间允许原始时长不同。不做“自动最佳偏移修正”后再评分。

先生成正式 RAW/ID 四臂，用作原生与声码器背景检查；再生成通过音频 QC 的 LEVEL/CONTOUR。背景优势是否显著仅限制解释，不据此挑样本或反复换队列。技术失败则不浪费资源继续该 pair。

### 6.2 每个接收方10个评分 cell

令 V_a 是 a音频驱动的视频，A_a 是该音轨，C(V,A) 是 Sync-C。

- own四项：(RAW,RAW)、(ID,ID)、(LEVEL,LEVEL)、(CONTOUR,CONTOUR)。
- 固定音轨看生成变化：(LEVEL,ID)、(CONTOUR,ID)。
- 固定视频看评分音轨变化：(ID,LEVEL)、(ID,CONTOUR)。
- 原音轨替换诊断：(ID,RAW)、(CONTOUR,RAW)。

每 pair 20 cell，24 pair最多480 cell；重复/延迟控制最多再4 cell。不需要480次模型提特征：每个视频/音频提取一次后缓存 embedding，交叉组合计算距离。不能把把音轨换入MP4的操作误当作重新生成视频。

每侧10 cell使用完全相同的有效时间窗口集合 U：对该侧全部可用臂的音视频特征长度，取足够容纳所有 lag∈[-15,15]、5帧SyncNet窗口的内部交集。每窗对应实际时间；至少50个窗口。不要把N侧第i行与T侧第i行认作同一发音。N/T原生比较是逐句整体分数配对，不需要共用声学时间轴。

保留逐窗口逐lag raw L2距离矩阵，不单位化embedding。每个lag先在U取平均 d(k)，再：
D=min_k d(k)
B=median_k d(k)
C=B-D
k*=argmin_k d(k)，并列取最小k。

同时报告 C/D/B/offset，以及固定本侧 ID/ID 的 k0 后各cell的 d(k0)。不能用各cell各自最优lag作固定lag证据。显示 C 保留3位小数，统计用原始精度。

### 6.3 小控制

按正式排序第一个可评分 pair，额外独立重生成 N_ID、T_ID（最多2视频），验证score曲线最大差≤1e-4；超过则定位确定性问题，记录修复。
在该pair N_ID视频上给评分音频注入±3200 samples延迟，仅通过末尾裁/补保持长度，不重生成视频；记录显式时间索引约定，最优lag变化应与±5帧相符（容差1帧；正负符号先由实现索引定义）。不满足则修复评分坐标，不搜索其他样本来通过控制。延迟样本若最佳lag原本在边界导致无法检验，记控制不可用并停止因果判定，不能当PASS。
重复与延迟控制不进入主统计。

## 7. 预注册统计

所有主统计使用同一套通过双向 QC、完整20cell的 pairs。某speaker仅余1个完整pair时，该speaker不进入主统计，其数据仍列入描述表；随后重新检查18 pairs/6 speakers支持门。完整案例结论仅条件于这些可成功处理的样本，报告缺失可能造成的选择偏差，不能外推为原始全队列平均效应。各speaker先对其完整pairs等权平均，再对speaker等权平均。用 speaker cluster bootstrap：PCG64 seed20260918，20,000次，整speaker有放回抽样；每次沿用该speaker全部pair均值，不独立重抽帧/phone。保存bootstrap索引，分位数method=linear。

### 7.1 四个主统计（正向均表示支持假设）

以 N_C=N_CONTOUR、T_C=T_CONTOUR 简写：

g_N = C(V_N_C,A_N_ID) - C(V_N_ID,A_N_ID)
l_T = C(V_T_ID,A_T_ID) - C(V_T_C,A_T_ID)

h_N = C(V_N_C,A_N_ID) - C(V_N_LEVEL,A_N_ID)
h_T = C(V_T_LEVEL,A_T_ID) - C(V_T_C,A_T_ID)

g_N：自然借TTS起伏的生成端得分收益。
l_T：TTS借自然起伏的生成端得分损失。
h_N/h_T：上述方向是否比仅换整体音高更明显。

四项报告普通95%与Bonferroni98.75%双侧区间；主要判定看校正区间。不能只用 g_N+l_T 为正，掩盖其中一方向失败。h阳性也不意味着LEVEL绝对无效。

### 7.2 背景门（不替代主统计）

同一complete cohort：
G_raw=C(V_T_RAW,A_T_RAW)-C(V_N_RAW,A_N_RAW)
G_id=C(V_T_ID,A_T_ID)-C(V_N_ID,A_N_ID)

两者普通95%CI下界都>0，才能说本队列原生优势复现且重合成后仍可研究。不是要求G_id=G_raw；同时描述 G_id-G_raw。仅因G_id区间跨0不能写“声码器破坏了优势”，应写未确认保留。

RAW/ID已成功但其他条件失败的较大样本集也报告背景分数，标明分母，不能拿其阳性替代主cohort的门。

### 7.3 辅助结果（描述性95%区间，不扩展主阳性）

- own gains：C(V_R_CONTOUR,A_R_CONTOUR)-C(V_R_ID,A_R_ID)。
- evaluator response：C(V_R_ID,A_R_CONTOUR)-C(V_R_ID,A_R_ID)。
- interaction：C(V_R_CONTOUR,A_R_CONTOUR)-C(V_R_CONTOUR,A_R_ID)-C(V_R_ID,A_R_CONTOUR)+C(V_R_ID,A_R_ID)。LEVEL同算。
- 原音轨替换：C(V_R_CONTOUR,A_R_RAW)-C(V_R_ID,A_R_RAW)，N侧最贴近replace目标。
- 主统计各自的D、B和固定k0距离分解。
- pilot/QC/剂量与结果散点仅描述，不做看到相关后再选子群的机制检验。
- 对g_N/l_T校正CI若完整落入[-0.100,+0.100]，可描述“在此协议下排除了超过0.100 Sync-C的双向单项效应”（逐项说）；区间更宽则只能不确定。0.100是本轮事先设定的实际效应参考值，非公认阈值。

不得报告“解释了多少百分比TTS增益”的中介比例：两侧时长、重合成域与评分音轨不同，不具备该因果中介分解条件。

## 8. 唯一判读表

先分别输出 engineering_status、manipulation_status、native_status、scientific_status，不把程序PASS当科学POSITIVE。

1. INPUT_UNAVAILABLE / AUDIO_BACKEND_UNAVAILABLE / MANIPULATION_NOT_VALIDATED / CONTROL_FAIL：实验不可解释，报告已做部分与具体原因。
2. INSUFFICIENT_SUPPORT / LOW_MANIPULATION_DOSE：不能下F0阴性结论。
3. g_N和l_T校正下界都>0：BIDIRECTIONAL_GENERATOR_SCORE_EFFECT。
   - 且G_raw/G_id背景门均通过：支持“这种F0起伏干预参与了本队列Wav2Lip/SyncNet优势”，仍非所有TTS的原因。
   - 背景门未通过：仅说明双向干预有效；不能说解释了本队列不存在/未确认的原生优势。
   - h_N和h_T也都阳性：再加 CONTOUR_OVER_LEVEL；否则不能定位为起伏独有。
4. 仅g_N阳性：NATURAL_DIRECTION_ONLY，候选增强线索，不是完整原因。
5. 仅l_T阳性：TTS_DIRECTION_ONLY，单向破坏敏感，不是可迁移收益。
6. 主统计不阳性但own阳性/均值偏正：按各区间描述评分端或联合响应，不宣称生成端改善。
7. 其余：INCONCLUSIVE；若支持窄区间才使用第7.3节的有限排除结论。
8. 两边own都下降并不能证明F0无用，必须讨论重合成损伤/声学配合。

任何阳性都只涉及评分响应。固定评分音轨减少了音频评价端变化，但Wav2Lip视频也可能更迎合SyncNet；无人工盲评不能写真人可感知嘴型更准。原音轨替换阳性才构成replace方向的初步线索，仍不启动头训练或宣称泛化。

## 9. 最小实现与产物

建议新增：
- `scripts/experiments/tts_f0_swap.py`
- `scripts/experiments/tts_f0_swap_recompute.py`
- `tests/experiments/test_tts_f0_swap.py`

CLI统一 `--run-id <id> --stage audit|pilot|audio|render|score|analyze|validate|report|all`。支持已有产物恢复；输入、公式或代码改变须失效对应缓存，禁止静默读旧成功状态。实现可局部调用已验证worker；不要直接调用其他实验的all流程。

产物根 `runs/tts_f0_swap_<run_id>/`：
- protocol.json / inputs.json：固定参数、排序、pilot/formal ID、真实speaker、路径/hash、环境/model/portrait/box、代码与spec hash。
- audio/*.wav 与 parameters/*.npz：原/目标/重提F0、sp/ap、时间、映射、w、缩放。
- audio_qc.csv：所有冻结pairs的质量、剂量、覆盖、失败理由。
- videos_manifest.json、features_manifest.json、distances/*.npz、scores.csv。
- analysis.json：逐pair与逐speaker统计、4主项、背景门、辅助项、缺失分母和bootstrap索引。
- validation.json、report.md；一张双向效应区间图及pilot F0曲线图即可。
- 不创建多套冗余计划/审批文件，不写进度日志。

阶段失败也生成可读报告。最终BM创建一篇结果笔记并关联此spec；更新此spec的execution_status与result链接，保留changelog。读写遵循Startup Router。

## 10. 必须自审的科学契约

测试以发现科学错误为目标，不复制实现输出作为expected：

- 已知解析F0曲线：常数偏移只有LEVEL变化，均值相同而起伏不同只有CONTOUR变化；identity供给必须两臂回到自身。
- CONTOUR全句voiced半音均值恒等式；无声、M外不改；两方向用各自接收方时间轴。
- donor q落在无声间隙/边界/重复同名phone时，不跨段插值、不串 occurrence；保留声调。
- sp/ap数组复用、精确长度、共同峰值缩放；没有用频率变化暗中改变时长。
- 渲染用正确驱动音频；固定音轨cell确实相同音频hash；固定视频cell相同解码帧hash；20cell无漏配/错配。
- 从距离矩阵独立算 C/D/B/k0 与主统计，符号测试保证“自然升、TTS降”两者都为正。
- bootstrap按speaker抽样，不能按phone/window抽样；缺失不会变0；pilot不能进入formal。
- 独立复算脚本不import生产统计/映射函数，至少独立复算全部主统计和区间，并抽查每侧1条真实F0映射、加权均值与sp/ap绑定；代码测试+产物验证都通过后才标工程完成。

完成后用普通中文回答：交换是否成功、原生优势是否复现、自然方向是否改善、TTS方向是否退步、起伏是否优于水平控制、固定原音轨结果如何，以及能否解释TTS优势。阴性/不确定同样是正确执行结果，不能为追求阳性改协议。

## 后续修订：均值公式与测量链

September 17, 2026 编写修复 spec 时，直接复算旧 NPZ 发现：旧代码保证的是 taper 加权均值不变，全部有声帧普通平均仍偏移约 −0.043627～+0.054398 半音。因此上文第4.2节追加的 w² 加权残差修正与原整句普通均值目标不一致；后续实现须删除该修正，按 [[TTS 音高起伏交换的 WORLD 与测量链修复 Spec]] 执行。该附件同时明确剂量在 w≥0.5 支持上计算、QC必须重读最终WAV，以及RAW的检测器对照。原 run 的1/4通过数仅保留为历史程序判定，不能作为修订后有效性证据；正式阶段仍未运行。

## Observations
- [status] concluded / pilot_blocked
- [result] 2026-09-17 本地执行 run `tts_f0_swap_f0spec_audit2`：4 对 pilot 完成音频生成，1/4 对双向通过 DIO→stonemask 数值 QC，正式阶段按规则阻断
- [diagnosis] 失败主要是 WORLD 重合成后独立 DIO 的 voiced mask 与原 harvest/stonemask 不一致、原 voiced coverage 不足；不能解释为 F0 无效
- [invariant] 通过 pair 的 CONTOUR 加权均值恒等式误差约 3e-15；参数、sp/ap、长度、增益和 hash 产物已保存
- [boundary] 没有 Wav2Lip/SyncNet 分数，因此没有自然方向、TTS 方向、CONTOUR 对 LEVEL 或 TTS 增益原因的结论
- [decision] 不放宽预注册 QC、不按分数补选；后续若继续须先更新协议并重新冻结

## Result

- [[TTS 音高起伏双向交换 Wav2Lip Pilot 结果 2026-09-17]]
- run: `runs/tts_f0_swap_f0spec_audit2/`
- status: `MANIPULATION_NOT_VALIDATED`
## Relations

- follows [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]
- relates_to [[18-tts-feature-control-validation]]
- relates_to [[TTS 声学变化重组轨迹与剩余项干预：实现与结果]]
- relates_to [[MFA-linear 轨迹增益的匹配项与背景项分解 2026-09-16]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求将双向F0实验限定为本地Wav2Lip，核实已有中文资产，冻结干预、评分矩阵、统计与下游验收规则 | September 17, 2026 | user request |
| 完成 runner、独立复算器与科学契约测试；执行 4 对音频 pilot，因 1/4 通过 QC 阻断正式阶段 | September 17, 2026 | agent execution |
| 编写后续修复spec时发现普通均值与加权均值契约冲突，补充修订指针及旧产物解释限制 | September 17, 2026 | agent audit |
