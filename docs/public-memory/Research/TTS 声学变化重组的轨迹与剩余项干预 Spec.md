---
title: TTS 声学变化重组的轨迹与剩余项干预 Spec
type: spec
permalink: tts-exp/research/tts-声学变化重组的轨迹与剩余项干预-spec
status: spec_ready
date: '2026-09-16'
protocol: tts_acoustic_reorganization_v1
tags:
- tts
- mechanism
- spec
- trajectory
- syncnet
---

# TTS 声学变化重组的轨迹与剩余项干预 Spec

🔄 状态：spec_ready；仅设计，尚未执行实验。面向下游 GPT Luna Max。
协议名：tts_acoustic_reorganization_v1。仓库：[redacted-local-path]
本文件是本轮科学协议的唯一真相源。任务是实现、执行、审阅并交付可复算结果；阴性或不可判读均为正常科学交付。冻结所有模型，不训练增强头、不新增 TTS、不扩样、不调参。

## 1. 只检验一个假设

假设：TTS 重新组织声学变化，减少部分对口型生成无益的变化，同时保留与发音有关的动态过程。

将其收敛为可操作的局部版本：
1. 同音素、相同相对位置上，能从其他句子重复观察到的特征轨迹，可能承载有用发音动态。
2. 相对于该轨迹的剩余变化，可能含有可抑制的干扰。
3. 若 TTS 已减少这些干扰，继续削弱剩余项对自然来源的帮助应大于对 TTS 来源的帮助。

“可重复轨迹”记 P，“剩余项”记 R。P 不是已证实的纯发音信息，R 不是预先认定的噪声；R 也可能包含共发音、声调、合法语境变化。必须由后续干预结果检验其功能。按 phone/phase 定义的模板可能不充分，模板失败不等于整个假设被推翻。

主检验在固定自然评分音轨下进行，测生成视频的变化。另做小型交叉评分定位评价端响应。SyncNet 改善不自动等于人类口型同步改善。

## 2. 数据、来源和固定划分

父运行 PARENT = runs/mfa_linear_trajectory_ablation_20260916。
读取 PARENT/inputs.json、audio_manifest.json，以及其中绑定的自然/TTS 波形、MFA tokens、Z_N/Z_T、参考脸视频和模型。
配对键使用 paired_key，顺序严格沿用 inputs.records：
- 前 5 条为 donor，sample_id 1–5，只建模板；
- 后 10 条为 evaluation，sample_id 6–15，接受干预并统计；
- 不按 Sync 分数、音质、模板表现更换样本；donor 与 evaluation 无 paired_key 重叠。
这不是新的盲测：全部是历史已见 S0765 单说话人；固定 donor 只避免目标句进入模板，不构成说话人泛化。

父 audio_manifest.records 的 z_n_path/z_t_path 文件用 torch.load(..., map_location="cpu") 后取 ["features"]，形状 [F,1024]。
Z_N 是自然音频的冻结 WavLM-Large L6 特征。
Z_T 是历史 MFA-linear 自然网格条件特征，已经过插值；不得再次对齐，也不得误取声码器输出重编码特征。
核对其值与父 conditions.N_100/T_100 的 conditioning 文件一致，并检查父记录中的文件 hash。没有旧 hash 的 Z_N/Z_T 文件本轮绑定新 hash，同时与已绑定 conditioning 内容逐元素核验。

为区分原生 TTS 与插值，另用同一冻结 WavLM 从 15 条父 tts_audio（16 kHz PCM16）提取 Z_T_RAW，保留自身时间轴；自然原生特征复用 Z_N。Z_T_RAW 只用于模板与原生特征诊断，不生成新 TTS、不直接用于自然音轨 replacement。
这是当前标准化波形链下的“原生”特征，不是原始 24 kHz TTS。

冻结模型：
- bshall/knn-vc revision c616845c4e309e24d5927f15adbdf277a3d65358；
- WavLM-Large L6，1024 维，16 kHz，hop=320；配套 prematched HiFi-GAN；
- Wav2Lip 和 SyncNet checkpoint、运行环境、参考视频、score_box 均沿用父 inputs 绑定。
模型 eval/inference_mode；任何资产缺失或 hash 不符，报具体缺项并 incomplete，不替换。

本次写 spec 只做了元数据可行性检查：按两侧原生 donor occurrence 至少 5 帧、至少 2 个 donor 句有同音素支持，evaluation 每句有 9–20 个候选 occurrence，可改内部帧约占非静音帧 32.5%–41.3%。这是按 tokens/帧中心估计的上限，尚未剔除零范数模板/分量；不是科学结果或模板数值检查。草稿的6帧要求在设计审查时因覆盖不足改为5帧，未运行新的科学评分。

## 3. 从独立 donor 构建一个 N/T 共享模板

所有分解计算使用 float64；最终给声码器的特征转 float32。
复用 scripts/knn_vc_retrieval.py 的 frame_owners、matched_span_map。帧中心固定 (i+0.5)*320/16000。使用该模块规范化后的完整 phone label，不另行去声调/合并近音。静音、未知、未匹配 occurrence 不参加。

每个 donor 的自然和原生 TTS：
1. 各自按各自 MFA tokens 分配帧；用 matched_span_map 关联自然/TTS occurrence。
2. 只保留两侧各至少 5 帧的 matched speech occurrence。
3. 每侧分别移除该 occurrence 首、尾各 1 帧。内部长度 m>=3。
4. 内部矩阵 Z_in 去掉自己的逐维时间均值，得 U = Z_in - mean_t(Z_in)。
5. 沿时间轴逐维线性插值，从 linspace(0,1,m) 到 linspace(0,1,8)，端点对齐；插值后再次逐维去时间均值。
6. Frobenius norm <=1e-8 的任一侧使该 occurrence pair 无效。其余两侧各自除以自己的 Frobenius norm，得到单位形状。该归一化只用于建模板，不用于声码器输入。
7. 先将一对 N/T 单位形状取平均，再在同 donor 句、同 phone 内平均，最后对 donor 句等权平均；避免长句、重复音素或某个来源主导。
8. 每个 phone 至少 2 个不同 donor paired_key；模板再去时间均值，norm<=1e-8 则无效，否则归一化为 Q_label，形状 [8,1024]。

写 templates.npz 和 templates.json：phone、donor paired_keys、occurrence 索引、每级计数、模板 norm 与 SHA。evaluation 的任何值不进入此平均。N/T 使用同一个模板，不给 TTS 单独学习更适合自己的坐标系。

### 模板语义检查

按有效 phone label 的 Unicode 排序建立循环错标签表：label_j -> label_(j+1 mod K)。K<2 时模板检查不可用。
evaluation 的原生 N/T matched occurrence，仅使用两侧都>=5帧且有真/错模板的 occurrence。按上述内部去均值、插值到 8、再去均值得 U8；不必归一化，以下指标已归一化。
分别用正确 Q 和错标签 Q_wrong 计算 explained = <U8,Q>² / (||Q||² ||U8||²)，零 U8 记缺失，不填 0。
每句先 occurrence 等权平均，再 N/T 等权平均，得到“正确减错标签 explained”的一个值。
正确标签优势不成立时，P 只能称模板方向，不可称已验证的发音相关成分；仍保留全部结果，不按此指标挑模板或 phone。

## 4. 分解与等幅干预：核心纯函数

evaluation 在自然时间网格操作 Z_N 和历史 Z_T。
一个 occurrence 须：父 eligible=true、连续 frame_indices 至少 5 帧、phone 有有效模板。使用同一批 N/T occurrence。
首尾各 1 帧保持原值，只修改内部 J；短音素、静音、fallback、无模板帧均保持原值。

对来源 X∈{N,T} 的内部矩阵：
- mu = mean_t(Z_X[J])；
- U = Z_X[J]-mu；
- Q_label 从 8 点插值到 |J| 点，再逐维去时间均值；
- 若 Q norm<=1e-8，此 occurrence 不操作；
- P_X = (<U,Q>/<Q,Q>)*Q；R_X=U-P_X，内积覆盖全部时间与 1024 维；
- P/R 必须零时间均值且正交；这是单一轨迹方向的正交投影，不是逐帧/逐维单独回归。投影系数允许为负，另报告负系数比例，不把投影强就说成正向匹配。
- 必须使用 U=P+R 重建，不另外平滑/归一化。

每个 matched occurrence 统一预算：
delta = 0.5 * min(||P_N||, ||R_N||, ||P_T||, ||R_T||)。
任一 norm<=1e-8 时两来源此 occurrence 都不操作。把原因计数，仍保留样本。
这样四种单项干预的特征改动 Frobenius norm 都是 delta，且任一项最多削弱 50%。预算使用配对目标的两来源值，只是匹配干预幅度，不是可部署增强算法。

三种状态：
- ID：直接复用原 Z_X，不经过重建浮点运算；
- R_DOWN：Z'_X[J] = mu + P_X + (1-delta/||R_X||)*R_X；
- P_DOWN：Z'_X[J] = mu + (1-delta/||P_X||)*P_X + R_X。

所有 occurrence 独立操作，同 phone 的不同 occurrence 不混合。
由此保持：内部及整个 occurrence 的逐维均值、首尾帧、跨 occurrence 的相邻边界差、帧数与特征网格时长。
不保证保持：首/尾帧到内部帧的差分、输出波形真实 phone 边界、音质、可懂度或响度；这些必须测量/限定。

数值验收：float64 相对误差<=1e-8（分母至少1），float32 输出均值/预算相对误差<=1e-5；源未修改帧用 array_equal。记录全局最大误差。
有效操作后每句至少 3 个 occurrence，活跃内部帧/全部非静音帧>=0.10；任一句不满足，feature 阶段交付 INSUFFICIENT_INTERVENTION_COVERAGE 并停止 GPU 音视频阶段。此时不能降低阈值或只留成功句；未来需独立修订协议。这是可读性门槛，不是效果筛选。
另报告每句实际特征改动 RMS、||deltaZ||/||Z||、P/R 削弱比例分布，防止“预算接近零”的阴性被过度解释。

## 5. 原始来源差异诊断

在 evaluation 原生自然和原生 TTS 的 matched occurrence 上做第3节相同的 8 点规范化时间比较，模板来自 donor。
沿用第3节可比较 mask，两侧一致。定义 f_R = 1-explained_true，每句两来源分别 occurrence 等权平均。
主统计 S = f_R_N - f_R_T_RAW。S>0 才支持“TTS 原生变化更集中于该共享模板方向”。
同步报告原生 ||P||²、||R||²、||U||² 的每元素能量与投影系数符号作为描述诊断。比例下降不能写成绝对剩余能量下降，也不能当作保留全部发音信息的证据。
历史 Z_T 在自然网格的 f_R 仅作旁证，单列“经过 MFA 插值的条件特征”。若只有它的 f_R 较低，而 Z_T_RAW 不低，应优先考虑插值/处理链效应。

## 6. 音频和视频：仅七臂

每个 evaluation 样本：
N_RAW、N_ID、N_R_DOWN、N_P_DOWN、T_ID、T_R_DOWN、T_P_DOWN。
10条×7臂=70条音频记录、70个视频。六个重合成臂都用本轮相同声码器 fresh decode；N_RAW复用绑定的自然PCM。
FLOAT WAV作母版，PCM16作为统一生成/评分输入，16 kHz mono；沿用 exact_natural_length 只在末尾裁剪/补零，记录调整。先检查 finite 和 [-1,1]，超界明确失败，不能静默 clip 或按臂归一化。
音频 QC：长度、RMS、peak、clipping、与同来源 ID 的 log-mel distance；不对齐后再算差距。log-mel仅QC，不作为机制终点。
所有视频用父对应 face_video 与固定 Wav2Lip 参数；独立 cwd/temp，固定随机种子42，使用父 crop，不根据分数换 box。父参考视频是已有生成参考，不是真人口型真值。

全部70视频 fresh render；不混历史分数。可复用本轮 smoke 的同一产物；不启动云端服务或下载新评估模型。
固定人工听检 evaluation 前3条（ID6/7/8），七臂同句对照，记录破音、吞字、点击、含糊。无听检能力则 QUALITY_NOT_ASSESSED，不伪造。人工口型结论需另行盲评；本轮交付可播放对照，但不把模型代评当真人标注。
操作破坏音质时只报告“该方向干预影响整条生成链”，不能独立归因为音素动态。
同时报告特征内部差分，以及首尾固定帧与被修改内部的两处接缝差分；跨音素边界不变不代表内部接缝没有伪影。

## 7. 210 个科学评分 cell 与三个控制

S(v,a)表示固定视频v、评分音频a的SyncNet分数；不通过AAC重封装换音轨，直接送统一PCM16到音频frontend。

每条21个唯一cell：
1. 七臂各自原生 S(a,a)：7。
2. 六个重合成视频配自然原音轨 S(a,N_RAW)：6。
3. X∈{N,T}，j∈{R_DOWN,P_DOWN}，加入
   S(X_j,X_ID) 和 S(X_ID,X_j)：2×2×2=8。
总计10×21=210。donor不渲染、不评分。
它们同时给出每个来源和干预的2×2：ID视频/干预视频 × ID音轨/干预音轨。

按父同步实现：
- 固定每条score_box，25 fps，visual[t]对应5视频帧，audio[t]对应从MFCC列4t开始的20列；
- lag k=-15…15，d(k)=mean_t ||visual[t]-audio[t+k]||_2；
- 每条21cell使用同一 F=min(所有涉及video/audio embedding长度)，共同t=range(15,F-15)，至少30窗口；
- 先时间均值，后D=min_k d(k)、B=median_k d(k)、C=B-D；k*=argmin，平局取第一个；
- 保存完整31点曲线、C/D/B/k*与共同支持。自然基线k_N来自 S(N_RAW,N_RAW)，保存所有自然音轨cell的d(k_N)。
- 一个cell失败则该样本结果不完整；不得悄悄缩小配对分母。支持不足也标incomplete，不临时降30。

另固定ID6三个控制（不计入210）：
- N_RAW/N_RAW完整独立重提特征评分一次；
- T_ID/T_ID同上；
- N_RAW/N_RAW仅把评分音频右移3200 samples、头部补零、末尾裁剪到原长（200ms）。
repeat要求曲线最大误差<=1e-4；delay要求在argmin没有触及lag边缘时，新k*=旧k*+5，允许±1帧。若触及边缘记控制不可判读，不能假装通过或挑另一句。此处k为取更晚音频的正号，official offset符号相反。
共213评分记录。重复仅用于数值稳定，不新增独立样本。

## 8. 预定义统计，不按结果改指标

主终点 q(a)=C(a,N_RAW)，评分音频全部固定。
每句先计算差，再在10条evaluation句间等权。donor不入分母，帧/phone不是独立样本。
b_X=q(X_R_DOWN)-q(X_ID)；h_X=q(X_ID)-q(X_P_DOWN)。
b正表示削弱剩余项有益，h正表示削弱模板轨迹有害。

六个主统计：
1. S：原生特征 f_R_N-f_R_T_RAW；
2. A：正确模板减错标签模板的 explained（每句先phone平均再N/T平均）；
3. b_N：自然剩余项削弱收益；
4. h_N：自然模板轨迹削弱损害；
5. h_T：TTS条件模板轨迹削弱损害；
6. J=b_N-b_T：削弱剩余项对自然来源的额外帮助。

使用同一10句索引矩阵，numpy.random.default_rng(20260917)，20000次有放回配对bootstrap。六项都报告均值、逐句值、普通95%区间及Bonferroni 99.1666667%区间，alpha=0.05/6，np.quantile(method="linear")。
模板固定，不在bootstrap里重建；区间仅表示条件于这5个donor的10句重采样不确定性，不覆盖模板估计不确定性，也不构成单说话人之外的泛化。
S/A任一句没有有效可比较occurrence时，该统计primary不可用，不插值填0，不降低其CI分母；其余下游统计仍按全部10句报告，整体假设不能判支持。

其余均辅助、描述性：
- b_T与b_N-b_T的逐句操作幅度；原始N/T来源优势q(T_ID)-q(N_ID)、replace参照q(T_ID)-q(N_RAW)。
- D改善按“基线D-候选D”，与主C方向同向解释；固定自然k_N的距离改善另列。
- 对每个来源/干预，G=S(X_j,X_ID)-S(X_ID,X_ID)，E=S(X_ID,X_j)-S(X_ID,X_ID)，I=S(X_j,X_j)-S(X_j,X_ID)-S(X_ID,X_j)+S(X_ID,X_ID)；验证native_delta=G+E+I。主q与G不同，不混用。
- 任意C差用 delta_C=(B_new-B_old)+(D_old-D_new)分解。B增大不自动证明错配识别更强，D下降不自动证明人类同步改善。
- 来源×干预差异受共同绝对预算与不同相对削弱比例影响，J只能表述为当前等幅干预下的来源差异。
- 可视化每句b/h散点、六统计区间、f_R来源对比；不做大量探索性相关检验。

## 9. 预先写好的结论规则

只有工程完整、模板A支持正确phone优于错phone、且b_N/h_N/h_T/S/J的校正区间都下界>0，才写：
“在该小样本和操作定义下，证据共同支持：原生TTS的变化更集中于跨句共享phone轨迹；削弱剩余项有利于自然音频驱动的视频，而削弱共享轨迹有害；该效应与声学变化重组假设相符。”
即使全成立，也保留“历史TTS处理链、声码器、音质、单speaker、非真人同步”的边界，不称为完整机制证明。听检未完成或明显质量恶化时，功能解释降级为含音质混杂。

常见分支：
- b_N>0而h_N<=0/不确定：不能说成功区分了有用动态和干扰；可能一般性处理作用。
- h_N/h_T>0而b_N不确定：仅支持保留共享轨迹，不支持R可被当成干扰去掉。
- b_N/h_N支持但S/J不确定：存在可干预方向，未解释TTS为何优于自然。
- 原生S不支持、只有aligned Z_T指标支持：处理链/插值解释优先，不能归因原始TTS。
- native C改善，q/G不改善且E改善：定位到评价端敏感性，不宣称生成端收益。
- C改善主要来自B，D/固定lag距离不改善：报告SyncNet置信度响应，真实匹配改善证据不足。
- A不支持：模板语义不足；P/R方向干预的数值结果可报告，不能称“发音/干扰分解”。
- 两项削弱都损害：剩余项也有用，或共同存在分布外/音质损害；不证明“更多变化总是好”。
- 所有区间跨0：本设计分辨率下不确定，不等同两成分等效或假设为假。
- 样本、覆盖或控制失败：incomplete/不可判读，与科学阴性分开。

不以本轮结果直接启动增强头训练。replace是否值得做，只能引用自然音轨终点相对于N_RAW的结果；本轮oracle式配对预算本身不可部署。

## 10. 实现保持简单

建议新增 scripts/experiments/tts_acoustic_reorganization.py 和对应 tests/experiments/test_tts_acoustic_reorganization.py。必要时一个评分worker，复用父工具，不重构旧pipeline。
CLI提供 prepare / features / audio / render / score / analyze / validate 与 --output-dir；默认新目录 runs/tts_acoustic_reorganization_20260916，不覆盖父run。
prepare绑定输入与划分；features建模板/分解/QC并先过覆盖门；随后以ID6做七臂smoke，再原参数跑余下9句。smoke依然使用全部5 donor，不把ID6加入模板。

可复用入口（先用CodeGraph读当前实现再复用）：
- scripts/experiments/mfa_linear_trajectory_ablation.py：输入、PCM写出、声码器、隔离render、评分封装；
- scripts/knn_vc_retrieval.py：frame_owners、matched_span_map；
- scripts/wavlm_knn_vc_adapter.py：提取/vocode；
- scripts/experiments/static_image_bridge/score_worker.py：SyncNet frontend与曲线。
本spec明确覆盖父八臂/225cell、50窗口和旧bootstrap常量，不能照搬。

最低科学测试：
1. 合成正交P/R重建、投影与符号；P/R各自零均值。
2. 四种单项干预的delta相同、系数在[0.5,1]；零分量导致N/T共同no-op。
3. 首尾、非活跃帧、整段均值与跨phone边界不变；同label occurrence不串段。
4. 用sentinel验证evaluation不进入模板；donor句和N/T等权；缺支持按规则失效。
5. occurrence总长5/7/10帧的两次插值、模板重新去均值；正确/错标签映射稳定。
6. 七臂/21cell、210+3控制计数与身份唯一；原生TTS自己的时间轴与自然网格不得混用。
7. 人工距离曲线验证C=B-D、G/E/I、lag符号、共同支持、共享配对bootstrap。
不要只断言函数返回自身结果。另用独立短脚本从保存的features/curves复算至少ID6全部干预和全部10句主统计，与正式输出比对。

交付：
- inputs.json：有序5/10划分、源/模型/hash、协议常量、spec hash；
- templates.npz/json、features/、feature_qc.csv：donor来源、正交/均值/预算/覆盖/原生诊断；
- audio/、videos/、audio_qc.csv；70行音频与70视频manifest；
- scores.csv（210科学行）、controls.json（3控制）、curves.npz或json；
- paired_effects.csv（10行）、analysis.json、validation.json、report.md、简洁图、固定3句的可播放对照索引。
报告首段必须写：完整性、覆盖、六主统计、质量评估状态与允许结论。分数3位小数，底层保留精度；JSON缺失用null+reason。
review按数据身份→模板泄漏→干预恒等式→cell身份/支持→统计方向→文字结论逐项核验；修复代码错误可重跑，不能据结果更改科学协议。
实验完成后按Startup Router在Experiments更新结果并链接本spec。本轮设计阶段不写“concluded”。

## Observations

- [status] spec_ready
- [question] 在等幅、保均值和保边界帧的干预下，跨句可重复phone轨迹与剩余项是否具有不同下游作用，且这种作用是否解释TTS来源差异？
- [design] 固定5 donor/10 evaluation；共享原生模板；七臂70视频、210科学评分及3控制；冻结模型。
- [boundary] 单说话人历史探索；剩余项不预先等同噪声；原生特征差异与MFA-linear条件干预分开解释。

## Relations

- follows [[MFA-linear 轨迹增益的匹配项与背景项分解 2026-09-16]]
- relates_to [[MFA-linear 连续轨迹机制消融 support30 2026-09-16]]
- relates_to [[TTS 原生增益来源的生成端与评估端交叉诊断]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 将声学变化重组假设写为独立模板、等幅双方向干预和固定音轨诊断协议；仅设计并核验已有资产元数据 | September 16, 2026 | user request |
| 设计自审：同时核对原生TTS donor时长，固定最短occurrence为5帧以保留足够覆盖；澄清原生/插值来源和等幅干预边界 | September 16, 2026 | user request / design review |
