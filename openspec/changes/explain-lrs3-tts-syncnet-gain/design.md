# 实验设计与计算契约

## 1. 问题和已有证据

N 是原自然音频，T 是原 TTS 音频；V_N/V_T 是分别由它们驱动生成的视频。这里研究原生比较 `C(V_T,T) − C(V_N,N)`。它同时改变音频与视频，不能单独归因为视觉改善。replacement 的比较是 `C(V_T,N) − C(V_N,N)`，本轮不新增这个四格实验。

按绑定的 multiset manifest 联结历史评分，得到以下 **50 条 record 等权均值**，用于输入校验，不是新独立复现，也不是下文 45-group 等权主估计：

| 模型 | ΔC | D 改善量 D_N−D_T | C 提高记录 |
|---|---:|---:|---:|
| Ditto | +1.124 | +0.008 | 45/50 |
| LeapTalk | +1.389 | +0.474 | 46/50 |

因此不得继续使用“英文不成立”作为实验前提。准确表述是：LRS3 上这些 TTS/TFG 条件有原生 Sync-C 优势；跨语料/模型有异质性。语言不作为本轮解释变量。

另外，`runs/lrs3_masked_tts_tfg_confirmation_20260902/05_analysis/report.md` 在 16 条/8 组上报告 TTS conditioning 相对 NAT_ONLY 的 modality C 中位增益 +1.501。那是 direct-mel 且相对特定重建基线的比较，不等价于 raw TTS 原生优势或超过 untouched-natural baseline，不能混入本轮均值。历史 raw/MFA-linear/bridge 阴性结果也保留各自口径。

本轮只检验三个可回答的问题：

1. C 的两个数学组成部分各变化多少？
2. 原生优势在移除特征 padding、限制 lag 搜索、控制有效窗口数量后是否仍然存在？
3. 自动分数能否获得人工感知方向的支持？人工部分单独交付，不阻断自动结果。

## 2. 固定输入、身份和选择

### 2.1 CPU 全队列

`input-bindings.json` 为固定入口。路径均相对 repo root，SHA-256 是文件字节 hash，不冒充 decoded PCM hash。

- 原 manifest：`data/dataset_samples/video_manifest_250.json`。`records` 的 **原始 1-based 行序号**就是历史 sample_id；筛选 `dataset == lrs3` 得到 151–200，共 50 条。
- source_group = `Path(video_local_path).parent.name`；核对 stem = `source_group + '_' + clip_id`。45 组，不使用恒为 `lrs3` 的 speaker_key。source group 是源视频组，不能无证据宣称为独立说话人。
- Ditto：`scores_250.json` 的 `(sample_id,condition,sync_c,sync_d)`；LeapTalk：`scores_leaptalk.json` 的 `(i,cond,c,d)`。转成共同 schema 后，键为 `(model,sample_id,condition)`，条件只有 natural_raw/tts_raw。
- 50×2×2 = **200 个历史 cell**、100 个模型内配对。禁止把两模型视为 100 条独立录音。检查重复键、缺失、有限值，并以 Ditto eval_meta 与绑定的逐 cell JSON 交叉核对。单一字段缺失不补零。
- 模型/TTS 来源是历史记录声明：faster_qwen3 0.6B ICL；历史配置含 language=Chinese，不能把配置字段当作实际语言/发音质量的证明。本轮不重合成，也不作 provider/音质因果判断。保存声明与可核验事实的差别。

### 2.2 曲线子队列

按原 manifest 顺序，每个来源保留首次记录，再取前 12 组：ID **151–162**（名单和 stem 已落 JSON）。这个规则不使用分数；总队列和一部分分数已观察，属于 seen-data 机制诊断，不能称未见确认集。

只使用已有 `runs/leaptalk_multiset/{natural_raw,tts_raw}/{id}.mp4`，24 个视频 hash 已绑定。视频内原生音轨为评分音频；不从易被覆盖的 `data/data/audio/{id}.wav` 寻找替代轨，不 time-stretch、不换 N 音轨。本轮能验证当前媒体身份，不能补造已缺失的历史生成命令/hash 链。

全部 12 对完成才标记曲线队列完整。单条失败保留其键和原因，不补其他条；CPU 全队列仍可完成。不得因 C 太低、offset 大、TTS 太长或跟踪失败换样本。

## 3. Stage A：无需模型的精确分解

官方每个 cell 的距离曲线为 d(k)，搜索 k∈{−15,…,15}。定义：

```text
D = min_k d(k)
B = median_k d(k)             # 距离背景，包含全部31个lag
C = B - D

B_hat = C_saved + D_saved     # 只能恢复标量背景，不能恢复完整曲线
gain_C = C_T - C_N
gain_match = D_N - D_T        # 正数：最佳特征匹配距离下降
gain_background = B_T - B_N   # 正数：距离背景上升
gain_C = gain_match + gain_background
```

使用 float64 读取已舍入评分，逐 cell、逐 record、逐 group、总体均检查恒等式绝对残差 ≤1e−10。历史单个 C/D 通常只有三位小数；相对于未舍入真实量，单 cell B_hat 误差至多 0.001、背景配对差至多 0.002。报告三位，分析保留原精度，不伪称恢复了未舍入值。

输入自检的 record 均值：Ditto background=1.11588，LeapTalk background=0.91544；与 bindings 检查值误差 ≤1e−9。不能因该检查未通过而修改历史数据来匹配预期。

**解释限制**：最佳距离下降仍只是 SyncNet 表征匹配，不能直接说嘴型更精确；背景上升也不证明“评分作弊”。更清楚的发音/视觉动作同样可能扩大错位距离。两个加项只是代数分解，不是可相加的声学因果贡献；不报告逐样本百分比贡献，避免除以近零或负 gain。

## 4. Stage B：固定小队列完整曲线

### 4.1 运行和支持冻结

不生成新 TFG 视频。审计 24 个 MP4 的视频/音频 stream、解码帧数、帧率、PTS 起点、音频 sample count，绑定 ffprobe 结果和解码 PCM hash。保留原媒体；播放音轨可能是 AAC，因此本轮叫原生媒体重评分，不叫 untouched 原始波形实验。

使用绑定官方 run_pipeline 的预处理，min_track=25、vshift=15、batch_size=20；其他参数按绑定源码默认值展开保存。先用固定检测配置产生 track/crop 清单，再开始读取新 C/D。多 track 时选实际帧数最多者，依次以起始时间更早、文件名字典序更小打破平局；选轨不读取分数。两臂可能有不同轨迹，必须记录这种限制，不能称逐帧视觉输入相同。无合格 track 标记失败，不引入首帧框等新 fallback。

N 和 T 时长不同，每个原生组合保留自己的真实音视频时钟。本轮不把等时长当成逐音素对齐。对选定 crop 用官方 SyncNetInstance.evaluate 返回矩阵 `M[t,j]`，shape `[T,31]`；T 为实际提取的同步窗口数。保存 crop 来源/帧索引、矩阵、日志、原始 dtype、每个前向输入及代码/权重 hash。严禁把任意全脸图直接传给评分模型。

每条主 cell 提取一次特征并保存完整矩阵即可，后续各种支持均在 CPU 计算。只允许身份和处理链完全匹配的本 run 完整缓存复用；没有 old cache 时新评分，不从 JSON 标量伪造曲线。

### 4.2 三种支持

矩阵第 j 列对应音频窗口 `t+j−15`，官方 offset 为 `15−j`。两者符号相反，代码字段分别命名 `audio_window_shift` 与 `official_offset`。

1. **FULL**：t=0,…,T−1。包含官方特征 padding，先按官方 float32 mean→min/median 重算 C/D/offset，与本次日志三位值误差 ≤0.000501；与未舍入返回值 ≤1e−4，offset 必须一致。另保存 float64 分析值。
2. **INTERIOR（曲线主分析）**：t=15,…,T−16，即 `range(15,T-15)`，两端所有 ±15 lag 均有真实音频特征。每臂独立使用自己的所有有效窗口，至少 25 行，否则该配对曲线不完整。先 mean_t 得31维曲线，再 min/median；不能先逐帧算 C 再平均。
3. **EQUAL_COUNT（敏感性）**：每对 n=min(T_N−30,T_T−30)，各臂在自己的 INTERIOR 中均匀抽 n 个不重复行，索引为 `floor(linspace(0,L−1,n))`。仍保持各自时钟，不能宣称这些行内容对齐。若 n=L 取全部。此分析控制窗口数量，不控制内容、语速或音素比例。

此外对同一 INTERIOR 31维曲线限定 |offset|≤5，取原列 j=10,…,20（Python切片 `curve[10:21]`，11列），计算该子曲线的median−min作为C_5；这是搜索范围敏感性，不能与 C_15 当成同一个量。零 lag 距离 `D0=d(offset=0)` 和搜索收益 `S=D0−D` 单独报告，不把 D0 称为已校准真实同步误差。

### 4.3 最小曲线诊断

每 cell 输出 B、D、C、D0、S、offset、边界最优标记、半深度谷宽 W，以及 FULL/INTERIOR 的有效行数。

W：对 `d(k) ≤ D + (B−D)/2`，取含最小值列的连续列段，W=列数×40 ms。最小值并列取官方首列；另保存 ties。若 C≤1e−6，W=null/flat_curve；若连续段碰到 ±15，W 记录下界并标记 censored，不把它当完整谷宽。宽度仅作描述，不据此自动声称精度改善。

画三张可导出的 PNG/SVG：两个模型的 ΔC=match+background 配对/组级分解图；LeapTalk 12 对在绝对 lag 上的完整曲线小多图；FULL/INTERIOR/EQUAL_COUNT 的配对效果图。先按每 cell 算端点再统计；不能从跨样本平均曲线求最小值后当平均 C。

### 4.4 重复和时间控制：总预算28个评分cell

- 主分析：12×2=24。
- 固定 ID151 两个原生臂各独立重复一次：2。复用同一冻结 crop/PCM，在独立工作目录重新跑 SyncNet，矩阵 max_abs_error ≤1e−4、offset 一致。这里只验证评分重复，不估计 TFG 生成随机性。
- 同 ID151 两臂各加一个 +200ms 音频延迟控制：2。对选定 crop 解码的16k PCM 右移3200样本，前补零、截去等量尾部，长度不变；视频帧、crop 和 PTS 保持相同，PCM 无损封装。不得重新检测人脸。
- 延迟比较用原始/延迟矩阵共同 `range(20,min(T_original,T_delayed)-20)`，至少25行。同支持内预期官方 offset 相对原始约 **−5帧**（容许±1帧）；原始 argmin 处的 delayed 距离应高于 original。若最优落边界、条件不满足或矩阵重复失败，标记 CONTROL_FAILED，不重选控制样本、不事后换符号。保留曲线和具体失败，限制时间解释；Stage A 不受影响。

这两个延迟控制只覆盖 ID151；不能宣称已验证全部记录或证明 SyncNet 全面可靠。搜索后 C 仍高不等于零延迟，这是本控制要展示的区别。

## 5. 统计契约

Stage A 每个模型分别计算 record 配对差→同 source_group 内均值→45组等权总体均值。主置信区间对应同一个组均值估计。另报 record 等权均值用于历史对账，不能把 record 点估计与 group CI 混在一列。重复记录不增加独立组数。

组按字符串排序，PCG64/default_rng seed=20260915，10000次有放回抽取45组；所有指标和两个模型共用抽样索引，quantile(method=linear)。Stage B同样规则，12组，重复/控制不进入主样本量。

Stage A两个模型×两个组成项共4项为主描述分解；同时给95%描述性CI和98.75% Bonferroni区间。只有对应98.75%区间下界>0，才称该组成项有正向支持；ΔC本身是历史现象复核。允许“两项均支持/仅背景有支持/仅最佳距离有支持/均不确定”。不以某项 p>0.05 断言它为零。

Stage B预定三个核心对比：INTERIOR ΔC、INTERIOR gain_match、ΔC_INTERIOR−ΔC_FULL。各报95%描述性区间和98.333333%区间（3项 Bonferroni）；其他 width、D0、S、C_5、EQUAL_COUNT、跨模型差异均为探索性，不作筛选或自动胜负门槛。小样本CI不代表总体泛化。

INTERIOR优势仍正只能支持“本子队列优势不完全依赖边界 padding”；支持变化明显只能说明该评分支持影响结果，不能宣布历史结果无效。模型差异不是语言效应，也不直接识别某种 frontend 因果机制。

## 6. 人工包：自动结果不等待人工

为同12条/LeapTalk导出24段原生播放片段，完整自然/TTS各自音轨和原帧率，不互换音轨、不拉伸；随机匿名A/B，seed=20260915，真值映射独立文件。使用已有视频引用或小体积预览，避免复制全量数据。

CSV包含 rater_id、sample_id、A/B随机顺序、同步感偏好(A/B/tie/unjudgeable)、各片段同步感1–5、各片段口型自然度1–5。明确主观任务是各自音画同步，不是更喜欢哪条声音。原生音轨可能暴露自然/TTS，说明匿名不等于完全盲法。

至少3位共同评审完整评12对；未完成时 `perception=NOT_ASSESSED`，不给预测评分。若有评分，先评审内配对再记录平均、最后12来源组bootstrap；不把36次判断当36独立样本。unjudgeable单独统计，缺失不当tie；不足完整共同评审则仅描述性PARTIAL。人工结果不用于选样、重跑或变更自动统计。

## 7. 性能、资源和恢复

- Stage A纯CPU，可在磁盘/GPU受限时立即完成。无新TTS/TFG/训练/API调用。
- Stage B先写预算估计：当前free、所选最长clip的解码帧临时需求和预计持久矩阵量；要求足够容纳预计峰值并留1GiB余量，不足返回 RESOURCE_WAIT。不要在447MiB空闲下盲目全量展开JPEG。
- GPU检测/评分持有同一排他lease，一次一个模型子进程。每30秒记录utilization、used_memory、compute PID；发现其他任务不杀进程、不抢占，返回RESOURCE_WAIT。检测与SyncNet不重叠加载。
- 每cell完成、矩阵及hash落盘核验后，可自动删除本run该cell新建的临时JPEG/检测中间文件；保留输入引用、crop参数、PCM hash、矩阵、日志和结果。删除路径须限于本run temp，禁止清理旧run或数据。
- 断点续跑由输入+代码+配置+crop+音轨hash确定；失败/partial输出不能冒充完成。代码或参数改变须新run，不覆盖历史文件。
- 默认 max_new_syncnet_cells=28，控制也计入；无需重复评分去凑稳定结果。不部署服务器、不要求外部账户。

## 8. 输出、验收和结论

```text
00_audit/inputs.json, cohort.json, resource_plan.json
01_decomposition/cells.csv, paired.csv, groups.csv, summary.json
02_curves/manifest.json, matrices/*.npy, logs/, controls.json
03_analysis/endpoints.csv, paired.csv, summary.json, figures/
04_perception/clips.json, assignments.csv, ratings_template.csv, private_key.json
report.md, final.json, validation.json
```

final.json分别保存 `historical_decomposition`、`curve_analysis`、`controls`、`perception`状态及counts/budget；允许CPU COMPLETE而curves RESOURCE_WAIT，不虚报整体完成。自动阶段全部完成时，即使人工NOT_ASSESSED，也可标记automatic_complete=true。

独立validator从原绑定评分和NPY重新计算，不读取summary数值作真值，检查200历史cells/100配对/45来源、24主曲线/4控制、选择顺序、bootstrap点估计与区间、加法恒等式、offset符号、媒体/模型hash。确认公式而不是只确认文件存在。

报告首先说明英文优势的证据，再解释分解与曲线，保留全部阴性与不可用结果。报告结尾明确下一步只应针对已定位的计算/时序现象做单一受控机制实验；本轮不自动启动训练。最终笔记按Startup Router写入Basic Memory的Experiments，完整报告留run目录引用；不写进度日志。
