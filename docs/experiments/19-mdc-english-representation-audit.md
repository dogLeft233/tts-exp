# 19 — MDC English natural-vs-F5-TTS 音素与表征审查

## 范围与问题

本报告归档 Effect AI Scripted Speech 1.0 English 上的第二套 natural-vs-TTS 音频表征实验。该数据集用于检验此前基于中文/AISHELL 与 LibriSpeech 的观察是否依赖语言、数据集或规整朗读条件，并进一步检查：

1. TTS 音频在音素/viseme 表征空间是否更紧凑、更易识别；
2. TTS 与 natural 的差异是否集中在某些 SSL 层或某些音素；
3. TTS 是否表现出不同的时长与停顿结构；
4. 这些表征差异是否可能由 source-author/speaker 混杂解释。

本轮只完成 representation-level 分析。**它没有完成 MDC English 的 Ditto/TFG/TensorRT 或 SyncNet 评估，因此不能据此声称 TTS 已改善 MDC English 的 lip-sync，也不能把表征指标当作 TFG 效果的代理终点。** B4“表征→TFG/SyncNet”仍处于阻塞状态。

## 数据与溯源

- 数据集：`mdc_tts`，Effect AI Scripted Speech 1.0 English，CC0-1.0。
- 配对：50 个稳定键 `en_001`–`en_050`，每个键各有一条 natural 与一条 TTS 音频。
- 运行目录：`runs/mdc_en_phoneme_20260807_f5_full/`。
- TTS：本地 F5-TTS，provider=`f5_tts`，model=`F5TTS_Base`；50/50 成功，0 失败。
- TTS 质量：canonical 音频中 17/50 条的 `clipped_fraction` 非零，最大约为 0.00251。因此本报告不声称 TTS 输出全部 clipping-free；clipping 是后续下游解释时需要保留的质量因素。
- 对齐：natural 与 TTS 使用独立 MFA staging、`.lab`、TextGrid 和 phone spans。最终 alignment 有 100 条记录（natural 50、TTS 50），全部 `alignment_source=mfa`；natural phone spans 没有复制给 TTS。`en_015` 的 natural TextGrid 使用了单独 MFA retry，并保留 retry provenance。
- 音频契约：canonical 音频为 16 kHz、mono、PCM-16 WAV；provider-native TTS 文件与 canonical TTS 文件分开保留。
- SSL：`facebook/hubert-base-ls960` 与 `facebook/wav2vec2-large-xlsr-53`（XLS-R），保存层 `[0, 6, 11, 12]`。每个模型有 100 条 JSON/NPY 记录，即总计 200 个 JSON 与 200 个 NPY 输入。
- 池化限制：有些 token span 没有 frame center，产生 null token pool；这些项被作为有效支持限制处理，不是 embedding NaN/Inf 或 alignment 损坏。Experiment 16 的 `per_sample` 诊断区还保留了支持不足产生的 NaN；其主 `results`、`comparisons` 与 `support` 区域才用于汇总，不能把整个旧 JSON 宣称为 strict finite。
- 所有结果按 `paired_key` 进行配对、分组或 leave-one-out；frame/token 数只用于支持信息，不能当作独立 utterance 样本。

## 方法

### Experiment 16：feature separability

对 HuBERT/XLS-R 的 phoneme 与 viseme token vectors 计算 intra-class distance、inter-class distance、Fisher ratio、silhouette、probe、boundary sharpness 与 segment stability。使用 50 个 paired keys，排除 non-speech tokens；全局 96 个 comparison 使用一个 BH-FDR family。结果包含 32 个 metric/model/layer/level 条目，phoneme 支持 55 类，viseme 支持 14 类。

现有生成协议是 `reference_conditioned_same_source`：每个 TTS 输出使用对应 natural 音频作为 reference。`tts:<provider>` 是 provider/condition scope 标签，不是经过独立验证的统一 synthetic speaker identity；source author 与 TTS provider 不应在 B5 中混为同一 speaker 语义。


使用 MFA 边界将 SSL frames 池化到 phoneme 或 viseme；保存层按 embedding metadata 直接索引。跨条件 probe 使用 paired-key leave-one-out：natural prototype→TTS 与 TTS prototype→natural 时，held-out paired key 不进入 source prototype。HuBERT 报告 phoneme 与 viseme；XLS-R 的 logistic probe 没有运行，不能写入 XLS-R probe 结论。

### Experiment 36/B1：duration JSD

使用两臂各自 TextGrid，分别计算 complete audio duration、speech span、phone duration、句内 duration CV、pause count，以及 raw/rate-normalized/base-phone duration distributions。逐 phone JSD 使用最低支持门槛和 FDR；逐句指标使用 paired permutation、bootstrap CI、Cohen’s d，并报告完整样本与 duration-ratio outlier 排除后的敏感性。

### Experiment 37/B2：layer curves

在实际保存层 `[0, 6, 11, 12]` 上计算每个模型和条件的 compactness/separation、Fisher、silhouette，以及 HuBERT probe。natural 与 TTS 使用相同的 50-key intersection；退化层不参与峰值选择。由于本轮只保存四层且未执行层级 confirmatory FDR，峰值只作描述性 argmax/argmin。

### Experiment 38/B3：逐 phone KLD 与判别探针

以两臂各自 MFA phone spans 为基础，按同一 base-phone 的 occurrence order 匹配 natural/TTS vectors。每层使用 PCA-30、Ledoit-Wolf 协方差与 symmetric Gaussian KLD；natural-vs-TTS probe 使用 PCA/标准化在分组交叉验证流程内拟合，并按 utterance 使用 `GroupShuffleSplit`。union phone classes 中，支持不足或一侧缺失的类显式排除。

### Experiment 39/B5：phoneme/speaker control

报告 raw phoneme probe、两臂对称的 per-utterance mean removal probe，以及 natural source-author 与 phoneme 的描述性关联。probe 按 paired utterance 分组，避免 token-level leakage。该设计不假设 source author 与 TTS provider speaker 已匹配。

## 结果

### 1. 表征分离度：TTS 优势是多指标、多层但并非单一指标结论

Experiment 16 的 96 个全局比较中有 58 个经 FDR 显著。总体方向是：TTS 在多个 HuBERT/XLS-R 层上具有更低的类内距离、更高的类间距离/Fisher 或更高的 probe/boundary 指标；但 silhouette、intra/inter distance、segment stability 等指标并不等价，不能压缩成一个“表征全面更好”的统计量。

在 HuBERT phoneme 层，layer 0 的 probe accuracy 为 natural `.3366`、TTS `.3926`；layer 6 为 `.5427/.6941`；layer 11 为 `.5911/.7310`；layer 12 为 `.5584/.7095`。HuBERT viseme 层对应为 `.4788/.5433`、`.6521/.7848`、`.6942/.8148`、`.6720/.7801`。这说明 TTS 优势不只存在于输入层；在本轮四个保存层中，layer 6–12 的差异更明显。

### 2. Phoneme/viseme probe：明确区分 self-LOO 与跨条件迁移

下面两张表中的 natural/TTS PER 与 `delta_tts_minus_nat_per` 是**同条件 utterance-held-out self-LOO**，不是跨条件迁移：

| 层 | natural PER | TTS PER | TTS−natural PER |
|---|---:|---:|---:|
| 0 | 76.61% | 71.97% | −4.64 pp |
| 6 | 57.02% | 45.88% | −11.14 pp |
| 11 | 55.66% | 44.99% | −10.67 pp |
| 12 | 57.56% | 47.28% | −10.27 pp |

HuBERT viseme PER 为：

| 层 | natural PER | TTS PER | TTS−natural PER |
|---|---:|---:|---:|
| 0 | 65.79% | 60.74% | −5.05 pp |
| 6 | 49.34% | 41.38% | −7.96 pp |
| 11 | 47.38% | 38.77% | −8.60 pp |
| 12 | 48.49% | 40.54% | −7.95 pp |

跨条件结果单独保存在 `natural_proto_to_tts` 与 `tts_proto_to_natural` 字段中，分别表示 paired-key leave-one-out 的两个方向。两个方向不对称，不能用上表的 self-LOO 数值代替。现有表格保留历史数值，但其语义按 self-LOO 解释。


### 3. B1 时长与停顿：TTS 更短、更规整、停顿更少

完整 50 对的逐句配对结果如下；`q` 为该输出中的 FDR 校正值。

| 指标 | natural mean | TTS mean | Cohen’s d | q | 结论 |
|---|---:|---:|---:|---:|---|
| complete duration (s) | 6.4926 | 4.9146 | 1.512 | .000875 | TTS 更短 |
| speech span (s) | 4.7668 | 4.2380 | .891 | .000875 | TTS 更短 |
| mean phone duration (s) | .10815 | .09710 | .867 | .000875 | TTS 更短 |
| within-utterance duration SD (s) | .07236 | .05610 | .432 | .000875 | TTS 更均匀 |
| duration CV | .65644 | .57337 | .323 | .01982 | TTS 句内相对时长更均匀 |
| pause count | 3.22 | 2.68 | .370 | .01982 | TTS 停顿更少 |
| phone count | 38.04 | 38.04 | null | 1.0 | 同文本、无差异 |

duration-ratio outliers 为 `en_011`、`en_027`、`en_035`、`en_039`、`en_044`、`en_047`。排除这 6 对后，duration CV 仍显著（q=.00210），而 pause count 的 q=.06647，不再显著。因此“整体更快”和“句内时长更均匀”比“停顿减少”更稳健。

rate-normalized phone duration 与 base-phone duration 均各有 31 个可测试 phone，均为 0 个 FDR 显著类。换言之，本轮没有证据表明某些 phone 的相对 duration 分布形状发生了系统性变化；差异主要表现为整体时长缩短与句内相对时长压缩。pooled pause JSD=.01593、permutation p=.10045；该比较是 unpaired 且 underpowered，只作描述性结果。

### 4. B2 layer curves：HuBERT 中高层承载较强 phone 信息，但峰值是描述性的

HuBERT phoneme probe 在本轮稀疏层网格的峰值为 layer 11：natural `.5911`，TTS `.7310`；HuBERT phoneme inter-class distance 与 Fisher ratio 也在 layer 11 达到各自条件的峰值，分别为 natural/TTS `.6876/.7278` 与 `6.957/8.858`。intra-class distance 的最小值在 layer 12（natural `.6467`、TTS `.5891`）。

XLS-R phoneme 的 Fisher ratio 在 layer 0 最大（natural `7.083`、TTS `8.294`），inter-class distance 的峰值分别在 natural layer 11、TTS layer 6；viseme 的峰值模式也不完全一致。由于只比较四个保存层，且没有层级 confirmatory FDR，不能据此宣称单一“最佳层”，尤其不能把层峰值直接等同于 TFG 部署所使用的末层。

### 5. B3 逐 phone 差异：支持充分的类中，KLD 与判别力在中间层相关更强

每一层共有 74 个 union classes，其中 30 个达到最低支持并进入测试，44 个被显式排除。Spearman(KLD, AUC) 及控制支持量 `n` 后的 partial correlation 为：

| 层 | 测试类 | 排除类 | Spearman | partial(n) |
|---|---:|---:|---:|---:|
| 0 | 30 | 44 | .235 | .349 |
| 6 | 30 | 44 | .723 | .722 |
| 11 | 30 | 44 | .426 | .439 |
| 12 | 30 | 44 | .392 | .395 |

代表性高 AUC phone 为：layer 0 的 `ʃ=.908`、`c=.888`、`ð=.827`；layer 6 的 `ŋ=.819`、`ð=.798`、`ɫ=.789`；layer 11 的 `ʃ=.825`、`ð=.818`、`iː=.803`；layer 12 的 `ð=.836`、`ʃ=.742`、`b=.729`。

这些结果表明 phone-specific natural-vs-TTS 差异并非均匀分布，且 layer 6 的 KLD/AUC 排序一致性较强；但相关性是描述性的，不能解释为 KLD 对 TFG 效果的因果预测。B3 的独立审查为 **PASS**：两臂使用各自 TextGrid，匹配发生在同 base-phone occurrence，probe 使用 utterance grouping，support/exclusion、PCA/CV、B3 主产物 strict finite JSON 与 provenance hashes 均通过核查。

### 6. B5 speaker/phoneme control：设计通过审查，但不是 speaker-balanced 实验

MDC 的 50 条 natural 记录并不对应 50 个独立作者；池化 token 中有 28 个重复且不均衡的 natural source-author IDs。所有 TTS 输出共享一个 provider-level synthetic identity：`tts:f5_tts`。因此 natural 与 TTS 的 speaker distribution 结构性不匹配。

HuBERT raw phoneme probe（natural/TTS）与对称 per-utterance mean removal 后的 probe 为：

| 层 | raw natural/TTS | utterance-controlled natural/TTS |
|---|---:|---:|
| 0 | .269/.315 | .293/.339 |
| 6 | .423/.560 | .430/.544 |
| 11 | .454/.573 | .461/.560 |
| 12 | .398/.508 | .388/.494 |

natural phoneme-speaker 的 Cramér’s V 为 `.167`；RV(Ph,Spk) 为 layer 0/6/11/12 的 `.0070/.0090/.0080/.0086`，置换 p 为 `.806/.915/.995/.975`。这些 RV 值与置换零分布不可区分，支持“在本设计和样本下没有检测到明显的 phoneme-speaker 子空间重叠”，但不构成 speaker independence 或正交的证明。

B5 独立审查为 **PASS with material interpretability caveat**：实现正确使用实际 `speaker_id`、paired-key grouping 与对称 residualization，且没有把该设计误写成 speaker-balanced。Per-utterance residualization 不是 speaker balancing，也不是因果控制；B5 只能作为 representation/control audit 使用。

## 综合结论

1. 在独立的 MDC English 50 对上，F5-TTS 表征相对于 natural 在多个 SSL 层、多个 phone/viseme 指标上表现出更紧凑或更易迁移识别的模式；该模式不局限于 layer 0。
2. TTS 的时长侧差异较清晰：同 phone count 下整体更快，平均 phone duration 更短，句内 duration CV 更低。逐 phone 的 rate-normalized duration shape 没有 FDR 显著差异，说明主要规律更像“全局缩短 + 句内相对时长压缩”，而不是 phone-specific duration 重排。
3. B3 显示差异具有 phone-specific 结构，layer 6 的 KLD/AUC 排序相关尤其明显；但该关联仍属于 representation-level descriptive evidence。
4. B5 没有发现与随机零分布明显不同的 phoneme-speaker 子空间重叠，且对称 utterance residualization 后 TTS probe 优势仍存在；不过 source-author 重复/不均衡与单一 TTS provider speaker 使其不能被称为 speaker-balanced 或 causal disentanglement。
5. 这些结果为后续 enhancement-head 设计提供了可操作目标：保留自然音频内容与节奏，同时尝试改善 phone-local compactness、跨 utterance phone prototype consistency、句内 duration regularity 和 pause structure。它们只能定义表征/声学训练目标，不能替代独立的 TFG/SyncNet 验证。
6. B4 暂不运行。只有在 MDC English 上完成独立 Ditto/TensorRT feature extraction、TFG inference 与 SyncNet/lip-sync evaluation 后，才能检验上述表征指标是否预测真实下游收益。

## 审查修复与状态

| 类别 | 审查结论与处理 |
|---|---|
| 数据隔离 | 使用独立 `mdc_en_phoneme_20260807_f5_full` run；未混入 AISHELL、LibriSpeech 或 `faster_qwen3` 主结果 |
| 配对/对齐 | 50 个 paired keys；natural/TTS 各自 MFA spans；`en_015` retry provenance 保留 |
| SSL 层语义 | layer 0 按输入 representation 解释；保存层直接按 metadata 映射，未发生旧的 off-by-one 或丢失 layer 12 问题 |
| 统计分组 | utterance/paired-key grouping、leave-one-out 与 global/per-family FDR 规则均记录在产物中 |
| B3 | 独立审查 **PASS** |
| B5 | 独立审查 **PASS with caveat**；明确 28 个重复/不均衡 author IDs 与一个 TTS provider speaker |
| 下游边界 | 未将 representation 结果写成 MDC English TFG/SyncNet 结果；B4 保持 blocked |

**状态：✅ 已完成（MDC English representation audit）；B4 downstream linkage 待独立 MDC English Ditto/TFG/SyncNet 评估。**

## 限制

- 样本为 50 个配对 utterances，TTS 仅使用一个 F5-TTS provider/model，不能代表所有 TTS 系统或语言。
- TTS canonical 音频有 17/50 条非零 clipping fraction；clipping、F5-TTS 声学伪影与表征差异不能在本轮完全分离。
- B5 不是 speaker-balanced design；per-utterance residualization 不消除 source-author 与 provider-level identity 的结构性不匹配。
- MFA、低支持 phone 排除及 token 无 frame center 会限制 phone-level coverage；B3 的 30/74 tested/excluded inventory 不能推广到全部 English phones。
- B2 仅比较四个保存层，峰值是描述性层选择，不是 confirmatory optimal-layer 结论。
- XLS-R 只提供分离度曲线，未提供本轮 logistic phoneme/viseme probe。
- pooled pause JSD 是 unpaired 且 underpowered；outlier 敏感性显示 pause-count 结论弱于 duration CV。
- 所有 B1–B5 结果是表征/声学关联证据，不证明 TFG 机制、因果关系或真实 lip-sync 改善。

## 相关文件

### 数据与运行产物

- 数据 manifest：`data/mdc_tts/manifest.json`
- Pair manifest：`runs/mdc_en_phoneme_20260807/00_pairs/pair_manifest.json`（由 alignment/TTS provenance hash 引用；`_f5_full` 目录不包含独立 pair manifest）
- TTS provenance：`runs/mdc_en_phoneme_20260807_f5_full/02_tts/tts_meta.json`
- MFA alignment：`runs/mdc_en_phoneme_20260807_f5_full/03_alignment/alignment.json`
- MFA retry provenance：`runs/mdc_en_phoneme_20260807_f5_full/03_alignment/mfa_retry_provenance.json`
- SSL embeddings：`runs/mdc_en_phoneme_20260807_f5_full/04_embeddings/`
- Experiment 16：`runs/mdc_en_phoneme_20260807_f5_full/05_separability/separability_metrics.json`
- Experiment 35 phoneme：`runs/mdc_en_phoneme_20260807_f5_full/06_probe_phoneme.json`
- Experiment 35 viseme：`runs/mdc_en_phoneme_20260807_f5_full/06_probe_viseme.json`
- B1：`runs/mdc_en_phoneme_20260807_f5_full/07_duration_jsd/duration_jsd.json`
- B2 phoneme：`runs/mdc_en_phoneme_20260807_f5_full/08_layer_curves/phoneme/layer_curves.json`
- B2 viseme：`runs/mdc_en_phoneme_20260807_f5_full/08_layer_curves/viseme/layer_curves.json`
- B3：`runs/mdc_en_phoneme_20260807_f5_full/09_per_phoneme_kld/per_phoneme_kld.json`
- B5：`runs/mdc_en_phoneme_20260807_f5_full/10_ph_spk_disentangle/ph_spk_disentangle.json`

### 实现脚本

- `scripts/prepare_mdc_english_pairs.py`
- `scripts/generate_mdc_english_tts.py`
- `scripts/prepare_mdc_english_alignment.py`
- `scripts/15_extract_ssl_embeddings.py`
- `scripts/16_feature_separability.py`
- `scripts/35_phoneme_recognition_probe.py`
- `scripts/36_duration_jsd.py`
- `scripts/37_layer_curves.py`
- `scripts/38_per_phoneme_kld.py`
- `scripts/39_ph_spk_disentangle.py`

### 前置与边界文档

- `docs/experiments/09-aishell100-phoneme-extension.md`
- `docs/experiments/10-next-steps-research.md`
- `docs/experiments/11-duration-jsd-analysis.md`
- `docs/experiments/12-layer-curves.md`
- `docs/experiments/13-per-phoneme-kld.md`
- `docs/experiments/14-ph-spk-disentangle.md`
- `docs/experiments/15-tfg-link.md`
