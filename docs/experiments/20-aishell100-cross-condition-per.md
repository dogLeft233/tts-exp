# 20 — AISHELL-100 跨条件音素/视位 PER

## 问题与范围

此前的 13 条中文小样本实验已经计算过 natural↔TTS 跨条件 PER，但 AISHELL-100 大样本扩展只记录了同条件 separability/probe accuracy，缺少同协议的跨条件迁移结果。本报告补跑 `scripts/35_phoneme_recognition_probe.py`，专门补齐 AISHELL-100 的 natural/TTS 跨条件结果。

本报告与 13 条小样本中文结果分开，不覆盖或重算旧的 `docs/experiments/phoneme_recognition_results.md`。它也不改变 AISHELL-100 原有的 separability 产物。

## 数据与输入

- 数据：AISHELL-1 test split 的 100 个 natural/TTS 配对，paired keys 为 `1`–`100`。
- TTS：`faster_qwen3`，同文本生成。
- 对齐：natural 与 TTS 各自的 MFA `phones` TextGrid，共 200 条；没有把 natural 的 token spans 复制到 TTS。
- 旧 manifest 的 token 是汉字及 initial/final 字段，不足以直接进行 phone-level PER。因此本次从 `results/aishell100_phoneme/mfa_full/out/{natural,tts}/` 的各自 TextGrid 重新构造了独立 phone-level manifest。
- 表征：现有 AISHELL-100 HuBERT embedding，来源为 `results/aishell100_phoneme/embeddings/`；legacy embedding 实际保存层为 0–11，本次运行层 0、6、11。没有运行 XLS-R probe。
- 隔离运行目录：`runs/aishell100_zh_cross_condition_per_20260808/`。

## 方法

使用 `scripts/35_phoneme_recognition_probe.py` 的最近质心分类：

1. 对每个 condition、每个 phone/viseme 类计算 source centroid；
2. self 评估使用 leave-one-utterance-out；
3. cross 评估使用 paired-key leave-one-out：
   - natural prototype → TTS；
   - TTS prototype → natural；
4. 被留出的 paired key 不进入 source centroid；
5. 报告定义的 PER 为 `1 - frame-level nearest-centroid accuracy`，即**逐帧错误率**，不是序列级 phoneme edit-distance PER；
6. MFA phone label 去除 tone 后作为类别标签，保留 phone identity；viseme 使用 MFA phone 对应的 viseme 映射。

本次 phone-level manifest 共包含 51 个去声调 phone 类和 11 个 viseme 类。音频/embedding 使用 16 kHz、320 samples/frame 的 HuBERT frame-time convention。

## 结果

### Phoneme frame-level PER

| Layer | Natural self | TTS self | Natural→TTS | TTS→Natural |
|---|---:|---:|---:|---:|
| L0 | 58.07% | 55.58% | 55.40% | 58.84% |
| L6 | **40.97%** | **38.38%** | **38.97%** | **40.70%** |
| L11 | 45.85% | 44.40% | 45.36% | 45.63% |

TTS self PER 在三个层都略低于 natural，差值为 L0 −2.49 pp、L6 −2.59 pp、L11 −1.45 pp。跨条件两个方向均接近，L6 的 natural→TTS 为 38.97%，TTS→natural 为 40.70%。这说明 prototype transfer 没有明显的单向崩溃。

### Viseme frame-level PER

| Layer | Natural self | TTS self | Natural→TTS | TTS→Natural |
|---|---:|---:|---:|---:|
| L0 | 52.60% | 51.70% | 51.60% | 53.07% |
| L6 | **39.93%** | **37.35%** | **38.08%** | **39.54%** |
| L11 | 41.89% | 39.73% | 40.26% | 41.26% |

Viseme 结果与 phoneme 结果一致：TTS self PER 略低，跨条件方向较为对称，L6 的 absolute PER 最低。

## 与 13 条中文小样本实验的关系

13 条小样本报告在 L6 记录了：

- phoneme self PER：natural 83.90%，TTS 83.34%；
- phoneme cross PER：natural→TTS 71.50%，TTS→natural 71.10%；
- viseme self PER：natural 69.57%，TTS 66.88%。

AISHELL-100 的方向与小样本实验一致：TTS self PER 不高于 natural，跨条件迁移没有出现 TTS 或 natural 一侧明显失效。但两个实验的绝对 PER 不应直接合并比较，因为 phone inventory、样本量、MFA manifest、frame support 和 pooling 记录不同；本报告的主要价值是补齐 AISHELL-100 内部的 cross-condition 结果，而不是重新估计 13 条实验的效应。

## 审查结论

独立只读审查结果：**PASS，无需修复**。

审查确认：

- `alignment_phone.json` 有 200 条记录，natural/TTS 各 100 条，100 个 paired keys，failures 为 0；
- 200 条 phone token/time span 均来自对应 condition 的 MFA TextGrid，重新解析后完全一致；
- natural/TTS embedding JSON/NPY 映射均指向预期的 AISHELL-100 legacy 文件，没有跨 condition 错配；
- arrays 均为 12 层、维度 768，实际运行层 0/6/11，没有错误请求不存在的 layer 12；
- probe 输出的帧数与重新计算的 labeled frame 数一致：natural 17,016，TTS 14,904；
- 100 个 utterance group 均参与 self 与 cross 评估，未发现 held-out paired key 泄漏；
- 输出写入新的隔离目录，没有覆盖旧的 AISHELL-100 separability 或 13 条中文 probe 结果。

## 限制

- 本报告的 PER 是逐帧最近质心错误率，不是序列级 phoneme edit-distance PER。
- tone 在分类时被去除，结果是 tone-stripped phone PER；不能解释为声调识别结果。
- 98 条记录继承了旧 manifest 的 `alignment_confidence=0.5` 元数据；本次重新解析的 phone tier 边界与 TextGrid 完全一致，但 confidence 字段没有在 staging manifest 中重新规范化。
- `ɲ`、`ʎ`、`ʔ` 等少数 IPA phone 在 viseme 映射中使用 parser 的 fallback 类；这影响 viseme 类的精细解释，不影响 phone label 本身。
- 本次只运行 HuBERT 0/6/11，没有补跑 XLS-R。
- 同条件与跨条件结果都是描述性 probe 指标；它们不能单独证明 TFG、SyncNet 或 lip-sync 的改善。

## 状态

✅ **已完成** — AISHELL-100 cross-condition frame-level phoneme/viseme PER 已运行并通过独立审查。

## 相关文件

- 运行 manifest：`runs/aishell100_zh_cross_condition_per_20260808/alignment_phone.json`
- phoneme 结果：`runs/aishell100_zh_cross_condition_per_20260808/phoneme_probe.json`
- viseme 结果：`runs/aishell100_zh_cross_condition_per_20260808/viseme_probe.json`
- probe 脚本：`scripts/35_phoneme_recognition_probe.py`
- phone parser：`scripts/33_prepare_mandarin_alignment.py`
- source alignment：`results/aishell100_phoneme/mfa_full/out/{natural,tts}/`
- source embeddings：`results/aishell100_phoneme/embeddings/`
- AISHELL-100 原 separability 报告：`docs/experiments/09-aishell100-phoneme-extension.md`
- 13 条中文跨条件 PER 报告：`docs/experiments/phoneme_recognition_results.md`
