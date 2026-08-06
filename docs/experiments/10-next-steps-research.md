# 10 — 深度调研：下一步研究方向与实验方案

## 问题

n=100 音素扩展实验（`09`）得出：35/126 配对比较 FDR<0.05 显著，主要来自 segment_stability（21）与 boundary_sharpness（12）；TTS 探针 accuracy 每层略高于 natural；旧"TTS 类间距离更优(+7%)"结论为均匀切片测量假象（MFA 修正后方向反转）。在文献层面，**下一步推荐怎么做**？调研覆盖四个方向：(1) 深化音素/表征分析；(2) 表征→TFG 连接；(3) 扩展语言/模型/规模；(4) 面向论文方法论。

## 调研方法

Deep-research harness（106 个 agent，578 次工具调用）：5 路并行 web search → 拉取 24 个源 → 对每条可证伪主张做 3 票对抗验证（2/3 否决才剔除）→ 语义合并、按置信度排序、附引用。产出本报告的结论全部标注置信度，被驳回的主张单独列出。

## 综述结论

### A1. 项目的方法框架已被文献验证（高置信）

SSL 特征几何 + 线性探针是评估音素保真度的被验证框架：

- **音素质心各向同性（Ph\Ph）是 phone 线性探针准确率的最强几何预测**：pooled Spearman ρ=0.94，六模型逐模型 0.69–0.90，全部 p<0.05（Mohamed et al., Interspeech 2024, [arXiv:2406.09200](https://arxiv.org/abs/2406.09200)）
- **说话人/音素子空间正交性（Ph\Spk）与 probe 显著相关**（Transformer 族 ρ=0.54–0.78、CPC-big 1.0、pooled 0.54）→ TTS-vs-natural 比较应控制说话人/音素解纠缠
- 帧级 articulatory-feature 探针 F1 是音素识别性能的强预测（within-language r=0.949、cross-language r=0.990，Ji et al., Interspeech 2022, [arXiv:2206.12489](https://ar5iv.labs.arxiv.org/html/2206.12489)）

项目所用的 intra/inter-class dist、fisher ratio、silhouette + GroupKFold 线性探针正是这套被验证的指标组合。

**层选择是系统性决定因素**（Pasad et al., ICASSP 2023, [arXiv:2211.03929](https://arxiv.org/abs/2211.03929)）：HuBERT 族（WavLM/AV-HuBERT）音素/词信息集中在高层；wav2vec2/XLS-R-53 族在**中间层达峰、高层下降**。项目当前 0/6/11/12 离散层对 HuBERT 合理，对 XLS-R 可能错过最优中间层，应单独标定。

### A2. segment_stability / boundary_sharpness 对应已知 TTS 伪影（高置信）

- **deepfake/TTS 的不一致是整段音素序列的时序不一致，而非孤立音素**（Zhang et al., AAAI 2025, [arXiv:2412.12619](https://mlanthology.org/aaai/2025/zhang2025aaai-phoneme/)）→ 项目的 segment_stability（21 显著）正是整段时序结构度量
- **确定性 L1/L2 时长模型均值塌缩 → 韵律扁平、停顿不足**：>20% 停顿出现在无标点词边界，但 L2 基线在所有词边界预测近零停顿（Abbas et al., Interspeech 2022, [arXiv:2206.14165](https://www.isca-archive.org/interspeech_2022/ammarabbas22_interspeech.pdf)）
- **时长分布 JSD 可区分模型**（停顿 JSD：L2 基线 0.36 > 短语条件 0.25 > 流模型 0.19）→ JSD 是可直接迁移、可证伪的客观指标

这为项目"TTS 时长均匀化"假说提供了可操作化版本和可引用数字。

### A3. 表征→TFG 有已发表的因果先例（高置信，注意边界）

**Wav2Sem（CVPR 2025）**（[论文页](https://www.openaccess.thecvf.com/content/CVPR2025/html/Li_Wav2Sem_Plug-and-Play_Audio_Semantic_Decoupling_for_3D_Speech-Driven_Facial_Animation_CVPR_2025_paper.html), [arXiv:2505.23290](https://arxiv.org/abs/2505.23290)）：对近同音词注入句级 BERT 语义向量后，SSL 空间 L2 可分离度提升（Wav2Vec2 0.0397→0.0701，约 +77%；HuBERT 0.2689→0.2909），并传递到六个 TFG 基线的唇形几何改善（VOCASET FaceFormer LVE 4.1090→3.9891、CodeTalker MVE 4.6132→4.4714；BIWI FaceDiffuse MVE 6.8088→6.5725）。

这是"干预可分离性 → 改善唇形同步"的因果先例，为项目把可分性指标与 SyncNet-C/TFG 关联提供已发表的范式。项目的 `scripts/34_near_homophone_analysis.py` 已按此范式用中文送气/发音部位/声调对照实现词级 L2 + t-SNE，可直接对接到 `scripts/17_link_features_to_tfg.py` 与 SyncNet 分析（scripts/20–25）。

### A4. 边界指标与逐音素 KLD 模板（medium 置信）

- **边界/过渡度量与音素边界共位**：谱变化率峰值（MFCC 40ms 窗回归系数）检测约 85% 的 TIMIT 人工音素边界（70% 落在 10ms、89% 落在 20ms 内）；汉语拼接 TTS 中音节边界谱连续性是听感自然度的主要决定因素；组合 5 种谱距离 + 2 种时域导数距离可预测四类音节边界处的可听不连续（Dušan & Rabiner, Interspeech 2006, [PDF](https://www.isca-archive.org/interspeech_2006/dusan06_interspeech.pdf)；Xu & Cai, ISCSLP 2006, [链接](https://www.isca-archive.org/iscslp_2006/xu06_iscslp.html)）
- **逐音素 KLD + 逐音素探针是现成模板**（Temmar et al., 2025 IEEE SMC, [DOI 10.1109/SMC58881.2025.11343334](https://ieeexplore.ieee.org/document/11343334)）：真实 vs 合成音素的 HuBERT 嵌入分布逐音素 KL 散度 → 给合成音素与真实对应物的对齐程度排序；逐音素真实/合成二分类器；分类器准确率与 KLD 相关——**元音类与分类器性能相关性最好**。可直接用于项目的音素/视位探针逐类分解

### A5. 论文定位论据（高置信）

前沿中文 TTS 诊断 **TTS-PRISM（小米, 2026）**（[arXiv:2604.22225](https://arxiv-org.ezproxy.obspm.fr/html/2604.22225v1)）用 MiMo-Audio 骨干 + LLM rationale+score 打分范式，**完全不用强制对齐、逐音素或 SSL 帧级分析**。项目基于 HuBERT/XLS-R + MFA 的帧级可分离性方法在其中占据差异化、互补的生态位，指标与这条前沿不冗余——可作为论文定位论据。

## 被驳回的主张（不可引用）

验证流程（3 票对抗，2/3 否决即剔除）驳回：

1. **TTS-PRISM"机械感源于子音素缺陷（n/l 混淆、声调 sandhi）"**——不可引用为因果证据
2. **"近同音耦合直接导致唇形生成 averaging effect"**（0-3）——不可把"表征可分离性是唇形同步的因果驱动"当已证事实；只能引用 Wav2Sem 的干预式结果
3. **"此前音素分析都孤立看待单音素、忽略整段时序不一致"**（1-2）——作为"研究空白"论述需弱化
4. **边界度量时间分辨率上界断言**——其强形式被驳回，只能作方法论注意点

## 下一步实验方案（按优先级）

全部基于已有产物增量扩展，不重跑大流水线：

| # | 实验 | 方法 | 现有资产 | 成本 | 预期结论 |
|---|---|---|---|---|---|
| **B1** | **时长分布 JSD**（最高优先） | 用已有 MFA TextGrids 计算 natural vs TTS 的逐音素时长 KLD/JSD + 停顿分布，对照 Abbas 2022 | n=100 TextGrids 在 `results/aishell100_phoneme/mfa_full/` | 纯脚本，无 GPU，半天 | 若 TTS 时长 JSD 大、停顿近零 → 直接验证"时长均匀化/韵律扁平"假说，给 segment_stability 发现客观佐证 |
| **B2** | **逐层可分离性曲线** | 层密度从 {0,6,11,12} 加密到每 2 层，输出每模型/每指标最优层曲线，标定 XLS-R 中间层 | 400 个 embedding 文件已缓存 | 只重跑 metric 脚本，GPU 小时级 | XLS-R 中间层达峰则支持文献规律；可选到最优层后重算可分离性结论 |
| **B3** | **逐音素 KLD + 逐音素探针** | 对每个音素类做 natural-vs-TTS 的 HuBERT KLD + 逐类二分类；重点看声母送气对立、声调对立 | 已有 embedding + MFA 音素标签 | 中等，复用 embedding | 元音/送气对立类最敏感 → 定位"哪些音素类贡献了 35 个显著项" |
| **B4** | **近同音对照 → TFG 传递**（关键因果实验） | 用 scripts/34 已备好的中文送气/发音部位/声调 minimal pairs，接到 17_link_features_to_tfg.py + SyncNet-C，做干预式正对照（仿 Wav2Sem） | scripts/34、17、20–25 已存在 | 需跑 TFG 推理，GPU 天级 | 音素对可分性差异 → SyncNet-C 变化方向一致 → 建立"可分性→唇形同步"项目内证据 |
| B5 | **Ph\Spk 解纠缠控制** | 在配对比较中正交化说话人子空间，重跑显著性 | 100 样本 16 说话人 | 小 | 排除"说话人混杂造成 TTS 差异"的备择解释 |

**建议顺序**：B1（半天，纯脚本）→ B2（小时级）→ B3（天级）→ B4（周末级）。B1+B2 直接产出两个可写进论文的新指标，风险最低。

## 风险与引用边界

- 多数关键文献基于**英语**（TIMIT/LibriSpeech）；中文 AISHELL-1 复现属推断；Wav2Sem 近同音分析是英文词级 L2，项目是中文帧/音素级，存在粒度转移
- 2006 拼接式 TTS 的边界连续性文献对 neural TTS（faster_qwen3）只是方法学类比
- 逐音素 KLD-准确率相关有**特征依赖**（PhonemeDF 对部分 SSL 嵌入报弱/反向相关），且 IEEE 源付费墙未核到逐字原文
- 框架网格量化下限（约 2.5–5ms）限制 boundary_sharpness/segment_stability 时间分辨率——与"均匀切片假象"同源，报告与论文需显式声明
- 源文献**没有**多 TTS/多语言音素保真度评估的效应量与功效基准；n=100→更大样本的功效计算需自行用 bootstrap 估计
- XLS-R 探针单条约 13 分钟，100 样本上已跳过；扩充样本时探针需降维或特征选择

## 状态

✅ **已完成调研** — 结论经 3 票对抗验证；实验方案待执行

## 相关来源（24 个源，按角度分组）

**前沿音素保真度分析（Wav2Sem 范式）**：Wav2Sem CVPR 2025（[openaccess](https://www.openaccess.thecvf.com/content/CVPR2025/html/Li_Wav2Sem_Plug-and-Play_Audio_Semantic_Decoupling_for_3D_Speech-Driven_Facial_Animation_CVPR_2025_paper.html)）、[arXiv:2406.09200](https://arxiv.org/abs/2406.09200)、[arXiv:2206.12489](https://ar5iv.labs.arxiv.org/html/2206.12489)、TTS-PRISM（[arXiv:2604.22225](https://arxiv-org.ezproxy.obspm.fr/html/2604.22225v1)）

**SSL 特征线性探针与音素编码**：[arXiv:2505.23290](https://arxiv.org/abs/2505.23290)、AAAI 2025 phoneme（[mlanthology](https://mlanthology.org/aaai/2025/zhang2025aaai-phoneme/)）、Temmar 2025（[IEEE](https://ieeexplore.ieee.org/document/11343334)）、[arXiv:2211.03929](https://arxiv.org/abs/2211.03929)

**TTS 韵律/时长伪影与边界/段内时序指标**：Dušan & Rabiner 2006（[PDF](https://www.isca-archive.org/interspeech_2006/dusan06_interspeech.pdf)）、Xu & Cai 2006（[链接](https://www.isca-archive.org/iscslp_2006/xu06_iscslp.html)）、Abbas 2022（[PDF](https://www.isca-archive.org/interspeech_2022/ammarabbas22_interspeech.pdf)）、[theses.hal.science/tel-04541736](https://theses.hal.science/tel-04541736)

**声学表征→唇形同步质量预测**：DL-Sync（[ACM](https://dl.acm.org/doi/fullHtml/10.1145/3536221.3556621)）、CVPR 35245（[bytez](https://bytez.com/docs/cvpr/35245/paper)）、TalkingFace 稳定同步（[ITU](https://research.itu.edu.tr/en/publications/audio-driven-talking-face-generation-withstabilized-synchronizati/)）、Shukla 2020（[icmlw](https://mlanthology.org/icmlw/2020/shukla2020icmlw-learning/)）、[Springer 978-981-92-3429-5_15](https://link-proxy.springer.com/chapter/10.1007/978-981-92-3429-5_15)、emergentmind phonetic-synchronization

**TTS 评测基准与统计功效设计**：Edinburgh 听者数量批判（[research.ed](https://www.research.ed.ac.uk/en/publications/are-we-using-enough-listeners-no-an-empirically-supported-critiqu/)）、[VQEG number-of-subjects](https://github.com/VQEG/number-of-subjects)、[ScienceDirect S088523082400130X](https://www.sciencedirect.com/science/article/pii/S088523082400130X)、Blizzard 2007（[clark07](https://www.isca-archive.org/blizzard_2007/clark07_blizzard.html)）、[JMLR 25(23-1318)](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf)、[SpeechArenaBench](https://huggingface.co/datasets/ai4bharat/SpeechArenaBench)

## 相关文件

- 调研原始产物：`/tmp/claude-1000/-mnt-e-Documents-tts-audio-tts-exp/a88bf733-b052-418e-9d23-572fedc80e3b/tasks/wfl67cb57.output`
- 前置实验：`docs/experiments/09-aishell100-phoneme-extension.md`
- 可复用脚本：`scripts/16_feature_separability.py`、`scripts/34_near_homophone_analysis.py`、`scripts/17_link_features_to_tfg.py`
