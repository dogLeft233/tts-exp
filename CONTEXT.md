# TTS-TFG 实验项目

## 核心问题

**TTS 生成的语音是否比自然语音产生更好的唇形同步（TFG）？**

使用 AISHELL-1 中文语音 + faster_qwen3 0.6B ICL 模型生成 TTS，经 Ditto/多模型 TFG 生成口型视频，SyncNet 评分。

## 关键结论

> **2026-10-30 中文20、多TFG**：统一RMS−24dBFS、同一3.png、每模型80逻辑视频/九评分器。[EchoMimicV3-Flash报告77](basic-memory/docs/experiments/77-echomimic-v3-chinese20-nine-scorers.md)全20人Sync-C −0.191 [−0.600,+0.218]，主要12人及全20九项Holm9均未显著；有效推理显存采样峰值14.86GiB，本地2376文件/25分层复算通过。[FlashHead报告76](basic-memory/docs/experiments/76-flashhead-chinese20-nine-scorers.md)全20 Sync-C +0.212、CI跨零，主要12人九项均未通过Holm9，全20 PEAVS偏自然（p=.04131）。6来源官方配置敏感性见[报告78](basic-memory/docs/experiments/78-echomimic-v3-official-control.md)：官方组合Δ+1.021，LUFS影响较小、帧数/seed明显，非全20复测。两轮固定五秒分类器各7人完整配对。2026-10-29英语[Wav2Lip40报告75](basic-memory/docs/experiments/75-ears40-english-wav2lip-nine-scorers.md)另列；所有分数均非感知真值。
> **2026-11-02 研究交接**：从[研究交接总览](basic-memory/Research/交接/研究交接总览.md)保留动机、六模块概览及整个BM文档库导航；详细实验按六模块独立保存，按需读取，133段说明覆盖84份编号报告及后续专题，518条来源与523条恢复快照可查；不是独立实验数。研究旨在解释TTS较高Sync-C，评分偏置是待检验解释。PC-Talk/DyStream全量计算完成，完整媒体检查及正式报告待补；HDTF TalkLikeYou及旧训练暂停不变。启动规则与研究路线统一到总览，详细修正链见[接手资料](basic-memory/Research/交接/第三部分-研究接手与历史资料.md#evidence-history)。

> **2026-10-15 输出所有者坐标**：[报告42](basic-memory/docs/experiments/42-output-owned-native-coordinate.md)修正Muse干预I→I+p4，两句raw/z口部Cp/Hp均plus>zero>minus，同H264 observer也2/2通过；但plus原自然音轨full Sync-C−0.169/−0.012且D0恶化，功能控制与评分目标分离。预编码非owned RGB相同而H264像素变化；约61.3%native扰动能量违反Whisper共享切片一致性，投影owned能量仅保留约17.5%。后续6条codec评分已完成（0TFG）：FFV1 plus full C−0.040/+0.024、D0仍均恶化；原事件lag0亦无双人共同增强。可微frontend两CPU零参数parity已逐值复现原mel/native；16参数真实波形6encoder/6VJP已核验，梯度FD最大相对误差0.124%；小幅box-LP中SEP/ENV线性可达、STAB受限，32参数分组strict-superset实际6forward/6VJP和saved-J三目标LP均已核验通过，解除原局部稳定性幅度瓶颈；暂停前下一计划为6forward有限剂量验证；当前不启动TFG/训练或该验证；[41](basic-memory/docs/experiments/41-native-articulatory-response.md)、[40](basic-memory/docs/experiments/40-functional-response-coordinate.md)、[39](basic-memory/docs/experiments/39-native-input-factor-mechanism-screen.md)保留各自冻结结论。

> **2026-10-13 增强头（✅原型可执行，科学结论仍探索性）**：三模型×10未见说话人独立 holdout 已完成，Wav2Lip 相对 RMS full/guard20 为 −0.008916/−0.005989，MuseTalk 为 +0.014064/+0.025367，Ditto 为 0/0；模型感知安全路由等权贡献约 full +0.004688、guard20 +0.008456。当前交付接口 `scripts/audio_enhancement/model_routing.py` + CLI `--model-id`：MuseTalk 走冻结 explicit11 反对称局部功率/谱形候选，Wav2Lip、Ditto、unknown 走 RMS fail-closed；增强音频只驱动生成，评分与换轨保留原自然 PCM/原时钟。单一样本 bundled-Python CPU smoke 已验证三路 finite、16kHz 与样本数保持；本机缺 PyTorch 时 MuseTalk 安全回退 RMS；远端 torch CPU probe 已让三路当前合同实际执行并保存 receipt。shared_audio_head.pt 仍为 FIT_ONLY research-only。未启动新 GPU 矩阵；旧9参数/四中心/final720 已 NOT_ADMITTED，不重跑。完整交接见 [HANDOFF](HANDOFF.md) 与 BM active Task。

> **局部谱形新证据（CONCLUDED描述性）**：固定8Fit/4说话人的三真实前端×32 natural/full/DC/RMS共96项已独立逐数组核验。step060投影mask 99.173–99.617%平方能量来自时间均值；同整体RMS的DC谱形仍产生局部40ms增益，886原时钟窗段内SD均值0.455143dB，平均段q05/q95 −0.391333/+0.949170dB。full/DC相对RMS的特征变化方向cos W/M/D=.995/.970/.992。静态频谱重分配通过随时间变化的谱成分形成局部条件变化；该结果不代替跨模型评分或TTS总体机制。BM《近静态谱形通过局部谱成分改变三类TFG的时变条件特征》保存设计、数值与source-only前端解释。

> **2026-10-03 机制结论**：AISHELL-1中文、cloud Qwen TTS与Wav2Lip链上已定位增益→log-mel→真实音频编码器方向→生成动作的正向响应。男性40说话人固定A方向 ΔC=+0.161，99%[+0.094,+0.232]；内部mel ΔC=+0.171 [+0.101,+0.245]。男性full-A/raw/guard15原生T−N时间超额从q00 RAW/RAW +0.235到q11 FIXED/FIXED +0.031，配对缩小+0.204 [+0.119,+0.295]，同时改变生成和评分输入；另列固定A男性自然生成G excess+0.185 [+0.108,+0.266]。女性20说话人的paired gap-null变化+0.245 [+0.118,+0.388]，保留不同队列与支持。

> **局部声学机制**：保持recipient F0/AP/清浊音、时钟、逐帧SP功率及全局RMS，mu平均相对频带log比与rho内部时间轨迹干预定位recipient谱比轨迹和发声/时序结构的相容性。自然供体→TTS recipient的rho−SELF_GRID fixed-A/raw/guard15 Cfixed −0.192，联合99%[−0.336,−0.047]；joint−grid两方向均负。FULL35不同局部负端点集中n/ŋ及少量a˥˩/早tɕ，约350ms mel覆盖含共发音，guard15局部区间含0。完整800视频/1600评分/11776声学端点（10304可估/1472短Tg20保留）与54动作端点、完整证据恢复和图QA均已独立通过。54族唯一联合99%非零项为该方向rho−grid平均开口+.001519[+.000086,+.002952]，开口和同步评分分别解释。见[正式报告25](basic-memory/docs/experiments/25-tts-tfg-mechanism.md)及BM任务“解释 TTS 的 TFG 与 SyncNet 优势”。

> **当前执行机（✅已资格）**：publickey `wjj@192.168.50.2:22`，Ubuntu/V100 UUID `GPU-d12b3309-c299-da83-064d-a3cd619bd560`，repo `/home/wjj/tts-audio/tts-exp/` 保持开机。本轮幅度attempt02终局及本地readback见HANDOFF；没有本轮GPU任务待运行。native研究在`.codex/scratch/native_attribute_response_probe_20261015_attempt01/`，16/32参数梯度与saved-LP已完成，head后续暂停；qualified lexical Python `/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python`不要resolve成MFA解释器。旧job_control/QUEUE关闭，SHM易失。AutoDL westd:18625为历史，DirectD0未知不重放；旧aliases陈旧。secret只runtime，访问来源见BM `docs/reference/server-access`。

> **2026-09-26 评价协议修正**：下列历史分数是原协议结果。Ditto 47对审计发现默认音频编码的高频损伤可造成大幅分差；统一源PCM后 ΔC=+0.287 [−0.090,+0.648]。全因子、频带双向干预及12真实视频复核见 `runs/syncnet_codec_mechanism_20260926/report.md` 和同名BM实验笔记。未经无损音轨复核，勿把原生高分升级为嘴型改善或发音动态机制。

> **同日跨队列复核**：Wav2Lip/云Qwen 74对、13说话人统一PCM后仍 +0.935；MFA-linear视频换回自然音为 −0.762，共同ROI及独立音频时间校正未救回。联合等时长与发音窗口控制后原生差仍 +0.482。编码机制不能概括全部队列，自身配对高分不等于自然音频迁移收益；详见 `runs/tts_pcm_residual_20260926/report.md` 与BM同名实验。

> **同日无人工独立视觉审计**：68对固定26/42校准/评估，纯视觉内容及已知时间扰动校准通过。原生T内容Q优势+0.716，同帧数后+0.190且区间跨0；不能把内容辨识等同同步或感知质量。原生T视频绕过声码器配自然PCM，字中心MFA仍−0.994；38条音素中心/边界细化仍−0.561/−0.573，插值损伤不足以解释主下降。支持已测迁移路线无收益，不支持“视频没有任何有用信息”；见 `runs/tts_independent_visual_20260926/report.md` 与BM同名实验。

> **同日生成顺序与静态源图复核**：38条/13说话人×3固定源图，原生T mel无声码器地先对齐再生成，比视频后重定时高+0.591，但相对同PCM自然基线仍−0.411；三图及13/13说话人均阴性，发音/边界敏感性稳健。主guard20原生自身差+0.553（B+0.584、D+0.031），包含边缘时仅+0.169；勿称全轨不变优势。固定窗口补偿也未救回，6/6测量对照和独立复核通过；见 `runs/tts_prepost_mel_20260926/report.md`、`runs/tts_window_clock_20260926/report.md` 与BM同名实验。

1. **历史原协议的中文原生 SyncNet 分差成立**，n=100 复现（Ditto: ΔSync-C=+1.02, d=1.24, p<0.0001, 92% 样本；详见 `15`）；编码/PCM及可见口型结论按上方审计分别判断。
2. **原协议评分优势跨模型出现**：Wav2Lip (+1.52) > IMTalker (+1.13) > 其余模型（n=13 旧数据），不直接代表同音轨的生成质量增益。
3. **英文也有原生优势，效应依赖数据与模型**：multiset 的 LRS3 50 对，Ditto ΔC=+1.124（45/50 正向）、LeapTalk +1.389（46/50）；LibriSpeech因用户报告泄漏已排除正/负独立推断，HDTF −0.14 ns 仅属该历史队列。数值与机制 spec 见 `openspec/changes/explain-lrs3-tts-syncnet-gain/`。
4. **差异在声学/韵律层而非音素可分离性**：早层（l0-2）最强（`12`/`13`）；TTS 更快、停顿少 63%、句内时长更均匀（`11`）；**无可分离性指标能预测逐样本 lip-sync 增益**（`15`，Wav2Sem 假设不被支持）
5. **音素结构非说话人混杂**（`14`）；早期"TTS 与自然在 HuBERT 空间完全重叠"结论已被 `13` 修正——逐音素差异在早层可检测（AUC 0.87-0.95）
6. **LRS3 MFA3/`english_mfa` 严格 replacement 未保留增益**：133 条记录的 Wav2Lip 四-cell 矩阵、532 个 strict mux 和 532 个 fresh SyncNet score 全部通过工程核验，但 natural-audio replacement 的 `benefit_C` 均值为 −1.201、`benefit_D` 均值为 −1.061，科学结论为 `NO_GO`；详见 `runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/04_statistics_exploratory_retry2/`
7. **Duration-compatible TTS visual teacher audit 当前工程 BLOCKED**：133 条 fit-only record 全部完成 real/natural/candidate landmark extraction 与独立 artifact review，但仅 115/133 达到固定 eligibility、21/23 effective groups 有 eligible record，低于预注册 minimum 122；scientific decision 为 `not_available`，不得进入 Stage02 或降低阈值；详见 `runs/lrs3_tts_visual_control_20260825/01_visual_teacher_audit_retry4/`

## 正在使用

| 类别 | 项目 | 说明 |
|------|------|------|
| TTS 引擎 | `faster_qwen3` (0.6B ICL) | 本地推理，24kHz 输出 |
| 机制确认 TTS | `qwen3-tts-vc-2026-01-22` | 云端自克隆；保留原24k及独立16k转换 |
| TFG 模型 | Ditto, Wav2Lip, MuseTalk, LatentSync, JoyVASA, V-Express, EchoMimic V2, IMTalker, StableAvatar | 9 个核心模型 |
| 评估 | SyncNet V2 | Sync-C (越高越好) + Sync-D (越低越好) |
| 参考帧预处理 | `38_prepare_frames.py` + `frame_quality.py` + `frame_detectors.py` | 第一帧策略（ADR-002）；mediapipe 检测器；原始分辨率输出 |
| 当前执行机 | Ubuntu/V100 `wjj@192.168.50.2` | 三模型共享确定性GPU环境；AutoDL为历史入口 |
| 流水线 | `00_datacheck → 02_tts → 03_ditto → 04_eval` | 可复现 |

## 已弃用

| 项目 | 原因 |
|------|------|
| EchoMimic V1 | 无 xformers 时 5.6h/样本 |
| AniPortrait | 256×256 fp16 OOM (11GB) |
| Hallo2 | CUDA 11.8 硬要求 (服务器 13.0) |
| 旧 `dashscope_vc` loudnorm 配置 | 旧响度预处理协议；新机制确认使用原始PCM与显式RMS干预 |
| `trt_online.pkl` | 输出 0.2s 视频 (已损坏) |
| 旧 1080Ti 集群 (10.1.114.114) | 11GB VRAM 不足 |
| `aishell1_ldn` loudnorm 实验 | 已是 null result |

## 数据

| 数据集 | 路径 | 说明 |
|--------|------|------|
| HDTF 英文原始音频 | `data/hdtf_audio_en/audio/` | 337 个 .m4a 文件（871MB），美国政客演讲录音 |
| LibriSpeech 配对音频 | `data/data/audio_en/` | 自然参考；Qwen3 TTS 在 `data/data/audio_en_qwen3_tts/` |
| 多语言 TTS 参考音频（旧版） | `data/multilingual_tts/` | 旧版 9 语言、每语言 20 条；CSS10/CML-TTS 存在已知来源重合风险 |
| 多语言 TTS 参考音频（MDC） | `data/mdc_tts/` | 当前 5 语言 × 20 条 = 100 条；英语/德语/意大利语/西班牙语/韩语已提取，另 4 种 MDC 条款仍返回 403 |
| TTS 资产目录 | `data/tts/catalog.json` | 本地 TTS 音频索引，支持按 provider/model/language/dataset 检索 |
| AISHELL-1 100 音素 corpus | `data/aishell1_100_zh/` | AISHELL-1 test split 100 条（16 说话人）+ 同文本 faster_qwen3 TTS；MFA 对齐 + HuBERT/XLS-R 分析 → `docs/experiments/09`，B 系列深化（`11-15`：时长 JSD / 逐层曲线 / 逐音素 KLD / 解纠缠 / TFG 关联）产物在 `results/aishell100_phoneme/` |
| TED-LIUM 3 (30%) | `data/tedlium3_30pct/` | SpeechColab TED-LIUM 3 子集，32 个 HF parquet 分片，81,760 条（15.97GB，≈277h，均值 12.2s）；字段 `audio(bytes)/text/speaker_id/gender/file/id`；16kHz 内嵌音频，尚未被代码引用 |

## 技术栈

- 当前V100：Python 3.10、PyTorch 2.5.1+cu121；历史AutoDL为cu124，runtime身份不得混用
- SyncNet V2 (评估)
- Ditto TRT (TensorRT 离线推理)
- HuBERT / XLSR (SSL 特征分析)
- MFA (强制对齐)

## 文档结构

文档由 Basic Memory 管理（`basic-memory/docs/`，经 MCP 索引）。仓库根不再维护独立 `docs/`。

```
CONTEXT.md              ← 你在这里（仓库根，agent 入口）
basic-memory/docs/
├── adr/                ← 架构决策记录
├── experiments/        ← 实验报告（按阶段编号；当前机制见报告25）
├── deployment/         ← 模型部署记录
├── results/            ← 评分汇总
└── reference/          ← 流水线/服务器/部署参考
```

当前机制报告：[报告25](basic-memory/docs/experiments/25-tts-tfg-mechanism.md)。B 系列（11-15）为 2026-08 的音素/韵律/TFG 深化实验，均经对抗子 agent 审查修复。
