# 多数据集样本准备（5×50）

**日期**: 2026-08-03  
**状态**: ✅ 可用  
**清单**: `data/dataset_samples/video_manifest_250.json`（250 条 records，schema v2）

## 目标

多数据集 TFG 泛化实验（ADR-002 第一帧参考帧协议）需要 5 个权威数据集 × 50 样本，每样本 = 视频 + 自然音频（配对）。

## 数据集与来源

| 数据集 | 样本 | 来源 | 规格 | 音频 |
|--------|------|------|------|------|
| **MEAD** | 50 | 官方百度盘 `W025`（40 neutral + 10 happy, front 视角） | 1080p h264 ~3-5s | mp4 内嵌 AAC |
| **VFHQ** | 50 | `hyb1124/liveface_vfhq` clip + YouTube 原音频 | 840-1310px 4-6s | YouTube 原音频按帧范围截取 |
| **TalkVid** | 50 | 官方 HF bench `videos_w_audios`（随机 seed 42） | 短视频段 | mp4 内嵌 |
| **LRS3** | 50 | HF `Ainncy/LRS3` pretrain.00.tar（45 说话人） | 224×224 人脸轨迹 6-14s | mp4 内嵌 + 词级 txt 转录 |
| **GRID** | 50 | Zenodo 官方 `s1.zip`（s1 前 50 句） | 360×288 mpg→mp4 3s | mpg 内嵌 mp2（44.1k→16k） |

## 关键方案：VFHQ 音频对齐

VFHQ 官方视频**无音频轨道**，音频来自 YouTube 原视频。对齐链：

1. `hyb1124/liveface_vfhq` 提供 961 个裁剪 clip，文件名编码帧范围：`Clip+<yt_id>+P<part>+C<clip>+F<start>-<end>.mp4`
2. liveface 帧号 = 原 YouTube 视频帧号（25fps 假设，经 0GRnvFxj8IU 时长验证：视频 6.48s vs F24901-25064/25=6.52s）
3. 下载 YouTube 原视频音频（yt-dlp `-f ba/b`，HTTP 代理 127.0.0.1:7890）→ `ffmpeg -ss <start/25> -t <dur/25>` 截取 → 16k mono wav
4. 截取前校验 `start+dur ≤ 音频实际时长`（防不完整下载导致空 wav）

**脚本**: `scripts/download_vfhq_v3.py`（输出 50aligned/ + align.json，含 aligned/yt_id/start_s/dur_s）

**已知失效源**（不可用）：`06OdQh1zcwQ`（YouTube 账号终止）、`-34OcNuUn7s`、`0sbnDLBp5Ag`（HF 文件 404）。

## 其他数据集要点

- **MEAD**：官方百度盘 `pan.baidu.com/s/1zYKz19kmQH6vnO2nogPk4Q`（code th8e），经 `bdpan transfer` 转存后按文件下载；`W025` 是 video.tar 最小的 actor（3.7GB）；neutral 仅 level_1，不足 50 由 happy/level_1 补
- **LRS3**：一卷 pretrain tar 含 1185 视频/46 说话人；采样策略 = 每说话人 1 个 + 前 5 说话人补第 2 个（seed 42）；txt 为官方词级转录（TTS 文本用官方转录，不用 ASR）
- **GRID**：Zenodo `s1.zip`（423MB，mpg 实际为 mpeg1video 360×288 + mp2 44.1kHz 内嵌音频）；`audio_25k.zip`（25kHz wav）时长仅 ~1.2s（剔除静音），**使用 mpg 内嵌音频（3.0s 与视频同步）**；bekalemu HF 镜像为 lip_roi 无音频版，弃用
- **TalkVid**：bench 4 类共 205 个带音频视频；随机采样（seed 42）

## 管线验证（14 样本先行）

2026-08-02 已用每数据集 2-3 样本跑通全链（抽帧→ASR→TTS→Ditto→SyncNet），两台服务器结果逐位复现（seed 42 确定性）。发现并修复 3 个 bug：

1. **音频提取音量异常**（-37dB 静音级）：首次提取偶发，rm 后重提取恢复（-21dB）；ASR 对静音产生幻觉文本（whisper "新闻腔"）→ 修复后转录正常
2. **ASR 质量**：wav2vec2-large-xlsr-53 greedy（facebook 版 vocab 32 与 tokenizer 不匹配）→ jonatasgrosman 英文微调版 → 最终 whisper-small + LRS3 官方转录
3. **SyncNet `--min-track 100` 过滤短视频**（3-6s 仅 75-150 帧）→ `--min-track 25`

结果：ΔSync-C 均值 +0.89，11/14 正向（LRS3/MEAD/GRID 系统性增益）；talkvid_C/grid_bbaf3s 为 TTS 模型对特定 ref 克隆失败（静音/弱音量输出），非管线问题。

## 局限与注意事项

- VFHQ fallback 已全部消除（50/50 aligned）；但 YouTube 音频依赖代理，且部分源随时间失效
- GRID/LRS3 低分辨率（360×288 / 224×224）压低绝对 Sync-C（见 `docs/experiments/08-reference-frame-selection.md` 协议差异讨论），配对 Δ 不受影响
- SyncNet 对音频音量敏感：静音音频可产生伪同步高分（第一轮验证数据已弃用）；正式实验需记录音频响度或统一 loudnorm
- 服务器磁盘管理：500 个 SyncNet 中间产物（pyavi/pycrop）需评完即删

## TTS 音频生成（250 样本，2026-08-03）

**产物**: `data/tts_audio_250/`（250 个 wav + transcript.json + tts_meta.json）与远端 `runs/multiset_pipeline/02_tts/`

| 步骤 | 方案 | 结果 |
|---|---|---|
| ASR | 百炼 DashScope `qwen3-asr-flash`（multimodal-generation 端点，base64 data URI） | 250/250 |
| TTS | faster_qwen3 0.6B ICL（本地权重 `/root/autodl-tmp/models/Qwen3-TTS-12Hz-0.6B-Base`，seed 42，24kHz） | 250/250, 673s |

### 配置

- ASR 响应为标准 choices 格式（`output.choices[0].message.content[0].text`，实测确认）；`parameters.asr_options.enable_itn=False`；retry 2 次
- 全部数据集统一走云端 ASR（**不用 LRS3 官方 txt**——txt 是 TED 整句级，与 6-14s 视频片段不匹配，TTS 输出会远超视频时长；ASR 片段转录后 TTS 时长/自然时长中位比 ≈ 0.84）
- 运行配置：`scripts/configs/multiset250.yaml`（250 全量）、`multiset250_retry*.yaml`（子集重试）

### 校验发现的问题与修复（TTS 失败模式）

1. **非英文文本循环**（157s 重复输出）：Qwen3-TTS 0.6B 仅支持中英；VFHQ YouTube 源含印地语/德语/法语样本 → ASR 语言检测逐个验证，替换为英文样本
2. **静音 ref → 静音输出**（rms 0.003-0.004）：talkvid bench 2 个源视频本身静音（非提取 bug）→ 从 HF 换样本（Al5MFMZvpY4、GBugVPs27Os）
3. **慢速生成**（GRID 7 词 8.2s，natural 3.0s）：同 seed 确定性 → seed 7 重生成修复（2.4s/3.2s）
4. **LRS3 长句**（15-26s 甚至 59-104s）：不是异常——LRS3 视频本身长（natural 最长 104s），TTS/natural 比 0.84 正常

### 最终校验标准（250 全过）

- TTS rms > 0.005（无静音输出）
- TTS/natural 时长比 ∈ [0.3, 2.5]（无循环/截断）
- TTS 时长 ≥ 1.0s

### 经验

- 云端 ASR 对静音音频产生幻觉文本（如"新闻腔"中文），故静音样本必须替换而非重试 ASR
- 模型权重排查先查 `autodl-tmp/models/`（hub cache 可能是残壳）；`snapshot_download(local_dir)` 会写 `.cache` 临时区，盘满时静默失败
- `faster_qwen3` 失败（循环/静音）在同一 seed 下确定性复现，换 seed 部分可解（GRID 慢速样本），文本语言不支持则必须换样本

## 正式实验推理（250 样本，2026-08-04）

**产物**: `runs/multiset_pipeline/04_eval/`（每样本 syncnet.json + eval_meta.json + scores_250.json）本地保存；Ditto mp4 留在服务器（已 shutdown，未下载）

| 步骤 | 配置 | 结果 |
|---|---|---|
| 参考帧 | `38_prepare_frames`（mediapipe 检测器，cv2 5.0 无 CascadeClassifier）+ 12 个拒绝样本 ffmpeg 第一帧补齐 | 250/250 |
| Ditto | 双臂 natural_raw/tts_raw，TRT online，seed 42，`--config multiset250.yaml`（修复：03_ditto 默认读 config.yaml 的 samples.ids=1-10，必须带 override） | 497/500 |
| SyncNet | `04_eval --min-track 25`（syncnet env 需 pip 装 librosa） | 496/500 |

**核心结果（配对 250/250 全完整，样本集修正后）**: **ΔSync-C 均值 +0.980，正向 204/250 (81.6%)**——多数据集泛化成立（14 样本验证 +0.89 → 正式 +0.98）

| 数据集 | n | Natural | TTS | ΔSync-C | 正向 | Cohen's d |
|---|---|---|---|---|---|---|
| GRID | 50 | 3.909 | 5.109 | +1.200 | 44/50 (88%) | 1.03 |
| LRS3 | 50 | 4.740 | 5.864 | +1.124 | 45/50 (90%) | 1.25 |
| VFHQ | 50 | 3.852 | 4.892 | +1.040 | 38/50 (76%) | 0.83 |
| TalkVid | 50 | 3.981 | 4.962 | +0.982 | 40/50 (80%) | 0.91 |
| MEAD | 50 | 3.677 | 4.234 | +0.557 | 37/50 (74%) | 0.55 |

**初始失败的 3 个样本（2026-08-04 已全部修复，250/250）**:
- 156、192（LRS3 长音频 78s/104s）：tts 臂 Ditto 300s 超时为**偶发**——重跑即成功（122s 完成 6 个视频）
- 248（grid bbizzn）：第一帧 insightface 检测失败（mediapipe 可检）→ 换 1.0s 帧（f1_0.png）后两臂成功；多候选帧测试 0.5-2.5s 全部可用（35-37s/视频）

### 响度混淆检验（2026-08-04，论文审稿防御）

**结论：响度差异不构成混淆，无需 loudnorm 重跑。**

| 指标 | natural | TTS | 配对检验 |
|---|---|---|---|
| RMS (dBFS) | -24.38 ± 6.25 | -25.52 ± 6.38 | Δ=-1.14 dB, p=2.6e-13 |
| LUFS (integrated) | -23.82 | -25.06 | Δ=-1.24 LUFS, p=4.7e-16 |

统计显著（n=250 功效高）但幅度可忽略（<2 dB）。决定性证据：
- corr(ΔLUFS, ΔSync-C): **r=-0.001, p=0.988**（零相关）
- 分层：\|ΔLUFS\|<1 dB 组 ΔSync-C=0.982 vs ≥1 dB 组 1.031（组间 p=0.735）
- → 主效应与响度差无关；论文中如实报告 Δ=-1.24 LUFS 并给出上述排除证据即可

### LeapTalk 全量推理（2026-08-04，模型无关性复现）

**产物**: `runs/leaptalk_multiset/{natural_raw,tts_raw}/{1..250}.mp4`（本地 178M）+ 服务器 `/root/autodl-tmp/repos/tts-exp/runs/leaptalk_multiset/`（已 shutdown）

- 服务器 21446（connect.weste，RTX 4080 SUPER，已 shutdown）：leaptalk 仓库 + SoulX-FlashHead-1_3B + wav2vec2-base-960h + leaptalk LoRA
- **关键修复**：Model_Pro 缺 config.json → 从 safetensors 权重推断生成（dim=1536, in_dim=32, ffn_dim=8960, out_dim=16, text_dim=4096, freq_dim=256, vae_stride=[4,8,8], patch_size=[1,2,2], num_heads=12, num_layers=30, has_image_input=False）；out_dim 按 head.head.bias 校验（16×4=64）
- venv 为 CPU torch → 重装 cu126（pip 双 index 参数冲突会装 CPU 版，须只用 --index-url）
- 参考帧 = ffmpeg 视频第一帧（⚠️ 与 Ditto 的 mediapipe 质量门控帧**不完全一致**——134/248 两个源视频第一帧无脸导致 LeapTalk 生成失败，见下）
- **batch 脚本** `batch_leaptalk.py`：模型加载一次循环 500 样本（inference.py 单次调用会 500 次加载）；import 遮蔽坑（flash_head/inference.py 优先于 leaptalk/inference.py）；PeftModel 来自 peft 库
- 推理：**498 ok + 2 smoke = 500/500，3177s（53 分钟）**，~5.5s/视频（无 flash_attn，pytorch attention）

### LeapTalk SyncNet 评分（2026-08-04，13201 服务器）

**产物**: `runs/leaptalk_eval/04_eval/`（scores_leaptalk.json 496 条 + 每样本 syncnet.json，本地）

**核心结果（配对 248/250）**: **ΔSync-C 均值 +1.256，正向 213/248 (85.9%)**——模型无关性复现成立（Ditto +0.980 vs LeapTalk +1.256，diffusion 与 SoulX 系两种技术路线均正效应）

| 数据集 | n | Nat | TTS | ΔSync-C | 正向 | Cohen's d |
|---|---|---|---|---|---|---|
| VFHQ | 50 | 5.148 | 6.968 | **+1.820** | 48/50 (96%) | 1.08 |
| GRID | 49 | 4.143 | 5.829 | **+1.685** | 42/49 (86%) | 1.09 |
| LRS3 | 50 | 5.560 | 6.949 | **+1.389** | 46/50 (92%) | 1.40 |
| TalkVid | 49 | 5.491 | 6.229 | +0.738 | 37/49 (76%) | 0.53 |
| MEAD | 50 | 5.747 | 6.391 | +0.644 | 40/50 (80%) | 0.62 |

**缺失 2 个样本（134 talkvid / 248 grid）**：LeapTalk 参考帧用 ffmpeg 第一帧，这两个源视频第一帧无人脸（134 = 网页截图帧、248 = 灰色帧）→ LeapTalk face crop 生成坏视频 → SyncNet parse failed。Ditto 不受影响（mediapipe 门控自动选视频中段人脸帧）。修复 = 用 Ditto 的正确帧重跑这 2 个样本（需重启 21446）。

- 评分在 13201 完成：eval 期间磁盘守护脚本（<1.5G 自动删 syncnet_data，syncnet.json 缓存不受影响）
- 注意 04_eval 的 run_dir 结构：LeapTalk 视频置于 `runs/leaptalk_eval/03_ditto/{cond}/` 以满足 find_videos（实际为任意模型视频）

### 样本集修正（2026-08-04，重要）

发现本地 manifest 的 VFHQ 段有 **9 个陈旧引用（v3 多轮替换后未同步）+ 3 个重复样本**（共 12 个位置内容错误/重复）。已用 `align.json` 权威集重建：
- 9 个 stale → align 中未引用样本（-ETAGhfvBSo_C1 等，全部 ASR 验证英文）
- 3 个重复 → 1ruwVc0t2c0_C1 / 2S-QW1nHpGQ_C1 / 3wgnbGStlzE_C0（二次替换剔除捷克语/德语/ASR 失败候选）
- 最终：vfhq 50 unique = align.json 全集，0 缺失 0 重复；manifest 全 250 校验通过
- **已完成重跑（2026-08-04）**：12 个位置（61/71/73/76/77/90/91/94/95/97/98/99）全流程重跑（ASR 全英文验证 → TTS 12/12 → Ditto 24/24 → SyncNet 500 ok 250/250 完整）。VFHQ Δ 修正为 +1.040（原 +1.210 含外语瞎读偏差）；修正 12 样本 7 正 5 负，属真实统计波动。**注意**：重跑前必须删除 12 位置的旧 mp4 + syncnet.json（Ditto/eval 按 sample_id 缓存跳过，不看内容）；02_tts 参数是 `--run_id`（下划线），`--run-id` 会报 required
- 教训：v3 多轮替换后必须整体重建 manifest 的 VFHQ 段（align.json 为唯一真相源），不能只 patch 单个位置

**坑**：
- `03_ditto.py --config` 必须指定 multiset250.yaml（否则 samples.ids=1-10）
- 03_ditto 调用本地脚本 `ditto_seeded_inference.py`（远端缺失会导致 "can't open file"）
- 磁盘：SyncNet syncnet_data 中间产物 ~4.4G，eval 后需 `find ... -name syncnet_data -type d -exec rm -rf`（浅 glob 删不干净）
