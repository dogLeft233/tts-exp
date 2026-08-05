# IndexTTS2 使用说明

本项目的 IndexTTS2 部署在独立的 SeetaCloud 服务器上进行批量 TTS 生成。

## 部署信息

**服务器**: `connect.westc.seetacloud.com:49961` (RTX 4080 SUPER 32GB, CUDA 13.0)  
**登录信息**: 见 `tmp/servers.sh` 中 `INDEXTTS` 段  
**仓库路径**: `/root/autodl-tmp/repos/index-tts/`  
**检查点**: `/root/autodl-tmp/repos/index-tts/checkpoints/` (5.5GB, 来自 ModelScope)  
**虚拟环境**: `/root/autodl-tmp/repos/index-tts/.venv/` (Python 3.10.20, torch 2.8.0+cu128)  
**关键环境变量** (已写入 `~/.bashrc`):
- `UV_CACHE_DIR=/root/autodl-tmp/.uv-cache`
- `UV_PYTHON_INSTALL_DIR=/root/autodl-tmp/.uv-python`
- `HF_ENDPOINT=https://hf-mirror.com`

**存储限制**: `/root` 仅剩 341MB, 所有数据必须放在 `/root/autodl-tmp` (50GB 可用)

## 网络状况

| 站点 | 可访问 | 延迟 |
|------|:------:|------|
| github.com | ✅ | 0.9s |
| hf-mirror.com | ✅ | 0.4s |
| modelscope.cn | ✅ | 0.2s (最快) |
| pytorch.org cu128 | ✅ | 1.1s |
| pypi.org | ✅ | 0.7s |
| mirrors.aliyun.com/pypi | ✅ | 7s (pip 默认使用) |
| mirrors.tuna.tsinghua.edu.cn | ✅ | 4s |
| huggingface.co | ❌ | — |

## 连接到服务器

```bash
# 见 tmp/servers.sh 中真正的密码
sshpass -p '<password>' ssh -p 49961 root@connect.westc.seetacloud.com
```

## 环境初始化 (每次登录后执行)

```bash
export PATH=/root/miniconda3/bin:$PATH
export UV_CACHE_DIR=/root/autodl-tmp/.uv-cache
export UV_PYTHON_INSTALL_DIR=/root/autodl-tmp/.uv-python
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/root/autodl-tmp/repos/index-tts:$PYTHONPATH
cd /root/autodl-tmp/repos/index-tts
```

## 单句推理

```bash
uv run python -c "
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path='checkpoints/config.yaml', model_dir='checkpoints', use_fp16=True)
tts.infer(spk_audio_prompt='examples/voice_01.wav', text='你好世界', output_path='out.wav')
"
```

### 推理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cfg_path` | `"checkpoints/config.yaml"` | 模型配置路径 |
| `model_dir` | `"checkpoints"` | 检查点目录 |
| `use_fp16` | `True` | 半精度推理 (降低显存, 加速) |
| `use_cuda_kernel` | `False` | 编译 CUDA kernel (可能更快但慢启动) |
| `use_deepspeed` | `False` | DeepSpeed 加速 (需 `--extra deepspeed`) |
| `spk_audio_prompt` | 必需 | 参考说话人音频 |
| `text` | 必需 | 要合成的文本 |
| `emo_audio_prompt` | `None` | 情感参考音频 |
| `emo_alpha` | `1.0` | 情感强度 (0.0–1.0) |
| `emo_vector` | `None` | 情感向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm] |
| `use_emo_text` | `False` | 根据文本自动推断情感 |
| `output_path` | 必需 | 输出 wav 路径 |

## WebUI

```bash
uv run webui.py --fp16
# 然后从本地: ssh -L 7860:127.0.0.1:7860 -p 49961 root@connect.westc.seetacloud.com
# 浏览器打开 http://127.0.0.1:7860
```

## 批量生成 (13 个测试样本)

### 文件结构

```
/root/autodl-tmp/indextts_batch/
├── ref_audio/        # 参考音频 {1..13}.wav
├── transcripts.json  # 转录文本映射
├── output/           # 生成结果 {1..13}.wav
└── batch_generate.py # 批量脚本
```

### 转录文本映射

```
 1: "组织了第一批七个地区城市开展三网融合试点"
 2: "进一步发挥市场在资源配置中的决定性作用"
 3: "为了规避三四线城市明显过剩的市场风险"
 4: "住房城乡建设部政策研究中心主任秦虹表示"
 5: "我国房地产市场过去从体偏紧部分地区过紧"
 6: "政府出台每平方米补贴五百元的托市政策"
 7: "对绿色高效宜居的高品质住房需求快速上升"
 8: "在经济增速放缓阶段运用货币政策工具时"
 9: "安徽铜陵结束了当地契税补贴政策"
10: "促使我国战略性新兴产业发展实现了良好开局"
11: "在市场整体从高速增长进入中高速增长区间的同时"
12: "国务院发展研究中心市场经济研究所副所长邓郁松认为"
13: "严厉打击春运期间违规上调票价价外收费等违法行为"
```

来源: 样本 1-10 来自 AISHELL-1 标注, 样本 11-13 来自 `data/test.csv` (BAC009S0764W0143/W0179/W0209)

### 执行批量生成

```bash
# 上传参考音频 (如需要)
tar -czf ref.tar.gz -C data/data/audio/ 1.wav ... 13.wav
scp -P 49961 ref.tar.gz root@connect.westc.seetacloud.com:/root/autodl-tmp/

# 运行
uv run python /root/autodl-tmp/indextts_batch/batch_generate.py
```

### 生成结果统计 (2026-07-08)

| 样本 | 输出时长 | RTF | 状态 |
|------|---------|------|:----:|
| 1 | 5.78s | 1.42 | ✅ |
| 2 | 5.14s | 1.32 | ✅ |
| 3 | 5.47s | 1.33 | ✅ |
| 4 | 6.22s | 1.27 | ✅ |
| 5 | 6.41s | 1.29 | ✅ |
| 6 | 6.08s | 1.27 | ✅ |
| 7 | 5.78s | 1.28 | ✅ |
| 8 | 5.69s | 1.28 | ✅ |
| 9 | 5.33s | 1.29 | ✅ |
| 10 | 5.99s | 1.28 | ✅ |
| 11 | 7.05s | 1.28 | ✅ |
| 12 | 7.55s | 1.28 | ✅ |
| 13 | 7.36s | 1.22 | ✅ |

全部 13 个样本均超过 SyncNet 最低 4 秒阈值。平均 RTF ~1.3 @ FP16。

## 输出格式

- **采样率**: 22050 Hz (IndexTTS2 原生输出)
- **声道**: 单声道
- **格式**: 16-bit PCM WAV
- **字号**: float32 内部, 写盘时自动转换

如需转为 16kHz (与自然参考音频对齐):
```bash
ffmpeg -i in.wav -ar 16000 -ac 1 out.wav
```

## 交付 TFG 管道评测

1. 将生成的 13 个 wav 文件上传至主评测服务器 (`:24599`) 的 `runs/<run_id>/02_tts/`
2. 运行 `python scripts/03_ditto.py --run_id <run_id>`
3. 运行 `python scripts/04_eval.py --run_id <run_id>`
4. 运行 `python scripts/05_report.py --run_id <run_id>`
5. 汇总: `python scripts/06_cross_model.py` 与其他 4 个提供者对比

## 模型恢复 (如果 checkpoint 丢失)

```bash
cd /root/autodl-tmp/repos/index-tts
uv run modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

## 情感向量实验 (2026-07-08)

### 研究问题

IndexTTS2 的 8 维情感向量 ([happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]) 能否改善下游 TFG (Ditto→SyncNet) 的唇同步质量？

### 情感接口原理

源码分析 (`indextts/infer_v2.py`):

- 情感向量通过预训练的 `emo_matrix` 与说话人情感 embedding 混合:
  ```
  emovec = emovec_mat + (1 - Σweight_vector) × emovec
  ```
  其中 `weight_vector = emo_vector × emo_bias`
- 内部偏置 `emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]`
  - calm (0.5625) 和 surprised (0.6875) 被大幅削弱
- `emo_alpha` 控制向量强度 (0.0–1.0), 低 alpha → 更多自然音频情感, 少矩阵影响
- 设 `emo_vector` 时自动禁用 `emo_audio_prompt`, 情感参考默认回退为同说话人音频
- `use_emo_text=True` 调用 QwenEmotion 从文本推断 8 维情感, 新闻文本大概率判为 calm

### 实验设计

| 阶段 | 条件 | 样本 | 文件数 |
|------|------|:---:|:---:|
| Phase 2 情感扫描 | 8 情感 × 5 样本, alpha=1.0 | 1,4,7,10,13 | 40 |
| Phase 3 强度剂量 | 2 情感 × 3 alpha × 3 样本 | 1,7,13 | 18 |
| 基线 (此前) | 无情感向量, 默认 infer() | 1–13 | 13 |

- Phase 2 情感: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
- Phase 3 情感: calm (中性), happy (表现力强)
- Phase 3 alpha: 0.3, 0.6, 1.0

### 输出位置

```
tmp/indextts2_output/              # 基线 (13 文件, 5.9 MB)
tmp/indextts2_emo_sweep/           # Phase 2 (40 文件, 11 MB)
  {sid}_{emotion}.wav
tmp/indextts2_emo_intensity/       # Phase 3 (18 文件, 4.5 MB)
  {sid}_{emotion}_{alpha}.wav
```

### Phase 2 输出时长统计

同文本不同情感产生的音频时长差异可达 49%:

| 情感 | 时长范围 | 特征 |
|------|---------|------|
| melancholic | 6.61–8.22s | 最长 (语速慢, 拖长) |
| afraid | 5.49–7.36s | 偏长 |
| sad | 5.83–7.49s | 偏长 |
| disgusted | 5.05–6.70s | 中等 |
| calm | 4.91–5.94s | 中等 |
| angry | 4.86–5.83s | 偏短 |
| happy | 4.77–5.99s | 偏短 |
| surprised | 4.63–5.50s | 最短 (语速快) |

全部 58 个文件 > 4s, 通过 SyncNet 最低阈值。

### 重新生成

```bash
# 在 :49961 服务器上
cd /root/autodl-tmp/repos/index-tts
uv run python /root/autodl-tmp/indextts_batch/batch_generate_emo.py

# 脚本副本保存在本地: tmp/batch_generate_emo.py
```

### TFG 评测结果 (2026-07-08)

评测在 :20398 (RTX 4080 SUPER, CUDA 13.2) 上完成。16 个运行目录, Ditto+SyncNet+Report 全管道耗时 69m27s, 0 失败。

原始数据: `data/r3_results/p2_emotion_sweep.csv`, `p3_intensity_sweep.csv`, `baseline_gsv.csv`  
分析脚本: `scripts/07_emo_analysis.py`

#### Phase 2: 情感扫描结果

所有 TTS 条件均优于同句自然音频 (SynCC 更高且 SyncD 更低)。但情感条件之间存在显著差异:

**One-way RM-ANOVA: F(8,32)=6.585, p=0.000028** ← 高度显著

| 条件 | SynCC | SyncD | vs Baseline ΔC | vs Baseline p |
|------|:-----:|:-----:|:------:|:---:|
| **angry** | 8.109 | 6.810 | +1.509 | 0.0013 ** |
| **calm** | 7.582 | 7.162 | +0.982 | 0.0263 * |
| **surprised** | 7.073 | 7.567 | +0.473 | 0.1945 |
| disgusted | 6.881 | 7.604 | +0.281 | 0.4070 |
| happy | 6.625 | 7.837 | +0.025 | 0.8754 |
| baseline | 6.600 | 7.822 | — | — |
| sad | 5.423 | 7.813 | −1.177 | 0.0096 ** |
| afraid | 5.426 | 7.777 | −1.174 | 0.0008 *** |
| melancholic | 5.234 | 7.814 | −1.366 | 0.0084 ** |

(基线 = IndexTTS2 无情感向量, 13 样本 baseline)

**核心发现**: angry (ΔC=+1.51) 和 calm (ΔC=+0.98) 显著提升唇同步质量; melancholic/sad/afraid 显著劣于基线。语速快、节奏清晰的情感 (angry, surprised) 对唇同步最有利; 拖长、含混的情感 (melancholic, sad) 则有害。

#### Phase 3: 强度剂量-反应

| 条件 | α=0.3 | α=0.6 | α=1.0 | 趋势 |
|------|:-----:|:-----:|:-----:|------|
| calm SynCC | 7.213 | 7.285 | 7.879 | ↗ 单调递增 |
| happy SynCC | 7.187 | 7.642 | 6.843 | ∩ 峰值 α=0.6 |

calm 随强度单调提升 (α=1.0 ≈ baseline +1.28), happy 呈倒 U (α=0.6 最高, α=1.0 反而下降), 说明过强的积极情感可能引入音高变异伤及唇同步。

#### IndexTTS2 vs GSV-TTS

| 模型 | SynCC | SyncD |
|------|:-----:|:-----:|
| IndexTTS2 (无情感) | 6.729 | 7.719 |
| GSV-TTS | 6.362 | 7.919 |
| 差值 | +0.367 | −0.200 |

IndexTTS2 在两方面均优于 GSV-TTS (效应量 d=0.418, p=0.20), 虽未达显著但趋势一致。

#### 最终排名 (SynCC ΔC 对自然语音)

| 排名 | 条件 | ΔC | n |
|:----:|------|:---:|:-:|
| 1 | angry | +2.799 | 5 |
| 2 | calm α=1.0 | +2.731 | 3 |
| 3 | happy α=0.6 | +2.494 | 3 |
| 4 | calm | +2.273 | 5 |
| … | … | … | … |
| 15 | afraid | +0.117 | 5 |
| 16 | sad | +0.114 | 5 |
| 17 | melancholic | −0.076 | 5 |

#### 结论

IndexTTS2 情感向量对 TFG 唇同步质量有**显著且方向不一的因果影响**: angry/calm 增强 (最大 ΔC≈+2.8), melancholic/sad/afraid 削弱。强度与积极度呈非线性关系。建议未来实验中默认使用 `emo_vector=[1,0,0,0,0,0,0,0]` (angry) 或 `emo_vector=[0,0,0,0,0,0,0,1]` (calm, α=1.0)。

## 音频物理属性对 TFG 唇同步质量的影响 (2026-07-08)

### 研究问题

IndexTTS2 情感实验发现的情感效应是由情感内容本身驱动, 还是由混淆的音频物理属性 (响度、时长) 驱动？

### Phase A: 观测性分析

**数据**: 全部 136 个 TTS 文件 (r2 跨模型 + r3 IndexTTS2/GSV), 提取 RMS、LUFS、时长、F0、频谱质心。

**相关分析** (Pearson r, n=134):

| 特征 | r(SyncC) | p | r(SyncD) | p |
|------|:---:|---|:---:|---|
| 时长 | −0.307 | 0.0003 | −0.092 | ns |
| RMS | +0.254 | 0.003 | −0.011 | ns |
| LUFS | +0.257 | 0.003 | −0.083 | ns |
| F0 均值 | −0.037 | ns | −0.014 | ns |

**多元回归** (SyncC ~ RMS + Duration + F0 + Spectral): R² = 0.128, Adjusted R² = 0.108 (n=134)

- Duration 是最强预测因子 (β=−0.247, p=0.008)
- RMS 是显著正向预测因子 (β=+4.531, p=0.020)
- 两者合计解释 12.8% 的 SyncC 方差

**偏相关分析** (控制 RMS + Duration 后):

| 情感 | 原始 SyncC | 残差 | 结论 |
|------|:-----:|:---:|------|
| angry | 8.109 | +1.06 | 情感内容仍有正向效应 |
| calm | 7.582 | +0.72 | 同上 |
| surprised | 7.073 | −0.06 | 效应被 RMS/时长完全解释 |
| disgusted | 6.881 | +0.02 | 同上 |
| happy | 6.625 | +0.08 | 同上 |
| sad | 5.423 | −1.10 | 情感内容仍有负向效应 |
| melancholic | 5.234 | −0.95 | 同上 |
| afraid | 5.426 | −1.18 | 同上 |

**核心发现**: angry 和 calm 的情感优势**不完全**由响度/时长解释 (残差仍为正); sad/afraid/melancholic 的劣势也不完全由响度/时长解释 (残差仍为负)。但 surprised/happy/disgusted 的效应则完全可由物理属性解释。

### Phase B: 干预性实验

**设计**: 79 个 TTS 文件 × 3 归一化条件, 33 次 Ditto+SyncNet 管道运行, 0 失败, ~3.5h。  
**数据**: `data/r4_results/`  
**脚本**: `08_audio_features.py`, `09_observational_analysis.py`, `10_normalize_audio.py`, `11_interventional_analysis.py`

#### Loudness Normalization (LUFS −23dB)

将所有音频归一化到相同的 LUFS 响度, 然后重跑管道:

| 提供者/情感 | 原始 SyncC | loudnorm SyncC | Δ |
|------|:-----:|:-----:|:---:|
| angry | 8.109 | 7.042 | **−1.066** |
| calm | 7.582 | 6.985 | −0.597 |
| fish_audio | 7.570 | 5.812 | **−1.758** ⚡ |
| surprised | 7.073 | 6.047 | −1.026 |
| indextts2 | 6.729 | 6.531 | −0.198 |
| GSV-TTS | 6.362 | 6.591 | **+0.229** ⚡ |
| melancholic | 5.234 | 4.904 | −0.330 |

**关键发现**:
1. **fish_audio 的"优势"几乎完全由高响度驱动** (RMS=0.177, 是最响的 3.5 倍)。归一化后从 7.57 暴跌至 5.81, 低于 indextts2 基线
2. **GSV-TTS 反而受益于响度归一化** (+0.23), 因为它原本最轻 (RMS=0.008), 被低估了
3. **angry 保留最强位置, 但优势缩小**: Δ 从 +1.51 降至 +0.32 (vs indextts2)
4. **情感排名不变**: angry > calm > surprised 的排序在所有归一化条件下保持一致

#### Duration Normalization (对齐到 baseline 时长)

使用相量声码器 (phase vocoder) 将每段音频拉伸/压缩到同一样本的 baseline 时长。

**⚠️ 注意**: durnorm 对所有条件都产生了 **全局性 SyncC 下降** (平均 −1.0), 说明相量声码器引入了可听人工产物, 伤害了唇同步质量。这本身是一个有价值的方法论发现: 时间拉伸不是控制时长混淆的有效手段。

#### 归一化后的排名稳定性

| 排名 | 原始 | loudnorm | durnorm | bothnorm |
|:----:|------|------|------|------|
| 1 | angry 8.11 | angry 7.04 | angry 6.95 | angry 7.06 |
| 2 | calm 7.58 | calm 6.99 | calm 6.63 | calm 7.02 |
| 3 | surprised 7.07 | surprised 6.05 | surprised 6.07 | disgusted 6.11 |
| 4 | disgusted 6.88 | disgusted 6.04 | disgusted 5.98 | surprised 6.01 |

**melancholic 排名上升**: 从第 8 升至第 6 (loudnorm/bothnorm), 因为其极低的原始响度 (RMS=0.01) 被修正后不再拖累 SyncC 分数。

### 综合结论

1. **响度 (RMS/LUFS) 是 SyncC 的显著混淆变量** — 更高的输出音量系统性推高 SyncC 分数
2. **但情感内容仍有独立于物理属性的因果效应** — angry/calm 在控制响度后依然优于 baseline
3. **TTS 提供者之间的比较必须归一化响度** — fish_audio 的"最佳"名次在归一化后消失
4. **时长归一化不可行** — 相量声码器引入的人工产物会伤害唇同步质量; 应使用统计控制 (ANCOVA) 替代物理拉伸
5. **实践建议**: TFG 评测管道应在 Ditto 输入前对齐 LUFS (−23dB), 以消除响度混淆

### 数据文件

| 文件 | 路径 |
|------|------|
| 音频特征 | `data/r4_results/audio_features.csv` (136 行) |
| 归一化音频 | `tmp/r4_normalized/{loudnorm,durnorm,bothnorm}/` (237 文件) |
| r4 运行时 | `runs/r4_{loudnorm,durnorm,bothnorm}_*/` (33 运行目录) |
| 观测分析 | `scripts/09_observational_analysis.py` |
| 干预分析 | `scripts/11_interventional_analysis.py` |

---

## 已知问题

1. `examples/voice_*.wav` 不在 git 仓库中, 通过 `indextts.utils.examples_downloader.ensure_examples_available()` 自动下载, 走 ModelScope
2. 辅助模型 (BigVGAN, semantic_codec, campplus) 首次推理自动下载至 `checkpoints/hf_cache/`, 走 HF 镜像
3. 生成音频时长最短 ~5s (最长的中文句子可到 ~7.5s) — 均满足 SyncNet 4s 最低要求
4. 参考音频必须是 >=15 秒以上才不会被截断——但较短音频 (5-8s) 也可以正常推理 (只是部分长度超 15s 会裁剪)
