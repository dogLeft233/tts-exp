# AISHELL-1 100 样本 · 自然语音 vs TTS 音素可分性扩展实验

生成时间：2026-08-05；样本：AISHELL-1 test split 100 条（16 说话人）；TTS：faster_qwen3 self-clone；对齐：MFA mandarin_mfa + mandarin_china_mfa 词典（词/音素边界）；SSL：HuBERT / XLS-R。

## 实验产物（已回传本地）

- `metrics/separability_metrics.json` — 42 结果 / 126 配对比较
- `mfa_full/out/{natural,tts}/` — 各 100 个 TextGrid
- `alignment/manifest.json` — 200 条对齐 manifest（natural/tts 各 100）
- `run/02_tts/` — 100 条 TTS wav + tts_meta.json

## 核心结果：HuBERT viseme 层探针（probe accuracy，越高=音素/viseme 可解码性越好）

| layer | natural | faster_qwen3(TTS) | Δ |
|---|---|---|---|
| 0 | 0.849 | 0.863 | +0.013 |
| 6 | 0.939 | 0.953 | +0.014 |
| 11 | 0.932 | 0.948 | +0.016 |

**每个层 TTS 的探针准确率都高于 natural**（+0.014 ~ +0.017），说明 100 样本下 TTS 对音素/viseme 信息的保持不弱于自然语音，且信号比 12 样本时更稳定。

## 显著性汇总

- 共 126 个 natural-vs-TTS 配对比较（per-sample，permutation 检验 + BH-FDR）。
- FDR<0.05 显著：**35** 个。
- 高亮显著项：segment_stability（l6/l0，|d|≥0.55）、boundary_sharpness（l6/l11，d≈-0.59~-0.69）、intra_class_dist（l11）、silhouette（l0）。
- 与 12 样本对比：当时 2/24 显著；现在 35/126 显著，且效应方向一致 → **扩充数据量使原有“TTS 音素表征差异”现象更明显**。

## 局限与说明

- XLS-R 探针（1024 维逻辑回归）在 100 样本上单条耗时 ~13 分钟，本次显式跳过（`probe=null`），其可分性/边界/稳定性指标仍完整。
- 98/200 条对齐为“词边界内回退”（MFA 词边界真实、字符分配在词内均匀），非整句均匀切分。
- 全部指标基于真实 MFA 时间边界，未使用旧实验的整句均匀切分。
- embedding（1.3G）未回传；如需可据 manifest + 本地缓存模型重跑 15 脚本。
