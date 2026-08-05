# 01 — TTS-TFG 基线实验（Ditto）

## 问题

TTS 合成语音是否比自然语音产生更好的 Ditto 唇形同步？

## 方法

- TTS: `faster_qwen3` 0.6B ICL（严格复现）/ `dashscope_vc` 云 API（首次尝试）
- TFG: Ditto TRT 离线引擎
- 评估: SyncNet V2（Sync-C / Sync-D）
- 样本: AISHELL-1 子集 9–10 人

## 关键结果

| 运行 | TTS 引擎 | Loudnorm | ΔSync-C | p 值 | 结论 |
|------|---------|:---:|:---:|:---:|:---:|
| `aishell1_ldn` | dashscope_vc | ON | ~0 | ns | ❌ null |
| `aishell1_strict` | faster_qwen3 | OFF | **+1.25** | 0.0005 | ✅ PASS |

## 发现

1. **TTS 优于 natural**，但仅在 loudnorm OFF 条件下
2. 响度归一化（EBU R128）抹平了 TTS 优势——TTS 天然高 ~12dB
3. `trt_online.pkl` 已损坏（产生 0.2s 视频），只能用 `trt.pkl`（离线）

## 状态

✅ **已完成** — 核心结论已得出

## 相关文件

- `docs/reference/ditto.md` — Ditto 部署指南
- `HANDOFF.md` — 严格复现协议
