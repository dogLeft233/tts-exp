# 08-参考帧选择对 Ditto 同步质量的影响

**日期**: 2026-08-02  
**结论**: 视频第一帧作为参考帧显著优于质量筛选抽帧（平均 ΔSync-C +1.42，3/3 样本一致）。决策记录见 [ADR-002](../adr/002-reference-frame-first-frame.md)。

## 1. 背景

多数据集 TFG 实验需要为每样本选取参考帧。初版 `38_prepare_frames.py` 从视频均匀采样 12 帧，按 yaw/人脸大小/清晰度加权打分选最优帧。本实验在真实 Ditto（TRT，4080 SUPER）上检验该策略的生成质量。

## 2. 方法

- 数据：HDTF 3 样本（RD_Radio47 / WRA_FredUpton / WDA_AdamSchiff），视频取自 NAS，配对演讲音频 16k mono
- 参考帧：A) 质量筛选抽帧（帧号 1132/1595/884，视频中段）；B) 第一帧（帧 0）
- 推理：Ditto TRT `v0.4_hubert_cfg_trt.pkl`，配对音频全时长驱动
- 评估：SyncNet V2（`run_pipeline.py --min_track 100` + `run_syncnet.py`），AV offset 均 1 帧
- 长视频（275s）在 SyncNet 音频提取阶段卡死，统一截取前 90s 评分

## 3. 结果

### 3.1 参考帧 A/B（同一音频，仅参考帧不同）

| 样本 | 抽帧 Sync-C | 第一帧 Sync-C | ΔSync-C | 抽帧 Sync-D | 第一帧 Sync-D |
|------|:---:|:---:|:---:|:---:|:---:|
| RD_Radio47 | 3.475 | **6.408** | **+2.93** | 11.804 | 8.361 |
| WRA_FredUpton | 3.567 | **4.505** | +0.94 | 11.553 | 9.535 |
| WDA_AdamSchiff | 4.830 | **5.225** | +0.40 | 9.256 | 8.876 |
| **平均** | **3.957** | **5.379** | **+1.42** | 10.871 | 8.924 |

### 3.2 与论文/项目历史对比

| 配置 | Sync-C | 备注 |
|------|:---:|------|
| 论文 Ditto HDTF100 | 8.939 | 100 样本平均，第一帧参考 |
| 论文 Ditto Talk9 | 8.069-8.111 | |
| 本项目历史 HDTF（13 说话人，6s 裁剪） | 6.451 | natural 条件 |
| **本实验第一帧（3 样本平均）** | **5.379** | 小样本 |
| 本实验抽帧（3 样本平均） | 3.957 | 中段帧 |

## 4. 机制分析

Ditto 以参考帧提取初始运动 `m_ref`、情绪 `e`、眼睛状态 `eye` 作为生成初始条件（论文 §3.2.1，ICS/ECS 条件信号）。质量筛选抽帧选中视频中段"说话状态"帧：

- 嘴张开（RD 抽帧 mouth=0.161、WRA 0.155）
- 接近闭眼（WDA 抽帧 EAR=0.007）

这些非中性状态被当作初始条件，污染整个生成序列的同步质量。第一帧（演讲开始）通常为中性状态，是论文协议（SadTalker/Ditto 均用第一帧）隐含选择的理由。

## 5. 实现变更

- `scripts/38_prepare_frames.py`：新增 `--frame-strategy first|quality`，默认 `first`（最早合格帧，不合格回退质量筛选）；采样起点改为 0.0
- `scripts/frame_quality.py` / `frame_detectors.py`：新增 `first_acceptable_frame`；mediapipe detector（Tasks API，复用 Ditto `face_landmarker.task`）
- 测试：+4（第一帧优先/回退/默认行为），全量 314 通过（3 个失败为遗留 `librispeech_phn_parsing` 问题）

## 6. 局限

- n=3 小样本；效应量（+1.42）个体差异大（+0.40 ~ +2.93）
- 抽帧质量打分的独立贡献未单独评估（作为门控/fallback 保留）
- 论文 HDTF100 对比为 100 样本平均，协议（音频预处理/SyncNet 实现细节）存在差异
