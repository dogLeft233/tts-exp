---
title: Hallo3 multiset 500 部署 (无卡模式完成)
type: experiment
permalink: tts-exp/experiments/hallo3-multiset-500-部署-无卡模式完成
status: concluded
tags:
- hallo3
- multiset
- deployment
- tfg
---

# Hallo3 multiset 500 部署 (无卡模式完成)

目标: [[跨数据集 TFG 测评（5×50 multiset）]] 复用到第三 TFG 模型 Hallo3 (CogVideoX-5B DiT), 同 250 样本双臂 (natural_raw / tts_raw)。

## 服务器状态 (2026-08-16 已 shutdown, 开卡后继续)

- **实例**: `[redacted-cloud-endpoint]` (root; 密码走环境变量, 见 server-access)
- **预期 GPU**: RTX 6000D (Blackwell sm_120, 48G) → 官方 torch 2.4.0 不可用, 已改 **torch 2.7.1+cu128 + xformers 0.0.30**
- **runbook**: 服务器 `[redacted-local-path]` (冒烟→全量→回传→评分)

## 无卡模式已完成

1. **代码**: `[redacted-local-path]` (官方 main)
2. **环境**: conda `hallo3` (py3.10)。踩坑记录:
   - requirements.txt 的 `pyav==14.0.1` py3.10 无轮子 → 整个 -r 安装中止, 删掉后 167 包装齐
   - requirements 内旧 `nvidia-*-cu12.1` 包会覆盖 torch 2.7.1 的 nccl 2.26 → torch 报 `undefined symbol: ncclCommResume`; 强制重装 torch+cu128 修复
   - 多个 pip 进程并发互踩 → 必须串行
   - `sat` 模块名即 SwissArmyTransformer, import 名是 `sat` 不是包名
   - 无卡时 `import sat` 因 triton 找不到 CUDA driver 失败 — 开卡后自愈
   - OpenCV 收敛 opencv-python-headless==[redacted-ip] (skill 既有方案)
3. **权重**: 52G 全 35 文件 (hallo3 29.1G + cogvideox 11.5G + t5 9.5G + 其余)。fudan 仓库的 buffalo_l.zip 是坏的 (1.5MB 假 zip) → det_10g.onnx 从 `deepinsight/insightface releases v0.7` 补, landmarker.task 从 googleapis 补
4. **数据** `[redacted-local-path]`:
   - 250 参考帧 `frames/`: 本地 `38_prepare_frames.py` haar 重新生成。原实验帧在旧远端已丢
   - **坑**: 38 脚本 sample_id = len(prepared)+1, 拒绝样本后编号全体前移错位 → 重写为按 manifest 位置定 id
   - LRS3 (224px) 全部过不了 20000px² 脸面积门 → 3000px² 回退救回; grid 27 条同回退; vfhq id98 Haar 检不到脸 → forced-mid 帧。manifest 有 detector 字段记录
   - natural 音频 = `data/data/audio/{1..250}.wav` (16k), TTS = `data/tts_audio_250/` (faster_qwen3 seed42), 转写在 `data/tts_audio_250/transcript.json`
   - input txt: `text@@image@@audio` 格式, hallo3 要求
5. **Runner**: `run_multiset.sh` 断点续跑 (按最终 mp4 存在与否过滤 todo), 输出收集到 `hallo3_outputs/{arm}_raw/{id}.mp4`, 固定 seed 42

## GPU 日待办
## GPU 日结果 (2026-08-16, 中途止损关机)

- GPU: RTX 6000D 实配 **85.6G** 显存, 推理峰值 ~59G
- 实测吞吐: **~16min/样本** (固定开销 ~9min 人脸掩码/嵌入 + 5.5min/s × 音频时长); 5.8s 样本 30min
- 500 全量 200h+ 不可行 → 用户选 5/数据集 (25×2 臂)
- 进度: natural 11 + tts 2 完成, **id101 处 IndexError 崩溃** (`cannot do a non-empty take from an empty axes`), runner 设计缺陷 = 只在整批成功后收集 → natural 首 batch 25 条成品丢失 (work 目录里只留下 11 个)
- 没钱止损 → 回传 13 视频 + run.log → shutdown
- 本地: `results/hallo3/{natural_raw(11),tts_raw(2),04_eval/run.log}`; 可用配对 = id15/id26 双臂 (n=2, 仅 smoke 级)

## 遗留问题 (下次开卡前必修)

1. runner 逐样本收集 (每 chunk 落盘即 cp), 不要 batch 末统一收集 — 本次丢 14 条成品就是这 bug
2. id101 IndexError 根因未查 (疑 LRS3 224px 帧或特定音频触发空轴); 复现: input 里单跑 101
3. 冒烟链路修了 3 个坑: setuptools 83 删了 pkg_resources → 降 <81; 缺 ffmpeg → apt 装 4.4.2; 这两个都是无卡模式测不出的
4. 模型加载时 "Missing keys: conditioner.embedders.0..." 警告 = T5 权重没进 checkpoint, 从 t5-v1_1-xxl 目录单独加载, 属正常 (冒烟出片正常)

1. 五步 GPU 检查 → 冒烟 input_smoke.txt (样本 1)
2. `run_multiset.sh both` 500 样本 (~60s/样本估 8.3h; 6000D 48G 预计 OOM 风险低)
3. rsync 回本地 `results/hallo3/{natural_raw,tts_raw}/`
4. SyncNet V2 评分 (min_track=25, 同 multiset250 协议), 对照 Ditto +0.980 / LeapTalk +1.256

## Observations

- [task] 开卡后执行 GPU_DAY_RUNBOOK.md
- [risk] hallo3 训练数据纯英文, multiset 音频中英德西韩混合 — 中文样本唇形质量存疑 (与 LeapTalk/Ditto 不可比处)
- [risk] GRID 3s 短音频 vs hallo3 49 帧输出, 尾部 padding 影响评分
- [decision] torch 2.7.1+cu128 替代官方 2.4.0 (Blackwell 兼容), 依赖差异已验证仅 nvidia 系
- [status] concluded
- [progress] 5×5 子集 25 对生成完成，SyncNet V2 完成 24 对可评分配对
- [result] Hallo3 multiset 5×5：24 对完整配对，Natural Sync-C 5.162、TTS 5.702、ΔSync-C +0.540（95% CI −0.098 至 +1.177，双侧配对 t p=0.0931）；16/24 对正向
- [result] Hallo3 配对 Sync-D：Natural 8.983、TTS 9.337、ΔSync-D +0.354（p=0.1254）；留档 natural_raw_fullrun 62/62 成功，Sync-C 均值 5.491
- [boundary] tts_raw ID114 无 SyncNet 人脸轨迹而失败；每数据集仅 4–5 对，结果属探索性，不能据此断言 Hallo3 上 TTS 优势稳定成立
- [report] 原始评分 results/hallo3/04_eval/hallo3_local_scores.json；统计 results/hallo3/04_eval/hallo3_analysis.json

## Relations

- continues [[跨数据集 TFG 测评（5×50 multiset）]]
- relates_to [[tts-exp]]
## 最终完成 (2026-08-21)

**25/25 natural + 25/25 tts 全部落本地 `results/hallo3/{natural_raw,tts_raw}/`** (5×5 multiset 子集, ids 15,26,32,42,49,51,64,66,70,97,101,112,114,136,147,155,170,174,175,194,204,205,206,207,208)。附赠 `natural_raw_fullrun/` 62 条全量 natural 臂视频 (199 上 input 忘换子集的意外产出, 留档)。未评分 (用户指示)。

### 101 IndexError 根因与修复
talkvid id101 (`videovideo0AvHEwYHsCo-scene61_scene13`, 1920×1080) 全片人脸仅 ~70k px² → hallo3 内部 insightface 置信度不足 → numpy "non-empty take from empty axes" 崩溃, 且首样本即炸会堵死整批。修复: 人脸中心 3:2 裁剪 (脸宽占 28%), 双臂均过。帧与 manifest (detector=haar-facecrop-3x2) 已本地同步。

### 199 服务器部署坑 (区别于 seetacloud)
- conda 26 需先 `conda tos accept` 两个 channel
- nvidia pip 包名混战: 新版无后缀 `nvidia-cublas` (cu13) 与 `nvidia-*-cu12` 共存互踩 → 全部卸载后单从 cu128 index 装 torch 全家
- 199 → github/hf-mirror 均近死 (5KB/s~1MB/s); hf 大权重靠 **seetacloud dd 分块中转** (本机两跳管道, ~1.2MB/s 单流 × 6 流), md5 双侧校验
- 服务器残留 (下次开卡可复用): /data/wangjiajun/{miniconda3,hallo3(代码+52G权重),hallo3_data,hallo3_outputs}; WEIGHTS_OK stamp

## SyncNet V2 评分（September 25, 2026）

对本地 5×5 子集双臂 results/hallo3/{natural_raw,tts_raw}/ 的 50 个 MP4，以及单臂留档 results/hallo3/natural_raw_fullrun/ 的 62 个 MP4，按 multiset250 协议运行 run_pipeline.py + run_syncnet.py（SyncNet V2，min_track=25，视频内嵌音轨）。评分程序为 scripts/04_eval.py:evaluate_video；逐视频分数、原视频 SHA-256 和模型 SHA-256 在 results/hallo3/04_eval/hallo3_local_scores.json，配对分析在 results/hallo3/04_eval/hallo3_analysis.json。

正式双臂 49/50 成功；tts_raw/114.mp4 可解码，但 SyncNet 提取到 0 条人脸轨迹，无法给分。完整配对 24/25，排除 ID 114 的自然臂分数后统计：

| 指标 | 配对 n | Natural | TTS | Δ=TTS−Natural | 95% 配对 t CI | 双侧配对 t p |
|---|---:|---:|---:|---:|---:|---:|
| Sync-C ↑ | 24 | 5.162 | 5.702 | +0.540 | [−0.098, +1.177] | 0.0931 |
| Sync-D ↓ | 24 | 8.983 | 9.337 | +0.354 | [−0.107, +0.815] | 0.1254 |

Sync-C 有 16/24 对为正，点估计偏向 TTS，但区间跨 0，不能据此断言 Hallo3 上已有稳定优势；Sync-D 方向反而较差，也未达统计显著。每数据集只有 4–5 对，分组结果仅作描述：

| 数据集 | n | ΔSync-C | 正向对数 |
|---|---:|---:|---:|
| MEAD | 5 | −0.400 | 2 |
| VFHQ | 5 | +1.297 | 4 |
| TalkVid | 4 | +0.779 | 3 |
| LRS3 | 5 | −0.249 | 2 |
| GRID | 5 | +1.320 | 5 |

另有 62/62 条 natural_raw_fullrun/ 视频成功评分：Natural Sync-C 均值 5.491、Sync-D 均值 9.256；这是无 TTS 配对的留档批次，不纳入上述效应估计。其中命名为 natural_15 等的 10 条与正式自然臂同 ID 视频哈希不同，作为独立输出留档。
