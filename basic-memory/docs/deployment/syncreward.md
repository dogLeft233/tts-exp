---
title: syncreward
type: guide
permalink: tts-exp/docs/deployment/syncreward
status: active
verified: '2026-10-27'
---

# syncreward

✅ 2026-10-26 已在 Ubuntu Tesla V100-SXM2-16GB 实际完成三条视频评分和独立进程重复验证。服务器 `wjj@192.168.50.2`，部署根 `/home/wjj/av-eval-20261026`，专用解释器 `/home/wjj/.venvs/syncreward/bin/python`。

| Date | Change | Requested by |
|---|---|---|
| 2026-10-26 | 部署官方 Stage-2；增加 V100 FP32 兼容入口并实际验证 | user |
| 2026-10-27 | 官方推理审计，更正默认批量/精度并增加版本化入口；保留历史调用 | user |

## 2026-10-27 BF16更正（当前入口）

当前入口为 `/home/wjj/av-eval-20261026/score_syncreward_official_20261027.py`，默认 `--precision bf16` 恢复官方CUDA autocast；`--precision float32` 显式保留历史重放。encode_batch_size1是官方文档支持的低显存参数（默认18），窗口/预处理/权重/公式不变。下方原score_syncreward_v100.py与原分数保留为历史FP32版本，不冒称官方BF16。

更正：V100没有原生BF16，并不等于当前Torch不能执行BF16。Torch2.6含仿真支持为true，完整模型实测成功。已完成20原生自然+40时移+3官方demo派生控制的BF16评分，下降14/40、持平7/40、上升19/40；旧FP32下降16/40，二者均未过本项目32/40门。精度最大单项分差0.065011，不可一概忽略。旧cloudTTS原生比例仍属于FP32，未重算为BF16。

原始score.main的FP32对照与旧包装器三代表项逐值相同，低显存分段batch1/2差<2.4e-7；48k重采样后移位保留。严格全Stage2加载与配置初始化优化保留，软件BF16不声称H20硬件bit-exact。新入口真实smoke1.453125（precision=bf16）与控制匹配、exit0。详见 [[PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27]]。

## 来源与环境

官方 [WeChatCV/SyncReward](https://github.com/WeChatCV/SyncReward)，源码版本 `f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615`，部署在 `SyncReward/`，原源码未改写。[Guan123/SyncReward](https://huggingface.co/Guan123/SyncReward) 权重版本 `04bc7701da8d8466c7b3aa2dec22a2a5051e6f0a`，Stage-2 文件 6,635,671,828 字节，实际 SHA-256 与官方 SHA256SUMS 一致：`6c09263ac41d59baf78a181f17e4116128a337d70a03637886de3734450a32a2`。

专用 venv 通过 `.pth` 只读引用现有 `/home/wjj/.venvs/qwen3/lib/python3.12/site-packages` 的 PyTorch 基础依赖，新增或覆盖版本安装在 syncreward venv 内，原 qwen3 环境未安装、卸载或改写包。实际版本：Python 3.12.3、Torch/Torchaudio 2.6.0+cu124、TorchCodec 0.2.1、Transformers 5.1.0、timm 1.0.26、NumPy 1.26.4、SciPy 1.13.1、huggingface_hub 1.33.0、safetensors 0.8.0、系统 FFmpeg 6.1.1。复用运行时路径需保留。

## 使用

```bash
/home/wjj/.venvs/syncreward/bin/python /home/wjj/av-eval-20261026/score_syncreward_v100.py \
  /absolute/path/clip.mp4 --output /absolute/path/syncreward_scores.jsonl
```

支持多个视频，每条必须带音轨且足够长。默认 `encode_batch_size=1`，全 FP32，18 个 0.5 秒片段、步长 0.25 秒、共同 center crop，总窗口 4.75 秒。保留官方解码、预处理、切窗、编码器、奖励头及计算公式。长视频只评分中心窗口，短视频依官方规则补帧/补零，不将此窗口分数误作全视频逐时评价。输出 JSONL 的 reward 为完整精度，范围 0–2，越高越同步，不能与 PEAVS 1–5 或 Sync-C 直接比较。

## V100 适配

官方 score.py 在 CUDA 上固定使用 BF16 autocast，V100 不支持原生 BF16；外置入口禁用 autocast，实际使用 FP32。它不声称逐值复现 H20 BF16 官方基准。

Stage-2 已含评分所需的全部编码器参数。入口用 facebook/pe-av-large 的官方配置构造同一模型结构，保留与官方 loader 相同的音频/视频 tower 和 head，再 `strict=True` 加载 Stage-2；缺失/多余键或形状不符立即失败。所有保留参数均由 Stage-2 覆盖，不下载将被覆盖的约 9GB backbone 权重；backbone 的配置和 processor 文件仍完整下载并在本地加载。

TimmWrapperConfig 会把视觉头解析为默认 2 类；入口从官方原始 JSON 的 `audio_video_config.video_config.vision_config.num_classes=1024` 恢复对应 `num_labels`，与 Stage-2 `head.weight [1024,1024]`/`bias [1024]` 严格匹配。SciPy 额外固定为 1.13.1，以兼容官方 NumPy 1.26.4；只覆盖专用环境。

## 实际验证

官方 `python -m tests.test_smoke` 全通过：Stage-1 GradCache 梯度检查、Stage-2、统计指标、safetensors 精确 round-trip。官方样例解码实际通过。

| 输入 | SyncReward FP32 |
|---|---:|
| 官方 case1-before.mp4 | 0.06744802743196487 |
| 官方 case1-after.mp4 | 1.7917524576187134 |
| 项目 BAC009S0913W0121/R.mp4 | 1.2807059288024902 |

after 样例在独立新进程完整重载后重复为 1.7917524576187134，差值 0。每条 forward 约 8.5–11 秒，另有约一分多钟 CPU 模型构造/加载。Torch 峰值分配显存 7,260,209,152 字节（约 6.762GiB），V100 正常执行。

证据：部署根 `outputs/syncreward_demo_project.jsonl`、`outputs/syncreward_repeat.jsonl`、`outputs/deployment_validation.json`、`syncreward-smoke-retry.log`、`syncreward-decode.log`、`syncreward-demo-project.log` 和 `syncreward-repeat.log`。加载日志有上游 Transformers 的 docstring `[ERROR]` 文案，该提示不阻断实际加载和评分；三条输出均 finite 且在范围内。before/after 为不同生成样例，评分差不作为受控因果或本项目人类感知效度证据。

## Relations
- relates_to [[peavs]]
- relates_to [[server-access]]
