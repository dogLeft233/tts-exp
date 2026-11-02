---
title: peavs
type: guide
permalink: tts-exp/docs/deployment/peavs
status: active
verified: '2026-10-27'
---

# peavs

✅ 2026-10-26 已在 Ubuntu V100 实际跑通完整视频评分。服务器 `wjj@192.168.50.2`，部署根 `/home/wjj/av-eval-20261026`，专用解释器 `/home/wjj/.venvs/peavs/bin/python`。

| Date | Change | Requested by |
|---|---|---|
| 2026-10-26 | 部署官方 PEAVS，验证两官方样例、一个项目视频和重复评分 | user |
| 2026-10-27 | 官方推理审计，更正默认批量/精度并增加版本化入口；保留历史调用 | user |

## 2026-10-27 官方默认设置审计（当前入口）

当前复现入口为 `/home/wjj/av-eval-20261026/score_peavs_official64_20261027.py`，默认 `--metric_batch_size 64`，与发布run_PEAVS.sh/metric.py默认一致。下方旧score_peavs.py实际固定batch1，只保留历史重放，不应称作完全默认结果。

发布dataset.py把视频有效长度写成音频长度；部分音视频特征长度不等时，同一视频受同批其他样本影响，最大单项batch1/64差0.811880。默认64仍保留此上游问题，不是可靠性修复。原生20来源/60cloud配对默认64重算Δ=-0.408085，TTS较高6/60、与Sync-C冲突42/60；原batch1Δ=-0.403537。时移下降从26/40改为25/40，仍未过本项目32/40门。

原源码/全部仓库模型/VGGish发布权重验证通过，CuPy兼容核误差2.38e-7、失败样本重新提取逐值复现。官方AVS基准排除人脸，不把该分数当中文唇同步真值。真实新入口双视频smoke通过。详见 [[PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27]]。

## 来源与环境

官方 [amazon-science/avgen-eval-toolkit](https://github.com/amazon-science/avgen-eval-toolkit)，源码版本 `9cf014ba2c02551f4494cae685607ca3b5328f43`。部署在 `peavs-source/`，按 Git tree 的 blob SHA-1 检查下载的源文件及仓库权重；没有改写官方源文件。VGGish 权重来自官方代码指定的 harritaylor/torchvggish release，SHA-256 `10086976245803799d9194e9a73d9b6c1549c71d1b80106f5cade5608a561f4b`。

专用 venv 通过只读共享路径复用既有 cogenav/shared PyTorch 运行时及其 Python 3.10 基础依赖；这些被引用的原环境未安装、卸载或改写包。新增包在 peavs venv 内。实际版本：Torch 2.5.1+cu121、Torchvision 0.20.1+cu121、NumPy 1.26.4、SciPy 1.15.2、CuPy 12.3.0、OpenCV 4.9.0、OmegaConf 2.3.0、resampy 0.4.3、系统 FFmpeg 6.1.1。该 venv 依赖原共享运行时路径继续存在。

## 使用

```bash
/home/wjj/.venvs/peavs/bin/python /home/wjj/av-eval-20261026/score_peavs.py \
  /absolute/path/clip.mp4 --output /absolute/path/peavs_scores.json
```

支持多个 MP4；输入必须带音轨、文件 stem 唯一。每次创建独立 work 目录，依次运行官方 I3D RGB+PWC flow、VGGish 和评分头。保留特征和原 CSV；JSON 输出不主动四舍五入。评分 1–5，越高越好；这不是 Sync-C。

外置 `peavs_extract_bootstrap.py` 将废弃的 CuPy `compile_with_cache` API 换成 RawModule，保持相同 PWC CUDA kernel 与当前 Torch stream。入口为子进程指定匹配共享 Torch 的 NVIDIA 库目录，避免 MFA 基础环境中另一版 CUDA 库造成符号冲突。官方 metric 的自动删除特征行为仅在内存执行副本中关闭，原源码不变。

## 实际验证

| 输入 | PEAVS |
|---|---:|
| SyncReward 官方 case1-before.mp4 | 1.3898629 |
| SyncReward 官方 case1-after.mp4 | 2.6804836 |
| 项目 BAC009S0913W0121/R.mp4 | 2.8051605 |

官方 before/after 是不同生成样例，不据其排序声称受控延迟因果效果。两官方样例重复独立提取/评分得到完全相同数值；固定库路径后复跑三条仍完全一致。项目样例为 `/home/wjj/tts-audio/tts-exp/runs/historical_aishell16_current_pipeline_20261023_attempt02/video/BAC009S0913W0121/R.mp4`，5.44 秒，H264+AAC，只读使用。

主要证据：部署根 `outputs/peavs_demo.json`、`outputs/peavs_demo_repeat.json`、`outputs/peavs_project_smoke.json`、`outputs/peavs_final_three.json` 及各自 work 目录。这里验证工程可评分，不证明 PEAVS 在本项目中文唇形任务上的感知效度。

## Relations
- relates_to [[syncreward]]
- relates_to [[server-access]]
