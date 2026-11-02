---
title: funasr
type: note
permalink: tts-exp/docs/deployment/funasr
status: in-use
updated: '2026-10-24'
tags:
- deployment
- funasr
- timestamp
---

# funasr

✅ 已部署并完成 CPU 最小验证（October 24, 2026）。用途是为已有 Qwen 0.6B／1.7B 中文音频的对应词段定位准备工具；未开展全批对齐、训练、新 TTS、口型生成或 SyncNet 评分。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial deployment and eight-file smoke | October 24, 2026 | user deployment request |

## 环境与固定模型

- [type] service
- [status] in-use
- [location] Ubuntu `/home/wjj/.venvs/funasr-cpu-20261024`；正式仓库 `/home/wjj/tts-audio/tts-exp`。
- [spec] CPython 3.11.13；FunASR 1.4.16；torch／torchaudio 2.5.1+cpu；transformers 4.51.3；huggingface-hub 0.36.2。完整依赖在 run 的 `requirements.lock.txt`，85 包 `uv pip check` 通过。
- [constraint] CPU-only、两线程；不修改现有 Qwen／Wav2Lip／SyncNet／MFA 环境。探测 V100 占用 322MiB，本次未用 GPU，所以无新增显存峰值。
- [constraint] 远端仅系统本地盘可运行，开始剩余18GB，部署后剩16GB；模型在远端 `/home/wjj/tts-audio/model_cache/funasr_20261024`，本机 E: 仅保存源码和小型凭据，不保存权重。

| 模型 | 固定 HF revision | model.pt SHA256 |
|:--|:--|:--|
| funasr/paraformer-zh | d7811ee3ac581fbcfdeb37c98c6ba674028433dc | 5bba782a5e9196166233b9ab12ba04cadff9ef9212b4ff6153ed9290ff679025 |
| funasr/fa-zh | d7701644d6e336f093501241beca010519c0986f | f34ede558af831fb504206b25f1c2f27ca2f77753c26c4dd38d03323153b6f73 |

两权重分别880502012／158469618字节，SHA与固定revision的Git-LFS pointer匹配；附属文件SHA详见 `provenance.json`。实际使用本地明确的模型目录，避免 `paraformer-zh` 别名在不同hub映射到不同模型。

## 官方接口依据

[FunASR 官方 README](https://github.com/modelscope/FunASR)区分具体checkpoint与任务能力；[fa-zh 固定模型卡](https://huggingface.co/funasr/fa-zh/blob/d7701644d6e336f093501241beca010519c0986f/README.md)给出 `generate(input=(audio,text), data_type=("sound","text"))`。本次同时运行 Paraformer 中文识别与 fa-zh 给定文本的字级 timestamp prediction，没有把其它模型的ASR输出能力当作通用时间戳保证。实际安装的 `MonotonicAligner`、`BiCifParaformer` 和 `timestamp_tools` 源码已复制到run便于核对，时间单位为毫秒。

## 已执行 smoke 与边界

凭据目录（本机与远端repo相同相对路径）：`runs/funasr_deployment_20261024_attempt01/`。

输入来自试听manifest的4来源×2规模、draw1，共8条16kHz mono PCM16 RMS归一化已有音频。S0002／S0007为旧批，S0013／S0014为新批；原始canonical WAV运行前后SHA均一致。文本来自旧批 `qwen17_level_comparison10_20261023_attempt02/protocol.json` 与新批 `qwen_prediction_mechanism_20261023_attempt02/C_protocol.json`，原转写、标点／空格去除后的字符映射和ASR差异操作全部保存。

- 进程exit0；CPU模型加载5.802秒，总8.575秒（不含Python导入）；单文件ASR0.196–0.420秒，fa0.086–0.156秒。
- ASR去空格／标点后8/8与给定文本一致，此次未观察到漏字／重复字；这不等于人工听辨确认。
- fa-zh 8/8非空、字符数绑定、单调性和音频范围检查通过；对应15／13／15／14字（各两规模）。
- Paraformer原生ASR时间戳只有2/8通过严格范围检查，6/8存在端点越界，原样标记、未裁剪。例：部分起点为−90ms，部分终点超出音频约80ms；内部顺序未发现倒置。后续定位采用fa输出并继续人工核对，不能直接使用这6条原生ASR边界。
- 初版validator把负首点同时记成顺序异常；已修正为 `previous=None`，仅用既有原始JSON重验，结果在 `validation_attempt02.json`。冻结 `smoke_attempt01.json` 与原runner保留，不覆盖历史输出。当前源码为 `scripts/funasr_deployment_smoke.py`；run内可复用修正版为 `funasr_deployment_smoke_validated.py`。
- 2个有意义单测通过，覆盖字符映射、token/边界绑定、范围、重叠及负首点分类。
- 这些时间戳是模型估计，不是人工准确性证据；给定文本对齐本身不能独立验证语音确实包含该文本。此次仅证明部署与最小接口可运行。

| 来源 | 两规模识别文本 |
|:--|:--|
| S0002 | 而对楼市成交抑制作用最大的限购 |
| S0007 | 对于长袖善舞的上市公司而言 |
| S0013 | 调控也逐渐淡出了政府的决议内容 |
| S0014 | 折合楼面价三点八万元每平方米 |

## 复用命令

在远端执行；output必须使用新文件名，runner拒绝覆盖，最多8条输入：

```bash
RUN=/home/wjj/tts-audio/tts-exp/runs/funasr_deployment_20261024_attempt01
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
 /home/wjj/.venvs/funasr-cpu-20261024/bin/python \
 "$RUN/funasr_deployment_smoke_validated.py" \
 --manifest "$RUN/manifest.json" \
 --model-root /home/wjj/tts-audio/model_cache/funasr_20261024 \
 --output "$RUN/smoke_replay_$(date -u +%Y%m%dT%H%M%SZ).json"
```

原始smoke、逐字边界与ASR文本：`smoke_attempt01.json`；最终边界结构审计：`validation_attempt02.json`；模型／包／音频SHA：`provenance.json`；原始进程stdout和exit：`smoke_attempt01.log`、`smoke_attempt01.exit`。完整包锁和安装／兼容修复日志都在同run。

## 部署修复记录

首次PyPI大依赖下载长时间重试，保留 `install_funasr.log` 后只停止本次uv进程，改清华PyPI镜像、600秒HTTP超时和4并发后安装成功（`install_funasr_retry02.log`）。默认transformers5.20.2要求Torch≥2.6，导入提示禁用Torch；在新venv固定4.51.3后导入与pip check均通过。HF直连不可用，固定revision经hf-mirror下载；Paraformer初始curl为避免600秒上限改断点续传，保留原日志。fa的seg_dict慢，核对两模型官方相同Git blob `fb19e07f2870b09b08226a42d7d5d6d26a3f3a16` 后从已完整下载的Paraformer复制，没有更改文件内容。未停止或修改其它任务。

## Relations

- relates_to [[研究交接总览]]
