---
title: PC-Talk 192服务器部署
type: note
permalink: tts-exp/docs/deployment/pc-talk
tags:
- deployment
- '192'
- pc-talk
- i2v
---

# PC-Talk 192服务器部署

✅ 2026-10-31 完成官方 PC-Talk 单张图片＋真实中文音频的服务器 GPU I2V 验收。仅为部署验收，未运行 SyncNet 或其它质量评分。

## 来源与安装位置

- 官方源码：https://github.com/BQ-Wang0511/PC-Talk ，commit `a0392744cbac857d96db1b99ceab70cf7b5f6c83`；模型源码未修改，editable 安装产生 `pc_talk.egg-info/`。
- 作者链接的权重：https://huggingface.co/doubi-killer/PC-Talk ，revision `e6220816118bd8714ff57dc83e5906122b4e9168`。从 hf-mirror 的固定 revision 下载，12 个权重逐一通过官方 `SHA256SUMS`，约 920 MB。
- 主机 `wangjiajun@10.108.16.192:22`；源码 `/home/data2/wangjiajun/PC-Talk`，checkpoint 在其 `pctalk/checkpoints/`。
- 部署根 `/home/data2/wangjiajun/PCTalkDeploy`；独立环境 `env/`，日志 `logs/`，缓存 `cache/`，临时 `tmp/`，Conda 缓存 `conda-pkgs/`，输出 `outputs/`，验收 `delivery/`。
- Python 3.10.22、torch 2.4.1+cu121、cuDNN 9.1、numpy 1.26.4、opencv-python 4.11.0.86、onnxruntime-gpu 1.21.0。完整版本见 `delivery/environment.freeze.txt`；`pip check` 无冲突。

## 服务器规则与隔离

本次先通过飞书 MCP 应用身份读取当次《【192服务器】1048服务器手册》实际正文，再连接。共享 `/home/data/public/model` 有限只读检查未发现同名 PC-Talk／LivePortrait 模型；代码、权重、环境、缓存、输出均在个人 data2 区。共用 `/home/user/anaconda3/bin/conda`，未修改共享 base、系统 CUDA／驱动或其他人的进程；长任务 nohup 后台执行。

原个人 `TalkModels/envs/infinitetalk` 环境为隔离复制来源。初次误将 venv 作为 Conda clone 来源已停止；第二次 offline clone 复制完成 pip 文件后因个人缓存缺少基础 Conda 包退出。保留失败日志，以原环境 `conda list --explicit` 的精确版本清单在新 prefix 补齐基础包，没有重新复制大依赖。随后仅在新环境移除无关且 wheel 平台元数据异常的 decord，收敛 OpenCV 为单发行包并安装 PC-Talk。原环境 torch/numpy/cv2 版本及导入路径的 before/after JSON 逐字相同。

运行固定单 GPU UUID `GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37`（A6000），运行前重查空余；设置 PCI_BUS_ID、4 CPU 线程、4 CPU affinity，以及 PyTorch allocator 上限 45%。这是本次保守设置，不是服务器新增配额；ONNX 显存另以 NVIDIA PID 采样计入。结束后进程退出，该 GPU 回到 2 MiB。

## 可复用命令

先 `nvidia-smi` 检查空闲 GPU，再通过 UUID 选择。`infer.sh` 设置个人缓存、线程与 allocator 限制，参数传给官方 CLI；换图时须重新检查人脸索引。

```bash
R=/home/data2/wangjiajun/PCTalkDeploy
CUDA_VISIBLE_DEVICES=GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37 \
nohup bash "$R/infer.sh" \
  --source /home/data2/wangjiajun/av-evaluation/datasets/face/3.png \
  --audio /home/data2/wangjiajun/av-evaluation/datasets/chinese20/audio/BAC009S0009W0121_N_0.wav \
  --face-index 1 --person-id 192 --lip-scale 0.6 \
  --emotion happy:all:0.5 --output-dir "$R/outputs/custom" \
  > "$R/logs/custom.log" 2>&1 < /dev/null &
```

`infer.sh --help` 与 shell 语法检查通过。复跑固定验收可用 `run_i2v.sh`，会重新生成视频并覆盖当前验收 JSON；已有证据需先封存。官方使用随机数字文件名。

## GPU与媒体验收

正式成片：`outputs/attempt02/8273583714.mp4`，SHA256 `a7043e05f50d8ef82bcb7407ec88249d4c9b7eaa16418a3cd09b3e7fbf0a1d5b`。

| 项目 | 实测 |
|---|---|
| 输入 | 3.png 人像＋BAC009S0009W0121_N_0.wav；原音频3.121秒、16kHz mono，输入SHA见inputs.sha256 |
| 画面 | H264、512×512、25fps、73帧、视频流2.920秒 |
| 音轨 | AAC、16kHz mono，2.944秒；解码RMS 0.038745，非静音 |
| 动态 | 连续帧平均绝对像素差均值0.443、最大1.183（0–255尺度）；首尾帧已目视，主体嘴部变化 |
| GPU真实前向 | AudioVisualEncoder、LAC／EMC相关网络及LivePortrait模块记录到cuda:0张量；三个ONNX会话均成功启用CUDAExecutionProvider |
| 耗时 | 9.289秒，含导入、权重加载、生成和保存；首轮冷缓存21.132秒选错人脸，不能当主体成片速度 |
| 显存 | 有效PID3819080 NVIDIA采样峰值2322MiB（2.268GiB）；PyTorch allocated1.508GiB、reserved1.570GiB |

73帧是该音频在官方音频窗口下的原生输出，少于完整3.121秒输入；保留原输入和原片，未补帧／拉伸。部署通过不意味着全时长覆盖、同步质量或TTS优势已验证。

### 人脸选择陷阱

官方默认按 left-right 排序且检测阈值较低。3.png 检出三个框：索引0为背景小框（score .109），索引1为主脸（bbox约137,109,362,384，score .837），索引2为另一背景小框（.102）。默认索引0虽然能成片，但画面变化几乎仅在背景，不作为最终验收；正式使用 `--face-index 1`，模型代码不改。首次 ONNX 验收 instrumentation 用函数替换可继承类导致 TypeError，后修为子类；失败证据保留。

## 证据入口

服务器 `PCTalkDeploy/delivery/` 保存官方SHA校验、版本锁、before/after环境、GPU前向和三ONNX会话、逐PID显存采样、媒体解码QA、首尾帧及失败日志。最终只认 `gpu_inference.json` 的success=true、PID3819080和 `media_validation.json` 的attempt02；face0与instrumentation失败另存文件。

本机独立读回：`runs/pctalk_deploy_192_20261031/`，含成片、脚本和delivery。用户密码未写入脚本、日志、笔记或产物。

## Observations

- [status] ✅ 192 A6000官方PC-Talk图片＋音频I2V真实生成通过。
- [pitfall] 3.png必须使用face-index1；默认最左侧会选中背景误检。
- [boundary] 本次无SyncNet／九评分器前向或科学结论。

## Relations

- part_of [[TFG 模型部署总览]]
- relates_to [[24GB快速I2V模型的Sync-C公开榜单筛选]]


## 评分协议核查（2026-10-31，只读）

PC-Talk本次部署只有I2V生成与工程QA，没有SyncNet或九评分器结果。实查已安装commit a0392744cbac857d96db1b99ceab70cf7b5f6c83的git ls-files与git grep，未附SyncNet离线评估器/评分CLI。官方论文v3 §4.1明确使用SyncNet LSE-C/LSE-D，与项目Sync-C/Sync-D属于同一指标家族；论文25fps/16kHz/512²、10秒跨身份测试，称train/test不重叠。源：https://arxiv.org/html/2503.14295v3#S4.SS1 。不能据同名指标确认作者权重版本/SHA、crop、offset窗口、轨迹门限、音轨与电平等完整协议相同。

本项目当前TLY评分入口使用joonson/syncnet_python固定revision907c0b579c2e2d83f0eae1b2ac9e720cde4e5623的run_pipeline.py/run_syncnet.py与syncnet_v2.model，FP32，默认vshift15/crop_scale.4，但min_track显式25（上游默认100）、规范RMS及无损PCM封装属于项目协议。官方评分器不等于所有论文测试协议完全一致，也不据此排除TLY输入/封装/裁剪问题。用户已明确暂停HDTF20评分；本核查没有启动PC-Talk或其他模型评分。


## 中文SyncNet检查点已交付（2026-10-31，后续授权）

用户后续明确授权中文推理与Sync-C评分，已完成[[81-pctalk-chinese20-syncnet-official-comparison]]：同既有20N+60T RMS−24 WAV/3.png，80逻辑/41唯一，person192/lip1/无EMC/face-index1/seed42。此前“仅部署未评分”描述为历史部署阶段；现在全20自然7.266、TTS7.756、Δ+.490、14/20提高，只有SyncNet，其余八项未启动。HDTF TLY仍暂停。

所有生成/评分媒体验收、41距离矩阵/stdout、6478本机文件SHA和来源统计复算PASS；GPU已释放。官方I2V HDTF9.370/MEAD8.190，本轮自然低2.104/.924，异协议不能直接当复现误差。官方SyncNet run_pipeline内部默认AVI为MP3，首对+1105样点/69.0625ms，首N145→147帧、全41offset−2；没有内部PCM对照，不归因低分。入口无损不代表全链无损。完整作者评分协议仍未对齐。


## 九评分器续评完成（2026-10-31，权威后续状态）

[[83-pctalk-chinese20-nine-scorers]]已补齐其余八项；报告81为历史检查点。80逻辑/41唯一媒体与原SyncNet不变，0新生成/SyncNet前向。主要12九项Holm9均未过；全20 CoGenAV/PEAVS/MTD/Voca四项负差过校正，Sync-C+.490 q=.15475未过。精确CPU敏感性判定全部不变，PEAVS精确q=.046143。七项20来源，两固定5秒P0各6完整来源，不插补。

MTD时移65/82低于80%门（至少66），不能把负差升级为嘴型退化；其余控制与PEAVS批次边界详见83。原数组、2339内容SHA、25分层统计及25精确p独立复核PASS，GPU已释放，HDTF仍暂停。首次SyncReward包装因缺GPU_UUID在模型加载前退出，私有环境修复后仅继续两项，715已完成科学文件SHA不变，旧失败保留。


## 新Chinese-LiPS21人像试跑验收（2026-10-31）

✅ [[84-pctalk-chinese-lips21-nine-scorers]]：复用已有云TTS，21人各一对，42唯一官方推理与九评分器完成；模型/12权重及环境均未升级，源码100 tracked SHA不变。执行源码 scripts/experiments/pctalk_lips21_20261031，个人run /home/data2/wangjiajun/av-evaluation/jobs/pctalk_lips21_rms24_nine_20261031_attempt01。47记录PID均退出；全局显存仍可能属于其他任务，不等于整卡空闲。HDTF TalkLikeYou仍用户暂停。

新图不是固定512²：官方pasteback按source_max_dim1280/source_division2保留源图尺寸，本批为454–904²。首N有效454²成片被旧512验收拒绝；仅修私有校验并复用同一raw/scoring MP4 SHA，不通过外部resize或重新生成修复。换图必须先核查实际检测框和排序，本批按原生left-right序选最大bbox，21张实际均index0；旧3.png仍须index1，不能把任一索引推广到所有图片。

生成27后的一次util门失败发生在下一条模型加载前；generate_resource_wait03.py仅把原16GiB空余/util≤85断言改45秒轮询，沿相同GPU/参数续缺项。source_freeze、两次历史失败和source/媒体SHA保留；旧FAILED/RUNNING不是当前状态。模板遗留20/80等描述字段另行明确更正，真实输入42、主要17分层及科学参数未变。评分控制、统计及正式结论只引用报告84。
