---
title: DyStream 192服务器部署
type: note
permalink: tts-exp/docs/deployment/dystream
tags:
- deployment
- '192'
- dystream
- i2v
---

# DyStream 192服务器部署

✅ 2026-10-31 完成192服务器单A6000的真实图片＋中文音频I2V验收。使用官方运动生成与渲染组件，提供无界面CLI适配；未启动Gradio服务，未做同步评分或在线流式延迟测试。

## 来源与目录

- 官方源码：https://github.com/RobinWitch/DyStream ，commit `f587541ccaa77f092d03487d7d89502a67f0a084`。
- Git直连较慢，改用官方GitHub codeload归档；归档PAX comment确认同一commit，SHA256 `5391fd7ec827ec39bdd4c3c0f0269d1fd76321ad588b18b366050588497bb15b`。21个源码/示例文件在服务器逐一校验通过。
- 官方配套资产：https://huggingface.co/robinwitch/DyStream ，revision `9ad5b1d3b0ef7aece4b9855d972c1f819f04dfbb`；约9.3GB，包含运动checkpoint、渲染checkpoint、MediaPipe模型和工具代码。下载结束码0，230个tools/checkpoints文件按官方LFS SHA256或Git blob SHA1全部通过首次校验。运行后另核实际源码/权重等非pyc资产；bundle自带的pyc属运行时派生缓存，最终入口禁止/重定向字节码写入。
- `checkpoints/last.ckpt`：7730126719B，SHA256 `15baf05196834ca54b4da7c2c9fd372b572e650b1edd84cb5a92c4be1689f29b`。
- `tools/pretrained_model/epoch=0-step=312000.ckpt`：1579672001B，SHA256 `a9f06f55912dfa12ea18d77315f1c1675c5f67669097db8f155b4a4d75f9d31d`。
- 主机 `wangjiajun@10.108.16.192:22`；官方源码和资产 `/home/data2/wangjiajun/DyStream`；部署根 `/home/data2/wangjiajun/DyStreamDeploy`，含env、cache、tmp、conda-pkgs、logs、outputs、delivery。

## 环境与服务器规则

当次通过飞书MCP读取《【192服务器】1048服务器手册》实际正文后连接，核对账号、HOME、目录所有者、磁盘及GPU。共享模型区有限只读检查未找到DyStream。所有新增文件均在本人data2目录；用共用 `/home/user/anaconda3/bin/conda` 新建Python3.11独立环境，未修改共享base、系统驱动/CUDA或其他人的环境与任务。下载、安装和推理均nohup后台执行；密码只用于交互式运行时登录，不入脚本/日志/笔记。

实际运行版本：torch2.6.0+cu124、torchvision0.21.0、CUDA runtime12.4、cuDNN9.1、NumPy1.26.4、opencv-contrib-python4.11.0.86、transformers4.57.6、diffusers0.36.0、MediaPipe0.10.21。完整Python/Conda版本见delivery锁文件，pip check无冲突。首次NJU PyTorch下载过慢，换用TUNA的官方PyPI发行包，torch/torchvision按官方PyPI JSON中的完整SHA256验证；不是降级为CPU版。

上游requirements有`mediapipee`拼写错误，并同时要求opencv-python4.13（NumPy2）与opencv-contrib4.11、NumPy1.26。实际清单修正为mediapipe0.10.21，只保留contrib4.11；torch/vision改为当前550.78驱动已实际验证的2.6/0.21 CUDA12.4组合。JAX固定0.4.35/jaxlib0.4.35，避免MediaPipe间接依赖回溯到需要NumPy2的新版本；其余主要固定版本保留上游设置。`requirements-resolved.txt`及环境freeze保留。

官方facebook/wav2vec2-base-960h固定revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`。复用本人SoulX目录的7个文件，逐文件核对官方LFS/Git对象后，以只读文件链接建立独立HF snapshot和refs；运行设置HF_HUB_OFFLINE，未再次下载或改写源模型。

## 推理适配与已解决问题

`infer.sh`调用部署根`gpu_infer.py`，由其调用官方`app.run_inference`，原官方`.py`不修改。固定4 CPU线程/4 CPU affinity，按单GPU UUID隔离，PyTorch allocator上限55%（本次保守设置，不是服务器配额）。

1. 原app会先构建训练用VGG/GAN损失。适配层直接按同一官方配置构造face_encoder、motion_encoder、flow_estimator、face_generator四组件，加载同一渲染checkpoint，省去不参与推理的损失模块及额外VGG下载。
2. 发布源码FaceGenerator定义了`warp_layer`，但其forward/manual_forward完全不使用，checkpoint也不含`warp_layer.weight`。第一次严格检查因此中止，失败日志保留；适配层删除这个未使用成员，避免其被意外调用后使用随机权重。其余渲染权重strict加载：114/52/11/226个键；运动模型完整加载0 missing/0 unexpected。EMA按官方app使用。base Wav2Vec2初始化时的masked_spec_embed提示发生在加载完整DyStream权重之前，最终运动权重无缺失。
3. 原MediaPipe delegate默认GPU，通过EGL另选3090，不服从CUDA_VISIBLE_DEVICES。首次可成片路径因此只作诊断保留；最终明确delegate=CPU，并将EGL vendor限定Mesa/software，避免额外NVIDIA OpenGL上下文。最终PID采样只出现所选A6000。
4. torch2.6的weights_only默认变化：仅对已校验的两个官方checkpoint明确使用weights_only=False；其余torch.load保持原行为。

## 正式验收

输出：`/home/data2/wangjiajun/DyStreamDeploy/outputs/attempt02/i2v.mp4`。

| 项目 | 实测 |
|---|---|
| 输入 | 现有3.png人像＋BAC009S0009W0121_N_0.wav；16kHz mono、3.121秒；输入SHA见inputs.sha256 |
| 推理 | 官方speaker-only路径，listener为原生零张量；5steps，seed222，CFG audio/listener=.5/.5、anchor=0、all=1 |
| 输出 | H264，512×512，25fps，77帧，3.080秒 |
| 音轨 | AAC，44.1kHz mono，3.080秒；独立解码RMS .037795、peak .229411，非静音 |
| 动态 | 连续帧平均绝对像素差均值1.496、最大4.655（0–255尺度）；首尾图已目视，主体头部/表情变化 |
| GPU前向 | 记录到WrapedWav2Vec、Audio2FaceGPTBlock、DiffusionHead、MotionEncoder、FaceEncoder、FlowEstimator、FaceGenerator等CUDA张量 |
| 主机GPU | A6000 UUID GPU-3464c5ee-e2ed-81e4-e275-b612a929bfe1；正式PID231087 |
| 耗时 | 32.573秒，含导入、权重加载、预处理、生成、封装；不是稳态流式速度 |
| 显存 | 本人PID NVIDIA采样峰值8274MiB=8.080GiB；PyTorch allocated6.438GiB、reserved7.734GiB |
| 完成 | 视频/音轨完整解码通过，PID已退出，选定GPU显存释放；本机读回成片SHA一致 |

视频SHA256：`39e05419d25e40bf209ab2b5a4ac8655f3d98e81470924a56b23dad064b0a29e`。

原音频3.121秒，原生视频3.080秒，未覆盖尾部约41ms。官方MoviePy封装将评分音轨重采样为44.1kHz AAC；未补帧、变速或额外移音。若后续用于既定PCM评价协议，需要单独制定适配与时钟检查，不能直接沿用此部署样片作为受控实验。未运行SyncNet/九评分器，不作同步质量或TTS优势判断。

## 可复用命令

先用nvidia-smi确认空闲显卡，再按UUID选择；避免覆写已有输出。

```bash
R=/home/data2/wangjiajun/DyStreamDeploy
CUDA_VISIBLE_DEVICES=GPU-3464c5ee-e2ed-81e4-e275-b612a929bfe1 \
nohup bash "$R/infer.sh" \
  --source /home/data2/wangjiajun/av-evaluation/datasets/face/3.png \
  --audio /home/data2/wangjiajun/av-evaluation/datasets/chinese20/audio/BAC009S0009W0121_N_0.wav \
  --output "$R/outputs/custom/i2v.mp4" \
  > "$R/logs/custom.log" 2>&1 < /dev/null &
```

支持`--listener-audio`、`--steps`、`--seed`；每片旁边保存gpu.json、reference.png、masked.png。`infer.sh --help`、shell/Python语法检查通过。`run_acceptance.sh`是本次固定验收入口，已有输出不允许覆盖。

## 证据

服务器`DyStreamDeploy/delivery/`含原始来源/版本/哈希、资产校验、环境freeze、GPU前向与逐PID显存、媒体QA、首尾帧、CLI帮助和失败记录。最终认`acceptance03.exit=0`、`outputs/attempt02`、PID231087；`acceptance01`是缺未使用warp参数的失败，`outputs/attempt01`与acceptance02是MediaPipe额外GPU诊断，不与正式显存合并。

本机读回在`runs/dystream_deploy_192_20261031/`，含正式MP4与delivery；9.3GB权重留在服务器，未在本机重复存储。

## Observations

- [status] ✅ DyStream在192单A6000的真实I2V部署验收通过。
- [pitfall] MediaPipe默认EGL会跨CUDA_VISIBLE_DEVICES选卡；最终预处理固定CPU并隔离EGL。
- [pitfall] FaceGenerator未使用的warp_layer参数不在官方checkpoint；显式移除后其余渲染权重严格加载。
- [boundary] 本次仅离线部署验收，没有同步评分或在线流式延迟实测。

## Relations

- part_of [[TFG 模型部署总览]]
- relates_to [[PC-Talk 192服务器部署]]
- relates_to [[24GB快速I2V模型的Sync-C公开榜单筛选]]
