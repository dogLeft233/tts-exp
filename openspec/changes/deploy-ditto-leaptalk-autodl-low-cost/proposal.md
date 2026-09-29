## Why

为后续跨 TFG 实验准备 Ditto、LeapTalk 两个可运行后端。本轮只要求低成本部署并各生成一条可播放视频，不验证 replacement 效应。先在 AutoDL 无卡模式完成可行的安装和下载，再短时开启单卡验证。

## What Changes

- 定义无卡准备 → GPU 短测 → 保存产物并关机的部署流程。
- Ditto 使用官方 PyTorch 离线推理；LeapTalk 使用官方 Lite 单步离线推理，两者独立环境、串行运行。
- 根据执行当天的兼容性、库存与价格选择单卡，记录报价；设置开卡前确认、时限和失败停止条件。
- 交付两个最小启动脚本、两条 smoke 视频及复现信息。

## Capabilities

### New Capabilities

- `autodl-two-model-smoke-deployment`: 无卡优先、费用有界的 Ditto / LeapTalk 单卡部署。

### Modified Capabilities

无。已有实验、Ditto TRT 环境及历史输出保持不变。

## Impact

实施时新增 `scripts/deployment/ditto_smoke.sh`、`scripts/deployment/leaptalk_smoke.sh`，允许增加一个确有必要的轻量验收脚本及对应测试。不增加通用部署框架。远端环境、模型和日志均保存在已确认的数据盘任务目录。

## Non-goals

不训练、不跑 SyncNet 或正式对照实验、不接入编号流水线、不部署 Web/实时对话、不申请云 ASR/TTS 密钥、不做 TensorRT 转换、量化或性能优化。本次提交只编写 spec，不租机、开关机或下载模型。
