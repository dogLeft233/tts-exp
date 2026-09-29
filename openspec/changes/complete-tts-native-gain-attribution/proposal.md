## Why

前轮 A 已发现固定 TTS 驱动视频对评价音频加噪敏感，但 B 缺少 LeapTalk 依赖且成功路径未实现，无法判断生成端效应。重复控制路径冲突、B 独立验收不足和人工包缺媒体也阻止完整交付。需要补全原协议，避免将工程阻塞误判为科学阴性。

## What Changes

- 新建不可变父证据的 continuation run，复用经核验的 A 音频、视频、特征、矩阵和统计抽样索引。
- 实现完整 LeapTalk provenance/消费输入证明、独立 repeat、冻结基线 ROI 和七格评分。
- 完成 144 科学视频、4 重复视频、336 科学评分、8 控制评分以及六项预注册统计。
- 修复 GPU/磁盘预算、异常分类、CLI 退出码、原子提交和可验证续跑。
- 补齐真实可播放盲评包、人工分析和 B 独立验证，审阅并修复所有影响协议的缺口。

## Capabilities

### New Capabilities

- `tts-native-gain-completion`: 恢复同家族生成环境并完整完成 TTS 原生增益归因实验。

### Modified Capabilities

无。原 change 的科学协议通过引用继承，原证据保留；本 change 的新增要求约束 continuation 实现。

## Impact

主要修改 `scripts/experiments/tts_native_gain_attribution/` 和对应测试，允许新增轻量 LeapTalk adapter、配置和部署说明。新增 run 自行冻结新代码和协议；不改写父 run 或旧 spec 的快照，不训练、不合成新 TTS、不搜索有利参数，不自动开通付费服务。
