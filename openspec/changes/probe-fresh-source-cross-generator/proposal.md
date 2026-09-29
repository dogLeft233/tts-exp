## Why

同批数据上的 phone-core、参考条件交互和视觉动态诊断均已完成，但没有建立新收益。下一步用新 source groups、第二生成器和独立视觉证据，检验固定波形变换的模型依赖，停止同批参数搜索。

## What Changes

- 冻结 12 个新来源、一个直接重合成候选、两种生成器和原自然评分音轨。
- A产出直接重建的配对同步收益；B独立视觉/人工验证；C检验整帧延迟的生成响应；D在测量门槛后检验TTS相对natural与解码对照的双重增量。
- 公共输入只写一次；每阶段最多三个Luna worker，复用既有模型与特征提取函数。

## Capabilities

### New Capabilities

- `fresh-source-replacement`: 新来源上的 Wav2Lip / Ditto 固定自然音轨探针。
- `fresh-source-visual`: 同一队列上的独立真实嘴部动态与人工验证。
- `fresh-source-timing-transfer`: 整帧音频延迟的跨生成器嘴部响应。
- `fresh-source-tts-increment`: 新来源上TTS对齐路径超出解码处理的增量。

### Modified Capabilities

无。旧实验、门禁和结论保留。

## Impact

仅新增 `scripts/experiments/fresh_source_{inputs,replacement,visual,timing}/`、对应测试、新run和本change。D复用inputs与replacement包，不另建框架。第一阶段依赖冻结WavLM/HiFi-GAN、Wav2Lip、SyncNet、MediaPipe及远端Ditto；D另用已有faster_qwen3 0.6B和MFA3。当前只交付spec；实施任务均未勾选。
