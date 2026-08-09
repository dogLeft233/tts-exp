# 16 — phone-local-warp 目标验证（V0–V2）

## 目的

验证一个自然时间轴 waveform target 是否能把 TTS 的局部声学特征迁移到 natural 音频，同时保留 natural 的总时长、token 时长和停顿位置：

```text
natural waveform + TTS token alignment
    -> phone-local-warp waveform
    -> Ditto + SyncNet
```

该实验验证的是 waveform enhancer 的训练 target 假设，不是已经训练好的 enhancer，也不是对原有 TTS→TFG 结果的重跑。

## 实现

实现文件：

- `scripts/pnp_alignment_warp.py`
- `scripts/pnp_cache_targets.py`
- `tests/test_pnp_alignment_warp.py`
- `tests/test_pnp_cache_targets.py`

处理流程：

1. 读取 natural/TTS 的 MFA token span：`label`, `start_s`, `end_s`, `confidence`。
2. 按时间顺序从 TTS 的当前位置向后寻找同 label token；不匹配 token 跳过并记录。
3. 对每个匹配 token，截取 TTS waveform 片段。
4. 用 `scipy.signal.resample` 将该片段重采样到 natural token 对应的 sample 数。
5. 以 natural waveform 为底板，仅覆盖匹配且 confidence ≥ 0.8 的 token 区域。
6. 未匹配、低置信度和 token 外的 pause/gap 保留 natural。
7. 在替换区间边界使用默认 8 ms 线性 crossfade。
8. natural/TTS 在缓存前统一重采样到 16 kHz；输出 sample count 严格等于 natural 输入。

这不是完美的 TTS 特征解耦。普通 waveform resampling 会同时影响速度、pitch、formant、相位和音色；逐 token splice 也可能破坏跨 token 的 coarticulation 和连续 prosody。因此 `phone-local-warp` 只能作为反事实 target/control，不能直接解释为“只迁移 TTS 的非时序特征”。

## V0：数据和 provenance audit

数据：AISHELL-1 100 条 natural/TTS 配对，provider 为 `faster_qwen3`。

修正后的审计命令显式使用 `--target-sample-rate 16000`，并使 audit 的 token matching 与实际 warp 逻辑一致。

结果：

- paired records：100
- rows：100
- missing audio：0
- natural：16 kHz
- raw TTS：24 kHz
- cache/warp target：16 kHz
- low coverage（coverage < 0.9）：48/100
- mean coverage：约 0.8807
- uniform fallback：0

V0 只证明数据和 alignment 具备可追溯性，不能证明 local-warp target 有效。

## V1：target 构造检查

每条 target 都保存为 WAV，并附带 JSON provenance metadata，包括：

- natural/TTS source path
- natural/TTS source sample rate
- target sample rate
- natural/TTS/output SHA-256
- matched/unmatched token 统计
- alignment source
- confidence threshold
- crossfade 参数
- exact sample count

100 条 local-warp target 均生成成功并满足 exact-N。当前缓存位于远端：

```text
/root/autodl-tmp/pnp_validation/v1_cache/
```

## V2：Ditto + SyncNet 配对验证

固定：

- 同一批 10 条 AISHELL 样本
- 同一参考图像
- Ditto checkpoint
- seed=42
- TRT online
- SyncNet `min_track=20`
- 所有视频生成和 SyncNet 评估均成功

比较条件：

```text
natural_raw
raw TTS
phone-local-warp
```

其中 `natural_raw` 在 raw-TTS 和 local-warp 两组运行中完全相同。

### 10 条配对结果

| 条件 | Sync-C 均值 | 相对 natural | 正向样本 |
|---|---:|---:|---:|
| natural | 3.2278 | — | — |
| raw TTS | 3.6453 | **+0.4175** | 7/10 |
| phone-local-warp | 2.7401 | **−0.4877** | 2/10 |

Sync-D：

| 条件 | Sync-D 均值 | 相对 natural |
|---|---:|---:|
| natural | 10.3976 | — |
| raw TTS | 10.3744 | −0.0232 |
| phone-local-warp | 11.0077 | +0.6101 |

Sync-C 是当前主要同步置信度指标，越高越好；Sync-D 越低越好。local-warp 在这组数据上没有保留 raw TTS 的 Sync-C 趋势。

### coverage 分层

首 10 条中：

- high coverage（≥ 0.9）：sample 1、2、4、9、10
- low coverage（< 0.9）：sample 3、5、6、7、8

Sync-C 相对 natural 的平均 delta：

| 子集 | raw TTS | phone-local-warp |
|---|---:|---:|
| high coverage，n=5 | +0.2420 | **−0.5546** |
| low coverage，n=5 | +0.5930 | **−0.4208** |
| all，n=10 | +0.4175 | **−0.4877** |

高 coverage 子集同样退化，因此不能把失败仅归因于 alignment coverage 不足。

## 结论和决策

1. 原有 raw TTS→TFG 结论没有被推翻：在这 10 条当前重跑中，raw TTS 的 Sync-C 仍然是 7/10 正向，平均 +0.4175。
2. 被否定的是更具体的假设：把 TTS token waveform 逐段重采样回 natural timing，不能在当前实现和 Ditto 条件下保留 raw TTS 的同步收益。
3. 该失败可能来自 pitch/formant 变化、跨 token prosody/coarticulation 破坏、逐段边界拼接，以及 TTS timing 本身就是收益组成部分。
4. **Go/No-Go：No-Go。** 当前 `phone-local-warp` 不进入大规模 waveform enhancer 训练 target。
5. 暂不扩大 residual enhancer，也不把当前结果称为 universal waveform enhancement。

后续应优先比较：

- pitch-preserving local time-stretch（WSOLA/phase vocoder）
- 保留部分 TTS timing/prosody 的受控变换
- duration-only、loudness-only、spectral-only controls
- 至少第二个 TFG 的复验
- feature-domain teacher ablation

如果这些对照显示 raw TTS 的收益依赖完整 timing/prosody，则应放弃“严格 natural timing preservation”作为主 target；如果 pitch-preserving 或受控 prosody 版本恢复趋势，才继续 waveform enhancer 路线。

## 试听文件

sample 1 的三个版本已下载到本地：

```text
artifacts/pnp_v2_sample1/natural.wav
artifacts/pnp_v2_sample1/raw_tts.wav
artifacts/pnp_v2_sample1/phone_local_warp.wav
artifacts/pnp_v2_sample1/phone_local_warp.json
```

可用 `ffplay`、VLC 或 Audacity 试听。

远端原始产物：

```text
/root/autodl-tmp/pnp_validation/v1_cache/1_faster_qwen3.wav
/root/autodl-tmp/pnp_validation/v1_cache/1_faster_qwen3.json
```

## 运行产物

Ditto/SyncNet runs：

```text
/root/autodl-tmp/repos/tts-exp/runs/pnp_v2_raw10_20260807
/root/autodl-tmp/repos/tts-exp/runs/pnp_v2_local10_20260807
```

审计：

```text
/root/autodl-tmp/pnp_validation/v0_audit_16k_matching.json
```

## 测试状态

- waveform MVP 专项测试：21 passed
- `py_compile`：通过
- 完整仓库回归：361 passed，1 failed
- 唯一失败来自既有英文 alignment fixture 缺少当前实现要求的 `source_manifest` provenance 字段，不属于本实验文件
