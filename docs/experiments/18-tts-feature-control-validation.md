# 18 — TTS 特征控制与 natural 节奏对齐验证（CPU pilot + Ditto/SyncNet n=10）

## 目的

验证实验 17 调研提出的下一步：不要直接把 TTS waveform 逐 phone 普通重采样，而是显式固定 natural 时间轴，并分别测试：

- pitch-preserving phone-local warp；
- duration interpolation；
- loudness/energy/spectral/F0 controls；
- 是否能在不改变 natural 总时长的情况下，单独改变可解释声学变量。

本报告是 CPU 侧 control validation，不是新的 Ditto/SyncNet 结果，也不是 trained enhancer 结果。

## 实现

新增：

- `scripts/pnp_feature_controls.py`
- `scripts/run_pnp_feature_control_experiment.py`
- `scripts/summarize_pnp_feature_controls.py`
- `tests/test_pnp_feature_controls.py`

实现边界：

1. 复用 `pnp_alignment_warp.py` 的 token matching、ordinary local warp 和 exact-N 约束；
2. 新增基于 `librosa.effects.time_stretch` 的 phone-level phase-vocoder 变换；
3. source segment 太短或为空时显式 fallback 到 natural，并记录原因；
4. duration-alpha 使用：

```text
d_out = d_natural * exp(alpha * (log(d_tts) - log(d_natural)))
```

并在 matched phone 内归一化，使 natural matched 总 sample count 不变，natural gap/pause 样本保留；
5. 统一输入到 16 kHz mono，原始 natural/TTS sample rate 写入 provenance；
6. `tts_raw_oracle` 保留 TTS 的实际时长，仅作上界参照，不要求 exact-N；其他 control 必须 exact-N；
7. spectral control 只调整长时 spectral tilt，不能称为完整 timbre disentanglement；F0 control 只做全局 median-F0 pitch shift，不能称为完整 prosody transfer；energy control 只传递归一化短时 envelope。

## Pilot 数据

固定使用实验 16 的前 10 条 AISHELL 配对和 `faster_qwen3` TTS：

- natural 本地 WAV：`data/aishell1_100_zh/audio/`；
- TTS 本地 WAV：`runs/r2_faster_qwen3_20260707T145233Z/02_tts/`；
- alignment manifest：`results/aishell100_phoneme/alignment/manifest.json`；
- natural 原始采样率：16 kHz；
- TTS 原始采样率：24 kHz；
- control 输出采样率：16 kHz；
- 每个 sample 生成 14 个条件，共 140 个 WAV/JSON。

产物：

```text
artifacts/pnp_feature_controls_pilot/
artifacts/pnp_feature_controls_pilot/manifest.json
artifacts/pnp_feature_controls_pilot/cpu_feature_summary.json
```

条件：

```text
natural_identity
tts_raw_oracle
global_duration
ordinary_phone_local
pitch_preserving_phone_local
loudness_transfer
energy_transfer
spectral_timbre_transfer
f0_transfer
duration_alpha_0
duration_alpha_0.25
duration_alpha_0.5
duration_alpha_0.75
duration_alpha_1
```

## CPU invariant 结果

除 `tts_raw_oracle` 外，130 个 exact-N control 全部满足：

- exact sample count：130/130；
- finite：130/130；
- clipped samples：0；
- natural 总时长保持；
- natural gap/pause 区域按 control 策略保留；
- 输出 provenance 和输入/输出 hash 均写入 JSON。

`tts_raw_oracle` 不要求 exact-N，因为它保留 TTS 的真实时长，作为 raw TTS 上界对照。

合成信号回归测试证明：

- phase-vocoder 将 220 Hz 单频信号变速后，频谱峰仍在 220 Hz ±2 Hz；
- 变换输出严格为目标 sample count；
- 短 segment 不静默使用 phase vocoder，而是显式 fallback；
- duration-alpha=0 的 matched phone duration 等于 natural；
- duration-alpha=1 的 phone duration 比例朝 TTS duration 方向变化；
- pause/gap 样本和 finite/exact-N 约束成立。

测试：

```text
50 passed
```

命令：

```bash
python -m pytest -q \
  tests/test_pnp_feature_controls.py \
  tests/test_pnp_alignment_warp.py \
  tests/test_pnp_cache_targets.py \
  tests/test_audio_interventions.py
```

## 关键 CPU 特征结果

以下是 10 条 paired sample 相对 `natural_identity` 的均值变化。所有 duration delta 为 0，说明这些 control 没有改变句子总时长。

| 条件 | Δ spectral centroid | Δ spectral tilt | Δ median F0 | Δ LUFS |
|---|---:|---:|---:|---:|
| ordinary phone-local | −161.1 Hz | −0.156 | −21.6 Hz | −1.02 dB |
| pitch-preserving phone-local | **−2.5 Hz** | −0.054 | +8.1 Hz | −2.27 dB |
| duration-alpha=0 | −2.5 Hz | −0.054 | +8.1 Hz | −2.27 dB |
| duration-alpha=1 | −14.0 Hz | −0.057 | +8.2 Hz | −2.54 dB |
| loudness transfer | +1.7 Hz | +0.004 | 0.0 Hz | −1.47 dB |
| energy transfer | +51.4 Hz | +0.188 | +0.2 Hz | −0.90 dB |
| spectral-timbre transfer | −165.6 Hz | −0.087 | +0.02 Hz | −1.23 dB |
| F0 transfer | −41.2 Hz | −0.247 | +4.4 Hz | −2.90 dB |

### 解释

1. **普通 waveform resampling 确实造成明显谱变化。** ordinary local warp 的 centroid 平均下降约 161 Hz；pitch-preserving local warp 下降约 2.5 Hz。这个结果支持“普通 resampling 的 pitch/phase/spectral distortion 是实验 16 退化的一个重要嫌疑”。
2. **这还不能证明 pitch-preserving warp 会恢复 TFG 收益。** phone boundary、coarticulation、crossfade、energy 和 TTS/global prosody 仍可能不同；必须通过固定 Ditto/SyncNet 评估才能判断。
3. **pitch-preserving control 仍有 gain/energy 偏移。** 其 LUFS 平均低于 natural 约 2.27 dB，说明 phase-vocoder 和 phone splice 不是干净的单因素控制；TFG 结果必须配合 loudness-matched 对照解释。
4. **duration-alpha 主要改变局部时长分配，而不改变句子总时长。** alpha=0 与 pitch-preserving local warp 相同，alpha=1 向 TTS phone duration 比例靠近；它仍使用 TTS phone waveform，因此不是纯 duration-only waveform control。
5. **energy/spectral/F0 controls 不是完全解耦器。** 它们是可解释的干预基线，用于测试方向性，不应被称为纯 energy、纯 timbre 或完整 prosody representation。
6. 每个 duration-alpha 条件共记录 3 个空/过短 token fallback，均回退到对应 natural span；这些 fallback 已写入 metadata，不能在后续结果中删除。

## GPU 端 Ditto/SyncNet 验证（固定 n=10）

在 CPU control invariant 通过后，使用同一批实验 16 的 10 条 AISHELL 配对，固定：

- Ditto checkpoint、参考图像、seed=42；
- TRT online；
- SyncNet `min_track=20`；
- 每个条件独立 run ID；
- 不修改远端源码、checkpoint 或已有 runs。

七个 exact-N control 均完成 Ditto 10/10、SyncNet 10/10。`tts_raw_oracle`、ordinary phone-local 和 natural identity 的历史 anchor 保留实验 16 的数值，不把本轮 control 结果与历史 raw TTS 结果混为同一批重新采样。

### paired Sync-C 结果

相对 natural identity 的 delta 按 sample 配对计算；bootstrap 是 n=10 的描述性 percentile 95% CI，exact sign permutation 是未做多重比较校正的参考量。

| 条件 | Sync-C 均值 | Δ vs natural | 正向样本 | Δ bootstrap 95% CI | Sync-D 均值 |
|---|---:|---:|---:|---:|---:|
| natural identity | 3.2278 | — | — | — | 10.3976 |
| raw TTS oracle（实验 16） | 3.6453 | **+0.4175** | 7/10 | — | 10.3744 |
| ordinary phone-local（实验 16） | 2.7401 | **−0.4877** | 2/10 | — | 11.0077 |
| pitch-preserving phone-local | 2.6247 | **−0.6031** | 2/10 | [−1.1878, +0.0294] | 10.0491 |
| duration-alpha=0 | 2.6247 | **−0.6031** | 2/10 | — | 10.0491 |
| duration-alpha=0.5 | 2.6807 | **−0.5471** | 3/10 | [−1.1979, +0.1304] | 10.0915 |
| duration-alpha=1 | 2.6836 | **−0.5442** | 3/10 | [−1.1627, +0.0792] | 9.9468 |
| loudness transfer | **3.1702** | **−0.0576** | 3/10 | [−0.2193, +0.1147] | 10.3813 |
| energy transfer | 3.1057 | **−0.1221** | 2/10 | [−0.2457, +0.0144] | 10.3170 |
| spectral-timbre transfer | 3.0016 | **−0.2262** | **6/10** | [−0.6024, +0.0801] | 10.3108 |

完整 sample-level 结果和远端评估 JSON 位于：

```text
artifacts/pnp_feature_controls_pilot/remote_syncnet_summary_20260807.json
artifacts/pnp_feature_controls_pilot/remote_syncnet_20260807/
```

### 结果解释

1. **没有一个 exact-N control 恢复 raw TTS 的正向 Sync-C 均值。** raw TTS 的历史 anchor 为 +0.4175；pitch-preserving 与 duration-alpha 仍约 −0.54 至 −0.60。
2. **phase-vocoder 只修复了一个音频层面的嫌疑，不足以修复 TFG 结果。** CPU 上它减少了 ordinary warp 的频谱偏移，但 GPU 上 Sync-C 反而为 −0.6031；因此 ordinary resampling 的 pitch/phase distortion 不是唯一主因。
3. **duration interpolation 没有形成恢复趋势。** alpha 从 0 增至 1 只使均值从 −0.6031 变为 −0.5442，且正向样本从 2/10 到 3/10；n=10 下不能视为有效改善。
4. **loudness control 最接近 natural，但不是正向迁移。** ΔSync-C 为 −0.0576，说明整体响度差异可能解释部分 loss，但没有解释 raw TTS 的 +0.4175。
5. **spectral-timbre control 的方向不稳定。** 6/10 样本相对 natural 上升，但均值仍为 −0.2262，且 sample 6/8 出现较大负 delta；不能称为 timbre-only 成功。
6. Sync-D 的变化没有与 Sync-C 形成一致的改善模式，因此当前判定仍以 Sync-C 为主，并同时报告二者。

这些结果只支持：在当前 Ditto/SyncNet、n=10 和这些 waveform controls 下，raw TTS 的收益不能由单独的响度、能量、长时 spectral tilt、phase-vocoder phone transfer 或固定 natural 总时长的局部 duration interpolation 复现。

## 与实验 16 的关系

实验 16 的 anchor 没有被改写：

```text
raw TTS Sync-C delta          = +0.4175
ordinary phone-local delta    = −0.4877
high-coverage ordinary delta  = −0.5546
```

本轮新增的 pitch-preserving、duration-alpha、loudness、energy 和 spectral 条件都保持 exact-N、finite、无 clipping，并在同一固定 Ditto/SyncNet 协议下完成。它们没有推翻 raw TTS→TFG 结论；它们进一步否定了“只要修复普通 waveform resampling 的谱失真，就能保留 raw TTS 优势”的较窄假设。

## 当前决策

### 已支持的中间结论

```text
普通 phone-local waveform resampling 会引入明显的谱/音高相关变化；
phase-vocoder 版本可以在 CPU 上显著减少该变化；
但这种修复不能在当前 n=10 Ditto/SyncNet 评估中恢复 raw TTS 的 Sync-C 优势；
loudness 是最接近 natural 的单因素 control，但仍未产生正向 Sync-C delta。
```

### No-Go / 保留方向

```text
No-Go：不训练当前 phone-local 或 strict-natural-timing waveform enhancer；
No-Go：不把 spectral-timbre、energy 或 duration-alpha control 称为已解耦的单属性迁移；
保留：若继续建模，应允许 conditional generator 保留部分 TTS timing/prosody，
并把 duration/pause 作为显式变量，而不是强制所有输出回到 natural timing。
```

在 n=10 pilot 下，不因 spectral 的 6/10 正向样本扩展到剩余 90 条，也不启动大规模 enhancer 或 latent generator 训练。若继续验证，应先把最接近 natural 的 loudness control、spectral control 和 raw TTS 放入第二个 TFG 的固定复验，并补充 global duration、F0 与组合 factorial controls。

## 远端运行产物

本轮远端只创建隔离 validation 输入、staging symlink、临时 YAML 和独立 run 输出；没有覆盖远端已有源码、checkpoint、旧 runs 或用户修改。

```text
/root/autodl-tmp/pnp_validation/pnp_v3_controls_20260807/
runs/pnp_v3_pitch_preserving_phone_local_20260807
runs/pnp_v3_duration_alpha_0_20260807
runs/pnp_v3_duration_alpha_0.5_20260807
runs/pnp_v3_duration_alpha_1_20260807
runs/pnp_v3_loudness_transfer_20260807
runs/pnp_v3_energy_transfer_20260807
runs/pnp_v3_spectral_timbre_transfer_20260807
```

## 下一步

1. 将当前七个 control 结果作为 n=10 pilot 归档，不扩展到 n=100；
2. 若需要继续，优先做第二个 TFG 的 raw/loudness/spectral/ordinary 对照，验证结果是否为 Ditto-specific；
3. 只有在第二个 TFG 复现方向，或 factorial control 明确定位到 timing/prosody 组合后，才设计 duration-controlled acoustic generator；
4. 训练 waveform enhancer 前必须先冻结 target 定义，并保留 raw TTS 作为上界、natural identity 作为 identity anchor。

## 与前序报告的关系

- 实验 16 的 ordinary phone-local No-Go 结论保持不变；
- 实验 17 的调研结论得到更强的负向边界证据：显式 duration/pause 接口是合理设计，但当前简单 waveform controls 还不足以实现声学因素迁移；
- 本报告不改写原有 raw TTS→TFG 结果。
