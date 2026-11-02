---
title: A1 Wav2Lip–SyncNet proxy parity 与 TTS 条件归因
type: task
permalink: tts-exp/tasks/a1-wav2-lip-sync-net-proxy-parity-与-tts-条件归因
assigned_to: gpt-luna
blockers: []
current_step: 10
parent_task: Wav2Lip replacement-aware cross-attention v2 修复
priority: high
started: '2026-08-27'
status: done
steps:
- Lock inputs and reproduce current evidence
- Build a minimal differential fixture
- Validate Wav2Lip input parity
- Validate Wav2Lip rendered-patch and compositor parity
- Resolve source/crop/box mismatch
- Validate SyncNet embedding and full-curve parity
- Run full-cohort TTS causal attribution
- Add boundary regression tests
- Run fixed-sample diagnostic overfit only after parity
- Issue decision report and update parent task
tags:
- task
- wav2lip
- syncnet
- parity
- cross-attention
- tts-conditioning
- lrs3
completed: '2026-08-27'
---

# A1 Wav2Lip–SyncNet proxy parity 与 TTS 条件归因

## Observations

- [description] 对 A1 replacement-aware cross-attention 实验建立可复现的 training-proxy → official Wav2Lip → official SyncNet 端到端 parity，并定量判断 paired TTS key/value 对 feature 与最终 waveform/video 的因果贡献；只有 parity 通过后才允许做固定样本优化能力诊断。
- [status] done
- [assigned_to] gpt-luna
- [current_step] 10
- [parent_task] [[Wav2Lip replacement-aware cross-attention v2 修复]]

## Objective

本任务不是继续扩大训练，也不是尝试通过更多 steps 挽救已经失败的 A1。目标是解释以下已观测事实：

```text
A1 differentiable proxy validation target distance 仅轻微下降，
但 official Wav2Lip + official SyncNet 的 strict natural-audio replacement endpoint 明显恶化。
```

必须区分：

1. differentiable Wav2Lip 的 face input、rendered patch、compositor 与 official inference 不一致；
2. direct SyncNet proxy 与 official file-level scorer 不一致；
3. paired TTS 条件进入 adapter 太弱，adapter 主要学成了 natural-only/generic residual；
4. parity 通过后，A1 固定 `d(s*)` objective 本身仍无法推动 Sync-C/Sync-D；
5. 20-step pilot 仅仅欠优化。

最终 deliverable 必须给出可复现 evidence 和明确的 decision tree，不能仅凭最终 Sync-C/Sync-D 猜原因。

## Repository and working state

主要仓库：

`/home/wjj/tts-audio/tts-exp`

当前实现所在 worktree：

`/home/wjj/tts-audio/tts-exp/.claude/worktrees/lrs3-wavlm-resynthesis-50`

当前 branch：

`worktree-lrs3-wavlm-resynthesis-50`

关键提醒：

- `scripts/experiments/lrs3_syncnet_finetune/` 与 `tests/experiments/lrs3_syncnet_finetune/` 在最近状态中仍为未跟踪文件；新 agent 开始前必须确认实际 git 状态，不能假设 main 已包含实现。
- 存在与本任务无关的 `data/wav2sem_analysis/mandarin_viseme_map.yaml` 删除；不得恢复、覆盖或提交该删除。
- CodeGraph 索引此前来自原始 checkout，而非该 worktree。若继续在 worktree 修改，必须使用 worktree-local 源码，不得把 stale CodeGraph 输出当作当前源码。
- 不覆盖任何历史 run，不上传本地音频、视频或数据到第三方服务。

## Key source files

- `scripts/experiments/lrs3_syncnet_finetune/replacement_model.py`
  - `CrossAttentionResidual`
  - natural WavLM = query；TTS WavLM = key/value
  - zero-initialized output projection
- `scripts/experiments/lrs3_syncnet_finetune/wav2lip_proxy.py`
  - differentiable mel、chunking、face input、Wav2Lip forward、224×224 compositor、SyncNet visual windows
- `scripts/experiments/lrs3_syncnet_finetune/syncnet_loss.py`
  - differentiable MFCC、SyncNet audio windows、official-style full distance curve、A1/A2 loss
- `scripts/experiments/lrs3_syncnet_finetune/replacement_train.py`
  - crop/teacher provenance、split、segment sampling、proxy training、validation selection
- `scripts/experiments/lrs3_syncnet_finetune/replacement_evaluate.py`
  - full-audio generation、official Wav2Lip rendering、strict PCM replacement、official SyncNet five-cell audit
- `scripts/experiments/lrs3_direct_audio/run_wavlm_hifigan_lrs3.py`
  - `render_wav2lip`、`strict_mux`、`score_syncnet`
- `tests/experiments/lrs3_syncnet_finetune/test_replacement_v2.py`

## Frozen artifacts

A1 training run：

`/home/wjj/tts-audio/tts-exp/runs/lrs3_wav2lip_replacement_cross_attention_v2_A1_pilot_20260827`

Checkpoint：

`/home/wjj/tts-audio/tts-exp/runs/lrs3_wav2lip_replacement_cross_attention_v2_A1_pilot_20260827/checkpoint.pt`

Official strict audit：

`/home/wjj/tts-audio/tts-exp/runs/lrs3_wav2lip_replacement_cross_attention_v2_A1_strict_audit_20260827`

Official Wav2Lip checkpoint hash：

`ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`

Official SyncNet checkpoint hash：

`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`

不得修改这些 run。所有新 diagnostic 输出必须写到新的、带日期或唯一后缀的目录。

## Frozen scientific endpoint

唯一主 endpoint：

```text
candidate audio
→ frozen official Wav2Lip
→ candidate-driven video
→ replace audio with untouched natural PCM
→ frozen official SyncNet
```

主比较：

```text
finetuned_video + untouched natural_audio
minus
natural_video + untouched natural_audio
```

有效 gain 必须同时满足：

- Sync-C mean 与 median 上升；
- Sync-D mean 与 median 下降；
- joint wins > 50%；
- target offset 不恶化；
- source-group cluster bootstrap 95% CI 支持正确方向。

candidate-video + candidate-audio 仅是 self-consistency/co-adaptation 诊断，不能作为成功结论。

## Known A1 results

Cohort/split：

```text
34 records / 27 source groups
train      23 records / 19 groups
validation  6 records /  4 groups
test        5 records /  4 groups
20 optimizer steps, batch size 1, 96 frames per step
```

固定 validation proxy：

```text
step 0 target_distance  = 9.0182846
step 20 target_distance = 9.0126354
mean delta              = -0.0056492  (~-0.063%)
4/6 target distances improved
proxy Sync-C delta      = -0.0034615
```

固定 4 条 train diagnostic：

```text
step 0 target_distance  = 8.7938838
step 20 target_distance = 8.8114698
mean delta              = +0.0175860
only 1/4 improved
```

不要把 step 1 的 `9.4221` 与 step 20 的 `8.5172` 当作 learning curve；二者来自不同样本和不同随机 segment。

Official replacement primary endpoint：

```text
Sync-C delta mean   = -0.2299
Sync-C delta median = -0.2655
95% CI              = [-0.2826, -0.1778]

Sync-D benefit mean   = -0.2400
Sync-D benefit median = -0.2500
95% CI                = [-0.2929, -0.1898]

joint wins = 2/34 = 5.9%
```

所有五个 cell、所有 34 条 official `av_offset=-2`。失败不是新的 offset drift。

Official split result：

```text
train vs natural: C=-0.1987, D-benefit=-0.2023
train vs zero:    C=-0.0345, D-benefit=-0.0299
validation vs zero: C=+0.0323, D-benefit=+0.0063
validation vs natural: C=-0.2037, D-benefit=-0.2890
test vs natural: C=-0.4046, D-benefit=-0.3548
```

## Known TTS-conditioning probe

现有 checkpoint 上对 6 条较短样本做了 paired-TTS → shuffled-TTS 消融，保持 natural query 不变：

```text
mean natural query RMS                       = 3.4167
mean adapter residual RMS                    = 0.007106
residual / query                             = 0.208%
paired-vs-shuffled feature RMS               = 0.000690
paired-vs-shuffled / residual                = 9.76%

paired candidate vs zero waveform:
  L1  = 0.00718
  RMS = 0.01870

paired candidate vs shuffled-TTS candidate:
  L1  = 0.00110
  RMS = 0.00372
```

解释边界：这些比值是 causal sensitivity，不是“多少百分比音频来自 TTS”的信息论归因。当前证据表明 paired TTS 确实影响输出，但大部分 residual 对具体 TTS 不敏感。

## Concrete suspected parity gaps

1. 训练从 `crop_video_path` 读取 official SyncNet 生成的 224×224 crop；official audit 把原始 `face_path` 交给 Wav2Lip。
2. proxy 使用 `static_full_frame_boxes`；official `render_wav2lip` 未传固定 `--box`，默认执行 face detector。
3. proxy 使用 PyTorch float bilinear resize/compositing；official inference 使用 OpenCV/uint8，并写入有损视频。
4. proxy 直接把生成帧送入 `syncnet.forward_lip`；official scorer 先运行 `run_pipeline.py`，重新做人脸检测、tracking、crop 和文件解码。
5. 当前 parity gate 只覆盖 Wav2Lip mel frontend/chunk count，不覆盖 6-channel face input、96×96 generated patch、224×224 compositor 或完整 SyncNet `d(s)`。
6. cross-attention 当前 `need_weights=False`，没有保存 attention map；而且 adapter 没有显式 TTS positional encoding。

这些必须通过 differential harness 验证，不能先假设无影响。

## Steps
1. [x] **Lock inputs and reproduce current evidence.** 记录当前 git commit/status、关键源码 hash、checkpoint/run hash；用只读脚本复算 A1 validation、official split 和 6-sample TTS ablation，确认数值一致。结果保存在 worktree-local `tmp/lrs3_wav2lip_proxy_parity_step1_20260827/summary.json`，明确记录 `failures: []`。不得修改历史 run。
2. [x] **Build a minimal differential fixture.** 固定 `lrs3_6VnKV1sr5VQ_00003`、前 96 帧、同一 natural/candidate waveform、同一 224×224 crop、同一 checkpoint；保存 WavLM、HiFi-GAN、mel/chunk、face input、Wav2Lip patch、compositor 和 SyncNet boundary 的 shape/hash/statistics。结果：93 effective chunks、88 visual windows。
3. [x] **Wav2Lip input parity.** 在 fixed box 下比较 official 与 proxy 的 mel、chunk starts、6-channel face tensor、BGR、mask、0–1 scale、resize。mel `max_abs_error=1.91e-5`，chunk starts 完全一致，face input `max_abs_error=4.41e-6`。
4. [x] **Wav2Lip rendered-patch/compositor parity.** 比较视频编码前 official 96×96 model output 与 proxy，以及 uint8/compositor/codec 差异。raw patch `MAE=1.08e-7`；official uint8 vs proxy uint8 `MAE=2.56e-5`；DIVX round-trip `MAE=2.63`，已与 pre-codec 差异分开。
5. [x] **Resolve source/crop/box mismatch.** 分开比较 `crop + fixed box`、`original face + fixed box` 和 `original face + detector/fallback`。original-fixed vs crop-fixed official output `MAE=11.57`；原始视频 detector 在该样本失败并使用 static first-frame box `[2,161,61,174]`，与 full-frame box 输出 `MAE=20.60`。
6. [x] **SyncNet embedding and curve parity.** 同一 tensor 的 direct audio/visual embeddings 与官方同权重模型完全一致；MFCC 完全一致；完整 31 点曲线实现最大差异约 `1.91e-6`，argmin/offset 一致。file-level scorer 的 codec/tracker/crop 差异单独记录。
7. [x] **Run full-cohort TTS causal attribution.** 对 34 条、27 个 source groups 运行 paired TTS、source-group-shuffled TTS、natural context、zero/mean context；mapping 无同组误配。paired-vs-shuffled feature RMS 均值 `7.34e-4`，cluster 95% CI `[6.24e-4,8.43e-4]`；少量 official replacement score 全部 PCM exact。
8. [x] **Add regression tests.** 新增 fixed-box OpenCV input convention、full curve shift indexing、paired/shuffled context sensitivity boundary tests；focused suite `11 passed`，相关模块 ruff 和 compileall 通过。全目录 ruff 仍有既有 `prepare_official_crop.py`、`train.py` 问题，未修改。
9. [x] **Run a diagnostic overfit only after parity.** 固定单样本/segment 20 步更新 adapter；proxy target delta `-0.0534`、official forced-box target delta `-0.0151`；proxy/official Sync-C 均同向上升。但 baseline argmin 为 proxy `+1`、official `+3`，不能作为科学 endpoint 放行。
10. [x] **Issue a decision report and update parent task.** 已完成 decision artifact，结论为 `FIX_PROXY`：先统一训练 proxy 与 strict audit 的 source/crop/box policy；禁止 A2/A3、延长 A1、扩大 cohort 或解冻模型。
## Acceptance gates

### Engineering completeness

- 新 diagnostic 有独立输出目录、summary、source/checkpoint/input hashes、完整失败列表；成功时也必须显式写入 `failures: []`。
- 所有比较使用同一 sample、segment、PCM、frame count、checkpoint、box 和 color convention。
- 每一个 comparison boundary 都保存可审计 statistics；不能只保存最终截图或一个 scalar。
- tests、ruff、compile 通过；不得改动无关文件。

### Wav2Lip parity

在同一 fixed-box、同一 float input 条件下：

- official/proxy mel 最大绝对误差保持 `<=1e-4`；
- chunk start/count 必须完全相同；
- 6-channel face input 的 shape、mask、channel order 和 scale 必须一致；
- 视频编码前 96×96 model output 应在数值容差内一致；若不一致，必须定位到第一个 diverging boundary；
- compositor 差异必须把 resize 与 uint8/codec 分开报告，禁止用 post-codec 误差掩盖 pre-codec mismatch。

不要在看过 finetuned-vs-zero official score 后放宽阈值。如某个 codec-dependent 阈值无法预先合理指定，先用 natural/zero-residual controls 冻结 engineering tolerance，再打开 finetuned comparison。

### SyncNet parity

- 同一 tensor 输入时 direct audio/visual embeddings 和完整 31-point distance curve 必须在固定数值容差内一致；
- fixed pre-encode frames 与 official-instrumented pre-encode path 的 target offset 必须完全一致；
- file-level path 必须单独报告 codec/tracker 导致的 curve error；
- `argmin`、Sync-C、Sync-D 的方向不能因为 proxy/file path 而翻转；若翻转，parity gate 失败。

### TTS attribution

- paired/shuffled/natural/zero-context 四臂必须保持 natural query、checkpoint 和 output length 不变；
- shuffled TTS 必须按 source group 处理并记录 mapping，避免意外 self-pair；
- attention weight 仅为诊断，不能单独作为 TTS 使用证据；paired-vs-control causal output difference 才是主证据；
- 报告全 cohort 分布和 source-group bootstrap，不只报告六条均值。

### Promotion decision

只有以下两项都成立才允许重新设计或训练下一 variant：

1. rendered-frame + SyncNet curve parity 通过；
2. fixed-sample diagnostic 能在 proxy 与相同 official path 上产生一致方向的可测变化。

即使通过，也不能复用当前 sealed test 做 checkpoint/超参数选择。

## Explicit prohibitions

- 不直接运行更长的 34-record A1 来“看看是否会好”。
- 不直接运行 A2/A3。
- 不扩大 cohort 来稀释 negative result。
- 不解冻 WavLM、HiFi-GAN、Wav2Lip 或 SyncNet。
- 不以 candidate self-pair 代替 untouched-natural-audio replacement。
- 不补零、伪造 audio coverage 或放宽 provenance 以增加样本。
- 不覆盖历史 run。
- 不上传本地实验数据到第三方服务。
- 不在发现 mismatch 后同时修改 box、color、resize、codec 和 loss；每次只改变一个可验证因素。

## Required final report format

后续 agent 的最终报告至少包含：

1. `Reproduction`：单一命令和 fixture；
2. `First divergence`：第一个出现差异的 boundary；
3. `Wav2Lip parity table`；
4. `SyncNet 31-point curve parity table/plot`；
5. `TTS causal ablation table`；
6. `Fixed-sample overfit result`，仅当 parity 已通过；
7. `Confirmed / Refuted / Unresolved hypotheses`；
8. `Decision`：STOP、FIX_PROXY、REDESIGN_OBJECTIVE 或允许的新 pilot；
9. 新 artifact 路径、branch/commit、测试命令和结果；
10. parent BM task 的更新位置。

## Execution result

本轮在 worktree-local `tmp/` 完成全部 10 步，未修改历史 run，也未上传本地数据。

- fixed-box Wav2Lip mel/chunk/input/raw-patch/compositor parity 通过；codec 差异单独量化。
- direct SyncNet MFCC、embedding、31-point curve parity 通过；file-level path 的 tracker/crop/codec 影响单独记录。
- 训练 proxy 与 strict audit 的 source/crop/box policy 不一致：训练使用 crop + full-frame box，strict audit 使用 original face；该样本 official detector 失败后使用 static first-frame box `[2,161,61,174]`。
- 34-record TTS attribution 显示 paired TTS 对输出有非零但较弱的 causal effect；attention 只作为诊断。
- 单样本 20-step overfit 在 proxy 和 original forced-box official raw path 上同方向改善，但 locked target offset 分别为 `+1` 与 `+3`。
- 最终 decision 为 `FIX_PROXY`。在统一 driver/crop/box policy 并重新通过 differential fixture 前，不得运行 A2/A3、延长 A1、扩大 cohort 或解冻模块。

主要报告：`tmp/lrs3_a1_proxy_parity_decision_20260827/decision_report.json`

主要 artifacts：

- `tmp/lrs3_wav2lip_proxy_parity_step1_20260827/summary.json`
- `tmp/lrs3_wav2lip_proxy_fixture_20260827/summary.json`
- `tmp/lrs3_wav2lip_input_parity_20260827/summary.json`
- `tmp/lrs3_wav2lip_render_parity_20260827/summary.json`
- `tmp/lrs3_wav2lip_source_box_20260827/summary.json`
- `tmp/lrs3_syncnet_curve_parity_20260827/summary.json`
- `tmp/lrs3_tts_attribution_20260827/summary.json`
- `tmp/lrs3_fixed_sample_overfit_20260827/summary.json`

## Relations

- parent_task [[Wav2Lip replacement-aware cross-attention v2 修复]]
- follows [[对齐 SyncNet loss 与官方评分]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]
- informed_by [[Fixed-video teacher-map distillation]]
