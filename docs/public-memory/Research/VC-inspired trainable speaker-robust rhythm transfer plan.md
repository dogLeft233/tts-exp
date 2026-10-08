---
title: VC-inspired trainable speaker-robust rhythm transfer plan
type: research_topic
permalink: tts-exp/research/vc-inspired-trainable-speaker-robust-rhythm-transfer-plan
date: '2026-08-14'
status: active
topic: speaker-robust VC-inspired rhythm transfer
tags:
- vc
- speaker
- duration
- prosody
- wavlm
- vocoder
- training-plan
---

# VC-inspired trainable speaker-robust rhythm transfer plan

## Observations

- [status] active
- [question] 如何在保持说话人鲁棒性的同时，将 TTS/MFA 的 rhythm 与 duration 信息转移到可部署的 waveform/VC 路径，并通过统一下游协议验证？

## Problem statement
目标输出不是普通 VC：需要保留 paired TTS 的内容/发音与目标音色，同时使用 paired natural 的 phone duration、pause 和节奏。当前 frozen MFA-linear 在 S0765 稳定正向，但在 S0901/S0912 speaker-dependent 失败；global alpha、residual norm/velocity、silence anchor、boundary smoothing、context crossfade 均不能稳定修复。

## Current evidence and failure taxonomy
- TTS 来自 Faster-Qwen3 self-clone：同一句 natural waveform 同时作为 ref audio，ref text == target text；因此 target TTS voice 随 paired natural speaker/utterance 变化。
- S0901 raw TTS 相对 natural 已偏负，更可能包含 self-clone quality、声学/face mismatch 或目标 TTS 本身的问题。
- S0912 raw TTS 尚可但 MFA-linear 后变负，更像 feature warp 或 vocoder-domain failure。
- alpha=1 MFA-linear 只使用 warped raw-TTS WavLM trajectory；natural 只提供 timing grid。因此不能把 alpha=1 的 speaker effect 简单解释为 natural feature speaker leakage。
- 当前 `prematched=True` HiFi-GAN 由 kNN-VC 在 LibriSpeech train-clean-100 的 prematched WavLM-L6 features 上训练；MFA-linear 输入却是 raw-TTS WavLM features 的 phone-local interpolation，并非 kNN-prematched features。regular/non-prematched vocoder 是一个必须先测的 distribution-match control。
- n25 使用固定 S0765 face；speaker-level SyncNet 同时受 audio/face pairing 和 codec/mux 影响，不能单独充当 audio training label。

## Available data
- Strict paired train: 290 pairs, 6 speakers: S0901 50, S0906 48, S0912 48, S0913 47, S0914 50, S0915 47.
- Speaker-disjoint valid: S0765 50 pairs.
- Train duration: natural 25.1 min + TTS 19.4 min = 44.5 min.
- Valid duration: 7.5 min total.
- Heldout S0770 remains excluded from design, tuning, selection and evaluation.
- This scale is suitable for small adapters and thousands of phone segments, not for training a full VC, neural codec, diffusion/flow decoder or codec language model from scratch.

## Desired factorization
```text
u_tts    = content/pronunciation representation from TTS
s_tts    = target TTS/self-cloned speaker/timbre embedding
d_nat    = natural phone durations and pauses
p_nat    = natural F0/energy/voicing, normalized into s_tts range
s_face   = visual identity used only for matched/mismatched evaluation control

y = Decoder(LengthRegulator(u_tts, d_nat), p_nat, s_tts)
```
Do not collapse `s_tts`, natural timing source and `s_face` into one speaker code. Content may be speaker-adversarial only if a probe confirms leakage; output/timbre must preserve speaker information.

## Ranked routes

### 0. Decisive pre-training split
Generate audio-only matrix with no time warp and with MFA-linear features, using both regular and prematched kNN-VC vocoders. Re-encode output and report per-speaker reconstruction, CER/PER, speaker cosine, quality/artifact metrics. If raw-TTS direct resynthesis fails, adapt vocoder; if direct succeeds but MFA-linear fails, adapt the temporal feature mapper; if audio passes and only SyncNet fails, fix face/evaluation rather than audio.

### 1A. Chinese/self-clone vocoder adapter
Fine-tune the regular WavLM-L6 HiFi-GAN checkpoint on raw TTS and natural self-reconstruction. Freeze most of generator; train input projection/normalization and one or two late blocks, optionally speaker-embedding FiLM/LoRA. Losses: MR-STFT/log-mel, periodicity/F0, WavLM re-encoding content, speaker embedding, clipping guard. Do not use natural waveform as target for an MFA-linear input because that would train away the TTS target.

### 1B. Phone trajectory neural decoder
Replace per-phone linear interpolation with an identity-initialized residual conditional neural field:
```text
segment latent = Encoder(raw TTS phone trajectory, phone/context, s_tts)
z_out(r) = linear_interpolation(r) + Adapter(segment latent, r, target/source duration ratio)
```
Train on raw TTS phone segments at native lengths with feature reconstruction, cosine/Huber, velocity/acceleration, phone/CTC preservation and random downsample/corruption consistency. At inference query at natural phone-frame positions. This provides arbitrary-length feature trajectories without requiring an unavailable ideal hybrid waveform target.

### 2. Pretrained unit/PPG duration-controlled VC
Use frozen PPG/ContentVec/HuBERT units from TTS, natural oracle durations/pauses and normalized natural F0/energy, plus continuous TTS speaker embedding. Feed a pretrained Mandarin TTS/VC decoder and train only FiLM/LoRA/prosody adapter. This is the canonical VC-inspired solution, but requires a compatible pretrained decoder; do not train AutoVC/FreeVC/SpeechSplit from scratch.

### 3. Qwen3-TTS latent/token bridge
If Faster-Qwen3 exposes its 12-Hz codec/semantic tokens and decoder, keep generation in the same latent/decoder domain as raw TTS. Train only a phone-duration/token expander and prosody adapter, then decode with the frozen Qwen codec. This may avoid the LibriSpeech WavLM-HiFi-GAN domain mismatch, but feasibility depends on internal API access.

### 4. Speaker-routed gate / MoE
Only after a global adapter works, add 2–3 low-rank experts routed by continuous `s_tts`, phone context, duration ratio and reconstruction confidence. Do not create one expert per known speaker. A phone quality gate can fall back from TTS trajectory to a canonical phone trajectory when raw-TTS/vocoder confidence is low.

### 5. Full factorized codec/VC model
FACodec/SpeechTokenizer/EnCodec or SpeechSplit-like factorization is conceptually strong, but use pretrained encoders/decoders only. From-scratch training is No-Go at 44.5 min.

## Training without an ideal hybrid waveform
There is no waveform ground truth that simultaneously has raw-TTS pronunciation/timbre and natural timing. Training directly toward natural waveform loses TTS traits; training toward raw TTS preserves TTS rhythm. Use self-reconstruction and factor constraints:
- reconstruct TTS with its own content/duration/prosody/speaker;
- reconstruct natural with its own factors;
- cross-compose TTS content/speaker + natural duration/prosody;
- enforce transcript/phone, exact duration/pause, normalized F0/energy, TTS speaker embedding, naturalness and cycle consistency on cross output;
- avoid waveform L1 against either non-hybrid arm.

## Validation
- Six-fold leave-one-training-speaker-out for architecture screening, including folds holding out S0901 and S0912.
- S0765 remains speaker-disjoint final valid after choices are frozen.
- S0770 remains untouched.
- Report each speaker separately: direct-resynthesis reconstruction, CER/PER, phone-duration error, pause F1, normalized F0/energy correlation, speaker cosine, artifact/quality, then matched-face downstream under unified audio-track contract.
- Do not optimize against SyncNet or n25 speaker scores; use them only at the final downstream gate.

## Recommendation
Next action is route 0. It is the smallest experiment with the highest information value and may expose that the current prematched vocoder is simply mismatched to interpolated raw WavLM features. Depending on that result, proceed to either 1A vocoder adapter or 1B phone trajectory decoder.

## References
- [[MFA-linear 冻结 VC 重合成与下游验证]]
- [[TTS Feature Alignment and Duration Model Research]]
- kNN-VC: https://arxiv.org/abs/2305.18975
- FastSpeech 2: https://arxiv.org/abs/2006.04558
- FastPitch: https://arxiv.org/abs/2006.06873
- HuBERT: https://arxiv.org/abs/2106.07447
- WavLM: https://arxiv.org/abs/2110.13900
- ContentVec: https://github.com/auspicious3000/contentvec
- AutoVC: https://proceedings.mlr.press/v97/qian19c.html
- SpeechSplit 2.0: https://arxiv.org/abs/2203.14156
- FreeVC: https://arxiv.org/abs/2210.15418
- NaturalSpeech 3 / FACodec: https://arxiv.org/abs/2403.03100
- LoRA: https://arxiv.org/abs/2106.09685

## First learned vocoder adapter result（2026-08-14）

The full n25 gate showed regular HiFi-GAN reduces speaker-dependent prematched peaks but loses some feature fidelity. A 2048-parameter bounded affine adapter was therefore trained to make frozen prematched HiFi-GAN imitate frozen regular HiFi-GAN on native natural/raw-TTS features from four train speakers; S0765 was speaker-disjoint validation.

The first attempt correctly failed at the autograd boundary because frozen WavLM `inference_mode()` tensors were passed directly into trainable code; `.detach().clone()` fixed this. The rerun completed 250 steps (`spectral 0.02910→0.01809`, waveform `0.02458→0.01310`) with output at `runs/aishell1_mfa_linear_n25_resample_poly_20260814/vocoder_input_adapter_20260814/` and tests `4 passed`.

On S0765 MFA-linear, prematched→adapted peak/crest was `0.0782/8.858→0.0699/8.394`, while output→conditioning distance was `0.1297→0.1312`. The adapter therefore calibrates amplitude/vocoder behavior but does not solve the phone trajectory mismatch. No downstream SyncNet was run.

[decision] move to learned phone-level trajectory modeling rather than increasing vocoder-adapter capacity. #speaker #trajectory #mfa-linear
