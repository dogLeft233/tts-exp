# V3 zero-parameter gradient supplement

**State:** prepared, source-pinned, not dispatched. This is an engineering check, not a model result.

## Frozen source bindings

| Source | Bytes | SHA-256 |
|---|---:|---|
| Worker `scripts/experiments/cross_tfg_audio_head_20261003/qualify_shared_recipient_transition_head_v3_zero_theta_gradient_attempt01.py` | 27,243 | `6193181ccb890fe1f9b49d97e7ab2960499d511816693d3de48669eca4a3c536` |
| Config `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_gradient_attempt01/qualification_config.json` | 5,396 | `594f5e4ff86e373d99a69815596148b72e7bdf58f9b9d6c2d65c37e4ebdbd1dd` |
| V3 head `scripts/audio_enhancement/shared_recipient_transition_head_v3.py` | 32,233 | `77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87` |
| NumPy baseline `scripts/audio_enhancement/baselines.py` | 15,123 | `7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2` |
| Frozen engineering config `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02/qualification_config.json` | 10,721 | `0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5` |
| Existing qualification worker reference `scripts/experiments/cross_tfg_audio_head_20261003/qualify_shared_recipient_transition_head_v3_cpu_attempt02.py` | 55,636 | `4f5a1d272eb57fa331241a2063bbb25e8c91b28b61d826487ec80a24d1cf9eb3` |
| Package initializer `scripts/audio_enhancement/__init__.py` | 1,503 | `db520f7b04343682e6df4ba53b22503de806287e70f2b8e41e84f16912d361fc` |
| Decoder `scripts/audio_enhancement/audio_io.py` | 14,294 | `7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004` |
| Package import `scripts/audio_enhancement/pipeline.py` | 5,975 | `b40ddce9b09ea42d56945eb66acca3326be6784acb6c1e6ab134b1b2899fb6c3` |
| Package import `scripts/audio_enhancement/overdub.py` | 12,002 | `0febe6f2b43ed4fb089bdf36e2e208cc1cd61681d8e0ebfd7f59cd56dd00b4b6` |

The contract/config/worker are new attempt01 files. Earlier V3 CPU qualification attempts remain unchanged. The runtime probe reuses the synthetic waveform, RMS baseline, objective, finite-difference steps, and tolerances from the frozen attempt02 qualification config without reading the 40-row fit manifest or any WAV.

## Procedure

The worker uses the frozen CPU runtime: CPython 3.10.12 at `/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python`, NumPy 1.26.4, Torch 2.5.1+cu124, CPU threads set to one, `PYTHONNOUSERSITE=1`, and `CUDA_VISIBLE_DEVICES=''`. It verifies every listed source hash before importing the V3 head. It rejects an existing output directory.

At exact float32 zero coefficients, it computes the existing synthetic probe's canonical NumPy RMS baseline and the V3 residual path, then saves the output/baseline digests, sample-count and bitwise-equality checks, finite autograd gradient matrix, nine coordinate central differences, and the normalized directional finite difference. The comparison steps and tolerances come directly from the attempt02 config. Zero/near-zero gradient entries are recorded, not treated as failure. It also records the raw/projected masks, gates, valid/active support, band basis, source/baseline/output 40 ms power, and sample-index frame centers. It performs the frozen silence-baseline equality check.

The NPZ does not contain input or output waveform samples. Outputs are limited to the NPZ, `result.json`, and `result_manifest.json`; the configured caps are 16 MiB uncompressed arrays, 8 MiB NPZ, 1 MiB JSON, and 20 MiB total. No frontend, generator, video, SyncNet, optimizer, fit-audio, development, confirmation, or API work is included.

## Packet-preparation command template

Before any future dispatch, the local packet builder should upload only the source closure above at the same repository-relative paths, verify every byte/hash from the config, and create the declared output directory only by running the worker. The future command uses the literal pinned interpreter and explicit fresh paths:

```sh
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python -B \
  "$REPO_ROOT/scripts/experiments/cross_tfg_audio_head_20261003/qualify_shared_recipient_transition_head_v3_zero_theta_gradient_attempt01.py" \
  --repo-root "$REPO_ROOT" \
  --config "$REPO_ROOT/runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_gradient_attempt01/qualification_config.json" \
  --output-dir "$REPO_ROOT/runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_gradient_attempt01/output"
```

`REPO_ROOT` must be bound to the actual staged repository path by the packet's pinned preflight; it is not inferred from the current shell. This command is a preparation template only and has not been run.

## Interpretation boundary

Finite-difference agreement checks numerical consistency of this synthetic objective around zero. It does not show that any parameter improves audio, mouth motion, SyncNet, or speech content. Positive-minus-negative relative magnitude flux also contains local total-magnitude change; it is not an articulation-only measure. The mask's frequency centering and final utterance RMS do not guarantee preservation of physical 40 ms frame power.
