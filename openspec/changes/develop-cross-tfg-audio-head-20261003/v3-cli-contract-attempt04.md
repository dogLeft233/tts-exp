# V3 enhancement CLI — executed engineering contract

The current entrypoint is `scripts/audio_enhancement/enhance_v3_attempt02.py`,
with `scripts/audio_enhancement/configs/v3_zero_engineering_attempt02.json`.
Its complete CPU qualification and independent result readback have run.
This profile has nine zero parameters and returns the canonical RMS baseline;
it is an engineering profile. A trained shared method is still in development.

```text
python scripts/audio_enhancement/enhance_v3_attempt02.py \
  --input original.wav \
  --output enhanced.float32.wav \
  --config scripts/audio_enhancement/configs/v3_zero_engineering_attempt02.json \
  --metadata enhanced.meta.json
```

Use the enhanced WAV to drive TFG generation. Attach the exact original input
file for delivery and scoring. The command refuses existing output or metadata
files, preserves source sample zero and count, and records original/canonical
PCM hashes and clock mapping. The executed input family here is mono PCM16,
16 kHz. Other accepted formats in the parser have not been qualified by this
three-input execution.

The schema, full constructor declaration, source checks and candidate loading
rules in [contract attempt03](v3-cli-contract-attempt03.md) still apply. Its
entrypoint and unexecuted-status statements are historical. Attempt02 changes
recursive finite diagnostic serialization and uses a new pinned config;
the head, nine zero parameters and RMS transform are unchanged. A
`METHOD_CANDIDATE` still requires genuine parameter, completed training and
frozen selection records. No such records are fabricated for this profile.

## Actual engineering evidence

The executed worker used Python 3.10.12, NumPy 1.26.4, Torch 2.5.1+cu124,
CPU with one intra/inter-op thread, and no CUDA initialization. It completed
in 6.200435 s. Conservative worker-plus-maximum-child peak RSS was
897,110,016 B. Existing-output rejection returned code 2 without changing
the enhanced WAV or metadata.

| Input | Source samples | Duration, s | Cold subprocess, ms | Cold RTF | Warm head median, ms |
|---|---:|---:|---:|---:|---:|
| S0065 j0 | 51,920 | 3.245 | 1,468.928 | 0.452674 | 9.397 |
| S0069 j0 | 70,128 | 4.383 | 1,509.725 | 0.344450 | 10.220 |
| S0064 j15 | 109,918 | 6.869875 | 1,527.201 | 0.222304 | 14.919 |

Cold subprocess time includes process startup and metadata output. The
CLI's separate internal totals stop before metadata output: 1.197437,
1.230501 and 1.237689 s. Warm components each exclude one warmup and retain
three measured calls; median and nearest-rank p90 were independently
recomputed for all 18 component groups. These small, engineering-only
measurements do not estimate trained-method or TFG throughput.

Root's independent RIFF parser decoded the original PCM16 and emitted
IEEE-float32 payloads. All three output arrays equal an independently
recomputed float32 RMS baseline bit for bit. Source hashes, sample counts,
sample-zero clock, metadata-to-report joins, source code pins, output hashes,
median/p90/RTF arithmetic and source preservation passed. A middle-row
scalar RMS reduction differed by one float64 ULP between NumPy versions;
the actual waveform payload is identical. The initial strict scalar
diagnostic is preserved.

Authoritative complete result archive:
`runs/cross_tfg_audio_head_20261003/` is the run family, with the downloaded
CLI tar at `.codex/scratch/v3_enhancement_cli_cpu_result_readback_attempt02/downloads/v3_cli02_selected_outputs_attempt02.tar`.
It has 7 regular members and 1,044,480 B, SHA256
`e79e3280bb5e630775f51e142c7f8a57d99db92a0cc147aec95bcab84d0af362`.
The complete 65,173 B report SHA256 is
`d6223fbbac16801003017a313fedc32c8150ac9ef5e3a2d2a5d689c15d9e59a7`.
Independent root receipt:
`.codex/scratch/root_cli02_full_archive_qc_attempt03.json`.

The raw source bundle accompanying this contract preserves exact executed
bytes, including config newlines, because Git newline conversion can change
source pins. Extract that bundle's source tree when replaying pinned CLI
configuration. The bundle is source and engineering evidence; it contains no
trained weights, original WAVs, credentials or untouched confirmation data.
