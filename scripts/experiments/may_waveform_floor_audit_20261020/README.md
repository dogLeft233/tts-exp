# May waveform zero and AVE floor audit

CPU-only replay of report50's actual May audio and pinned AVE frontend, with
report56's fourteen post-normalization input hashes. No neural forward,
training, rendering, or SyncNet scoring.

Dependencies: Python 3.12, numpy 1.26.4, scipy 1.15.2, librosa 0.11.0,
soundfile 0.14.0. Plotting additionally uses matplotlib and a Chinese font.
Python/OS differ from the historical Ubuntu runtime; the four audio package
versions match it. The independent explicit FFT readback matches all nine
recomputed mel arrays exactly.

Use a fresh output directory; the audit refuses to overwrite an executed run.
Historical roots remain read-only. On the current Windows host:

```powershell
$python = 'C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/env/Scripts/python.exe'
$deps = 'F:/Datasets/tts-exp/may_waveform_floor_audit_runtime_20261020/deps'
$result = 'F:/Datasets/tts-exp/may_waveform_floor_audit_20261020_v1'
# For a new replay, replace $result with a fresh path.
& $python -X utf8 scripts/experiments/may_waveform_floor_audit_20261020/audit.py --repo E:/Documents/tts-exp --data F:/Datasets/tts-exp/synctalk_gain_training_20261017 --parent F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017 --postnorm F:/Datasets/tts-exp/may_postnorm_scale_20261020 --output $result --deps $deps
& $python -X utf8 scripts/experiments/may_waveform_floor_audit_20261020/verify.py --output $result --deps $deps
& $python -X utf8 scripts/experiments/may_waveform_floor_audit_20261020/plot.py --output $result --plotdeps F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps
```

`results.json` contains the zero-count, inverse-gain, provenance, and spectral
summaries. `arrays/` holds decoded waveforms, mel amplitudes, dB/mapped mel and
all six masks per split. The frequency/time CSVs locate every affected band
and frame; energy strata use within-split RMS quantiles, not speech/noise truth.

All reported spectral percentages use the complete split, including frontend
context. They do not count overlapping AVE windows repeatedly. PCM16 controls
are deliberately additional quantization diagnostics and were not used in the
historical pure-gain training data.

Formal Chinese report: `basic-memory/docs/experiments/57-may-waveform-floor-audit.md`.
The scientific run is sealed by its delivery manifest and final receipt.
