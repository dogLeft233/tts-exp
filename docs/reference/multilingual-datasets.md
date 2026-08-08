# Multilingual Audio Acquisition

## Target

Collect at least 20 text-labeled audio utterances for each of the ten languages
listed in the Qwen3-TTS report: Chinese, English, German, Italian, Portuguese,
Spanish, Japanese, Korean, French, and Russian.

Ditto does not publish a complete language support list. Treat every
non-English language as experimentally supported only after the local Ditto
pipeline succeeds.

## Evidence Boundary

The Qwen3-TTS report (`arXiv:2601.15621`, 22 January 2026) says the model was
trained on over five million hours across ten languages, but does not disclose a
complete training-corpus manifest. Ditto (`arXiv:2411.19509v3`) reports about 50
hours of broadcast video from 330 identities and evaluates on `Talk9` and
`HDTF100`; it does not publish a complete training manifest either.

Therefore "unseen" is not provable from public information. The dataset table
uses these labels:

- **A**: explicit evidence of non-overlap, currently none.
- **B**: temporal/source separation with no identified overlap.
- **C**: known overlap or unsuitable source; do not use for the strict result.

Release dates are useful evidence but do not prove when the underlying
recordings were collected or what entered model training.

## Current Progress (2026-08-06)

| Language | Status | Dataset | License | Risk |
|---|---|---|---|---|
| Chinese | candidate identified | MSceneSpeech HF mirror | TBD | B |
| English | **downloaded** (50 MP3 + CSV) | Effect AI Scripted Speech 1.0 | CC0 | B |
| German | **downloaded** (20 MP3 + TSV) | CV Spontaneous Speech 4.0 | CC0 | B |
| Italian | **downloaded** (20 WAV + TSV) | Italian TTS - female voice | Apache-2.0 | B |
| Portuguese | blocked: MDC returned 403 | Faber 1.0 | CC0 | B |
| Spanish | **downloaded** (20 MP3 + TSV) | CV Spontaneous Speech 4.0 | CC0 | B |
| Japanese | blocked: MDC returned 403 | CV Scripted Speech 26.0 | CC0 | B |
| Korean | **downloaded** (20 MP3 + TSV) | CV Scripted Speech 26.0 | CC0 | B |
| French | blocked: MDC returned 403 | CV Spontaneous Speech 4.0 | CC0 | B |
| Russian | blocked: MDC returned 403 | CV Spontaneous Speech 4.0 | CC0 | B |

The current MDC artifact set contains 50 English samples plus 20 samples for each of German, Italian, Spanish, and Korean (130 samples total).
Audio is under `data/mdc_tts/audio/<language>/`; the manifest is at
`data/mdc_tts/manifest.json`. Italian WEBM clips were converted to 24 kHz mono
WAV. Only selected audio and provenance metadata are kept in the repository;
the full archives remain in the MDC download cache (`~/.mozdata/datasets/`).

Chinese remains a candidate-only HF source. Portuguese, Japanese, French, and
Russian returned HTTP 403 with the message that terms must be accepted before
downloading; their exact errors and retry instructions are recorded in the
manifest. The candidate table below documents the full source details.

## Candidate Matrix

MDC downloads require an API key (`MDC_API_KEY`) and acceptance of each
dataset's terms via the dataset page on the MDC website.

| Language | Candidate | Evidence and size | License / constraint | Risk |
|---|---|---|---|---|
| Chinese | [MSceneSpeech HF mirror](https://huggingface.co/datasets/malaysia-ai/MsceneSpeech) | 5,106 paired rows, 491 MB; original project reports about 15 h, four scenes, `.wav`/`.txt` structure | Original license must be verified; HF mirror is not proof of licensing | B |
| English | [Effect AI Scripted Speech 1.0](https://mozilladatacollective.com/datasets/cmkfm9fbl00nto0070sdcrak2) | 10.2 h, 11,000+ rows, CSV/MP3, released 2026-01-15 | CC0; do not identify speakers | B |
| German | [Common Voice Spontaneous Speech 4.0](https://mozilladatacollective.com/datasets/cmqi29u41004umf07amojxtop) | 319 clips, 65 validated, 33.36 MB, released 2026-06-17 | CC0; do not re-host or re-share | B |
| Italian | [Italian TTS - female voice](https://mozilladatacollective.com/datasets/cmoiuyem401j5mr07s0jx8rqr) | About 10 h, paired WEBM/TSV, 680.16 MB, released 2026-04-28 | Apache-2.0; one anonymous speaker; prompted public-domain texts | B |
| Portuguese | [Faber 1.0](https://mozilladatacollective.com/datasets/cmiupazq801ftnv079bo1zu4h) | About 1.5 h, 30.98 MB, TTS-oriented WEBM/metadata, released 2025-12-06 | CC0; recording/text were not post-validated | B |
| Spanish | [Common Voice Spontaneous Speech 4.0](https://mozilladatacollective.com/datasets/cmqi28y2v004imf076oh7e5zs) | 403 clips, 37 validated, 27.59 MB, released 2026-06-17 | CC0; do not re-host or re-share; only 37 validated rows | B |
| Japanese | [Common Voice Scripted Speech 26.0](https://mozilladatacollective.com/datasets/cmqim4lxy00tunr07cjkcupeg) | Released 2026-06-17; paired `sentence`/MP3 metadata; full archive is large | CC0; do not re-host or re-share | B |
| Korean | [Common Voice Scripted Speech 26.0](https://mozilladatacollective.com/datasets/cmqi922c5001pnq07dmj0oypw) | 7,178 clips, 1,758 validated, 208.26 MB, 210 speakers | CC0; do not re-host or re-share | B |
| French | [Common Voice Spontaneous Speech 4.0](https://mozilladatacollective.com/datasets/cmqi2a71t005uo507j5vzspfu) | 667 clips, 166 validated, 42.30 MB, released 2026-06-17 | CC0; do not re-host or re-share | B |
| Russian | [Common Voice Spontaneous Speech 4.0](https://mozilladatacollective.com/datasets/cmqi2c2eu0062o5075atr17rs) | 552 clips, 442 validated, 70.57 MB, released 2026-06-17 | CC0; do not re-host or re-share | B |

The MDC/Common Voice archives must not be committed or mirrored into this
repository. Keep only the local experiment artifacts and source-row metadata
needed for reproducibility.

## Rejected Existing Sources

The existing local manifest in `data/multilingual_tts/manifest.json` contains
20 rows for each of nine languages, but it is a legacy baseline rather than a
strict unseen set:

- `CSS10-Multilingual-LJSpeech` is derived from CSS10/LibriVox and is **C**.
- `CML-TTS` is based on Multilingual LibriSpeech (MLS) and is **C**.
- `KSS` is not an identified overlap, but its 2018 release and opaque model
  training history make it only **B**, not A.
- HDTF audio is explicitly part of Ditto's evaluation ecosystem and must not be
  used as the unseen English source.

Do not delete these files; preserve them as the old comparison baseline and do
not describe them as unseen in results.

## Reproducible Commands

### Legacy Hugging Face baseline (CSS10 / CML-TTS / KSS)

```bash
python scripts/acquire_multilingual_tts.py --dry-run
```

Writes `data/multilingual_tts/manifest.json` and ignored audio under
`data/multilingual_tts/audio/`. CSS10/CML-TTS rows are legacy data with known
overlap risk.

### MDC acquisition (Effect AI, CV Spontaneous/Scripted, Italian TTS, Faber)

Requires `MDC_API_KEY` exported at runtime and each dataset's terms accepted on
the MDC website before download.

```bash
export MDC_API_KEY='provided-at-runtime-only'

# Download one archive; existing files in ~/.mozdata/datasets/ are reused.
python3 -c "from datacollective import download_dataset; download_dataset('DATASET_ID')"

unset MDC_API_KEY
```

The selected artifacts and per-language status are recorded in
`data/mdc_tts/manifest.json`. Raw MDC archives must remain outside the
repository.

### HDTF audio (NAS)

```bash
export NAS_DHG_MM_PASSWORD='provided-at-runtime-only'
python scripts/download_hdtf_english_audio.py
unset NAS_DHG_MM_PASSWORD
```

The HDTF command writes `data/hdtf_audio_en/manifest.json` and uses the verified
FileStation path `/dhg_mm/HDTF/_audio_raw`; it does not use SSHFS.
