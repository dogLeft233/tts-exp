# Physical alignment overlay

Frozen before WORLD EVAL synthesis or model forwards, 2026-09-30 UTC.
Bind this annex, the original specification and execution-bindings annex
alongside the independent CAL QA and the actual EVAL producer source.

## Scope and reason

The original male EVAL qualification remains 79/80. Its one rejected pair,
`S0064_BAC009S0064W0121`, contains publisher text
`确实侵蚀了不少房企的净利润`. The frozen official dictionary has the
compound `侵蚀`, but lacks single-character `蚀`. The original character LAB
therefore produced an unknown phone in each condition.

For this physical experiment only, use the successful compound-tokenization
probe for this pair. Merge the adjacent LAB tokens `侵` and `蚀` into `侵蚀`
in both N and T. Concatenating the new tokens must equal the original
normalized publisher transcript exactly. Audio, transcript, dictionary,
acoustic model, MFA version and phone-label normalization remain identical.
The two new grids satisfy the original structural rules, contain zero OOV
phones, and end at the decoded audio duration (N 3.616 s; T 3.440 s).

## Input seals

Local review root:
`runs/recovered_20260930/remote/mfa_20261001_cpu/oov_repair_probe_20261001/probe_v2_review/`.

- Archive `s0064_compound_oov_probe_v2.tar.gz`, SHA256
  `15bedb7d85beec2c5bb45003997d1e305db259ffe5984109a5e880a0714810a1`.
- `files/probe_manifest.json`, raw SHA256
  `2ce7ece5eceac956f259d6a562a985deac8495502757b46f98e3a895adf377f6`.
- `files/logs/mfa_align.json`, SHA256
  `7ceda2101eed910164e0667d0c3ab82ef51566f456dca038f935f23ce90b2464`.
- N grid SHA256
  `05e5da63a3ec3290325e959d76e053dfba442d3f931290140791aecfb63346bd`.
- T grid SHA256
  `0b1e087de976336c16925e1cc4e816e1953e33d99b0bc2e4e6d9e1d44ca997f8`.
- Both new LAB files SHA256
  `bb21a656d88777dc5ded6ed3f34f84de653ca6258971caaed945631f776713ff`.
- Original qualification raw SHA256
  `acc6cacfb8f5b35c13163157c6de70f90bc6e34ae54364036935923982a85ba5`.
- Original MFA protocol raw SHA256
  `aeb753c937a8c1dfd8d6d837f4ee79d6b653a1aa6cfdb6479b2a86e154dc6f18`.
- Official dictionary SHA256
  `f790bbd70b398d20b9e510dafc4c94ee76b52e9bf8597b4ed188cc9ff7ba23a0`.

Remote probe root, relative to the portable research root:
`inputs/hf_selected/mfa_oov_repair_20261001_v1/probe_v2/`.
Exact grid and audio paths/hashes are authoritative in the sealed manifest.
Verify the original and staged audio hashes agree before applying the overlay.

## Execution and inference

Create a new optional-overlay EVAL producer version. Do not change or
overwrite the already executed CAL source, waveforms, grids or QA seals.
Record that CAL waveform mathematics were executed with the old source SHA,
and EVAL input selection was executed with the new source SHA plus this annex.

Only this sample may substitute its paired LAB and TextGrid inputs. Verify
the other 158 grids against the original seals. Use the same exact-phone,
within-word matching algorithm on the new paired token lists; do not alter
phone identity, tone handling, LCS tie rules or transfer mathematics.

Keep the complete intended cohort at 80 clips / 40 two-clip speakers.
Report physical alignment coverage after overlay separately from original
coverage, and preserve the original short-T guard20 unavailability. Never
change the four primary criteria or promote guard15 to the primary endpoint.

The source-axis visual primary and its availability subset continue using
the original qualification and grids. Any visual use of the repaired pair
must be a separately labeled sensitivity analysis.
