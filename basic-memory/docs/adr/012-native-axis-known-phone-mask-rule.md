---
title: 012-native-axis-known-phone-mask-rule
type: decision
permalink: tts-exp/docs/adr/012-native-axis-known-phone-mask-rule
status: open
project: tts-exp
decided: 2026-10-13
---

# 012-native-axis-known-phone-mask-rule

## Decision
For the native-axis pilot, SEP and STAB may operate only on phone intervals whose MFA label is known. Intervals labeled `spn`, empty, or otherwise unknown stay byte-for-byte at the RMS identity input and are excluded from the manipulation mask. Every row must report known-phone coverage and the excluded interval list. No unknown interval may be assigned a guessed phone label.

## Context
CPU MFA completed for the two unseen target clips and produced complete TextGrid coverage, but Mandarin OOV regions are explicitly labeled `spn`. S0044 has 31 known non-silence phones out of 34 phone intervals; S0045 has 21 out of 29. The original all-or-nothing parser would block both rows even though most intervals are valid.

## Rationale
Keeping `spn` untouched preserves the fixed-content and fixed-clock design and avoids inventing phoneme boundaries. A partial mask makes the requested causal probe executable while leaving the unknown regions auditable. Coverage is a manipulation check, not evidence that unknown regions were correctly classified.

## Alternatives considered
- Guess phone labels from neighboring characters: rejected because it fabricates the target coordinate.
- Reuse historical numeric TextGrids or phone prototypes: rejected because target IDs and speaker provenance do not match.
- Drop both clips and select new speakers: rejected because it would discard a valid unseen cohort and delay the bounded pilot.

## Consequences
SEP and STAB remain unavailable for any row whose known-phone coverage falls below a predeclared threshold of 70 percent of non-silence phone time. For S0044 and S0045 this threshold must be checked from the qualification manifest before a factor launch. ENV is unaffected. Results must distinguish known-phone effects from whole-utterance effects.

## Affected work
- `basic-memory/Experiments/native-axis-factor-pilot-20261013.md`
- `basic-memory/Tasks/native-axis-factor-pilot-20261013.md`
- `basic-memory/Research/research-map.md`
- `HANDOFF.md`
\n\n## Cohort amendment\nThe first scorer qualification clips were S0044 and S0045. S0044 passes the 70 percent known-phone time gate; S0045 has 50 percent known-phone coverage and is retained only as a valid Wav2Lip scorer baseline. The bounded factor cohort is therefore S0044 plus BAC009S0052W0386 (speaker S0052), whose CPU MFA qualification has 84.905660 percent known-phone coverage of its non-silence phone region. S0052's one spn interval remains excluded. This preserves the two-speaker budget while avoiding a hidden eligibility failure.