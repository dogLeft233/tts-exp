# WORLD evaluation aliases

This execution annex enumerates the already specified four-cell comparisons
before WORLD neural outcomes are read. It preserves all nine waveform arms,
both donor-to-recipient directions, and the RAW and within-WORLD score families.
Bind its file SHA256 in the GPU plan alongside the original spec, execution
bindings, physical waveform plan, and CAL QA freeze.

## Recipient identity

- `tts_to_natural`: donor T, recipient N; EX-minus-RT positive contrasts.
- `natural_to_tts`: donor N, recipient T; EX-minus-RT negative contrasts.

Each of 80 pairs has both recipients, so there are 160 recipient contexts.
RAW scoring means that recipient's original canonical, unnormalized PCM.
FIXED generation means its unsynthesized, fixed-RMS PCM. RT is that FIXED
recipient's own WORLD reconstruction. An arm X uses the physical producer's
saved waveform. No label reverses donor and recipient.

## Explicit four-cell mapping

| Family | q00 | q10 | q01 | q11 |
|---|---|---|---|---|
| RAW, each X | V_RT/A_RAW | V_X/A_RAW | V_RT/A_X | V_X/A_X |
| within-WORLD, each X | V_RT/A_RT | V_X/A_RT | V_RT/A_X | V_X/A_X |
| synthesis bridge | V_FIXED/A_RAW | V_RT/A_RAW | V_FIXED/A_RT | V_RT/A_RT |

X includes RT, EX, SP, BOTH, F0, AP, EX_EM, SP_EM and E. Thus some RT cells
are identities. They remain explicit labels and refer to shared artifacts.
The bridge measures WORLD synthesis relative to fresh unsynthesized FIXED
generation; its q10-minus-q00 is not a RAW-to-FIXED loudness intervention.

Each family uses G=q10-q00, E=q01-q00, I=q11-q10-q01+q00. The RAW and
within-WORLD families retain their own score-audio baselines and supports.
Do not combine them into one counterfactual table.

## Unique computation and retained labels

Per recipient, RAW arm cells have 26 unique V/A pairs. The within-WORLD
family adds eight V_X/A_RT pairs. The bridge adds two V_FIXED pairs.
Total: 36 unique matrices, ten unique videos, and ten audio-feature contexts.
The 9x4x2 family cells plus four bridge cells produce 76 explicit aliases.
Across the full cohort: 5,760 unique V/A matrices, 1,600 unique videos,
and 12,160 labeled cells. Store an exact alias-to-artifact map and verify
every label, including identities, without duplicating its computation.

Shared embeddings and compressed full raw/unit cross matrices permit
independent recomputation of full/valid/guard20/guard15 lag curves, C,
C_anchor, B, D and search gain. Temporary crop video/JPEG files can be
removed before sealing; sealed inputs, videos, embeddings, matrices and
alias references remain. Record actual face, model and runtime identities.
