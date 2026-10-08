# Future Dev20 three-condition protocol

**Status:** source and cohort evidence are verified. The protocol is prospective and awaits ROOT review of the numeric support/content/visual admission-policy pins and an actual qualified Fit40 checkpoint. No scoring, training, ASR/API, model, GPU, or Confirm access occurred while preparing it.

## Cohort and source identity

The frozen source cohort contains twenty mono 16-kHz PCM16 utterances from speakers S0102–S0111, two fixed clips per speaker. The mirror is a third-party Hugging Face copy of AISHELL-1. All twenty WAV byte hashes, lengths, sample clocks and canonical little-endian float32 decodes match the sealed manifest; the actual remote source-stage receipt was downloaded and independently checked against all 48 outputs. No score or performance value selected these clips, and no Confirm audio or scores were opened.

The earlier completed 360-row controls use a different ten-speaker cohort (S0005, S0022, S0024, S0037, S0038, S0039, S0041, S0044, S0049, S0050). They are pinned as context and integrity evidence only. Their identity/RMS rows cannot stand in for any of the future cohort's planned rows.

## Three-condition matrix

Generate all three conditions on this new cohort: original-audio `identity`, `rms_fixed`, and one Fit40-only DirectD0/unit-direction candidate frozen after a qualified 120-step fit. Use the same three native TFGs, face, and seeds 20261003, 20261004, 20261005. This is 27 create-only model×condition×seed jobs and exactly 540 planned rows (180 per condition). The Fit candidate's method ID and checkpoint path/hash remain null until a real Fit40 terminal and ROOT freeze receipt exist. The same-theta GateDC control and post-projection checks remain Fit-only diagnostics; they do not add Dev conditions.

The primary endpoint is `native_roi_original_pcm16` Cbest, computed as median lag distance minus the minimum lag distance. The fixed prospective practical-effect rule is candidate-minus-RMS Cbest greater than zero for each of the three models and at least +0.050 Sync-C units for the equal-model mean, using the complete planned 20-clip/10-speaker/three-seed model denominators. Candidate-versus-identity is reported as a predeclared secondary contrast.

## Audio, support, and admission

Enhancement changes only the generator condition. The complete original natural waveform remains the scorer input and the sole final-mux audio. Each output must bind actual generator PCM hashes, source-clock sample count, full audio readback and exact video timestamps. No best-offset retiming is permitted.

D0, Canchor, content, visual, native support, and simultaneous-family limits must use the exact ROOT-owned independent admission policy, pinned before future effects are read. GUARD20 uses the frozen three-condition common query support, removes twenty queries from each end, then requires at least forty-one remaining queries. Duration alone never proves support eligibility. All 540 planned rows and all failures remain in the denominator; there is no replacement or complete-subset rescue.

The candidate objective, source cohort, actual receipt identities, old controls distinction, and code lineage are recorded in the self-hashed JSON. Current execution bindings and candidate checkpoint identity are intentionally unset until real qualified receipts exist.
