# Executed CLI and native CPU engineering evidence

`v3-cli02-executed-engineering-source-attempt01.zip` preserves the nine exact
CLI source/configuration/qualification files, its complete actual report, two
independent receipts and contract attempt04. The external
`member_index_cli02_attempt01.json` pins all 13 members and the whole archive.
The source tree is under `src/`. Source bytes must be restored from this ZIP
for replaying the hash-bound configuration; Git newline conversion can alter
the otherwise equivalent checkout.

The native CPU qualification's raw source, configuration, manifest and root QC
are preserved together in `v3-native07-executed-bindings-attempt01.zip`, with
`member_index_native07_attempt01.json` pinning all four members. Restore the
configuration from that outer ZIP to preserve its executed byte hash across
Git newline conversion. The nested source archive is the actual deployed
`v3-native07-same-cpu-source-attempt01.zip`, copied without recompression. Its one
directory entry is not counted among the 19 source files. The config refers
to machine-local historical-fit/model paths and an immutable output namespace;
replaying it requires those assets and a fresh, explicitly rebound output.
It is not a standalone inference profile.

`native07_root_arrays_qc_attempt01.json` records independent readback of the
complete ten-member result archive and all 29 finite arrays. Direct canonical
RMS and the zero-parameter head produce bit-identical waveforms and exactly
equal 31-lag curves and 27×1024 visual embeddings on the same CPU. Nine
parameter gradients remain nonzero, and the waveform Jacobian after RMS
projection has rank nine. The earlier GPU/batch cache comparison remains
outside its fixed tolerance; the overall actual qualification status and
exit code remain `COMPLETE_WITH_CONTROL_FAILURE` and 2. This failure is
retained separately from the same-CPU pass.

These archives provide engineering replay and measured evidence. They contain
no original audio, trained shared-method checkpoint, confirmation data or
credentials. They do not establish an improvement in Sync-C, mouth quality
or cross-model transfer. The full40 training, complete development comparisons
and untouched confirmation remain the method's next stages.
