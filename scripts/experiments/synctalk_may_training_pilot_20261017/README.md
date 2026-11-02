# May gain training pilot

✅ Training and all nine primary test cells completed and independently verified on October 17, 2026. One natural-video identity, one common random initialization, three equally budgeted training gains. Data preparation is a separate verified task. `protocol.json`, `common.py`, `train.py` and `supervisor.py` remain immutable execution sources. Persistent delivery is confirmed separately by the F-disk delivery manifest and cloud readback receipt.

The source data are official May, with independent temporal splits: 4150/550/1100 supervised train/validation/test frames (166/22/44 seconds). Gain is a scalar audio transformation only. The AVE encoder is frozen. Each generator performs 60000 successful FP32 AdamW updates with 4096 rays per update. Initial tensors, image order and every actual sampled ray index are checked across arms. This is a first-stage pilot, not the full official two-stage training recipe.

`supervisor.py` runs training in order 0, −6, +6 dB, then invokes the separately sealed `postprocess.py`. Rendering uses fixed final EMA weights; the test set never selects a checkpoint. All nine cells render 1100 frames and score the same four predefined 250-frame clips. FFV1 preserves RGB, PCM16 preserves the original natural scoring audio. Shared source-video face tracks fix the SyncNet crop. FP32 guard20 is primary, full is secondary; retain C, Dmin, D0, B and offsets separately.

Validation every 2000 steps uses the same 32 frames at all input gains; final evaluation adds all 550 validation frames. `analyze_plot.py` applies the predeclared late-curve plateau screen and draws learning curves and the nine-cell test matrix. A plateau screen is not proof of sufficient training data. One identity and one seed cannot support a population or best-volume claim.

## Execution and receipts

- Ubuntu work root: `/dev/shm/synctalk-may-training-pilot-20261017` (volatile).
- Ubuntu data: `/dev/shm/synctalk-gain-training-data-20261017/dataset` (volatile).
- Persistent Windows backup: `F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote`.
- `supervisor_receipt.json`: live process identity and original training-source hashes. Check it before any process action; do not relaunch a completed/active arm.
- `postprocess_admission.json`: separately frozen evaluation-source hashes, bound to the same protocol.
- `training/*/training_receipt.json`: successful updates, initial-state hash, actual ray schedule and final checkpoint identity.
- `independent_results_qa.json`: independent initialization, update counts, raw prediction, original-PCM, crop, feature-distance and metric checks.
- `analysis/analysis.json`, `learning_curves.png`, `validation_by_input_gain.png`, `test_matrix.png`, `report.md`: descriptive outputs.
- `experiment_complete.json`: evaluation completed. This alone does not prove persistent delivery; check the Windows `backup_status.json`, per-file SHA receipts and final delivery manifest.

`backup_watch.py` runs on Windows and only copies and verifies experiment files. It never starts training. Large model/video writes stay out of the nearly full Ubuntu root filesystem. Existing prepared data are already independently verified in Baidu cloud storage; F disk holds the persistent working backup.

Supplemental pixel response and viewing code were sealed before testing. `ground_truth_reference_attempt01/reference.json` is explicitly a post-hoc diagnostic: the original natural video also gives −3-frame SyncNet offsets on all four clips. It does not replace any primary result. Final results and scientific boundaries are recorded in `basic-memory/docs/experiments/50-synctalk-may-gain-training-pilot.md`; task acceptance is tracked in `basic-memory/Tasks/训练 May 电平交叉先导.md`.
