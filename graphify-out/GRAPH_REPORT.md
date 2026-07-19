# Graph Report - .  (2026-07-18)

## Corpus Check
- 224 files · ~443,470 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1297 nodes · 2238 edges · 93 communities (86 shown, 7 thin omitted)
- Extraction: 73% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 310 edges (avg confidence: 0.72)
- Token cost: 35 input · 3,250 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Feature TFG Linking|Feature TFG Linking]]
- [[_COMMUNITY_GxE Matrix|GxE Matrix]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Wav2Sem Paper|Wav2Sem Paper]]
- [[_COMMUNITY_Stability Tests|Stability Tests]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Wav2Sem Report|Wav2Sem Report]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Acoustic Interventions|Acoustic Interventions]]
- [[_COMMUNITY_Cross-TFG Validation|Cross-TFG Validation]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Stability Intervention|Stability Intervention]]
- [[_COMMUNITY_Tests Test Identity|Tests Test Identity]]
- [[_COMMUNITY_Tests Test Dose|Tests Test Dose]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_SSL Embeddings|SSL Embeddings]]
- [[_COMMUNITY_Acoustic Interventions|Acoustic Interventions]]
- [[_COMMUNITY_Acoustic Interventions|Acoustic Interventions]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Concept Aishell1 Sample9|Concept Aishell1 Sample9]]
- [[_COMMUNITY_SSL Embeddings|SSL Embeddings]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_Acoustic Interventions|Acoustic Interventions]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Concept Boundary Sharpness|Concept Boundary Sharpness]]
- [[_COMMUNITY_Dose Response|Dose Response]]
- [[_COMMUNITY_Concept Code Patch|Concept Code Patch]]
- [[_COMMUNITY_Concept No Loudnorm|Concept No Loudnorm]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_Acoustic Interventions|Acoustic Interventions]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_SSL Embeddings|SSL Embeddings]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Tests Test Ssl|Tests Test Ssl]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_SSL Embeddings|SSL Embeddings]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Identity Correction|Identity Correction]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_Identity Control|Identity Control]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_SSL Embeddings|SSL Embeddings]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_Feature Separability|Feature Separability]]
- [[_COMMUNITY_Tests Test Cross|Tests Test Cross]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Tests Test Audio|Tests Test Audio]]
- [[_COMMUNITY_Concept Adr Conflict|Concept Adr Conflict]]
- [[_COMMUNITY_TTS Providers|TTS Providers]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Tests Test Audio|Tests Test Audio]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Experiment Scripts|Experiment Scripts]]
- [[_COMMUNITY_Concept Triage Labels|Concept Triage Labels]]
- [[_COMMUNITY_Concept Intra Class|Concept Intra Class]]
- [[_COMMUNITY_Configs Nfe Step|Configs Nfe Step]]
- [[_COMMUNITY_Doc Agents Issue|Doc Agents Issue]]

## God Nodes (most connected - your core abstractions)
1. `TTSProvider` - 43 edges
2. `TTSResult` - 38 edges
3. `_make_clustered_embeddings()` - 20 edges
4. `FishAudioProvider` - 17 edges
5. `SiliconFlowProvider` - 17 edges
6. `Remote GPU Deploy Skill` - 17 edges
7. `Experiments Summary` - 17 edges
8. `SyncNet Evaluation Metrics` - 17 edges
9. `process_all()` - 16 edges
10. `DashScopeQwen3VCProvider` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Pairwise Feature Relationships Matrix` --analyzes_metric--> `SyncNet Evaluation Metrics`  [1.0]
  figures/obs_v2_pairplot.png → docs/tfg_models/TFG_DEPLOY_SUMMARY.md
- `PCA of Audio Features Colored by SyncD` --visualizes_metric--> `SyncNet Evaluation Metrics`  [1.0]
  figures/obs_v2_pca.png → docs/tfg_models/TFG_DEPLOY_SUMMARY.md
- `Random Forest Feature Importance for SyncD` --predicts_metric--> `SyncNet Evaluation Metrics`  [1.0]
  figures/obs_v2_rf_importance.png → docs/tfg_models/TFG_DEPLOY_SUMMARY.md
- `Top-4 MFCC Features vs SyncD/SyncC Scatter` --analyzes_metric--> `SyncNet Evaluation Metrics`  [1.0]
  figures/obs_v2_scatter_top4.png → docs/tfg_models/TFG_DEPLOY_SUMMARY.md
- `Wav2Sem (CVPR 2025)` --likely_uses_evaluation--> `SyncNet Evaluation Metrics`  [0.55]
  papers/Wav2Sem_CVPR2025.pdf → docs/tfg_models/TFG_DEPLOY_SUMMARY.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **TTS Lip-Sync Evaluation Pipeline** — concept_evaluation_pipeline, concept_ditto_animation_pipeline, concept_syncnet_evaluation, dataset_aishell1_test [INFERRED]
- **Model-Server Compatibility** — concept_model_compatibility_matrix, server_gtx1080ti, server_rtx4080, model_wav2lip, model_musetalk, model_latentsync, model_joyvasa, model_v_express, model_echomimic, model_aniportrait, model_hallo2 [INFERRED]
- **SyncD Feature Prediction Pipeline** — metric_syncd, concept_audio_feature_correlation, analysis_lasso_regression_syncd, analysis_random_forest_syncd, dataset_scored_rows [INFERRED]
- **Wav2Sem Analysis Pipeline (Phase 1)** — script_14_prepare_alignment, script_15_extract_ssl, script_16_feature_separability, script_17_link_features, script_18_render_report, encoder_hubert_base, dataset_aishell1, doc_wav2sem_analysis_report [INFERRED]
- **Phase 2 Causal Intervention Pipeline** — script_20_causal_interventions, script_22_dose_response, script_23_identity_control, script_24_identity_corrected, script_25_stability_perturbation, concept_gxe_matrix, concept_ditto_nondeterminism, concept_gxe_codaptation, doc_phase2_execution [INFERRED]
- **TFG Model Comparison Matrix** — model_ditto_talkinghead, model_wav2lip, model_musetalk, model_latentsync, model_joyvasa, model_echomimic, model_aniportrait, model_hallo2, concept_syncnet, model_faster_qwen3, dataset_aishell1 [INFERRED]
- **Lip-Sync Model Comparison Pipeline** — tfg_models_wav2lip, tfg_models_musetalk, tfg_models_latentsync, tfg_models_joyvasa, tfg_models_vexpress, tfg_models_syncnet_metric, tfg_models_syncnet_rankings [INFERRED]
- **Audio Feature Analysis for Lip-Sync Prediction** — figures_nat_vs_tts_features, figures_obs_v2_corr_heatmap, figures_obs_v2_pairplot, figures_obs_v2_pca, figures_obs_v2_rf_importance, figures_obs_v2_scatter_top4 [INFERRED]
- **Round 1 TTS Provider Comparison Experiment** — r1_configs_six_providers, r1_configs_faster_qwen3, r1_configs_dashscope_vc, r1_configs_dashscope_flash, r1_configs_dashscope_cv3, r1_configs_fish_audio, r1_configs_siliconflow_cv2, config_yaml_global_config, tts_tfg_experiment_qwen3_tts [INFERRED]
- **TTS Synthesis → ASR Transcription Evaluation Loop** — configs_f5_tts_provider, configs_f5tts_base_model, transcripts_tts_asr_evaluation_pipeline, transcripts_1_doc, transcripts_1_1_doc, transcripts_2_doc, transcripts_2_2_doc, transcripts_3_doc, transcripts_3_3_doc, transcripts_4_doc, transcripts_4_4_doc, transcripts_5_doc, transcripts_5_5_doc, transcripts_6_doc, transcripts_6_6_doc, transcripts_7_doc, transcripts_10_doc, transcripts_10_10_doc, transcripts_11_doc, transcripts_11_11_doc, transcripts_12_doc, transcripts_12_12_doc, transcripts_13_doc, transcripts_13_13_doc [HIGH 0.85]
- **Chinese Housing Policy Discourse Cluster** — transcripts_chinese_real_estate_market, transcripts_government_housing_subsidy_500, transcripts_third_tier_city_oversupply, transcripts_green_efficient_housing_demand, transcripts_qin_hong_researcher, transcripts_market_decisive_role, transcripts_market_growth_transition [HIGH 0.80]
- **F5-TTS Model Configuration Parameters** — configs_r2_f5_tts_yaml, configs_f5_tts_provider, configs_f5tts_base_model, configs_nfe_step_32, configs_cfg_strength_2, configs_cuda_device, configs_non_autoregressive_flow_matching [HIGH 0.95]
- **hyperedge_duplicate_transcript_8** —  [INFERRED]
- **hyperedge_duplicate_transcript_9** —  [INFERRED]
- **hyperedge_economic_policy_theme** —  [INFERRED]

## Communities (93 total, 7 thin omitted)

### Community 0 - "Feature TFG Linking"
Cohesion: 0.06
Nodes (49): _build_feature_delta_df(), _build_sync_df(), _get_per_sample_feature_deltas(), _load_separability_metrics(), _load_syncnet_for_run(), main(), _multivariate_analysis(), _parse_args() (+41 more)

### Community 1 - "GxE Matrix"
Cohesion: 0.06
Nodes (31): build_ffmpeg_command(), classify_sample(), complete_matrix_samples(), condition_for_axis(), decompose(), find_existing_results(), find_existing_videos(), gxe_key() (+23 more)

### Community 2 - "TTS Providers"
Cohesion: 0.06
Nodes (52): LASSO Regression for SyncD, PCA Feature Space Analysis, Random Forest for SyncD, Audio Feature-SyncNet Correlation, Audio Transform Controlled Experiments, Ditto Animation Pipeline, Emotion-Aware TTS Synthesis, concept_emotion_vector (+44 more)

### Community 3 - "Wav2Sem Paper"
Cohesion: 0.06
Nodes (42): SyncNet Evaluation Configuration, Report Generation Configuration, TTS Provider Abstraction, Case Study: Per-Sample SyncD Comparison, Natural vs TTS Feature Distribution Comparison, Spectrogram Comparison: Natural vs f5_tts, Audio Feature-SyncD/SyncC Correlation Heatmap, Pairwise Feature Relationships Matrix (+34 more)

### Community 4 - "Stability Tests"
Cohesion: 0.06
Nodes (23): ndarray, _random_source(), Unit tests for scripts/25_stability_perturbation_syncnet.py.  These tests exerci, If a sample is missing from the manifest AND no audio_duration_s is     provided, Tiny differentiable stand-in for HuBERT: produces (1, T, D)     hidden states fr, For the stub model, raise_cost should *raise* the loss value., The Intervention.run_intervention_pipeline calls transform(y_tts, y_nat, sr, sid, post_hoc_verify should return a dict with the expected fields, even     on near- (+15 more)

### Community 5 - "TTS Providers"
Cohesion: 0.09
Nodes (34): check_audio(), check_images(), main(), Path, Return (width, height) from a PNG's IHDR chunk, or None on failure.      PNG spe, read_png_size(), load_api_key(), load_config() (+26 more)

### Community 6 - "Wav2Sem Report"
Cohesion: 0.12
Nodes (36): Figure, _generate_boundary_sharpness_curve(), _generate_candidate_ranking(), _generate_confusables_detail(), _generate_feature_vs_sync_scatter(), _generate_metrics_forest_plot(), _generate_probe_confusion(), _generate_report() (+28 more)

### Community 7 - "TTS Providers"
Cohesion: 0.12
Nodes (21): ABC, Any, Path, Any, Path, TTSResult, Abstract TTS provider interface.  All backends return raw audio as a numpy array, Output of a single TTS generation. (+13 more)

### Community 8 - "Acoustic Interventions"
Cohesion: 0.12
Nodes (30): compute_lufs(), compute_spectral_centroid(), evaluate_pilot(), InterventionResult, load_sample_pairs(), main(), _mean_lufs(), ndarray (+22 more)

### Community 9 - "Cross-TFG Validation"
Cohesion: 0.12
Nodes (28): aggregate_per_sample_deltas(), build_mechanism_results(), build_summary_table(), check_latentsync_neutral(), classify_consistency(), compute_aggregate_deltas(), derive_ditto_sign_by_mechanism(), load_ditto_data() (+20 more)

### Community 10 - "TTS Providers"
Cohesion: 0.12
Nodes (10): _identity_record(), _intervention_record(), _make_synthetic_files(), Path, Unit tests for scripts/24_identity_corrected_analysis.py.  These tests cover the, Build tiny dose_response.json + identity_control.json files., TestClassify, TestComputeResiduals (+2 more)

### Community 11 - "Stability Intervention"
Cohesion: 0.11
Nodes (26): apply_uniform_fallback_boundaries(), _build_interventions(), _ensure_hubert_model(), _load_manifest_samples(), _load_script_module(), load_token_boundaries(), pgd_perturb(), pgd_stability_transform() (+18 more)

### Community 12 - "Tests Test Identity"
Cohesion: 0.10
Nodes (9): Unit tests for scripts/23_identity_control_syncnet.py.  These tests cover the id, Sanity-check the interpretation thresholds encoded in main().      The threshold, Identity must not peek at y_nat to decide the output., Script 23 must reuse Intervention from script 22, not redefine it., Identity intervention must be a valid Intervention instance., TestAggregationWithIdentity, TestIdentityIntervention, TestScript22MachineryImport (+1 more)

### Community 13 - "Tests Test Dose"
Cohesion: 0.10
Nodes (8): Unit tests for scripts/22_dose_response_syncnet.py.  These tests cover the inter, Registry names must align with intervention_results.json keys., TTS-source interventions must baseline against tts_raw and vice versa., TestAggregateDeltas, TestBaselineLookup, TestInterventionRegistry, TestPipelineSkipFlags, TestSaveTransformedAudio

### Community 14 - "Experiment Scripts"
Cohesion: 0.13
Nodes (21): apply_eq_band(), apply_gain(), apply_lowpass(), collect_syncnet(), find_python(), get_ditto_bin_path(), list_ditto_bin(), main() (+13 more)

### Community 15 - "SSL Embeddings"
Cohesion: 0.15
Nodes (20): _auto_detect_device(), _load_audio_mono(), _load_manifest(), main(), _parse_args(), _parse_comma_list(), _parse_model_keys(), process_all() (+12 more)

### Community 16 - "Acoustic Interventions"
Cohesion: 0.17
Nodes (15): measure(), Dispatch wrapper for metric computation., _make_loud_tone(), _make_pinkish_noise(), _make_quiet_tone(), _make_sine(), _make_white_noise(), ndarray (+7 more)

### Community 17 - "Acoustic Interventions"
Cohesion: 0.14
Nodes (14): apply_spectral_tilt_match(), compute_spectral_tilt(), Compute spectral tilt as the slope of log-mag vs log-freq (linear regression)., Match the spectral tilt of *y_src* to *target_slope*.      Applies an inverse fi, _build_interventions(), Define the 8 interventions mirroring script 20.      For TTS-source intervention, White noise has near-zero spectral tilt (flat spectrum)., Low-pass filtered noise has negative tilt (high-frequency roll-off). (+6 more)

### Community 18 - "Experiment Scripts"
Cohesion: 0.15
Nodes (20): Transcript 13_13: Market Growth Transition (paired), Transcript 13: Market Growth Transition, Transcript 2_2: Market Decisive Role (paired), Transcript 2: Market Decisive Role, Transcript 3_3: City Oversupply Risk (paired), Transcript 3: City Oversupply Risk, Transcript 4_4: Qin Hong Statement (paired), Transcript 4: Qin Hong Statement (+12 more)

### Community 19 - "Concept Aishell1 Sample9"
Cohesion: 0.13
Nodes (19): concept_aishell1_sample9_exclusion, concept_benjamini_hochberg, concept_cohens_d, concept_confusable_pairs, concept_mandarin_viseme, concept_mfa_alignment, concept_paired_permutation_test, concept_tsne_fallback (+11 more)

### Community 20 - "SSL Embeddings"
Cohesion: 0.15
Nodes (13): _frame_to_token_pooling(), Mean-pool frame embeddings into per-token vectors.      For each token span, col, _make_fake_embeddings(), ndarray, Token exactly at frame boundaries should include boundary frames., Token covering exactly one frame., Return deterministic fake embeddings for testing., Tests for _frame_to_token_pooling. (+5 more)

### Community 21 - "Feature Separability"
Cohesion: 0.19
Nodes (18): _compare_natural_vs_tts(), _compute_metrics_for_pool(), _compute_metrics_per_sample(), _discover_embedding_files(), _gather_layer_data(), main(), _parse_args(), _parse_comma_ints() (+10 more)

### Community 22 - "Acoustic Interventions"
Cohesion: 0.12
Nodes (12): apply_lufs_match(), Apply linear gain so *y_src* reaches *target_lufs*., LUFS matching moves audio toward target LUFS., Increase test case with 'increase' expected -> passes., Decrease test case with 'increase' expected -> fails., Decrease test case with 'decrease' expected -> passes., Verify works with energy_env_std metric., TTS->NAT and NAT->TTS LUFS adjustments should have opposite deltas. (+4 more)

### Community 23 - "Experiment Scripts"
Cohesion: 0.13
Nodes (12): A single token (syllable) with alignment and viseme information.      Attributes, TokenSpan, Unit tests for scripts/14_prepare_mandarin_alignment.py., Smoke test for build_manifest in check-only mode., Verify the YAML file loads and has correct structure., Verify TokenSpan from tfg_feature_common is compatible., Tests for MFA binary presence check., TestBuildManifestCheckOnly (+4 more)

### Community 24 - "Concept Boundary Sharpness"
Cohesion: 0.21
Nodes (18): concept_boundary_sharpness, concept_ditto_nondeterminism, concept_identity_control, concept_pgd_perturbation, concept_pipeline_noise_verdict, concept_segment_stability, concept_silhouette_score, doc_phase2_execution (+10 more)

### Community 25 - "Dose Response"
Cohesion: 0.19
Nodes (17): aggregate_deltas(), Intervention, load_baseline_results(), _load_script_module(), main(), ndarray, Path, Return {condition: {sample_id: {sync_c, sync_d, av_offset}}}.      Mirrors scrip (+9 more)

### Community 26 - "Concept Code Patch"
Cohesion: 0.17
Nodes (17): Common Code Patch Patterns, Conda Environment Management, Lip-Sync Model Compatibility Matrix, Remote GPU Deployment Workflow, SyncNet Evaluation, Weight Download Strategy, doc_readme_aniportrait, doc_readme_hallo2 (+9 more)

### Community 27 - "Concept No Loudnorm"
Cohesion: 0.23
Nodes (17): concept_no_loudnorm_in_tfg, concept_syncnet, doc_readme_echomimic, doc_readme_latentsync, doc_readme_musetalk, doc_tfg_deployment_design, doc_tfg_deployment_plan, doc_tfg_progress (+9 more)

### Community 28 - "TTS Providers"
Cohesion: 0.21
Nodes (8): Any, ndarray, Path, TTSResult, DashScopeQwen3VCProvider, POST /services/aigc/multimodal-generation/generation → (audio_np, sr)., Get or register a voice_id for a sample, caching by audio hash., POST /services/audio/tts/customization → voice_id.

### Community 29 - "TTS Providers"
Cohesion: 0.23
Nodes (7): Any, ndarray, Path, TTSResult, FishAudioProvider, POST /v1/tts → (audio_np, sample_rate)., POST /model with multipart/form-data → model_id.

### Community 30 - "TTS Providers"
Cohesion: 0.23
Nodes (7): Any, ndarray, Path, TTSResult, POST /v1/audio/speech → (audio_np, sample_rate)., POST /v1/uploads/audio/voice → uri., SiliconFlowProvider

### Community 31 - "Experiment Scripts"
Cohesion: 0.22
Nodes (14): build_summary(), find_syncnet_results(), _fmt(), generate_figures(), generate_report(), main(), paired_stats(), Path (+6 more)

### Community 32 - "Experiment Scripts"
Cohesion: 0.21
Nodes (6): load_viseme_map(), pinyin_to_viseme(), Load and cache the viseme map from YAML., Return the viseme label for a pinyin (initial, final) pair.      Prefers the ini, Tests for mandarin_viseme_map.yaml loading and classification., TestVisemeMap

### Community 33 - "Experiment Scripts"
Cohesion: 0.16
Nodes (14): apply_linear_lufs_gain(), bootstrap_paired_ci(), ensure_output_dirs(), load_audio_mono(), ndarray, Path, Shared data structures and utilities for TFG audio feature analysis.  Provides d, Apply a pure linear gain to the waveform so it reaches *target_lufs*.      No dy (+6 more)

### Community 34 - "TTS Providers"
Cohesion: 0.23
Nodes (14): concept_bootstrap_ci, concept_dynamic_range, concept_gxe_codaptation, concept_gxe_matrix, concept_lufs_normalization, concept_spectral_tilt, concept_stability_inconclusive, doc_syncnet_limitation (+6 more)

### Community 35 - "Experiment Scripts"
Cohesion: 0.25
Nodes (13): build_manifest(), _get_audio_duration(), _load_transcript(), main(), _mfa_available(), _parse_args(), Namespace, Path (+5 more)

### Community 36 - "Feature Separability"
Cohesion: 0.18
Nodes (7): Silhouette score using cosine distance.      Falls back to sklearn if available,, _silhouette_cosine(), Unit tests for scripts/16_feature_separability.py.  All tests use synthetic data, Smoke test using synthetic embedding files., TestEmptyInput, TestProcessAllSmoke, TestSilhouetteCosine

### Community 37 - "TTS Providers"
Cohesion: 0.18
Nodes (9): fdr_bh_correction(), Benjamini-Hochberg FDR correction for multiple hypothesis testing.      Paramete, 100 independent null correlations → FDR < 0.1 rate., TestFdr, Unit tests for scripts/tfg_feature_common.py., test_fdr_bh_empty(), test_fdr_bh_extreme(), test_fdr_bh_identical_pvalues() (+1 more)

### Community 38 - "Feature Separability"
Cohesion: 0.23
Nodes (8): inter_class_separation(), linear_probe_cv(), Mean cosine distance between class centroids.      Parameters     ----------, Grouped cross-validation logistic regression probe.      Splits are performed on, _make_clustered_embeddings(), Generate synthetic embeddings with known class structure.      Centroids are pla, TestInterClassSeparation, TestLinearProbeCv

### Community 39 - "Acoustic Interventions"
Cohesion: 0.21
Nodes (10): apply_dynamic_transform(), compute_energy_env_std(), compute_energy_envelope(), Compute RMS energy envelope with window., Compute standard deviation of the energy envelope., Apply an exponent transform to the RMS envelope.      exponent < 1.0 compresses, Compress dynamic range, verify energy_env_std decreases., Expand dynamic range, verify energy_env_std increases. (+2 more)

### Community 40 - "Experiment Scripts"
Cohesion: 0.24
Nodes (6): Convert Chinese *text* to a list of (character, initial, final, tone).      Retu, Split a single pinyin syllable into (initial, final, tone).      Examples     --, _split_pinyin_syllable(), text_to_pinyin_tokens(), Tests for Chinese text -> pinyin conversion., TestPinyinConversion

### Community 41 - "SSL Embeddings"
Cohesion: 0.24
Nodes (10): _compute_frame_stride(), extract_frame_embeddings(), _get_embedding_dim(), load_model(), Any, Compute the CNN encoder downsampling factor from the model config.      Multipli, Load a frozen SSL model from HuggingFace.      Parameters     ----------     mod, Extract frame-level embeddings for the requested layers.      Parameters     --- (+2 more)

### Community 42 - "TTS Providers"
Cohesion: 0.21
Nodes (9): Any, Path, TTSResult, Path, TTSProvider, F5TTSProvider, get_tts_provider(), Build provider instance based on tts.provider config entry.      Args:         t (+1 more)

### Community 43 - "Experiment Scripts"
Cohesion: 0.27
Nodes (10): apply_durnorm(), apply_loudnorm(), get_baseline_duration(), load_audio(), main(), process_file(), Apply normalization and return output path., Load baseline IndexTTS2 file and return its duration in seconds. (+2 more)

### Community 44 - "Experiment Scripts"
Cohesion: 0.42
Nodes (5): parse_ctm_file(), Parse a CTM alignment file.      Returns (tokens, stats) where *tokens* is a lis, Path, Tests for MFA CTM output parsing., TestCtmParsing

### Community 45 - "Tests Test Ssl"
Cohesion: 0.18
Nodes (8): _FakeConfig, _FakeConfigNoStride, Unit tests for scripts/15_extract_ssl_embeddings.py.  Tests helper functions wit, Simulate a config missing conv_stride (fallback path)., Tests for audio input validation without model loading., 1-second 440 Hz sine wave produces correct shape and properties., Write sine wave to WAV and verify round-trip properties., TestSineWaveInput

### Community 46 - "TTS Providers"
Cohesion: 0.22
Nodes (10): CFG Strength 2.0, CUDA GPU Device Target, F5-TTS Provider, F5TTS_Base Model, Non-Autoregressive Flow-Matching TTS, R2 F5-TTS Local Config (r2_f5_tts.yaml), Transcript 1_1: Tri-Network Convergence (paired), Transcript 1: Tri-Network Convergence (+2 more)

### Community 47 - "Experiment Scripts"
Cohesion: 0.36
Nodes (9): find_videos(), load_config(), main(), parse_syncnet_output(), Path, Parse Confidence, Min dist, AV offset from SyncNet stdout., Scan 03_ditto for existing .mp4 files., Run full SyncNet pipeline: detect+crop faces, then evaluate.      Step 1: run_pi (+1 more)

### Community 48 - "SSL Embeddings"
Cohesion: 0.31
Nodes (4): Validate layer indices and return a deduplicated, sorted list.      Raises ``Val, _validate_layers(), Tests for _validate_layers., TestLayerValidation

### Community 49 - "Feature Separability"
Cohesion: 0.31
Nodes (6): boundary_sharpness(), Cosine change rate at token boundaries and within-segment stability.      Parame, _make_frame_embeddings_with_boundary(), ndarray, Create frame embeddings with a sharp transition at boundary_at., TestBoundarySharpness

### Community 50 - "TTS Providers"
Cohesion: 0.33
Nodes (5): Any, ndarray, Path, TTSResult, HiggsTTSProvider

### Community 51 - "Experiment Scripts"
Cohesion: 0.36
Nodes (7): apply_hf_shelf(), apply_linear_gain(), get_meter(), main(), Gentle high-frequency shelf boost using IIR filter., run_ditto(), run_eval()

### Community 52 - "Identity Correction"
Cohesion: 0.33
Nodes (8): classify(), compute_residuals(), get_identity_per_sample(), load_inputs(), main(), Path, Return {sample_id: per-sample record} for the identity intervention., For one intervention, compute per-sample residuals & summary stats.      The res

### Community 53 - "Experiment Scripts"
Cohesion: 0.22
Nodes (6): paired_permutation_test(), Paired permutation test for the null hypothesis *mean(a) == mean(b)*.      The t, Verify paired permutation works when natural > TTS for silhouette., TestStatisticalIntegration, test_paired_permutation_test_known_result(), test_paired_permutation_test_null()

### Community 54 - "TTS Providers"
Cohesion: 0.43
Nodes (8): Ditto TRT Online Configuration, Global Experiment Configuration, Experiment Pipeline Stages, R1: Local Faster Qwen3 Baseline, AISHELL-1 & HDTF Datasets, Ditto Talking Head Model, Qwen3-TTS-12Hz-0.6B-Base, Sync-C & Sync-D Evaluation

### Community 55 - "Experiment Scripts"
Cohesion: 0.36
Nodes (7): load_summary(), main(), paired_stats(), ndarray, Path, Return {sample_id: {natural_raw_c, natural_raw_d, tts_raw_c, tts_raw_d}}., Compute paired t-test, Wilcoxon, Cohen's d, mean Δ.

### Community 56 - "Experiment Scripts"
Cohesion: 0.32
Nodes (7): extract_features_v2(), _load_syncnet_from_run(), main(), _parse_audio_file(), Parse metadata from an audio filename in a run directory., Load all SyncNet scores from a run directory., Extract 35+ acoustic features from a WAV file.

### Community 57 - "Experiment Scripts"
Cohesion: 0.36
Nodes (4): _clean_text(), Strip punctuation and whitespace, keeping only Chinese characters., Tests for Chinese text cleaning., TestCleanText

### Community 58 - "Feature Separability"
Cohesion: 0.36
Nodes (5): _cosine_dist(), _cosine_sim(), ndarray, Row-wise cosine similarity between two sets of vectors.      Parameters     ----, TestCosineDist

### Community 59 - "Identity Control"
Cohesion: 0.29
Nodes (7): _build_identity_interventions(), _identity_tts(), _load_script_module(), main(), Intervention, Load a sibling script as a module by file path (mirrors script 22)., No-op transform: return TTS audio unchanged.      The point is to exercise the d

### Community 60 - "Experiment Scripts"
Cohesion: 0.38
Nodes (4): _discover_audio(), Return the audio file path for a given (sample_id, condition)., Tests for _discover_audio., TestDiscoverAudio

### Community 61 - "SSL Embeddings"
Cohesion: 0.38
Nodes (4): _make_output_stem(), Return the filename stem for a given sample/model combination.      >>> _make_ou, Tests for _make_output_stem., TestOutputPaths

### Community 62 - "Feature Separability"
Cohesion: 0.43
Nodes (3): intra_class_variance(), Mean pairwise cosine distance within each class, averaged across classes.      C, TestIntraClassVariance

### Community 63 - "Experiment Scripts"
Cohesion: 0.47
Nodes (5): load_normalized_scores(), load_original_scores(), main(), Load original SyncC scores from r3_results CSVs., Load normalized SyncC scores from r4 runs (local or server).

### Community 64 - "Feature Separability"
Cohesion: 0.47
Nodes (3): cohens_d_paired(), Cohen's d for paired samples., TestCohensDPaired

### Community 65 - "Feature Separability"
Cohesion: 0.47
Nodes (3): confusable_pairs(), Top-k most confusable class pairs by centroid cosine distance.      Parameters, TestConfusablePairs

### Community 66 - "Feature Separability"
Cohesion: 0.47
Nodes (3): fisher_ratio(), Between-class variance / within-class variance, averaged across dimensions., TestFisherRatio

### Community 67 - "Feature Separability"
Cohesion: 0.47
Nodes (3): _pool_frames_for_layer(), Pool frame embeddings into per-token vectors for one layer.      Returns     ---, TestPoolFramesForLayer

### Community 68 - "Tests Test Cross"
Cohesion: 0.33
Nodes (5): Tests for cross-TFG mechanism result loading., Phase 1 candidates use ``mechanism`` rather than ``name``., Per-sample model results must be usable by consistency analysis., test_aggregate_per_sample_deltas_produces_model_delta(), test_mechanism_name_accepts_feature_link_schema()

### Community 69 - "TTS Providers"
Cohesion: 0.40
Nodes (3): Default interventions must use the TTS source used by Phase 1., The CLI input root must select real natural/tts audio directories., TestAudioDirectoryResolution

### Community 70 - "TTS Providers"
Cohesion: 0.50
Nodes (3): run_r1_matrix.sh script, HF_ENDPOINT, LD_LIBRARY_PATH

### Community 71 - "Experiment Scripts"
Cohesion: 0.67
Nodes (3): extract_features(), main(), Extract acoustic features from a WAV file.

### Community 73 - "Concept Adr Conflict"
Cohesion: 0.67
Nodes (3): concept_adr_conflict, concept_domain_glossary, doc_agents_domain

### Community 75 - "Experiment Scripts"
Cohesion: 0.67
Nodes (3): AudioItem, A single audio sample in the analysis dataset.      Attributes     ----------, test_audio_item_instantiation()

### Community 77 - "Experiment Scripts"
Cohesion: 1.00
Nodes (3): Transcript 10_10: Strategic Industries (paired), Transcript 10: Strategic Industries, Strategic Emerging Industries Development

### Community 78 - "Experiment Scripts"
Cohesion: 1.00
Nodes (3): Transcript 11_11: Deng Yusong Quote (paired), Transcript 11: Deng Yusong Quote, Deng Yusong - Development Economist

### Community 79 - "Experiment Scripts"
Cohesion: 1.00
Nodes (3): Transcript 12_12: Spring Festival Enforcement (paired), Transcript 12: Spring Festival Enforcement, Spring Festival Transport Fare Enforcement

### Community 80 - "Experiment Scripts"
Cohesion: 1.00
Nodes (3): transcripts_7_7_green_efficient_housing, transcripts_7_7_high_quality_housing, transcripts_7_7_housing_demand_rise

### Community 81 - "Experiment Scripts"
Cohesion: 0.67
Nodes (3): transcripts_8_8_economic_slowdown, transcripts_8_8_monetary_policy_tools, transcripts_8_8_phased_deployment

### Community 82 - "Experiment Scripts"
Cohesion: 0.67
Nodes (3): transcripts_8_economic_slowdown, transcripts_8_monetary_policy_tools, transcripts_8_phased_deployment

### Community 83 - "Experiment Scripts"
Cohesion: 1.00
Nodes (3): transcripts_9_9_anhui_tongling, transcripts_9_9_gas_subsidy_policy, transcripts_9_9_subsidy_termination

### Community 84 - "Experiment Scripts"
Cohesion: 1.00
Nodes (3): transcripts_9_anhui_tongling, transcripts_9_gas_subsidy_policy, transcripts_9_subsidy_termination

## Ambiguous Edges - Review These
- `Chinese Real Estate Market` → `Market's Decisive Role in Resource Allocation`  [AMBIGUOUS]
  graphify-out/transcripts/2.txt · relation: applies_to
- `transcripts_7_7_green_efficient_housing` → `transcripts_7_7_housing_demand_rise`  [AMBIGUOUS]
  graphify-out/transcripts/7_7.txt · relation: rising demand targets this housing category
- `transcripts_7_7_high_quality_housing` → `transcripts_7_7_housing_demand_rise`  [AMBIGUOUS]
  graphify-out/transcripts/7_7.txt · relation: rising demand targets this housing category
- `transcripts_8_economic_slowdown` → `transcripts_8_monetary_policy_tools`  [AMBIGUOUS]
  graphify-out/transcripts/8.txt · relation: intended to mitigate or address
- `transcripts_8_8_economic_slowdown` → `transcripts_8_8_monetary_policy_tools`  [AMBIGUOUS]
  graphify-out/transcripts/8_8.txt · relation: intended to mitigate or address

## Knowledge Gaps
- **60 isolated node(s):** `ndarray`, `Namespace`, `Namespace`, `ndarray`, `Namespace` (+55 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Chinese Real Estate Market` and `Market's Decisive Role in Resource Allocation`?**
  _Edge tagged AMBIGUOUS (relation: applies_to) - confidence is low._
- **What is the exact relationship between `transcripts_7_7_green_efficient_housing` and `transcripts_7_7_housing_demand_rise`?**
  _Edge tagged AMBIGUOUS (relation: rising demand targets this housing category) - confidence is low._
- **What is the exact relationship between `transcripts_7_7_high_quality_housing` and `transcripts_7_7_housing_demand_rise`?**
  _Edge tagged AMBIGUOUS (relation: rising demand targets this housing category) - confidence is low._
- **What is the exact relationship between `transcripts_8_economic_slowdown` and `transcripts_8_monetary_policy_tools`?**
  _Edge tagged AMBIGUOUS (relation: intended to mitigate or address) - confidence is low._
- **What is the exact relationship between `transcripts_8_8_economic_slowdown` and `transcripts_8_8_monetary_policy_tools`?**
  _Edge tagged AMBIGUOUS (relation: intended to mitigate or address) - confidence is low._
- **Why does `fdr_bh_correction()` connect `TTS Providers` to `Feature TFG Linking`, `Experiment Scripts`, `Experiment Scripts`, `Feature Separability`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `TokenSpan` connect `Experiment Scripts` to `Experiment Scripts`, `Experiment Scripts`, `Experiment Scripts`, `Experiment Scripts`, `Experiment Scripts`, `Experiment Scripts`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._