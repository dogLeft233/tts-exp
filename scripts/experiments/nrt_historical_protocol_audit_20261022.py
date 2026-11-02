"""Posthoc CPU comparison of sealed NRT supports and historical provenance.

No model load, audio synthesis, video generation or replacement of main results.
The padded cached-feature diagnostic is explicitly NOT an official pipeline score.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np

ROOT = Path('/home/wjj/tts-audio/tts-exp')
OUT = ROOT / 'runs/nrt_historical_protocol_audit_20261022'
CURRENT = ROOT / 'runs/qwen_nrt_expansion40_20261022_attempt01'
bindings = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    bindings[str(path)] = sha(path)
    return json.loads(path.read_text())


def summary(values, seed=20261022):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(len(x), size=(10000, len(x)))].mean(axis=1)
    return dict(n=len(x), mean=float(x.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
                positive=int((x > 0).sum()), values=x.tolist())


def cscore(matrix, selection):
    curve = np.mean(matrix[selection], axis=0, dtype=np.float64)
    return float(np.median(curve)-np.min(curve))


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'root_cpu_comparison.json').exists()
    analysis = read(CURRENT / 'analysis.json')
    p = read(CURRENT / 'protocol.json')
    audio = {(r['sample_id'], r['role']): r for r in read(CURRENT / 'audio_manifest.json')['records']}
    features = {(r['sample_id'], r['role']): r for r in read(CURRENT / 'feature_manifest.json')['records']}
    samples = {s['sample_id']: s for s in p['samples']}
    roles = ['N', 'R', 'T1', 'T2', 'T3']
    records = []
    max_core_error = 0.
    for r in analysis['records']:
        sid = r['sample_id']
        matrices = {}
        counts = r['counts']
        padded_cells = {}
        raw_level = []
        for role in roles:
            path = CURRENT / 'distances' / sid / (role + '_native.npy')
            bindings[str(path)] = sha(path)
            matrices[role] = np.load(path, allow_pickle=False)
            n = counts[role]
            assert matrices[role].shape == (n-30, 31)
            assert abs(cscore(matrices[role], np.arange(n-30))-r['policies']['native_full'][role]['C']) < 1e-12
            f = features[sid, role]
            for key in ['audio_path', 'visual_path']:
                assert sha(f[key]) == f[key.replace('_path', '_sha256')]
                bindings[f[key]] = f[key.replace('_path', '_sha256')]
            a = np.load(f['audio_path'], allow_pickle=False)[:n]
            v = np.load(f['visual_path'], allow_pickle=False)[:n]
            pad = np.pad(a, ((15, 15), (0, 0)))
            # New diagnostic changes only inclusion of synthetic padded edge rows.
            # Preserve the exact sealed interior matrix, not a changed reduction.
            def distances(q):
                d = v[q, None, :] - pad[q[:, None]+np.arange(31)[None, :]] + np.float32(1e-6)
                return np.sqrt(np.sum(d*d, axis=-1, dtype=np.float32)).astype(np.float64)
            rebuilt = distances(np.arange(15, n-15))
            error = float(abs(rebuilt-matrices[role]).max())
            assert error < 1e-5
            max_core_error = max(max_core_error, error)
            full = np.concatenate([distances(np.arange(15)), matrices[role], distances(np.arange(n-15, n))])
            assert full.shape == (n, 31)
            padded_cells[role] = cscore(full, np.arange(n))
            if role.startswith('T'):
                raw_level.append(20*np.log10(audio[sid, role]['rms_before']/audio[sid, 'N']['rms_before']))
        pooled = {}
        for policy in ['common', 'native_full', 'equal_count_spread']:
            cells = r['policies'][policy]
            t = float(np.mean([cells[role]['C'] for role in roles[2:]]))
            pooled[policy] = {'T-N': t-cells['N']['C'], 'T-R': t-cells['R']['C']}
        per_pair = {'T-N': [], 'T-R': []}
        for role in roles[2:]:
            for baseline, contrast in [('N', 'T-N'), ('R', 'T-R')]:
                k = min(counts[role], counts[baseline])-30
                selection = np.arange(k)
                per_pair[contrast].append(cscore(matrices[role], selection)-cscore(matrices[baseline], selection))
        pooled['pair_common_no_padding'] = {key: float(np.mean(value)) for key, value in per_pair.items()}
        t = float(np.mean([padded_cells[role] for role in roles[2:]]))
        pooled['cached_fixedROI_full_padded_NOT_OFFICIAL'] = {'T-N': t-padded_cells['N'], 'T-R': t-padded_cells['R']}
        records.append(dict(sample_id=sid, cohort=r['cohort'], comparisons=pooled, per_pair=per_pair,
                            raw_T_minus_N_dB=raw_level, padded_cells=padded_cells))
    cohorts = {}
    for cohort in ['new30', 'all40', 'pilot10']:
        rows = [r for r in records if cohort == 'all40' or r['cohort'] == cohort]
        cohorts[cohort] = {policy: {key: summary([r['comparisons'][policy][key] for r in rows])
                                   for key in ['T-N', 'T-R']} for policy in records[0]['comparisons']}
        for policy in ['common', 'native_full', 'equal_count_spread']:
            for key in ['T-N', 'T-R']:
                z = cohorts[cohort][policy][key]
                old = analysis['cohorts'][cohort]['summaries'][policy]['C']['contrasts'][key]
                assert abs(z['mean']-old['mean']) < 1e-12
                assert np.max(abs(np.asarray(z['ci95'])-old['CI95'])) < 1e-12
    raw = np.array([value for r in records for value in r['raw_T_minus_N_dB']])
    level = dict(draws=120, mean_dB=float(raw.mean()), median_dB=float(np.median(raw)),
                 T_louder_draws=int((raw > 0).sum()), quantiles_dB=np.percentile(raw,[0,25,50,75,100]).tolist(),
                 after_PCM_RMS_max_absolute_error=max(abs(v['rms_after']-p['target_rms']) for v in audio.values()))
    # Historical100 is recomputed from scores, grouped using the actual manifest.
    old100 = read(ROOT / 'results/aishell100_phoneme/tfg_link/syncnet_100.json')
    manifest = read(ROOT / 'data/aishell1_100_zh/manifest.json')['records']
    speakers = {str(r['sample_id']): r['speaker_id'] for r in manifest}
    assert len(old100) == 100 and set(old100) == set(speakers)
    differences = {sid: v['tts']['c']-v['natural']['c'] for sid,v in old100.items()}
    names = sorted(set(speakers.values()))
    grouped = [float(np.mean([d for sid,d in differences.items() if speakers[sid] == name])) for name in names]
    historical100 = dict(n_recordings=100, n_speakers=len(names), recording_mean=float(np.mean(list(differences.values()))),
                         recording_positive=sum(d>0 for d in differences.values()),
                         speaker_equal=summary(grouped), speakers=names,
                         overlap_with_current_speakers=sorted(set(names)&{s['speaker_id'] for s in p['samples']}))
    oldlevel = read(ROOT / 'runs/tts_native_level_generation_cross_20260927/summary.json')['results']['raw/guard20/common71']['C']
    original = oldlevel['baseline_T_minus_N']; treatment = oldlevel['conditions']['LEVEL']
    for z in [original, treatment['total']['T_minus_N'], treatment['processed_T_minus_N']]:
        assert abs(float(np.mean(list(z['group_means'].values())))-z['mean']) < 1e-12
    assert abs(original['mean']+treatment['total']['T_minus_N']['mean']-treatment['processed_T_minus_N']['mean']) < 1e-12
    level_control = dict(original=original, change=treatment['total']['T_minus_N'], after=treatment['processed_T_minus_N'],
                         scope='Same71 pairs/15 speakers and Wav2Lip/cloud; pair-geometric RMS, not new40 fixed global RMS')
    oldlocal = read(ROOT / 'runs/lrs3_local_tts_n100_20260921/01_qwen_local/tts_manifest.json')
    cloud = read(ROOT / 'runs/aishell1_qwen_mfa_linear_n100_20260816/01_tts_retry/tts_meta.json')
    result = dict(status='POSTHOC_CPU_REANALYSIS_COMPLETE_NOT_OFFICIAL_RESCORING',
        source_sha256=sha(__file__), no_training=True, no_new_model_forward=True,
        nrt_cohorts=cohorts, nrt_records=records, raw_level_diagnostic=level,
        historical_Ditto100=historical100, matched_level_control=level_control,
        metadata=dict(old_Chinese_local='faster_qwen3 0.6B from canonical reports/config; historical checkpoint hash not independently established',
                      old_LRS3_local={key:oldlocal.get(key) for key in ['model','checkpoint','generation']},
                      cloud_control={key:cloud.get(key) for key in ['provider','model','reference_policy','duration_gate']},
                      current_model=p['model_dir'],current_generation=p['generation_kwargs'],current_RMS=p['target_rms']),
        numeric_checks=dict(all18_original_cohort_policy_contrasts_mean_CI_exact=True,padded_core_max_float32_error=max_core_error),
        boundaries=['Posthoc policies do not replace frozen primary new30 or certify an official score.',
                    'Cached padded full diagnostic holds crop/features/PCM fixed; cannot simulate S3FD tracking, mux or official crop.',
                    'Historical and current cohort/model/TFG/scoring differences cannot be subtracted into causal percentages.',
                    'Scores and ASR are not human mouth-accuracy or naturalness truth.'],inputs=bindings)
    with (OUT / 'root_cpu_comparison.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],cohorts=cohorts,raw_level=level,historical100=historical100,
                         oldlevel={k:level_control[k]['mean'] for k in ['original','change','after']}),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
