import array
import collections
import hashlib
import json
import math
import random
import statistics
import struct
from pathlib import Path

ROOT = Path('E:/Documents/tts-exp')
BASE = ROOT / '.codex/scratch/dev360_controls360_rms_diagnostic_attempt01'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()

def check_self(value, field):
    clone = dict(value)
    expected = clone.pop(field)
    assert digest(canonical(clone)) == expected

protocol = json.loads((BASE / 'analysis_protocol_attempt01.json').read_bytes())
manifest = json.loads((BASE / 'analysis_manifest_attempt01.json').read_bytes())
effects = json.loads((BASE / 'effects_table_attempt01.json').read_bytes())
check_self(manifest, 'manifest_self_sha256')
check_self(effects, 'table_self_sha256')
for pin in manifest['artifacts'] + list(protocol['source_inputs'].values()):
    raw = (ROOT / pin['path']).read_bytes()
    assert len(raw) == pin['bytes'] and digest(raw) == pin['sha256']
assert protocol['source_inputs'] == manifest['source_inputs']
rows = [json.loads(line) for line in (ROOT / protocol['source_inputs']['ledger']['path']).read_bytes().splitlines()]
assert len(rows) == 360 and all(r['status'] == 'PASS' for r in rows)
fields = protocol['population_and_pairing']['paired_identity_fields']
grouped = collections.defaultdict(dict)
for row in rows:
    key = tuple(row[field] for field in fields)
    assert row['method_id'] not in grouped[key]
    grouped[key][row['method_id']] = row
assert len(grouped) == 180 and all(set(pair) == {'identity', 'rms_fixed'} for pair in grouped.values())
models = protocol['population_and_pairing']['models']
speakers = protocol['population_and_pairing']['speakers']
names = [spec['name'] for spec in protocol['estimands']['metric_family']]
targets = models + ['equal_model_mean']
clusters = {(model, speaker, name): [] for model in models for speaker in speakers for name in names}
denominator = collections.Counter()
guard_fail = []
max_metric_residual = 0.0
for key, pair in grouped.items():
    ident, rms = pair['identity'], pair['rms_fixed']
    speaker, model = ident['speaker_id'], ident['model_id']
    denominator[(speaker, model)] += 1
    assert ident['score']['support_by_support'] == rms['score']['support_by_support']
    assert ident['score']['support_by_support']['full']['eligible']
    if not ident['score']['support_by_support']['guard20']['eligible']:
        guard_fail.append(dict(zip(fields, key)))
    for row in (ident, rms):
        metrics = row['score']['metrics_by_support']['full']
        assert metrics['anchor_column'] == metrics['zero_lag_column'] == 15
        max_metric_residual = max(max_metric_residual,
                                  abs(metrics['Cbest'] - (metrics['B'] - metrics['Dmin'])),
                                  abs(metrics['Canchor'] - (metrics['B'] - metrics['D0'])))
    for name in names:
        vals = []
        for row in (ident, rms):
            metrics = row['score']['metrics_by_support']['full']
            vals.append(abs(metrics['best_lag_frames']) if name == 'abs_best_lag_frames' else metrics[name])
        clusters[(model, speaker, name)].append(vals[1] - vals[0])
assert len(denominator) == 30 and set(denominator.values()) == {6}
assert len(guard_fail) == 9 and {r['sample_id'] for r in guard_fail} == {'BAC009S0022W0123'}
for row in rows:
    m = row["score"]["metrics_by_support"]["full"]
    for c, d in (("Cbest", "Dmin"), ("Canchor", "D0")):
        assert m[c] == struct.unpack("<f", struct.pack("<f", m["B"] - m[d]))[0]
matrix = []
cells = []
for target in targets:
    for name in names:
        cell = []
        for speaker in speakers:
            model_means = [statistics.fmean(clusters[(model, speaker, name)]) for model in models]
            cell.append(statistics.fmean(model_means) if target == 'equal_model_mean' else
                        statistics.fmean(clusters[(target, speaker, name)]))
        cells.append((target, name))
        matrix.append(cell)
observed = [statistics.fmean(cell) for cell in matrix]
replicates = protocol['uncertainty']['resamples']
rng = random.Random(protocol['uncertainty']['seed'])
boots = [array.array('d') for _ in cells]
for _ in range(replicates):
    counts = [0] * len(speakers)
    for _ in range(len(speakers)):
        counts[rng.randrange(len(speakers))] += 1
    for column, data in zip(boots, matrix):
        column.append(math.fsum(value * count for value, count in zip(data, counts)) / len(speakers))
sd = [statistics.stdev(column) for column in boots]
max_statistics = sorted(max(abs((boots[k][j] - observed[k]) / sd[k]) if sd[k] else 0.0
                            for k in range(len(cells))) for j in range(replicates))
critical = max_statistics[math.ceil(.99 * replicates) - 1]
assert abs(critical - manifest['bootstrap']['simultaneous_max_abs_t_99_critical_value']) < 2e-12
recomputed = []
for k, cell in enumerate(cells):
    row = next(r for r in effects['effects_table'] if (r['target'], r['metric']) == cell)
    expected = {'paired_delta_mean_rms_fixed_minus_identity': observed[k], 'cluster_bootstrap_se': sd[k],
                'simultaneous_99_ci_low': observed[k] - critical * sd[k],
                'simultaneous_99_ci_high': observed[k] + critical * sd[k]}
    assert all(abs(row[field] - value) < 2e-12 for field, value in expected.items())
    recomputed.append({'target': cell[0], 'metric': cell[1], **expected})
report = {'status': 'PASS_ROOT_INDEPENDENT_NUMERIC_RECOMPUTE', 'rows': len(rows), 'pairs': len(grouped),
          'speaker_count': len(speakers), 'all_pairs_retained': True, 'family_size': len(cells),
          'bootstrap_resamples': replicates, 'seed': protocol['uncertainty']['seed'],
          'simultaneous_99_critical_value': critical, 'max_metric_identity_residual': max_metric_residual,
          'guard20_eligible_pairs': 171, 'guard20_ineligible_pairs': guard_fail,
          'effects_table_raw_sha256': digest((BASE / 'effects_table_attempt01.json').read_bytes()),
          'recomputed_effects': recomputed, 'gpu_api_ssh_or_experiment_replay': False}
destination = ROOT / '.codex/scratch/root_controls360_rms_numeric_qc_attempt02.json'
with destination.open('xb') as output:
    output.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode() + b'\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('recomputed_effects', 'guard20_ineligible_pairs')}))
