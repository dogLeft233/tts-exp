"""Rebuild all reported means from individual clips with independent stdlib math."""
import hashlib,itertools,json,math
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
IDS=['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']
SEEDS=[42,2026];GAINS=[-6,0,6]
METRICS=['sync_c','sync_d','distance_d0','background_median']
AXES=['training_gain_db','driver_gain_db','evaluation_gain_db']
CELLS=list(itertools.product(GAINS,repeat=3))
def equal(actual,expected):
    assert isinstance(actual,(int,float)) and math.isfinite(actual)
    assert abs(actual-expected)<=1e-10,(actual,expected)
def reduce_rows(rows):
    assert rows
    values={k:math.fsum(float(r[k]) for r in rows)/len(rows) for k in METRICS}
    for v in values.values():assert math.isfinite(v)
    offsets=[int(v) for r in rows for v in r['offsets']]
    assert offsets and all(-15<=v<=15 for v in offsets)
    return dict(values,offsets=offsets,offset_mean=math.fsum(offsets)/len(offsets))
def check_cell_order(cells):
    assert [tuple(c[k] for k in AXES) for c in cells]==CELLS,'incomplete, reordered or duplicate factorial cells'
def compare(actual,expected):
    for k in METRICS+['offset_mean']:equal(actual[k],expected[k])
    assert actual['offsets']==expected['offsets']
def seed_rows(raw,identity,seed):
    assert raw['passed'] and raw['identity']==identity and raw['seed']==seed
    assert raw['source_owned_ROI'] and raw['evaluation_zero_original_PCM16'] and raw['evaluation_plusminus_float64_original_PCM_units']
    assert raw['repeat']['visual_exact'] and raw['repeat']['matrix_exact'] and raw['actual_FP32_MFCC_input_exact']
    check_cell_order(raw['cells']);rows=[]
    for c in raw['cells']:
        assert [r['clip_id'] for r in c['clips']]==[f'test_{i:02d}' for i in range(6)]
        result={k:c[k] for k in AXES}
        for support in ['guard20','full']:
            clips=[dict(r[support],offsets=[r[support]['av_offset']]) for r in c['clips']]
            expected=reduce_rows(clips)
            for k in METRICS:equal(c['mean_'+support][k],expected[k])
            assert c['mean_'+support]['offsets']==expected['offsets']
            result[support]=expected
        rows.append(result)
    return rows
def run():
    output=ROOT/'delivery';analysis=json.loads((output/'analysis_receipt.json').read_text());assert analysis['passed']
    for row in analysis['files']:
        p=output/row['path'];assert p.is_file() and p.stat().st_size==row['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    tables=json.loads((output/'tables27.json').read_text());assert tables['passed'] and tables['identities']==IDS and tables['seeds']==SEEDS
    reconstructed={};by_id={};input_hashes=[]
    for identity in IDS:
        for seed in SEEDS:
            path=ROOT/'scores'/identity/f'seed{seed}'/'scores.json';raw=json.loads(path.read_text())
            qa=json.loads(path.with_name('independent_score_qualification.json').read_text());assert qa['passed'] and qa['clip_matrices']==162
            actual=tables['per_identity_seed'][f'{identity}/seed{seed}'];check_cell_order(actual)
            rows=seed_rows(raw,identity,seed);reconstructed[identity,seed]=rows
            for a,b in zip(actual,rows):
                for support in ['guard20','full']:compare(a[support],b[support])
            input_hashes.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        by_id[identity]=[];check_cell_order(tables['per_identity'][identity])
        for index in range(27):
            expected={s:reduce_rows([reconstructed[identity,seed][index][s] for seed in SEEDS]) for s in ['guard20','full']}
            for support in ['guard20','full']:compare(tables['per_identity'][identity][index][support],expected[support])
            by_id[identity].append(expected)
    check_cell_order(tables['equal_identity'])
    for index in range(27):
        for support in ['guard20','full']:
            expected=reduce_rows([by_id[identity][index][support] for identity in IDS])
            compare(tables['equal_identity'][index][support],expected)
    curves=json.loads((output/'learning_curves.json').read_text());assert len(curves)==408
    indexed={(r['identity'],r['seed'],r['training_gain_db'],r['epoch']):r for r in curves};assert len(indexed)==408
    for identity,seed,gain in itertools.product(IDS,SEEDS,GAINS):
        totals={'coarse':0,'fine':0}
        for epoch in range(1,18):
            phase='coarse' if epoch<=10 else 'fine';r=indexed[identity,seed,gain,epoch];assert r['phase']==phase
            p=ROOT/'training'/identity/f'seed{seed}'/f'gain_{gain:+d}'/phase/f'ep{epoch:04d}/attempt01/epoch_receipt.json'
            receipt=json.loads(p.read_text());assert receipt['passed'] and receipt['epoch_attempts']==6000 and receipt['global_step']==epoch*6000
            equal(r['training_original_total_loss'],receipt['post_validation_stats']['loss'][-1])
            assert r['validation_by_driver_gain']==receipt['validation']['rows']
            assert [v['gain_db'] for v in r['validation_by_driver_gain']]==GAINS
            for v in r['validation_by_driver_gain']:assert math.isfinite(v['mean_mse']) and v['mean_mse']>=0
            assert r['amp_skips']==receipt['epoch_amp_skips'] and r['successful_optimizer_updates']==receipt['epoch_successful_updates']
            assert r['amp_skips']+r['successful_optimizer_updates']==6000
            totals[phase]+=r['successful_optimizer_updates']
        for phase,total in totals.items():
            q=json.loads((ROOT/'blocks'/identity/f'seed{seed}'/(phase+'_qualification.json')).read_text());assert q['passed']
            assert set(map(int,q['actual_updates_by_gain']))==set(GAINS) and len(set(q['actual_updates_by_gain'].values()))==1
            assert q['actual_updates_by_gain'][str(gain)]==total
    proof=dict(passed=True,identity_seed_tables=8,identity_tables=4,combined_cells=27,
        clip_cells_recomputed=8*27*6,learning_curve_rows=408,
        independent_stdlib_fsum_hierarchy=True,all_offsets_preserved=True,actual_phase_update_budgets_equal=True,
        inputs=input_hashes,analysis_receipt_sha256=hashlib.sha256((output/'analysis_receipt.json').read_bytes()).hexdigest(),
        execution_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='aggregation, original curves and budgets; not human perception or final cloud delivery')
    target=output/'independent_aggregate_qualification.json';assert not target.exists()
    target.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
if __name__=='__main__':run()
