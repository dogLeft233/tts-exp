"""Independent acceptance of actual paired sampling and optimizer budgets."""
import hashlib,json
from pathlib import Path
GAINS=[-6,0,6]
FIELDS=['epoch','global_step','frame_index','source_audio_id','lip_mode','pixel_index_sha256','actual_rays','rect']

def qualify_epoch(receipts,traces):
    assert set(receipts)==set(traces)==set(GAINS)
    base=receipts[0];epoch=base['epoch'];phase=base['phase'];common=None
    result=dict(identity=base['identity'],seed=base['seed'],epoch=epoch,phase=phase,arms=[])
    for gain in GAINS:
        receipt=receipts[gain]
        for field in ['identity','seed','phase','epoch','global_step','protocol_sha256','execution_contract_sha256','epoch_order_sha256','sampling_sequence_sha256']:
            assert receipt[field]==base[field],(gain,field)
        assert receipt['gain_db']==gain and receipt['passed'] and receipt['epoch_attempts']==6000 and receipt['global_step']==epoch*6000
        raw=Path(traces[gain]).read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['trace_sha256']
        rows=[json.loads(x) for x in raw.splitlines()];assert len(rows)==6000
        assert sorted(x['frame_index'] for x in rows)==list(range(6000))
        assert [x['global_step'] for x in rows]==list(range((epoch-1)*6000+1,epoch*6000+1))
        assert all(x['epoch']==epoch and x['source_audio_id']==x['frame_index']+50 for x in rows)
        assert all(x['actual_rays']==65536 if not x['lip_mode'] else 0<x['actual_rays']<=512*512 for x in rows)
        if phase=='coarse':assert not any(x['lip_mode'] for x in rows)
        else:assert all(rows[i]['lip_mode']!=(rows[i+1]['lip_mode']) for i in range(5999))
        updates=sum(x['successful_update'] for x in rows)
        assert updates==receipt['epoch_successful_updates'] and 6000-updates==receipt['epoch_amp_skips']
        previous=receipt['cumulative_successful_updates']-updates
        for row in rows:
            assert row['successful_update']!=row['amp_skip'];previous+=int(row['successful_update'])
            assert previous==row['successful_updates']
        sampling=[{k:row[k] for k in FIELDS} for row in rows]
        if common is None:common=sampling
        else:assert sampling==common,'measured cross-gain sampling differs'
        result['arms'].append(dict(gain_db=gain,successful_updates=updates,amp_skips=6000-updates,
            cumulative_successful_updates=receipt['cumulative_successful_updates'],phase_successful_updates=receipt['phase_successful_updates']))
    result.update(passed=True,actual_sampling_equal=True,actual_successful_updates_equal_this_epoch=len({x['successful_updates'] for x in result['arms']})==1)
    return result

def qualify_phase(epoch_results,final_receipts):
    base=final_receipts[0];phase=base['phase'];expected=list(range(1,11)) if phase=='coarse' else list(range(11,18))
    assert sorted(x['epoch'] for x in epoch_results)==expected and all(x['passed'] and x['actual_sampling_equal'] for x in epoch_results)
    assert all(x['identity']==base['identity'] and x['seed']==base['seed'] and x['phase']==phase for x in epoch_results)
    totals={g:sum(next(a['successful_updates'] for a in row['arms'] if a['gain_db']==g) for row in epoch_results) for g in GAINS}
    assert all(totals[g]==final_receipts[g]['phase_successful_updates'] for g in GAINS)
    same=len(set(totals.values()))==1
    return dict(identity=base['identity'],seed=base['seed'],phase=phase,epochs=expected,
        attempted_steps_per_arm=len(expected)*6000,actual_updates_by_gain=totals,passed=same,
        status='QUALIFIED' if same else 'CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES',
        no_scaler_policy_change=True,no_extra_steps=True)
