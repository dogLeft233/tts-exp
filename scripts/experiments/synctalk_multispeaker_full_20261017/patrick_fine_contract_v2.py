"""Pure guards shared by fine entry, qualification, and host scheduling."""
from pathlib import PurePosixPath

VERSION='patrick_fp16_fine_v2'

def validate_start(contract,gain,checkpoint,receipt_path,receipt,checkpoint_sha,receipt_sha):
    assert contract['version']==VERSION and contract['descriptive_only'] and contract['parent_control_failed_retained']
    pin=contract['startpoints'][str(gain)]
    assert (checkpoint,receipt_path,checkpoint_sha,receipt_sha)==(pin['checkpoint_path'],pin['receipt_path'],pin['checkpoint_sha256'],pin['receipt_sha256'])
    assert receipt['identity']=='PatrickLeahy' and receipt['seed']==42 and receipt['gain_db']==gain
    assert receipt['passed'] and receipt['phase']=='coarse' and receipt['epoch']==10 and receipt['global_step']==60000
    assert receipt['cumulative_successful_updates']==contract['parent_updates_by_gain'][str(gain)]
    assert PurePosixPath(checkpoint).is_relative_to(PurePosixPath(contract['parent_run'])/'training/PatrickLeahy/seed42'/f'gain_{gain:+d}')
    return True

def route(persisted,remote_state):
    if persisted:return 'reuse_verified'
    if remote_state is None:return 'launch_once'
    assert remote_state in ['starting','running','completed'],'lost job must never replay'
    return 'observe_existing'

def qualify_cases(contract_sha,cases):
    assert set(cases)=={-6,0,6}
    sampling=None
    fields=['epoch','global_step','frame_index','source_audio_id','lip_mode','pixel_index_sha256','actual_rays','rect']
    for g,r in cases.items():
        assert r['passed'] and r['gain_db']==g and r['contract_sha256']==contract_sha
        assert r['qualification_attempts']==2 and 0<=r['qualification_optimizer_updates']<=2 and r['formal_optimizer_updates']==0
        assert r['qualification_optimizer_updates']==sum(x['successful_update'] for x in r['rows'])
        assert r['qualification_amp_skips']==sum(x['amp_skip'] for x in r['rows'])
        assert all(x['amp_skip']!=x['successful_update'] for x in r['rows'])
        assert all(r[k] for k in ['full_checkpoint_roundtrip_exact','counters_roundtrip_exact','first_lip_then_nonlip','original_parent_unchanged','fp16'])
        assert r['patch_size']==64 and [x['lip_mode'] for x in r['rows']]==[True,False]
        current=[{k:x[k] for k in fields} for x in r['rows']]
        if sampling is None:sampling=current
        else:assert current==sampling,'fine qualification sampling differs across gains'
    return dict(passed=True,contract_sha256=contract_sha,qualification_attempts=6,qualification_optimizer_updates=sum(r["qualification_optimizer_updates"] for r in cases.values()),qualification_amp_skips=sum(r["qualification_amp_skips"] for r in cases.values()),formal_optimizer_updates=0,ordinary_amp_skips_preserved=True,three_gain_sampling_exact=True,full_checkpoint_roundtrip_exact=True,no_precision_policy_change=True)
