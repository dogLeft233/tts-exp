"""CPU-only candidate gates for repeat coarse10 readiness and newly rendered media."""
from pathlib import PurePosixPath

VERSION='patrick_ave_rms_coarse_repeat_v1'
RUN=PurePosixPath('/dev/shm/synctalk-multispeaker-data-20261017/trials')/VERSION

def digest(value):
    assert isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)

def validate_ready(ready,training_contract_sha256):
    digest(training_contract_sha256)
    assert ready['passed'] and ready['identity']=='PatrickLeahy' and ready['version']==VERSION
    assert ready['contract_sha256']==training_contract_sha256
    assert ready['no_training_job_active'] and ready['new_27_grid_required']
    assert set(ready['final_receipts'])=={'-6','0','6'}
    # A CONTROL_FAILED phase remains descriptive; this gate never upgrades it.
    assert isinstance(ready['phase']['passed'],bool) and isinstance(ready['phase']['status'],str)
    for gain in (-6,0,6):
        r=ready['final_receipts'][str(gain)]
        assert r['passed'] and r['identity']=='PatrickLeahy' and r['seed']==42 and r['gain_db']==gain
        assert r['trial_version']==VERSION and r['independent_repeat'] and not r['old_fine_resumed']
        assert r['trial_contract_sha256']==r['normalization_contract_sha256']==training_contract_sha256
        assert r['phase']=='coarse' and r['epoch']==10 and r['global_step']==60000
        assert r['epoch_attempts']==6000 and r['epoch_successful_updates']+r['epoch_amp_skips']==6000
        assert r['original_full_checkpoint_restored'] and not r['new_independent_initialization']
        assert 0<=r['epoch_successful_updates']<=r['cumulative_successful_updates']<=60000
        assert r['phase_successful_updates']==r['cumulative_successful_updates'] and r['phase_start_updates']==0
        digest(r['checkpoint_sha256']);digest(r['trace_sha256']);digest(r['normalized_binding_sha256'])
        expected=RUN/'training/PatrickLeahy/seed42'/f'gain_{gain:+d}'/'coarse/ep0010/attempt01/workspace/checkpoints/ngp_ep0010.pth'
        assert '\\' not in r['checkpoint_path'] and PurePosixPath(r['checkpoint_path'])==expected
    return True

def validate_render(receipt,contract,training_contract_sha256,gain,driver):
    assert receipt['passed'] and receipt['rgb_and_pcm_readback_exact']
    assert receipt['experiment_version']==VERSION
    assert receipt['normalization_contract_sha256']==training_contract_sha256
    assert receipt['identity']=='PatrickLeahy' and receipt['seed']==42
    assert receipt['phase']=='coarse' and receipt['epoch']==10 and receipt['formal_fine17_model'] is False
    assert receipt['train_gain_db']==gain and receipt['driver_gain_db']==driver
    assert gain in (-6,0,6) and driver in (-6,0,6)
    ema=receipt['ema'];assert ema['passed'] and ema['gain_db']==gain
    assert ema['checkpoint_sha256']==contract['checkpoints'][str(gain)]['sha256']
    assert ema['ema_updates']==60 and ema['epoch']==10 and ema['global_step']==60000
    frames=receipt['frames'];assert len(frames)==1500
    assert [r['index'] for r in frames]==list(range(1500))
    assert [r['source_audio_id'] for r in frames]==list(range(50,1550))
    return True
