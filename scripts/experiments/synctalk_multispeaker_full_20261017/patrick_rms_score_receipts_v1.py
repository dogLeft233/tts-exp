"""Reject stale non-RMS/coarse metadata before reusing a rendered video for scores."""
def validate_render(receipt,contract,training_contract_sha256,gain,driver):
    assert receipt['passed'] and receipt['rgb_and_pcm_readback_exact']
    assert receipt['experiment_version']=='patrick_ave_rms_coarse_v1'
    assert receipt['normalization_contract_sha256']==training_contract_sha256
    assert receipt['identity']=='PatrickLeahy' and receipt['seed']==42
    assert receipt['phase']=='coarse' and receipt['epoch']==10 and receipt['formal_fine17_model'] is False
    assert receipt['train_gain_db']==gain and receipt['driver_gain_db']==driver
    assert gain in [-6,0,6] and driver in [-6,0,6]
    ema=receipt['ema'];assert ema['passed'] and ema['gain_db']==gain
    assert ema['checkpoint_sha256']==contract['checkpoints'][str(gain)]['sha256']
    assert ema['ema_updates']==60 and ema['epoch']==10 and ema['global_step']==60000
    frames=receipt['frames'];assert len(frames)==1500
    assert [r['index'] for r in frames]==list(range(1500))
    assert [r['source_audio_id'] for r in frames]==list(range(50,1550))
    return True
