"""Phase and checkpoint checks for the descriptive Patrick fine17 score run."""


def fine_metadata():
    return dict(epoch=17, global_step=102000, phase='fine',
                formal_fine17_model=False, fine17_descriptive_model=True,
                parent_control_failed_retained=True)


def validate_render_receipt(receipt, contract, gain, driver):
    """Reject a coarse video or another arm before extracting score features."""
    assert receipt['passed'] and receipt['rgb_and_pcm_readback_exact']
    assert receipt['identity'] == contract['identity'] == 'PatrickLeahy'
    assert receipt['seed'] == contract['seed'] == 42
    assert receipt['train_gain_db'] == gain and receipt['driver_gain_db'] == driver
    for key, value in fine_metadata().items():
        assert receipt[key] == value, key
    assert receipt['training_control_status'] == contract['training_control_status']
    ema = receipt['ema']
    assert ema['passed'] and ema['full_state_finite']
    assert ema['epoch'] == 17 and ema['global_step'] == 102000
    assert ema['ema_updates'] == 102 and ema['ema_decay'] == .95
    assert ema['checkpoint_sha256'] == contract['checkpoints'][str(gain)]['sha256']
    assert ema['independent_named_overlay_equals_torch_ema']
    assert ema['raw_state_restored_exact'] and ema['buffers_from_full_checkpoint']
    assert len(receipt['frames']) == 1500
    assert [row['index'] for row in receipt['frames']] == list(range(1500))
