"""Frozen-weight density-refresh diagnostic; never constructs a trainer/optimizer."""
import argparse, copy, fcntl, hashlib, json, math, os, random, sys, time
from pathlib import Path

ROOT = Path('/dev/shm/synctalk-multispeaker-data-20261017')
RUN = ROOT / 'trials/patrick_ave_rms_coarse_v1'
TERMINAL = RUN / 'training/PatrickLeahy/seed42/gain_+0/coarse/ep0004/attempt01/terminal_state.pth'
TERMINAL_SHA = '21ef29fcb575ff9ad03c1d5af1381ef615fbf0d4b0f59b7cac02c8699b3f4f3f'
CONTRACT_SHA = '5cc4d22b57a32e5b6c3672d1e782dd40914b33850da5ae9b1e819b34ebb17859'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--cases', nargs='+', default=['rms_amp_a'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-sha')
    parser.add_argument('--checkpoint-label', default='terminal_raw_global18992_weights')
    a = parser.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lock = (ROOT / 'preparation_GPU.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert sha(TERMINAL) == TERMINAL_SHA
    assert sha(RUN / 'contract.json') == CONTRACT_SHA
    contract = json.loads((RUN / 'contract.json').read_text())
    for name, digest in contract['sources'].items():
        assert sha(RUN / 'sources' / name) == digest
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert b'train_patrick_rms_epoch_v1.py' not in cmd, proc.name
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(RUN / 'sources'))
    import numpy as np
    import torch
    import runtime_common as rc
    rc.configure()
    from patrick_rms_runtime_v1 import install
    install()
    from execution_state import tree_sha, restore_model_counters
    from nerf_triplane.network import NeRFNetwork
    import nerf_triplane.renderer as renderer
    import audio_context

    # Any accidental attempt to perform training must fail immediately.
    def forbidden(*args, **kwargs):
        raise AssertionError('backward/optimizer forbidden in density-only diagnostic')
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.optim.Optimizer.step = forbidden
    torch.optim.AdamW.step = forbidden
    torch.optim.Adam.step = forbidden
    state = torch.load(TERMINAL, map_location='cpu', weights_only=False)
    assert state['global_step'] == 18993 and state['successful_updates'] == 18956
    model_original_sha = tree_sha(state['model'])
    counters_original_sha = tree_sha(state['model_counters'])
    selected = state
    if a.checkpoint:
        assert a.checkpoint_sha and sha(a.checkpoint) == a.checkpoint_sha
        selected = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    # Load exact original data/options, including eye preprocessing and intact-recording context.
    data = rc.dataset('PatrickLeahy', 'train', 0, 42, 'coarse', preload=0)
    opt = data.opt
    model = NeRFNetwork(opt).cuda()
    model.load_state_dict(selected['model'], strict=True)
    restore_model_counters(model, state['model_counters'])
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    model.aud_features, model.eye_area, model.poses = data.auds, data.eye_area, data.poses
    weights_before = tree_sha(dict(model.named_parameters()))
    source_ids = np.asarray(data.original_source_ids)
    # Resolve the parent dataset via the saved original options closure without changing the wrapper.
    normalized_binding = json.loads((Path(opt.path) / 'binding.json').read_text())
    from cohort_binding import identity_root
    parent_dir = identity_root('PatrickLeahy', ROOT) / 'datasets/PatrickLeahy/train/gain_0'
    parent_binding = json.loads((parent_dir / 'binding.json').read_text())
    assert sha(parent_dir / 'binding.json') == normalized_binding['parent_binding_sha256']
    old_whole = torch.from_numpy(np.load(parent_dir / 'whole_ave.npy', allow_pickle=False)).float().unsqueeze(1)
    assert sha(parent_dir / 'whole_ave.npy') == parent_binding['whole_ave_sha256']
    old_aligned = old_whole[torch.from_numpy(source_ids)].clone()
    audio_context.register(old_aligned, old_whole, source_ids)
    assert old_aligned.shape == data.auds.shape

    def stats(x):
        x = x.detach()
        finite = torch.isfinite(x)
        xf = x[finite]
        return dict(dtype=str(x.dtype), shape=list(x.shape), finite=int(finite.sum()),
                    nan=int(torch.isnan(x).sum()), posinf=int(torch.isposinf(x).sum()),
                    neginf=int(torch.isneginf(x).sum()),
                    min=float(xf.min()) if xf.numel() else None,
                    max=float(xf.max()) if xf.numel() else None)

    report = dict(version='patrick_rms_density_refresh_diagnostic_v1', source_sha256=sha(__file__),
                  terminal_sha256=TERMINAL_SHA, contract_sha256=CONTRACT_SHA,
                  scope='frozen terminal weights; only density refresh, zero backward/optimizer/trainer/epoch replay',
                  cuda=dict(name=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda),
                  occupancy_step=18992, seed_domain='occupancy_randomness', seed=42,
                  model_state_sha256=model_original_sha, terminal_counters_sha256=counters_original_sha,
                  weight_label=a.checkpoint_label, selected_checkpoint_sha256=a.checkpoint_sha or TERMINAL_SHA,
                  selected_weight_global_step=selected['global_step'],
                  normalized_feature_sha256=normalized_binding['whole_ave_sha256'],
                  old_feature_sha256=parent_binding['whole_ave_sha256'],
                  options=vars(opt), data=dict(auds=stats(data.auds), old_auds=stats(old_aligned), eye=stats(data.eye_area)),
                  cases=[])
    (out / 'input_receipt.json').write_text(json.dumps(report, indent=2, default=str))
    terminal_bad = torch.isposinf(state['model']['density_grid']).cuda()
    # Hooks are observational: no tensor replacement or clipping.
    active = {}
    hooks = []
    for name, mod in model.named_modules():
        if name in ('audio_net','audio_att_net','encoder_xy','encoder_yz','encoder_xz','aud_ch_att_net','eye_att_net','sigma_net') or name.startswith('sigma_net.net.'):
            def observe(module, inputs, output, name=name):
                active.setdefault('modules', []).append(dict(name=name, input=stats(inputs[0]), output=stats(output)))
                if name == 'sigma_net':
                    logits = output[..., 0]
                    threshold = math.log(torch.finfo(torch.float32).max)
                    active['sigma_logit'] = stats(logits)
                    active['float32_exp_log_limit'] = threshold
                    active['logits_above_float32_exp_limit'] = int((logits.float() > threshold).sum())
                    indices = torch.nonzero(logits.float() > threshold).flatten()
                    active['overflow_logit_values'] = logits[indices].float().cpu().tolist()
                    active['overflow_point_linear_indices'] = indices.cpu().tolist()
                    active['max_logit_exp_float64'] = math.exp(float(logits.max())) if float(logits.max()) < 709 else None
            hooks.append(mod.register_forward_hook(observe))
    original_density = model.density
    def observe_density(*args, **kwargs):
        result = original_density(*args, **kwargs)
        active.setdefault('density', []).append(dict(x=stats(args[0]), enc_a=stats(args[1]),
             eye=stats(args[2]) if len(args)>2 and args[2] is not None else None,
             sigma=stats(result['sigma']), geo_feat=stats(result['geo_feat'])))
        active['xyz_sha256'] = tree_sha(args[0])
        active['encoded_audio_sha256'] = tree_sha(args[1])
        bad = torch.nonzero(~torch.isfinite(result['sigma'])).flatten()
        active['first_nonfinite_xyz'] = args[0][bad[:100]].cpu().tolist()
        return result
    model.density = observe_density
    original_dilate = renderer.raymarching.morton3D_dilation
    def observe_dilation(grid):
        active['pre_dilation'] = stats(grid)
        active['pre_dilation_bad_indices'] = torch.nonzero(~torch.isfinite(grid), as_tuple=False).cpu().tolist()
        result = original_dilate(grid)
        active['post_dilation'] = stats(result)
        active['post_dilation_bad_indices'] = torch.nonzero(~torch.isfinite(result), as_tuple=False).cpu().tolist()
        active['new_bad_mask_matches_terminal'] = bool(torch.equal(torch.isposinf(result), terminal_bad))
        return result
    renderer.raymarching.morton3D_dilation = observe_dilation
    try:
        for case in a.cases:
            assert case in ('rms_amp_a','rms_amp_b','old_amp_a','old_amp_b','rms_fp32','old_fp32','rms_eye_zero_amp')
            active = dict(case=case, amp=('fp32' not in case), modules=[])
            model.load_state_dict(selected['model'], strict=True)
            restore_model_counters(model, state['model_counters'])
            model.aud_features = old_aligned if case.startswith('old') else data.auds
            model.eye_area = torch.zeros_like(data.eye_area) if 'eye_zero' in case else data.eye_area
            rc.seed_domain(42, 18992, 'occupancy_randomness')
            py_state = random.getstate()
            idx = random.randint(0, len(data.auds)-1)
            random.setstate(py_state)
            active['rand_idx'] = idx
            active['source_audio_id'] = int(source_ids[idx])
            active['selected_eye'] = model.eye_area[idx].cpu().tolist()
            start = time.monotonic()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=active['amp']):
                model.update_extra_state()
            torch.cuda.synchronize()
            active['elapsed_seconds'] = time.monotonic()-start
            active['result_grid'] = stats(model.density_grid)
            active['mean_density_repr'] = repr(model.mean_density)
            active['parameters_unchanged'] = tree_sha(dict(model.named_parameters())) == weights_before
            assert active['parameters_unchanged']
            active['gradients_absent'] = all(p.grad is None for p in model.parameters())
            assert active['gradients_absent']
            (out / (case+'.json')).write_text(json.dumps(active, indent=2, allow_nan=False))
            report['cases'].append(active)
            print(json.dumps(dict(case=case, rand_idx=idx, selected_eye=active['selected_eye'], density=active['density'], pre=active['pre_dilation'], post=active['post_dilation'], mask_match=active['new_bad_mask_matches_terminal'], elapsed=active['elapsed_seconds'])), flush=True)
    finally:
        renderer.raymarching.morton3D_dilation = original_dilate
        for h in hooks:
            h.remove()
    assert tree_sha(state['model']) == model_original_sha
    assert tree_sha(state['model_counters']) == counters_original_sha
    assert sha(TERMINAL) == TERMINAL_SHA
    assert sha(RUN / 'contract.json') == CONTRACT_SHA
    for name, digest in contract['sources'].items():
        assert sha(RUN / 'sources' / name) == digest
    report.update(passed=True, backward_calls=0, optimizer_calls=0, trainer_instantiations=0,
                  terminal_file_unchanged=True, original_source_pins_unchanged=True,
                  original_trial_not_resumed=True, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
    print('DIAGNOSTIC_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
