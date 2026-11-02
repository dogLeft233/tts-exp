"""Paired frozen-encoder layer/first-GroupNorm diagnostics; never trains."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from pathlib import Path
import sys, json, hashlib, time, traceback, gc
sys.path.insert(0, 'F:/Datasets/tts-exp/encoder_survey_20261018/deps')
import numpy as np
import torch
import soundfile as sf
import transformers
from transformers import AutoModel, AutoFeatureExtractor

OUT = Path('F:/Datasets/tts-exp/encoder_layer_epsilon_20261020_v1')
PARENT = Path('F:/Datasets/tts-exp/may_postnorm_scale_20261020')
WEIGHTS = Path('F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/weights')
DATA = Path('F:/Datasets/tts-exp/synctalk_gain_training_20261017')
EPSILONS = (1e-5, 0.0, 1e-7, 1e-3)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)

def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def moments(x, y):
    """Float64 moments over ALL coordinates; rows are time, columns channels."""
    assert x.shape == y.shape and x.ndim == 2
    n = len(x)
    acc = np.zeros((n, 5), dtype=np.float64)
    for start in range(0, n, 1024):
        a = x[start:start+1024].astype(np.float64)
        b = y[start:start+1024].astype(np.float64)
        d = b-a
        acc[start:start+len(a)] = np.stack([
            np.einsum('ij,ij->i', a, a), np.einsum('ij,ij->i', b, b),
            np.einsum('ij,ij->i', d, d), np.einsum('ij,ij->i', a, b),
            np.max(np.abs(d), axis=1)], axis=1)
    return acc

def summarize(acc):
    a, b, d, dot, mx = acc.T
    valid = (a > 0) & (b > 0)
    cos = np.clip(dot[valid]/np.sqrt(a[valid]*b[valid]), -1, 1)
    angle = np.degrees(np.arccos(cos))
    return dict(relative_l2=float(np.sqrt(d.sum()/a.sum())), maxabs=float(mx.max()),
                mean_angle_degrees=float(angle.mean()), p95_angle_degrees=float(np.quantile(angle,.95)),
                mean_norm_ratio=float(np.sqrt(b[valid]/a[valid]).mean()),
                baseline_energy=float(a.sum()), changed_energy=float(b.sum()), delta_energy=float(d.sum()))

class Capture:
    def __init__(self, model):
        self.handles=[]; self.arrays={}; self.axes={}
        def register(name, module, conv=False):
            self.axes[name] = 'time,channel'
            def hook(_module, _inputs, output):
                while isinstance(output, (tuple, list)): output=output[0]
                value=output.detach().cpu().numpy()[0]
                if conv: value=value.T
                assert value.ndim == 2 and np.isfinite(value).all(), name
                self.arrays[name]=value.copy()
            self.handles.append(module.register_forward_hook(hook))
        register('conv0_linear', model.feature_extractor.conv_layers[0].conv, True)
        register('conv0_groupnorm', model.feature_extractor.conv_layers[0].layer_norm, True)
        for i, block in enumerate(model.feature_extractor.conv_layers): register(f'conv{i}_activation',block,True)
        register('projection_norm', model.feature_projection.layer_norm)
        register('projection_linear', model.feature_projection.projection)
        register('positional_embedding', model.encoder.pos_conv_embed)
        register('transformer_input_norm', model.encoder.layer_norm)
        for i, block in enumerate(model.encoder.layers): register(f'transformer_{i+1:02d}',block)
        self.names=list(self.axes)
    def close(self):
        for handle in self.handles: handle.remove()

def state_sha(model):
    h=hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.4)
    free,total=torch.cuda.mem_get_info(); assert free > 4<<30
    protocol=dict(status='FROZEN_BEFORE_FORWARD', parent=str(PARENT), models=[],
        epsilon_grid=EPSILONS, scales=[.5,1.,2.], splits=['train','test'], chunk_samples=320080,
        transform='whole-split float32 native zscore THEN scale THEN first 320080 samples; no renormalization',
        epsilon_intervention='only feature_extractor.conv_layers[0].layer_norm.eps',
        compensated_control='scale a with epsilon=1e-5*a*a versus scale1 epsilon1e-5',
        comparisons='within-model matching stages; cross-model coordinates not subtracted',
        training_updates=0,tf_g_generation=0,syncnet_calls=0, planned_full_forwards=60,
        precision='FP32 CUDA eager, TF32 off, eval/inference_mode, frozen weights',
        source_script_sha256=sha(__file__),runtime=dict(torch=torch.__version__,transformers=transformers.__version__,
        numpy=np.__version__,python=sys.version,gpu=torch.cuda.get_device_name(),initial_free_bytes=free))
    sources={}
    for mid in ('hubert_base','wav2vec2_base'):
        source=json.loads((PARENT/'features'/mid/'receipt.json').read_text())
        for item in source['source']['files']: assert sha(WEIGHTS/mid/item['file'])==item['sha256']
        sources[mid]=source
        protocol['models'].append(dict(id=mid,source=source['source']))
    write(OUT/'protocol.json',protocol)
    rows=[]; controls=[]; calls=[]; norm_diagnostics=[]
    def persist():
        write(OUT/'progress.json',dict(status='RUNNING',completed_forwards=len(calls),comparisons=len(rows),latest=calls[-1] if calls else None))
        write(OUT/'layer_metrics.json',rows); write(OUT/'controls.json',controls)
        write(OUT/'forward_calls.json',calls); write(OUT/'groupnorm_diagnostics.json',norm_diagnostics)
    for mid in ('hubert_base','wav2vec2_base'):
        model,info=AutoModel.from_pretrained(WEIGHTS/mid,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
        assert not info['missing_keys'] and not info['mismatched_keys']
        model.eval().requires_grad_(False).cuda()
        norm=model.feature_extractor.conv_layers[0].layer_norm
        assert isinstance(norm,torch.nn.GroupNorm) and norm.eps==1e-5
        assert model.feature_extractor.conv_layers[0].conv.bias is None
        before=state_sha(model); capture=Capture(model)
        processor=AutoFeatureExtractor.from_pretrained(WEIGHTS/mid,local_files_only=True)
        write(OUT/mid/'loading.json',dict(loading=info,state_sha256=before,stage_names=capture.names,
            runtime_config=model.config.to_dict(),first_norm=dict(groups=norm.num_groups,channels=norm.num_channels,eps=norm.eps)))
        for split in ('train','test'):
            path=DATA/'arms'/split/'gain_0/aud.wav'
            raw,sr=sf.read(path,dtype='float64'); assert sr==16000 and raw.ndim==1
            original=next(c for c in sources[mid]['conditions'] if c['split']==split and c['input_scale']==1.)
            assert sha(path)==original['source_audio_sha256']
            x=raw.astype(np.float32)
            z=processor(x,sampling_rate=sr,return_tensors='np').input_values[0]
            assert array_sha(z)==original['native_processor_sha256']
            z=z[:320080].copy()
            np.save(OUT/mid/(split+'_standardized_input.npy'),z)
            default=None
            def forward(eps,scale,label):
                norm.eps=float(eps); capture.arrays={}
                start=time.time()
                with torch.inference_mode(): hidden=model(input_values=torch.from_numpy((z*np.float32(scale))[None]).cuda()).last_hidden_state[0].cpu().numpy()
                assert set(capture.arrays)==set(capture.names)
                arrays=capture.arrays; arrays['last_hidden_state']=hidden.copy()
                folder=OUT/mid/split/label; folder.mkdir(parents=True)
                np.save(folder/'last_hidden_state.npy',hidden)
                samples={name:arr[np.linspace(0,len(arr)-1,min(32,len(arr)),dtype=int)] for name,arr in arrays.items()}
                np.savez_compressed(folder/'stage_samples.npz',**samples)
                calls.append(dict(encoder=mid,split=split,eps=eps,scale=scale,label=label,seconds=time.time()-start,
                    path=str(folder.relative_to(OUT)),last_sha256=array_sha(hidden)))
                print(json.dumps(calls[-1]),flush=True)
                return arrays,folder
            def compare(base,changed,folder,meta):
                evidence={}
                for name in base:
                    m=moments(base[name],changed[name]); evidence[name]=m
                    rows.append(dict(encoder=mid,split=split,stage=name,shape=list(base[name].shape),**meta,**summarize(m)))
                np.savez_compressed(folder/'full_frame_moments.npz',**evidence)
            for eps in EPSILONS:
                label=f'eps_{eps:.0e}_scale1'
                base,folder=forward(eps,1.,label)
                if default is None:
                    default=base
                    prior=np.load(PARENT/'features'/mid/f'{split}_s1.00.npy',mmap_mode='r').reshape(-1,768)[:len(base['last_hidden_state'])]
                    metrics=summarize(moments(prior,base['last_hidden_state']))
                    controls.append(dict(encoder=mid,split=split,kind='original_cache_bridge',**metrics))
                    assert metrics['relative_l2']<1e-4, metrics
                    repeat,rep_folder=forward(eps,1.,'default_repeat')
                    exact=all(np.array_equal(base[k],repeat[k]) for k in base)
                    controls.append(dict(encoder=mid,split=split,kind='default_repeat_all_stages_exact',passed=exact)); assert exact
                    del repeat
                else: compare(default,base,folder,dict(kind='epsilon_baseline_drift',epsilon=eps,scale=1.))
                linear=base['conv0_linear']
                variance=np.var(linear.astype(np.float64),axis=0); mean=np.mean(linear.astype(np.float64),axis=0)
                assert np.all(variance>0)
                gamma=norm.weight.detach().cpu().numpy().astype(float); beta=norm.bias.detach().cpu().numpy().astype(float)
                np.savez_compressed(folder/'norm_channel_stats.npz',variance=variance,mean=mean,gamma=gamma,beta=beta)
                for scale in (.5,2.):
                    changed,c_folder=forward(eps,scale,f'eps_{eps:.0e}_scale{scale:g}')
                    compare(base,changed,c_folder,dict(kind='scale_response',epsilon=eps,scale=scale))
                    # Explicit GroupNorm mathematical prediction, separately from downstream measurements.
                    ratio=np.sqrt((variance+eps)/(variance+eps/scale**2))
                    delta_energy=np.sum(gamma**2*variance/(variance+eps)*(ratio-1)**2)
                    base_energy=np.sum(gamma**2*variance/(variance+eps)+beta**2)
                    observed=next(r for r in reversed(rows) if r['stage']=='conv0_groupnorm')
                    norm_diagnostics.append(dict(encoder=mid,split=split,epsilon=eps,scale=scale,
                        predicted_groupnorm_relative_l2=float(np.sqrt(delta_energy/base_energy)),
                        observed_groupnorm_relative_l2=observed['relative_l2'],
                        conv_scaling_maxabs=float(np.max(np.abs(changed['conv0_linear']-linear*np.float32(scale)))),
                        channel_variance_quantiles=np.quantile(variance,[0,.01,.1,.5,.9,.99,1]).tolist(),
                        channels_variance_below_1e_5=int(np.sum(variance<1e-5))))
                    del changed
                    persist()
                if eps!=1e-5: del base
            for scale in (.5,2.):
                changed,folder=forward(1e-5*scale*scale,scale,f'compensated_scale{scale:g}')
                compare(default,changed,folder,dict(kind='epsilon_compensated',epsilon=1e-5*scale*scale,scale=scale))
                controls.append(dict(encoder=mid,split=split,kind='compensated_postnorm_all_stages_exact',scale=scale,
                    passed=all(np.array_equal(default[k],changed[k]) for k in default if k!='conv0_linear')))
                del changed; persist()
            del default
            gc.collect()
        norm.eps=1e-5
        after=state_sha(model); assert before==after
        controls.append(dict(encoder=mid,kind='state_dict_unchanged',passed=True,state_sha256=after))
        capture.close();del model,capture;gc.collect();torch.cuda.empty_cache();persist()
    assert len(calls)==60
    persist();write(OUT/'completion.json',dict(status='COMPLETED',forwards=len(calls),comparisons=len(rows),
        weights_unchanged=True,training_updates=0,peak_cuda_reserved=torch.cuda.max_memory_reserved(),protocol_sha256=sha(OUT/'protocol.json')))

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if OUT.exists(): write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
