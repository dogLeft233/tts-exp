"""Swap only first-GroupNorm affine parameters; frozen, no training."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,json,hashlib,importlib.util,time,gc,traceback
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/deps')
import numpy as np
import torch
import transformers
from transformers import AutoModel

PREV=Path('F:/Datasets/tts-exp/encoder_layer_epsilon_20261020_v1')
OUT=Path('F:/Datasets/tts-exp/encoder_groupnorm_swap_20261020_v1')
WEIGHTS=Path('F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/weights')
HELPER=Path(__file__).parents[1]/'encoder_layer_epsilon_20261020/run.py'
spec=importlib.util.spec_from_file_location('frozen_epsilon_helper',HELPER)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,write,moments,summarize=old.sha,old.write,old.moments,old.summarize
MIDS=('hubert_base','wav2vec2_base')
ARMS=('native','swapped','self_permuted')

def non_gn_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name in ('feature_extractor.conv_layers.0.layer_norm.weight','feature_extractor.conv_layers.0.layer_norm.bias'):continue
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.4)
    assert torch.cuda.mem_get_info()[0]>4<<30
    parent=json.loads((PREV/'protocol.json').read_text())
    assert sha(HELPER)==parent['source_script_sha256']
    manifest={x['path']:x['sha256'] for x in json.loads((PREV/'manifest.json').read_text())}
    params={}
    for mid in MIDS:
        p=PREV/mid/'train/eps_1e-05_scale1/norm_channel_stats.npz'
        assert sha(p)==manifest[p.relative_to(PREV).as_posix()]
        with np.load(p) as z:params[mid]=(z['gamma'].astype(np.float32),z['beta'].astype(np.float32))
        source=next(x['source'] for x in parent['models'] if x['id']==mid)
        for item in source['files']:assert sha(WEIGHTS/mid/item['file'])==item['sha256']
    permutation=np.random.default_rng(20261020).permutation(512)
    np.save(OUT/'channel_permutation.npy',permutation)
    protocol=dict(status='FROZEN_BEFORE_FORWARD',parent=str(PREV),parent_manifest_sha256=sha(PREV/'manifest.json'),
        models=parent['models'],arms=list(ARMS),scales=[1.,.5,2.],splits=['train','test'],samples_per_chunk=320080,
        swap='first GroupNorm gamma and beta exchanged by raw channel index; epsilon 1e-5, groups512 and all other weights unchanged',
        control='own gamma/beta jointly permuted by one fixed seed20261020 permutation, shared between models; no channel matching or optimization',
        input='exact report58 standardized 20.005s chunk; then float32 times scale',
        direction_metrics='all-coordinate paired per-frame cosine/angles and relativeL2; cross-model raw cos descriptive only; 128 deterministic time samples for centered linear CKA and perturbation-direction comparisons',
        limitation='512 convolution channels and 768 hidden coordinates are not aligned across separately trained models; raw channel swap is an off-manifold intervention',
        full_forwards=40,training_updates=0,syncnet_calls=0,tfg_calls=0,source_sha256=sha(__file__),helper_sha256=sha(HELPER),
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),precision='FP32 eager TF32 off deterministic eval'))
    write(OUT/'protocol.json',protocol)
    calls=[];rows=[];controls=[];predictions=[]
    def save():
        write(OUT/'forward_calls.json',calls);write(OUT/'layer_metrics.json',rows);write(OUT/'controls.json',controls);write(OUT/'groupnorm_predictions.json',predictions)
        write(OUT/'progress.json',dict(status='RUNNING',forwards=len(calls),comparisons=len(rows),latest=calls[-1] if calls else None))
    for mid in MIDS:
        donor=next(x for x in MIDS if x!=mid)
        model,loading=AutoModel.from_pretrained(WEIGHTS/mid,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
        assert not loading['missing_keys'] and not loading['mismatched_keys']
        model.eval().requires_grad_(False).cuda();capture=old.Capture(model)
        gn=model.feature_extractor.conv_layers[0].layer_norm
        assert gn.eps==1e-5 and gn.num_groups==gn.num_channels==512
        for a,b in zip(params[mid],(gn.weight,gn.bias)):assert np.array_equal(a,b.detach().cpu().numpy())
        original_state=old.state_sha(model);unaffected=non_gn_sha(model)
        write(OUT/mid/'loading.json',dict(loading=loading,original_state_sha256=original_state,non_gn_sha256=unaffected))
        for split in ('train','test'):
            input_path=PREV/mid/(split+'_standardized_input.npy')
            assert sha(input_path)==manifest[input_path.relative_to(PREV).as_posix()]
            x=np.load(input_path);assert x.shape==(320080,) and x.dtype==np.float32
            native_base=None
            def forward(arm,scale,label=None):
                folder=OUT/mid/split/arm/(label or f'scale{scale:g}');folder.mkdir(parents=True)
                capture.arrays={};start=time.time()
                with torch.inference_mode():last=model(input_values=torch.from_numpy((x*np.float32(scale))[None]).cuda()).last_hidden_state[0].cpu().numpy()
                arrays=capture.arrays;arrays['last_hidden_state']=last.copy()
                np.save(folder/'last_hidden_state.npy',last)
                sampled={name:a[np.linspace(0,len(a)-1,min(128,len(a)),dtype=int)] for name,a in arrays.items()}
                np.savez_compressed(folder/'stage_samples.npz',**sampled)
                calls.append(dict(encoder=mid,split=split,arm=arm,scale=scale,folder=folder.relative_to(OUT).as_posix(),repeat=bool(label),seconds=time.time()-start,last_sha256=old.array_sha(last)))
                if arm=='native':
                    previous=PREV/mid/split/f'eps_1e-05_scale{scale:g}'/'last_hidden_state.npy'
                    expected=np.load(previous);assert np.array_equal(last,expected)
                    controls.append(dict(encoder=mid,split=split,scale=scale,kind='parent_last_state_exact',passed=True))
                print(json.dumps(calls[-1]),flush=True)
                return arrays,folder
            def compare(a,b,folder,kind,arm,scale):
                evidence={}
                for name in a:
                    m=moments(a[name],b[name]);evidence[name]=m
                    result=summarize(m)
                    valid=(m[:,0]>0)&(m[:,1]>0)
                    result['mean_cosine']=float(np.mean(np.clip(m[valid,3]/np.sqrt(m[valid,0]*m[valid,1]),-1,1)))
                    rows.append(dict(encoder=mid,split=split,arm=arm,scale=scale,kind=kind,stage=name,shape=list(a[name].shape),**result))
                np.savez_compressed(folder/(kind+'_moments.npz'),**evidence)
            for arm in ARMS:
                g,b=params[mid] if arm=='native' else params[donor] if arm=='swapped' else tuple(v[permutation] for v in params[mid])
                with torch.no_grad():gn.weight.copy_(torch.from_numpy(g).cuda());gn.bias.copy_(torch.from_numpy(b).cuda())
                assert gn.eps==1e-5
                base,folder=forward(arm,1.)
                np.savez_compressed(folder/'applied_parameters.npz',gamma=g,beta=b)
                if arm=='native':
                    native_base=base
                    repeat,repeat_folder=forward(arm,1.,'repeat_scale1')
                    exact=all(np.array_equal(base[k],repeat[k]) for k in base);assert exact
                    controls.append(dict(encoder=mid,split=split,kind='all_stage_repeat_exact',passed=True));del repeat
                else:
                    assert np.array_equal(base['conv0_linear'],native_base['conv0_linear'])
                    controls.append(dict(encoder=mid,split=split,arm=arm,kind='upstream_conv_unchanged',passed=True))
                    compare(native_base,base,folder,'baseline_drift',arm,1.)
                v=np.var(base['conv0_linear'].astype(float),axis=0)
                for scale in (.5,2.):
                    changed,c_folder=forward(arm,scale)
                    compare(base,changed,c_folder,'scale_response',arm,scale)
                    q=np.sqrt((v+1e-5)/(v+1e-5/scale**2))
                    expected=float(np.sqrt(np.sum(g.astype(float)**2*v/(v+1e-5)*(q-1)**2)/np.sum(g.astype(float)**2*v/(v+1e-5)+b.astype(float)**2)))
                    actual=next(z for z in reversed(rows) if z['stage']=='conv0_groupnorm')['relative_l2']
                    predictions.append(dict(encoder=mid,split=split,arm=arm,scale=scale,predicted=expected,actual=actual))
                    assert abs(actual-expected)<1e-6
                    del changed;save()
                assert non_gn_sha(model)==unaffected
                if arm!='native':del base
            del native_base;gc.collect()
        with torch.no_grad():gn.weight.copy_(torch.from_numpy(params[mid][0]).cuda());gn.bias.copy_(torch.from_numpy(params[mid][1]).cuda())
        assert old.state_sha(model)==original_state
        controls.append(dict(encoder=mid,kind='original_full_state_restored_exact',passed=True))
        capture.close();del model,capture;gc.collect();torch.cuda.empty_cache();save()
    assert len(calls)==40 and len(rows)==832
    save();write(OUT/'completion.json',dict(status='COMPLETED',forwards=40,layer_comparisons=len(rows),training_updates=0,
        original_states_restored=True,peak_cuda_reserved=torch.cuda.max_memory_reserved(),protocol_sha256=sha(OUT/'protocol.json')))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
