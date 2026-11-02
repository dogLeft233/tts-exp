"""Baseline-preserving GN response interventions and matched-dose tail probes."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,json,hashlib,importlib.util,gc,time,traceback
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/deps')
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel
from metrics import anchor,matched,moments,summarize,norm

OUT=Path('F:/Datasets/tts-exp/encoder_gn_baseline_preserving_20261020_v1')
PREV=Path('F:/Datasets/tts-exp/encoder_layer_epsilon_20261020_v1')
WEIGHTS=Path('F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/weights')
HELPER=Path(__file__).parents[1]/'encoder_layer_epsilon_20261020/run.py'
spec=importlib.util.spec_from_file_location('original_capture_helper',HELPER);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
write,sha=old.write,old.sha
MIDS=('hubert_base','wav2vec2_base')
STAGES=('gn','conv7','projection','T1','T4','T8','T9','last')
DOSES=(1e-4,3e-4,1e-3,3e-3)
RANDOM_DOSES=(3e-4,1e-3)
SEEDS=(20261020,20261021)

class Capture:
    def __init__(self,model):
        self.target=None;self.capture_reference=True;self.arrays={};self.handles=[]
        self.gn=model.feature_extractor.conv_layers[0].layer_norm
        def inject(module,inputs,output):
            if self.capture_reference:
                self.conv=inputs[0].detach().clone()
            return output if self.target is None else self.target
        self.handles.append(self.gn.register_forward_hook(inject))
        def add(name,module,conv=False):
            def collect(module,inputs,output):
                while isinstance(output,(tuple,list)):output=output[0]
                a=output.detach().cpu().numpy()[0]
                if conv:a=a.T
                assert np.isfinite(a).all()
                self.arrays[name]=a.copy()
            self.handles.append(module.register_forward_hook(collect))
        add('gn',self.gn,True);add('conv7',model.feature_extractor.conv_layers[6],True)
        add('projection',model.feature_projection.projection)
        for number in (1,4,8,9,12):add('last' if number==12 else 'T'+str(number),model.encoder.layers[number-1])
    def close(self):
        for h in self.handles:h.remove()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.4)
    assert torch.cuda.mem_get_info()[0]>4<<30
    previous=json.loads((PREV/'protocol.json').read_text());assert sha(HELPER)==previous['source_script_sha256']
    manifest={x['path']:x['sha256'] for x in json.loads((PREV/'manifest.json').read_text())}
    params={}
    for mid in MIDS:
        path=PREV/mid/'train/eps_1e-05_scale1/norm_channel_stats.npz';assert sha(path)==manifest[path.relative_to(PREV).as_posix()]
        with np.load(path) as p:params[mid]=(p['gamma'].astype(np.float32),p['beta'].astype(np.float32))
        source=next(x['source'] for x in previous['models'] if x['id']==mid)
        for item in source['files']:assert sha(WEIGHTS/mid/item['file'])==item['sha256']
    permutation=np.random.default_rng(20261020).permutation(512);np.save(OUT/'gamma_permutation.npy',permutation)
    protocol=dict(status='FROZEN_BEFORE_FORWARD',models=previous['models'],parent=str(PREV),parent_manifest_sha256=sha(PREV/'manifest.json'),
        source_sha256=sha(__file__),metrics_sha256=sha(Path(__file__).with_name('metrics.py')),helper_sha256=sha(HELPER),
        inputs='same whole-split-standardized first320080-sample May train/test chunks; model input always scale1 during mediator replay',
        anchor='original_GN(scale1) + [functional_GN_theta(scale_a)-functional_GN_theta(scale1)], CPU float64 arithmetic then float32; beta stays original',
        intervention_location='first GroupNorm output before GELU; all model parameters and eps property unchanged',
        stages=STAGES,scales=[.5,2.],epsilon_grid=[0.,1e-7,1e-3],gamma_grid=['donor_raw_index','own_fixed_permutation'],
        natural_matched_relative_frobenius_doses=DOSES,random_doses=RANDOM_DOSES,random_families=['iid_gaussian','channel_modulated'],random_seeds=SEEDS,random_signs=[-1,1],
        random_definition='iid Gaussian in own GN coordinates OR 512 independent Gaussian coefficients times own (GNbase-beta); normalize global Frobenius dose; not acoustically realizable probes',
        comparisons='terminal unit-vector RMS change per measured GN relativeFrobenius dose; natural-direction and random-direction sensitivities separate',
        control='scale1 anchored target exact for every theta; full replay of one large-epsilon and one donor-gamma zero-response target; default repeat; native gain replay against old last states',
        full_forward_budget=160,per_model_chunk_forward_budget=40,training_updates=0,tfg_calls=0,syncnet_calls=0,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),precision='FP32 CUDA eager, TF32off deterministic eval; frozen'))
    write(OUT/'protocol.json',protocol)
    calls=[];rows=[];controls=[];target_ledger=[]
    def persist():
        write(OUT/'forward_calls.json',calls);write(OUT/'layer_metrics.json',rows);write(OUT/'controls.json',controls);write(OUT/'target_ledger.json',target_ledger)
        write(OUT/'progress.json',dict(status='RUNNING',forwards=len(calls),latest=calls[-1] if calls else None))
    for mid in MIDS:
        donor=next(x for x in MIDS if x!=mid)
        model,info=AutoModel.from_pretrained(WEIGHTS/mid,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
        assert not info['missing_keys'] and not info['mismatched_keys']
        model.eval().requires_grad_(False).cuda();capture=Capture(model)
        assert capture.gn.eps==1e-5 and capture.gn.num_groups==512
        for a,b in zip(params[mid],(capture.gn.weight,capture.gn.bias)):assert np.array_equal(a,b.detach().cpu().numpy())
        state_before=old.state_sha(model)
        write(OUT/mid/'loading.json',dict(loading=info,state_sha256=state_before))
        for split in ('train','test'):
            p=PREV/mid/(split+'_standardized_input.npy');assert sha(p)==manifest[p.relative_to(PREV).as_posix()]
            waveform=torch.from_numpy(np.load(p)[None]).cuda();base=None;local_calls=0
            def forward(label,target,meta):
                nonlocal base,local_calls
                folder=OUT/mid/split/label;folder.mkdir(parents=True)
                capture.target=None if target is None else torch.from_numpy(np.ascontiguousarray(target.T[None])).cuda()
                capture.arrays={};started=time.time()
                with torch.inference_mode():last=model(input_values=waveform).last_hidden_state[0].cpu().numpy()
                arrays=capture.arrays;assert tuple(arrays)==STAGES
                if base is None:base=arrays
                measurements={}
                for name in STAGES:
                    m=moments(base[name],arrays[name]);measurements[name]=m
                    rows.append(dict(encoder=mid,split=split,label=label,stage=name,**meta,**summarize(m)))
                np.savez_compressed(folder/'frame_moments.npz',**measurements)
                samples={k:a[np.linspace(0,len(a)-1,min(64,len(a)),dtype=int)] for k,a in arrays.items()}
                np.savez_compressed(folder/'stage_samples.npz',**samples);np.save(folder/'last_hidden_state.npy',last)
                if target is not None:
                    assert np.array_equal(target,arrays['gn'])
                    target_ledger.append(dict(encoder=mid,split=split,label=label,target_sha256=old.array_sha(target),target_relative_l2=rows[-8]['relative_l2'],metadata=meta))
                local_calls+=1
                calls.append(dict(encoder=mid,split=split,label=label,folder=folder.relative_to(OUT).as_posix(),**meta,seconds=time.time()-started,last_sha256=old.array_sha(last)))
                if meta['family'] in ('baseline','baseline_repeat','anchor_control'):
                    expected=np.load(PREV/mid/split/'eps_1e-05_scale1/last_hidden_state.npy');assert np.array_equal(last,expected)
                    controls.append(dict(encoder=mid,split=split,label=label,kind='baseline_exact',passed=True))
                    if meta['family']!='baseline':assert all(np.array_equal(base[k],arrays[k]) for k in STAGES)
                if meta['family']=='natural':
                    expected=np.load(PREV/mid/split/f"eps_1e-05_scale{meta['scale']:g}/last_hidden_state.npy");assert np.array_equal(last,expected)
                    controls.append(dict(encoder=mid,split=split,label=label,kind='natural_gain_bridge_exact',passed=True))
                if meta['family']=='anchored_epsilon' and meta['epsilon']==0:
                    assert all(np.array_equal(base[k],arrays[k]) for k in STAGES)
                    controls.append(dict(encoder=mid,split=split,label=label,kind='zero_epsilon_response_all_stages_exact',passed=True))
                print(json.dumps(dict(encoder=mid,split=split,label=label,forward=len(calls))),flush=True)
                persist();return arrays
            forward('baseline',None,dict(family='baseline'))
            capture.capture_reference=False
            y0=base['gn'];g0,b0=params[mid]
            forward('baseline_repeat',None,dict(family='baseline_repeat'))
            def functional(eps,gamma,scale):
                with torch.inference_mode():
                    out=F.group_norm(capture.conv*scale,512,torch.from_numpy(gamma).cuda(),capture.gn.bias,eps)
                    return out[0].cpu().numpy().T.copy()
            native0=functional(1e-5,g0,1.);assert np.array_equal(native0,y0)
            natural={scale:functional(1e-5,g0,scale) for scale in (.5,2.)}
            for scale in natural:
                assert np.array_equal(anchor(y0,natural[scale],native0),natural[scale])
                forward(f'natural_{scale:g}',natural[scale],dict(family='natural',scale=scale))
            reference_folder=OUT/mid/split
            np.savez_compressed(reference_folder/'mediator_reference.npz',conv=capture.conv[0].cpu().numpy().T,
                gn=y0,natural_half=natural[.5],natural_double=natural[2.],gamma=g0,beta=b0)
            for eps in (0.,1e-7,1e-3):
                theta0=functional(eps,g0,1.)
                zero=anchor(y0,theta0,theta0);assert np.array_equal(zero,y0)
                controls.append(dict(encoder=mid,split=split,kind='epsilon_theta_scale1_target_exact',epsilon=eps,passed=True))
                if eps==1e-3:forward('anchor_control_epsilon',zero,dict(family='anchor_control',intervention='epsilon'))
                for scale in (.5,2.):
                    changed=functional(eps,g0,scale);target=anchor(y0,changed,theta0)
                    forward(f'anchored_eps_{eps:.0e}_{scale:g}',target,dict(family='anchored_epsilon',epsilon=eps,scale=scale))
                del theta0
            for kind,gamma in [('donor_raw_index',params[donor][0]),('own_fixed_permutation',g0[permutation])]:
                theta0=functional(1e-5,gamma,1.);zero=anchor(y0,theta0,theta0);assert np.array_equal(zero,y0)
                controls.append(dict(encoder=mid,split=split,kind='gamma_theta_scale1_target_exact',gamma=kind,passed=True))
                if kind=='donor_raw_index':forward('anchor_control_gamma',zero,dict(family='anchor_control',intervention='gamma'))
                for scale in (.5,2.):
                    target=anchor(y0,functional(1e-5,gamma,scale),theta0)
                    forward(f'anchored_gamma_{kind}_{scale:g}',target,dict(family='anchored_gamma',gamma=kind,scale=scale))
                del theta0
            for scale in (.5,2.):
                direction=natural[scale].astype(float)-y0.astype(float)
                for dose in DOSES:
                    target=matched(y0,direction,dose)
                    forward(f'matched_natural_{scale:g}_{dose:.0e}',target,dict(family='matched_natural',scale=scale,dose=dose))
                del direction
            signal=y0.astype(float)-b0[None,:]
            probe_ledger=[]
            for family in ('iid_gaussian','channel_modulated'):
                for seed in SEEDS:
                    rng=np.random.default_rng(seed)
                    direction=rng.standard_normal(y0.shape,dtype=np.float32) if family=='iid_gaussian' else (signal*rng.standard_normal(512)[None,:]).astype(np.float32)
                    probe_ledger.append(dict(family=family,seed=seed,direction_sha256=old.array_sha(direction),direction_norm=norm(direction)))
                    for dose in RANDOM_DOSES:
                        for sign in (-1,1):
                            target=matched(y0,direction,sign*dose)
                            forward(f'probe_{family}_{seed}_{dose:.0e}_{sign:+d}',target,dict(family='random_probe',probe_family=family,seed=seed,dose=dose,sign=sign))
                    del direction
            write(reference_folder/'probe_ledger.json',probe_ledger)
            assert local_calls==40
            capture.target=None;del base,y0,natural,native0,signal,waveform;gc.collect()
            capture.capture_reference=True
        assert old.state_sha(model)==state_before
        controls.append(dict(encoder=mid,kind='all_model_parameters_unchanged_exact',passed=True,state_sha256=state_before))
        capture.close();del model,capture;gc.collect();torch.cuda.empty_cache();persist()
    assert len(calls)==160 and len(rows)==1280
    write(OUT/'completion.json',dict(status='COMPLETED',forwards=len(calls),layer_comparisons=len(rows),training_updates=0,
        weights_unchanged=True,peak_cuda_reserved=torch.cuda.max_memory_reserved(),protocol_sha256=sha(OUT/'protocol.json')))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
