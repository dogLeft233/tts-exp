"""Known additive-noise factorial audit of frozen May wav2vec2. Never trains."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import sys,argparse,json,hashlib,importlib.util,time,gc,traceback,shutil
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/deps')
import numpy as np
import torch,transformers,soundfile as sf
from transformers import AutoModel

PRIOR=Path('F:/Datasets/tts-exp/wav2vec2_weak_signal_20261020_v1')
WEIGHTS=Path('F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/weights/wav2vec2_base')
SEEDS=[20261020,20261021,20261022]
STAGES=['first_gn','cnn7','projection','transformer4','transformer8','transformer9','last']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)


def noise_template(length,kind,seed):
    x=np.random.default_rng(seed).standard_normal(length)
    if kind=='pink':
        f=np.fft.rfft(x);weights=np.arange(len(f),dtype=float);weights[0]=1
        f/=np.sqrt(weights);f[0]=0;x=np.fft.irfft(f,n=length)
    elif kind!='white':raise ValueError('unsupported noise')
    x-=x.mean();x/=np.sqrt(np.mean(x*x));return x.astype(np.float32)


def mixture(s,n,arm):
    if s.shape!=n.shape or s.dtype!=np.float32 or n.dtype!=np.float32:raise ValueError('input contract')
    if arm=='A':return (s+n).astype(np.float32)
    if arm=='B':return ((s+n)*np.float32(.5)).astype(np.float32)
    if arm=='C':return (s+n*np.float32(.5)).astype(np.float32)
    if arm=='D':return (s*np.float32(.5)+n).astype(np.float32)
    raise ValueError('unknown arm')


def response(m):
    xx,yy,dd,xy=m[:,:4].sum(0)
    if xx<=0:raise ValueError('zero reference energy')
    valid=(m[:,0]>0)&(m[:,1]>0)
    cos=np.clip(m[valid,3]/np.sqrt(m[valid,0]*m[valid,1]),-1,1)
    unit_sum=float(np.sum(np.maximum(0,2-2*cos)))
    return dict(xx=float(xx),yy=float(yy),dd=float(dd),xy=float(xy),frames=len(m),
                relative_noise_change=float(np.sqrt(dd/xx)),
                unit_delta_square_sum=unit_sum,unit_valid_frames=int(valid.sum()),
                unit_noise_change=float(np.sqrt(unit_sum/valid.sum())))


class Capture:
    def __init__(self,model):
        self.arrays={};self.handles=[]
        modules=[('first_gn',model.feature_extractor.conv_layers[0].layer_norm,True),
                 ('cnn7',model.feature_extractor.conv_layers[6],True),
                 ('projection',model.feature_projection.projection,False)]+[
                 (f'transformer{i}',model.encoder.layers[i-1],False) for i in [4,8,9]]
        for name,module,conv in modules:
            def hook(_m,_inputs,result,name=name,conv=conv):
                while isinstance(result,(tuple,list)):result=result[0]
                x=result.detach().cpu().numpy()[0]
                if conv:x=x.T
                assert x.ndim==2 and np.isfinite(x).all()
                self.arrays[name]=x.copy()
            self.handles.append(module.register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def main(out):
    out.mkdir(parents=True,exist_ok=False);assert shutil.disk_usage(out).free>5<<30
    prior_final=json.loads((PRIOR/'final_receipt.json').read_text())
    assert prior_final['passed'] and sha(PRIOR/'delivery_manifest.json')==prior_final['manifest_sha256']
    prior_manifest=json.loads((PRIOR/'delivery_manifest.json').read_text())['members']
    prior_protocol=json.loads((PRIOR/'protocol.json').read_text())
    src=Path(__file__).resolve().parents[1]/'wav2vec2_weak_signal_20261020/run.py'
    assert sha(src)==prior_protocol['script_sha256']
    spec=importlib.util.spec_from_file_location('sealed_metrics',src);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    for row in prior_protocol['source']['files']:assert sha(WEIGHTS/row['file'])==row['sha256']
    (out/'sources').mkdir();shutil.copyfile(__file__,out/'sources/run.py');shutil.copyfile(src,out/'sources/paired_metrics.py')
    shutil.copyfile(PRIOR/'protocol.json',out/'sources/prior_protocol.json')
    pins=json.loads((PRIOR/'input_pins.json').read_text());inputs={};noises={};noise_pins=[]
    for ix,split in enumerate(['train','val','test']):
        p=next(p for p in pins if p['split']==split)
        assert sha(p['source_audio'])==p['source_sha256']
        raw,sr=sf.read(p['source_audio'],dtype='float64');assert sr==16000
        x=raw.astype(np.float32);s=(x-x.mean())/np.sqrt(x.var()+1e-7)
        assert digest(s)==p['native_sha256'] and digest(s*.5)==p['half_sha256']
        inputs[split]=s;np.save(out/f'{split}_native.npy',s)
        rms=float(np.sqrt(np.mean(s.astype(float)**2)))
        for kind in ['white','pink']:
            for seed in SEEDS:
                actual_seed=seed+ix*100000
                unit=noise_template(len(s),kind,actual_seed)
                noise=(unit*np.float32(rms*10**(-30/20))).astype(np.float32)
                noises[(split,kind,seed)]=noise
                pth=out/f'noise/{split}_{kind}_{seed}.npy';pth.parent.mkdir(exist_ok=True)
                np.save(pth,noise)
                nrms=float(np.sqrt(np.mean(noise.astype(float)**2)))
                noise_pins.append(dict(split=split,kind=kind,seed=seed,generator_seed=actual_seed,
                    path=str(pth.relative_to(out)),sha256=sha(pth),samples=len(noise),
                    native_rms=rms,noise_rms=nrms,added_noise_snr_db=float(20*np.log10(rms/nrms)),
                    note='SNR relative to original May content including its intrinsic noise; not clean-speech SNR'))
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.4);assert torch.cuda.mem_get_info()[0]>4<<30
    model,loading=AutoModel.from_pretrained(WEIGHTS,local_files_only=True,attn_implementation='eager',
        output_loading_info=True,low_cpu_mem_usage=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys']
    model.eval().requires_grad_(False).cuda();norm=model.feature_extractor.conv_layers[0].layer_norm
    assert isinstance(norm,torch.nn.GroupNorm) and norm.eps==1e-5
    # Local state hash has no dependency on the prior producer's main routine.
    def state_sha():
        h=hashlib.sha256()
        for key,tensor in model.state_dict().items():h.update(key.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=state_sha();assert before==prior_protocol['state_sha256']
    capture=Capture(model)
    protocol=dict(status='FROZEN_BEFORE_FORWARD',source=prior_protocol['source'],prior=str(PRIOR),
        input_pins=pins,noise_pins=noise_pins,seeds=SEEDS,noise_types=['white','pink'],
        snr_db=30,arms=dict(A=[1,1],B=[.5,.5],C=[1,.5],D=[.5,1]),
        arm_snr_db=dict(A=30,B=30,C=36.020599913,D=23.979400087),
        clean_anchors='Original May without added noise, scales1/.5; not assumed noise-free',
        transform='whole-split historical zscore once, then additive mixing; no per-mixture normalization or clipping',
        chunk_hop=320000,chunk_max_samples=320080,stages=STAGES,
        planned_forwards=449,training_updates=0,generator_forwards=0,syncnet_calls=0,
        primary_endpoint='last layer unit-vector noise-induced change versus matching-scale no-added-noise anchor',
        secondary_endpoint='relative L2 noise-induced change; baseline waveform RMS q20/q80 strata',
        primary_contrasts=['B/A same added-noise SNR, lower whole scale','C/A only added noise halved','D/B add noise back at low May scale'],
        compensated='B input, firstGN epsilon2.5e-6; compare A input epsilon1e-5',
        script_sha256=sha(__file__),paired_metrics_sha256=sha(src),state_sha256=before,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,python=sys.version,gpu=torch.cuda.get_device_name()))
    write(out/'protocol.json',protocol)
    rows=[];controls=[];calls=[]
    def persist():
        write(out/'noise_response_rows.json',rows);write(out/'controls.json',controls)
        write(out/'forward_calls.json',calls);write(out/'progress.json',dict(completed_forwards=len(calls),planned=449,latest=calls[-1] if calls else None))
    for split in ['train','val','test']:
        s=inputs[split];pin=next(p for p in pins if p['split']==split);thresholds=np.array(pin['thresholds'])
        for chunk,start in enumerate(range(0,len(s),320000)):
            end=min(start+320080,len(s));piece=s[start:end]
            if len(piece)<400:continue
            folder=out/split/f'chunk{chunk:02d}';folder.mkdir(parents=True)
            def forward(y,label,eps=1e-5):
                norm.eps=eps;capture.arrays={};begin=time.monotonic()
                with torch.inference_mode():last=model(input_values=torch.from_numpy(y[None]).cuda()).last_hidden_state[0].cpu().numpy()
                values=capture.arrays;values['last']=last.copy();assert set(values)==set(STAGES)
                p=folder/label;p.mkdir(parents=True)
                np.save(p/'last.npy',last)
                sample={}
                for name,a in values.items():
                    idx=np.linspace(0,len(a)-1,32,dtype=int);sample[name]=a[idx];sample[name+'__indices']=idx
                np.savez_compressed(p/'samples.npz',**sample)
                calls.append(dict(split=split,chunk=chunk,label=label,epsilon=eps,seconds=time.monotonic()-begin,input_sha256=digest(y),last_sha256=digest(last)))
                return values
            anchor1=forward(piece,'anchor1');anchorhalf=forward(piece*.5,'anchorhalf')
            for scale,anchor in [(1,anchor1),(.5,anchorhalf)]:
                old=PRIOR/split/f'chunk{chunk:02d}'/('baseline' if scale==1 else 'half')/'last_hidden_state.npy'
                expected=next(p for p in prior_manifest if p['path']==old.relative_to(PRIOR).as_posix())
                assert sha(old)==expected['sha256'] and np.array_equal(np.load(old),anchor['last'])
            labels={}
            for name,a in anchor1.items():
                centers=start+np.arange(len(a))*(5 if name=='first_gn' else 320)+(4 if name=='first_gn' else 199)
                energy=helper.energy_at(s,centers)
                labels[name]=np.where(energy<=thresholds[0],0,np.where(energy>thresholds[1],2,1)).astype(np.int8)
            for kind in ['white','pink']:
                for seed in SEEDS:
                    noise=noises[(split,kind,seed)][start:end];case=f'{kind}_{seed}'
                    a_reference=None
                    for arm in ['A','B','C','D']:
                        y=mixture(piece,noise,arm)
                        if arm=='B':assert np.array_equal(y,piece*.5+noise*.5)
                        changed=forward(y,f'{case}/{arm}')
                        anchor=anchor1 if arm in ['A','C'] else anchorhalf
                        frames={};channels={}
                        for name,x in anchor.items():
                            fm,cm=helper.pair_moments(x,changed[name]);frames[name]=fm;channels[name]=cm
                            frames[name+'__stratum']=labels[name]
                            for k,stratum in [(-1,'all'),(0,'low20'),(1,'middle60'),(2,'high20')]:
                                use=fm if k==-1 else fm[labels[name]==k]
                                if len(use):rows.append(dict(split=split,chunk=chunk,kind=kind,seed=seed,arm=arm,stage=name,stratum=stratum,**response(use)))
                        destination=folder/case/arm
                        np.savez_compressed(destination/'frame_moments.npz',**frames)
                        np.savez_compressed(destination/'channel_moments.npz',**channels)
                        if arm=='A':
                            a_reference=changed
                            if split=='train' and chunk==0 and kind=='white' and seed==SEEDS[0]:
                                repeat=forward(y,f'{case}/repeatA')
                                assert all(np.array_equal(changed[k],repeat[k]) for k in STAGES)
                                controls.append(dict(kind='repeatA_all_stages_exact',passed=True));del repeat
                        else:del changed
                    compensated=forward(mixture(piece,noise,'B'),f'{case}/B_compensated',2.5e-6)
                    assert all(np.array_equal(a_reference[k],compensated[k]) for k in STAGES)
                    controls.append(dict(kind='epsilon_compensation_all_stages_exact',split=split,chunk=chunk,
                                         noise_kind=kind,seed=seed,passed=True))
                    del a_reference,compensated
                    persist();print(json.dumps(dict(forwards=len(calls),split=split,chunk=chunk,case=case)),flush=True)
            del anchor1,anchorhalf;gc.collect()
    norm.eps=1e-5;assert state_sha()==before
    capture.close();del capture,model;gc.collect();torch.cuda.empty_cache();persist()
    assert len(calls)==449
    write(out/'completion.json',dict(status='COMPLETED',forwards=len(calls),chunks=14,training_updates=0,
        generator_forwards=0,syncnet_calls=0,state_unchanged=True,protocol_sha256=sha(out/'protocol.json'),
        peak_cuda_reserved=torch.cuda.max_memory_reserved()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:main(args.output)
    except BaseException as exc:
        if args.output.exists():write(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
