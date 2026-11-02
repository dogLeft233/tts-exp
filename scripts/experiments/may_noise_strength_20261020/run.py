"""Frozen May wav2vec2 noise dose scan; independent of training and video."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import sys,json,argparse,hashlib,importlib.util,time,gc,shutil,traceback
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/deps')
import numpy as np
import torch,transformers,soundfile as sf
from transformers import AutoModel

PRIOR=Path('F:/Datasets/tts-exp/may_known_noise_20261020_v1')
WEIGHTS=Path('F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/weights/wav2vec2_base')
GRID=list(range(0,101,10))+list(range(120,261,20))
SEEDS=[20261020,20261021,20261022]
STAGES=['first_gn','cnn7','projection','transformer4','transformer8','transformer9','last']
SOURCE=Path(__file__).resolve().parents[1]/'may_known_noise_20261020/run.py'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)

def metric_frames(ref,value):
    """Stable even below sqrt(machine epsilon): never use 2-2*cos."""
    if ref.shape!=value.shape:raise ValueError('shape mismatch')
    out=np.empty((len(ref),6),dtype=np.float64)
    for start in range(0,len(ref),1024):
        stop=min(start+1024,len(ref));x=ref[start:stop].astype(np.float64);y=value[start:stop].astype(np.float64)
        xx=np.sum(x*x,axis=1);yy=np.sum(y*y,axis=1)
        if np.any(xx==0) or np.any(yy==0):raise ValueError('undefined direction')
        d=y-x;u=y/np.sqrt(yy[:,None])-x/np.sqrt(xx[:,None])
        out[start:stop]=np.column_stack((xx,yy,np.sum(d*d,1),np.sum(u*u,1),np.max(np.abs(d),1),np.sum(d!=0,1)))
    return out

def summarize(m,channels):
    return dict(frames=len(m),channels=channels,unit_delta=float(np.sqrt(m[:,3].mean())),
        relative_l2=float(np.sqrt(m[:,2].sum()/m[:,0].sum())),max_abs=float(m[:,4].max()),
        changed_coordinate_fraction=float(m[:,5].sum()/(len(m)*channels)),
        exact_frame_fraction=float(np.mean(m[:,5]==0)))

def dose(s,n30,q,precision):
    dtype=np.float32 if precision=='fp32' else np.float64
    return s.astype(dtype),n30.astype(dtype)*dtype(10**((30-q)/20))

def waveform_audit(s,n):
    a=s+n;b=a*s.dtype.type(.5);r=s*s.dtype.type(.5)
    da=a-s;db=b-r
    assert np.array_equal(a*.5,b) and np.array_equal(da,db*2)
    rms=lambda x:float(np.sqrt(np.mean(x.astype(np.float64)**2)))
    return dict(input_changed_fraction=float(np.mean(da!=0)),half_changed_fraction=float(np.mean(db!=0)),
        exact_power_two_difference=True,intended_noise_rms=rms(n),realized_noise_rms=rms(da),
        measured_prefix_snr_db=float(20*np.log10(rms(s)/rms(n))),
        addition_error_relative_l2=float(np.linalg.norm(da.astype(float)-n.astype(float))/np.linalg.norm(n.astype(float))))

def main(out):
    out.mkdir(parents=True,exist_ok=False);assert shutil.disk_usage(out).free>10<<30
    prior_final=json.loads((PRIOR/'final_receipt.json').read_text());assert prior_final['passed']
    assert sha(PRIOR/'delivery_manifest.json')==prior_final['manifest_sha256']
    manifest={r['path']:r for r in json.loads((PRIOR/'delivery_manifest.json').read_text())['members']}
    protocol=json.loads((PRIOR/'protocol.json').read_text())
    assert sha(PRIOR/'protocol.json')==manifest['protocol.json']['sha256']
    assert sha(SOURCE)==protocol['script_sha256']=='f7a6d930c34656de02853c9710b2adc6d3ccd3a082ee842910062d0f5b17e575'
    spec=importlib.util.spec_from_file_location('sealed_capture',SOURCE);prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    for r in protocol['source']['files']:assert sha(WEIGHTS/r['file'])==r['sha256']
    inputs={};noises={};pins=[]
    for split in ['train','test']:
        p=next(r for r in protocol['input_pins'] if r['split']==split)
        assert sha(p['source_audio'])==p['source_sha256']
        raw,sr=sf.read(p['source_audio'],dtype='float64');assert sr==16000
        x=raw.astype(np.float32);s=(x-x.mean())/np.sqrt(x.var()+1e-7)
        assert digest(s)==p['native_sha256'];inputs[split]=s[:320080].copy()
        np.save(out/(split+'_native.npy'),inputs[split]);pins.append(p)
        for kind in ['white','pink']:
            for seed in SEEDS:
                r=next(r for r in protocol['noise_pins'] if r['split']==split and r['kind']==kind and r['seed']==seed)
                assert sha(PRIOR/r['path'])==r['sha256']==manifest[Path(r['path']).as_posix()]['sha256']
                noises[split,kind,seed]=np.load(PRIOR/r['path'])[:320080].copy()
                pth=out/r['path'];pth.parent.mkdir(exist_ok=True);np.save(pth,noises[split,kind,seed])
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.4)
    model,loading=AutoModel.from_pretrained(WEIGHTS,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys']
    model.eval().requires_grad_(False).cuda();norm=model.feature_extractor.conv_layers[0].layer_norm
    state_sha=lambda:hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest()
    before=state_sha();assert before==protocol['state_sha256']
    capture=prior.Capture(model)
    design=dict(status='FROZEN_BEFORE_FORWARD',prior=str(PRIOR),prior_manifest_sha256=prior_final['manifest_sha256'],
        source=protocol['source'],input_pins=pins,noise_pins=[r for r in protocol['noise_pins'] if r['split'] in inputs],
        snr_grid_db=GRID,fp64_grid_db=[120,200,260],compensation_grid_db=[30,100,180,260],seeds=SEEDS,
        split_chunks={'train':[0,320080],'test':[0,320080]},stages=STAGES,script_sha256=sha(__file__),capture_sha256=sha(SOURCE),
        scale=0.5,arms={'A':'s+n','B':'0.5*(s+n)'},anchors={'A':'s','B':'0.5*s'},
        normalization='whole-split original float32 standardization once; no mixture normalization/clipping',
        dose='same sealed whole-split30dB noises; n_q=n30*10**((30-q)/20); nominal whole-split SNR, actual prefix SNR also recorded',
        primary_endpoint='last RMS direct float64 unit-vector subtraction; ratio B/A',secondary='relative L2, all seven stages, changed-coordinate and exact-frame fractions',
        threshold_rule='candidate: B/A<=0.8 at three consecutive resolved grid levels, all 3 seeds and both chunks within same noise profile; descriptive only',
        precision_rule='resolved_fp32 iff both A and B unit deltas exceed 10x respective clean FP32-vs-FP64 directional discrepancy',
        numerical_controls='clean repeats exact; first GN epsilon2.5e-6 at scale0.5 versus default1e-5 at scale1; FP64 same originallyFP32 weights',
        planned_forwards=538,training_updates=0,generator_forwards=0,syncnet_calls=0,state_sha256=before,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name()))
    write(out/'protocol.json',design);(out/'sources').mkdir();shutil.copyfile(__file__,out/'sources/run.py');shutil.copyfile(SOURCE,out/'sources/sealed_capture.py')
    rows=[];calls=[];controls=[];waveforms=[]
    def persist():
        for name,v in [('rows',rows),('forward_calls',calls),('controls',controls),('waveform_audit',waveforms)]:write(out/(name+'.json'),v)
        write(out/'progress.json',dict(completed=len(calls),planned=538,latest=calls[-1] if calls else None))
    def forward(y,folder,label,eps=1e-5,all_arrays=False):
        norm.eps=eps;capture.arrays={};start=time.monotonic()
        with torch.inference_mode():last=model(input_values=torch.from_numpy(y[None]).cuda()).last_hidden_state[0].cpu().numpy()
        arrays=capture.arrays;arrays['last']=last.copy();assert set(arrays)==set(STAGES)
        path=folder/label;path.mkdir(parents=True);np.save(path/'last.npy',last)
        sample={}
        for k,a in arrays.items():
            idx=np.linspace(0,len(a)-1,64,dtype=int);sample[k]=a[idx];sample[k+'__indices']=idx
            if all_arrays and k!='last':np.save(path/(k+'.npy'),a)
        np.savez_compressed(path/'samples.npz',**sample)
        calls.append(dict(path=path.relative_to(out).as_posix(),dtype=str(y.dtype),epsilon=eps,input_sha256=digest(y),last_sha256=digest(last),seconds=time.monotonic()-start))
        return arrays
    def compare(anchor,arrays,folder,meta):
        frames={}
        for stage in STAGES:
            fm=metric_frames(anchor[stage],arrays[stage]);frames[stage]=fm
            rows.append(dict(**meta,stage=stage,**summarize(fm,arrays[stage].shape[1])))
        np.savez_compressed(folder/'frames.npz',**frames)
    for precision in ['fp32','fp64']:
        if precision=='fp64':model.double();gc.collect();torch.cuda.empty_cache()
        for split in ['train','test']:
            folder=out/precision/split;folder.mkdir(parents=True)
            s=inputs[split].astype(np.float32 if precision=='fp32' else np.float64)
            anchors={arm:forward(s*scale,folder,'anchor'+arm,all_arrays=True) for arm,scale in [('A',1),('B',.5)]}
            if precision=='fp32':
                for arm in ['A','B']:
                    old=f'{split}/chunk00/'+('anchor1' if arm=='A' else 'anchorhalf')+'/last.npy'
                    assert sha(PRIOR/old)==manifest[old]['sha256'] and np.array_equal(np.load(PRIOR/old),anchors[arm]['last'])
                repeat=forward(s,folder,'repeatA');assert all(np.array_equal(repeat[k],anchors['A'][k]) for k in STAGES)
                controls.append(dict(kind='clean_repeat',split=split,all_stages_exact=True));del repeat
            else:
                for arm in ['A','B']:
                    a32={k:np.load(out/'fp32'/split/('anchor'+arm)/(k+'.npy'),mmap_mode='r') for k in STAGES}
                    p=folder/('precision_anchor'+arm);p.mkdir()
                    compare(a32,anchors[arm],p,dict(precision='clean32vs64',split=split,kind='none',seed=0,snr_db=-1,arm=arm))
                    del a32
            for kind in ['white','pink']:
                for seed in (SEEDS if precision=='fp32' else SEEDS[:1]):
                    for q in (GRID if precision=='fp32' else [120,200,260]):
                        content,n=dose(s,noises[split,kind,seed],q,precision)
                        waveforms.append(dict(precision=precision,split=split,kind=kind,seed=seed,snr_db=q,**waveform_audit(content,n)))
                        a=content+n;case=f'{kind}_{seed}/snr{q:03d}'
                        reference_a=None
                        for arm,scale in [('A',1),('B',.5)]:
                            arrays=forward(a*scale,folder,case+'/'+arm)
                            compare(anchors[arm],arrays,folder/case/arm,dict(precision=precision,split=split,kind=kind,seed=seed,snr_db=q,arm=arm))
                            if precision=='fp32' and q==30:
                                old=f'{split}/chunk00/{kind}_{seed}/{arm}/last.npy'
                                assert sha(PRIOR/old)==manifest[old]['sha256'] and np.array_equal(arrays['last'],np.load(PRIOR/old))
                                controls.append(dict(kind='prior_30dB_bridge',split=split,noise=kind,seed=seed,arm=arm,exact=True))
                            if arm=='A' and precision=='fp32' and q in [30,100,180,260]:reference_a=arrays
                            else:del arrays
                        if reference_a is not None:
                            compensated=forward(a*.5,folder,case+'/B_compensated',eps=2.5e-6)
                            exact={k:bool(np.array_equal(reference_a[k],compensated[k])) for k in STAGES};assert all(exact.values())
                            controls.append(dict(kind='epsilon_compensation',split=split,noise=kind,seed=seed,snr_db=q,stages_exact=exact))
                            del reference_a,compensated
                    persist();print(json.dumps(dict(forwards=len(calls),precision=precision,split=split,kind=kind,seed=seed)),flush=True)
            del anchors;capture.arrays={};gc.collect()
    model.float();norm.eps=1e-5;assert state_sha()==before
    assert len(calls)==538;persist();capture.close()
    write(out/'completion.json',dict(status='COMPLETED',forwards=len(calls),training_updates=0,generator_forwards=0,syncnet_calls=0,state_unchanged=True,protocol_sha256=sha(out/'protocol.json'),peak_cuda_reserved=torch.cuda.max_memory_reserved()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:main(args.output)
    except BaseException as exc:
        if args.output.exists():write(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
