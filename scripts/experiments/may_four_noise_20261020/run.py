"""Four known interference families on frozen May wav2vec2; never trains."""
from pathlib import Path
import importlib.util,sys,json,argparse,time,gc,shutil,traceback,hashlib
PREVIOUS=Path('F:/Datasets/tts-exp/may_noise_strength_20261020_v2')
BASE_SOURCE=Path(__file__).resolve().parents[1]/'may_noise_strength_20261020/run.py'
spec=importlib.util.spec_from_file_location('sealed_dose_scan',BASE_SOURCE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np=base.np;torch=base.torch;sf=base.sf
from scipy.signal import welch
SEEDS=base.SEEDS
STAGES=base.STAGES
KINDS=['single_speech','babble4','modulated_ssn','clicks']
GRID=list(range(0,101,10))+[120,140]
DOUBLE_GRID=[80,90,100,120]
SPEAKERS=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
MATERIAL=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/material')

def unit_rms(x,center=True):
    x=np.asarray(x,dtype=np.float64).copy()
    if center:x-=x.mean()
    r=np.sqrt(np.mean(x*x))
    if r<=0:raise ValueError('zero noise')
    return (x/r).astype(np.float32)

def speech_psd(waves):
    psds=[]
    for x in waves:
        _,p=welch(x[:60*16000],fs=16000,window='hann',nperseg=1024,noverlap=512,detrend='constant',scaling='density')
        psds.append(p/p.sum())
    p=np.mean(psds,axis=0);p=np.maximum(p,p.max()*1e-8);p[0]=0
    return p

def generate_noise(kind,length,seed,waves,psd,instance):
    rng=np.random.default_rng(seed);meta=dict(generator_seed=seed)
    def take(index):
        x=waves[index];start=int(rng.integers(0,len(x)-length+1))
        return x[start:start+length],dict(speaker=SPEAKERS[index],start_sample=start,samples=length)
    if kind=='single_speech':
        index=[0,1,3][instance];x,p=take(index);meta['components']=[p];return unit_rms(x),meta
    if kind=='babble4':
        parts=[];components=[]
        for index in range(4):
            x,p=take(index);parts.append(unit_rms(x).astype(float));components.append(p)
        meta['components']=components;meta['speaker_count']=4
        return unit_rms(np.sum(parts,axis=0)/2),meta
    if kind=='modulated_ssn':
        carrier=rng.standard_normal(length);f=np.fft.rfftfreq(length,1/16000)
        shaping=np.sqrt(np.interp(f,np.linspace(0,8000,len(psd)),psd));shaping[0]=0
        carrier=np.fft.irfft(np.fft.rfft(carrier)*shaping,n=length)
        t=np.arange(length)/16000;freqs=[float(rng.uniform(2,5)),float(rng.uniform(5,8))];phases=rng.uniform(0,2*np.pi,2)
        env=.15+.85*(.5+.5*np.sin(2*np.pi*freqs[0]*t+phases[0]))*(.5+.5*np.sin(2*np.pi*freqs[1]*t+phases[1]))
        meta.update(envelope_frequencies_hz=freqs,envelope_phases=phases.tolist(),envelope_min=float(env.min()),envelope_max=float(env.max()),shape='mean normalized Welch PSD of first60s of four independent interference recordings; floor1e-8peak, DC0')
        return unit_rms(carrier*env),meta
    if kind=='clicks':
        x=np.zeros(length);events=[];position=float(rng.exponential(16000/4))
        while position<length:
            start=int(position);width=int(rng.integers(32,81));freq=float(rng.uniform(1500,5000));phase=float(rng.uniform(0,2*np.pi));amplitude=float(rng.uniform(.5,1.5)*rng.choice([-1,1]))
            t=np.arange(width)/16000;p=np.hanning(width)*np.exp(-3*np.arange(width)/width)*np.sin(2*np.pi*freq*t+phase)
            p[1:-1]-=p.sum()/(width-2);stop=min(start+width,length);x[start:stop]+=amplitude*p[:stop-start]
            events.append(dict(start=start,width=width,carrier_hz=freq,phase=phase,amplitude=amplitude));position+=float(rng.exponential(16000/4))
        meta.update(events=events,event_rate_hz=4,note='synthetic sparse zero-mean windowed/damped clicks, not real keyboard recordings')
        return unit_rms(x,center=False),meta
    raise ValueError(kind)

def noise_stats(s,n):
    s=s.astype(float);n=n.astype(float);mask=n!=0
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    return dict(active_fraction=float(mask.mean()),noise_peak=float(np.max(np.abs(n))),noise_rms=rms(n),crest_factor=float(np.max(np.abs(n))/rms(n)),
        active_support_snr_db=float(20*np.log10(rms(s[mask])/rms(n[mask]))))

def main(out):
    out.mkdir(parents=True,exist_ok=False);assert shutil.disk_usage(out).free>12<<30
    previous_final=json.loads((PREVIOUS/'final_receipt.json').read_text());assert previous_final['passed']
    assert base.sha(PREVIOUS/'delivery_manifest.json')==previous_final['manifest_sha256']
    members={r['path']:r for r in json.loads((PREVIOUS/'delivery_manifest.json').read_text())['members']}
    prior=json.loads((PREVIOUS/'protocol.json').read_text());assert base.sha(BASE_SOURCE)==prior['script_sha256']
    capture_source=base.SOURCE;assert base.sha(capture_source)==prior['capture_sha256']
    spec=importlib.util.spec_from_file_location('sealed_capture',capture_source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for r in prior['source']['files']:assert base.sha(base.WEIGHTS/r['file'])==r['sha256']
    full={};inputs={};noises={};pins=[];noise_pins=[];source_pins=[];waves=[]
    for name in SPEAKERS:
        p=MATERIAL/name/'main_original.wav';x,sr=sf.read(p,dtype='float64');assert sr==16000 and x.ndim==1 and np.isfinite(x).all()
        waves.append(x);source_pins.append(dict(speaker=name,path=str(p),sha256=base.sha(p),samples=len(x),sample_rate=sr,description='existing other-identity natural recording; not source-separated or assumed noise-free'))
    psd=speech_psd(waves);np.save(out/'speech_shape_psd.npy',psd)
    for split_ix,split in enumerate(['train','test']):
        pin=next(p for p in prior['input_pins'] if p['split']==split);assert base.sha(pin['source_audio'])==pin['source_sha256']
        raw,sr=sf.read(pin['source_audio'],dtype='float64');assert sr==16000
        x=raw.astype(np.float32);s=(x-x.mean())/np.sqrt(x.var()+1e-7);assert base.digest(s)==pin['native_sha256']
        full[split]=s;inputs[split]=s[:320080].copy();pins.append(pin)
        assert np.array_equal(inputs[split],np.load(PREVIOUS/(split+'_native.npy')))
        np.save(out/(split+'_native.npy'),inputs[split]);rms=float(np.sqrt(np.mean(s.astype(float)**2)))
        for kind in KINDS:
            for instance,seed in enumerate(SEEDS):
                generator_seed=seed+split_ix*100000
                noise,meta=generate_noise(kind,len(s),generator_seed,waves,psd,instance)
                n30=noise*np.float32(rms*10**(-30/20));noises[split,kind,seed]=n30[:320080].copy()
                path=out/f'noise/{split}_{kind}_{seed}.npy';path.parent.mkdir(exist_ok=True);np.save(path,n30)
                noise_pins.append(dict(split=split,kind=kind,seed=seed,instance=instance,path=path.relative_to(out).as_posix(),sha256=base.sha(path),samples=len(n30),native_rms=rms,full_split_snr_db=float(20*np.log10(rms/np.sqrt(np.mean(n30.astype(float)**2)))),full_stats=noise_stats(s,n30),construction=meta))
    del waves;gc.collect()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.4)
    model,loading=base.AutoModel.from_pretrained(base.WEIGHTS,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys'];model.eval().requires_grad_(False).cuda()
    norm=model.feature_extractor.conv_layers[0].layer_norm;assert norm.eps==1e-5
    def state_sha():
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=state_sha();assert before==prior['state_sha256'];capture=old.Capture(model)
    design=dict(status='FROZEN_BEFORE_FORWARD',previous=str(PREVIOUS),previous_manifest_sha256=previous_final['manifest_sha256'],source=prior['source'],state_sha256=before,
        input_pins=pins,noise_pins=noise_pins,interference_source_pins=source_pins,speech_psd_sha256=base.sha(out/'speech_shape_psd.npy'),
        kinds=KINDS,seeds=SEEDS,snr_grid_db=GRID,fp64_grid_db=DOUBLE_GRID,fp64_seeds=SEEDS[:1],compensation_grid_db=[30,100],stages=STAGES,
        chunks={'train':[0,320080],'test':[0,320080]},scale=.5,arms={'A':'s+n','B':'0.5*(s+n)'},anchors={'A':'s','B':'0.5s'},
        transform='original full-split float32 zscore once; noise calibrated full-split; first20.005s scanned; no mixture normalization, clipping, integer quantization or time shift',
        endpoint='last per-frame unit-vector difference RMS, calculated by direct float64 subtraction; compare B/A noise responses versus matching-scale no-added-noise references',
        secondary='relativeL2, seven stages, per-frame exactness, prefix SNR, click support SNR and peak',
        resolution_rule='FP32 both A/B directional responses >10x respective cleanFP32-vs-FP64 discrepancy, diagnostic not error bound',
        descriptive_suppression_rule='B/A<1 is smaller response; highlight <=0.8 only if resolved; call profile-wide consistent only if all three instances and both prefixes agree at a level; no claims of harmful noise, training or perceptual quality',
        counts=dict(fp32=678,fp64=68,total=746),planned_forwards=746,training_updates=0,generator_forwards=0,syncnet_calls=0,
        script_sha256=base.sha(__file__),metric_source_sha256=base.sha(BASE_SOURCE),capture_source_sha256=base.sha(capture_source),
        runtime=dict(torch=torch.__version__,numpy=np.__version__,transformers=base.transformers.__version__,gpu=torch.cuda.get_device_name()))
    base.write(out/'protocol.json',design);(out/'sources').mkdir();shutil.copyfile(__file__,out/'sources/run.py');shutil.copyfile(BASE_SOURCE,out/'sources/paired_metrics.py');shutil.copyfile(capture_source,out/'sources/capture.py')
    rows=[];calls=[];controls=[];audits=[]
    def persist():
        for name,value in [('rows',rows),('forward_calls',calls),('controls',controls),('waveform_audit',audits)]:base.write(out/(name+'.json'),value)
        base.write(out/'progress.json',dict(completed=len(calls),planned=746,latest=calls[-1] if calls else None))
    def forward(y,folder,label,eps=1e-5):
        norm.eps=eps;capture.arrays={};start=time.monotonic()
        with torch.inference_mode():last=model(input_values=torch.from_numpy(y[None]).cuda()).last_hidden_state[0].cpu().numpy()
        arrays=capture.arrays;arrays['last']=last.copy();p=folder/label;p.mkdir(parents=True);np.save(p/'last.npy',last);samples={}
        for name,a in arrays.items():
            idx=np.linspace(0,len(a)-1,64,dtype=int);samples[name]=a[idx];samples[name+'__indices']=idx
        np.savez_compressed(p/'samples.npz',**samples)
        calls.append(dict(path=p.relative_to(out).as_posix(),dtype=str(y.dtype),epsilon=eps,input_sha256=base.digest(y),last_sha256=base.digest(last),seconds=time.monotonic()-start));return arrays
    def compare(anchor,arrays,p,meta):
        frames={}
        for stage in STAGES:
            fm=base.metric_frames(anchor[stage],arrays[stage]);frames[stage]=fm;rows.append(dict(**meta,stage=stage,**base.summarize(fm,arrays[stage].shape[1])))
        np.savez_compressed(p/'frames.npz',**frames)
    for mode in ['fp32','fp64']:
        if mode=='fp64':model.double();gc.collect();torch.cuda.empty_cache()
        for split in ['train','test']:
            folder=out/mode/split;folder.mkdir(parents=True);s=inputs[split].astype(np.float32 if mode=='fp32' else np.float64)
            anchors={}
            for arm,scale in [('A',1),('B',.5)]:
                anchors[arm]=forward(s*scale,folder,'anchor'+arm)
                for stage,a in anchors[arm].items():
                    old_path=PREVIOUS/mode/split/('anchor'+arm)/(stage+'.npy');assert base.sha(old_path)==members[old_path.relative_to(PREVIOUS).as_posix()]['sha256']
                    assert np.array_equal(a,np.load(old_path,mmap_mode='r'))
                controls.append(dict(kind='previous_all_stage_anchor_bridge',mode=mode,split=split,arm=arm,exact=True))
            if mode=='fp32':
                repeat=forward(s,folder,'repeatA');assert all(np.array_equal(repeat[k],anchors['A'][k]) for k in STAGES)
                controls.append(dict(kind='deterministic_repeat',split=split,all_stages_exact=True));del repeat
            else:
                for arm in ['A','B']:
                    refs={k:np.load(PREVIOUS/'fp32'/split/('anchor'+arm)/(k+'.npy'),mmap_mode='r') for k in STAGES}
                    p=folder/('precision_anchor'+arm);p.mkdir();compare(refs,anchors[arm],p,dict(precision='clean32vs64',split=split,kind='none',seed=0,snr_db=-1,arm=arm));del refs
            for kind in KINDS:
                for seed in (SEEDS if mode=='fp32' else SEEDS[:1]):
                    for q in (GRID if mode=='fp32' else DOUBLE_GRID):
                        _,n=base.dose(s,noises[split,kind,seed],q,mode)
                        audits.append(dict(precision=mode,split=split,kind=kind,seed=seed,snr_db=q,**base.waveform_audit(s,n),**noise_stats(s,n)))
                        case=f'{kind}_{seed}/snr{q:03d}';a=s+n;reference=None
                        for arm,scale in [('A',1),('B',.5)]:
                            arrays=forward(a*scale,folder,case+'/'+arm)
                            compare(anchors[arm],arrays,folder/case/arm,dict(precision=mode,split=split,kind=kind,seed=seed,snr_db=q,arm=arm))
                            if arm=='A' and mode=='fp32' and q in [30,100]:reference=arrays
                            else:del arrays
                        if reference is not None:
                            comp=forward(a*.5,folder,case+'/B_compensated',eps=2.5e-6);exact={k:bool(np.array_equal(reference[k],comp[k])) for k in STAGES};assert all(exact.values())
                            controls.append(dict(kind='epsilon_compensation',split=split,noise=kind,seed=seed,snr_db=q,stages_exact=exact));del comp,reference
                    persist();print(json.dumps(dict(forwards=len(calls),mode=mode,split=split,kind=kind,seed=seed)),flush=True)
            del anchors;capture.arrays={};gc.collect()
    model.float();norm.eps=1e-5;assert state_sha()==before;capture.close();assert len(calls)==746;persist()
    base.write(out/'completion.json',dict(status='COMPLETED',forwards=746,training_updates=0,generator_forwards=0,syncnet_calls=0,state_unchanged=True,protocol_sha256=base.sha(out/'protocol.json'),peak_cuda_reserved=torch.cuda.max_memory_reserved()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:main(args.output)
    except BaseException as e:
        if args.output.exists():base.write(args.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
