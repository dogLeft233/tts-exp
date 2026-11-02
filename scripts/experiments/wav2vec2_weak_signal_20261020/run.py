"""Frozen May wav2vec2 layer audit, separating time energy and channel response.

Default scale1/.5 + first-GN epsilon-compensated .5; no training or TFG.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import argparse,sys,json,hashlib,importlib.util,gc,time,traceback,shutil
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/deps')
import numpy as np
import torch,soundfile as sf,transformers
from transformers import AutoModel

PARENT=Path('F:/Datasets/tts-exp/may_postnorm_scale_20261020')
DATA=Path('F:/Datasets/tts-exp/synctalk_gain_training_20261017')
WEIGHTS=Path('F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/weights/wav2vec2_base')
PREVIOUS=Path('F:/Datasets/tts-exp/encoder_layer_epsilon_20261020_v1')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(p)


def pair_moments(a,b):
    """Time rows: xx,yy,dd,xy,maxdiff; channel rows: sumx,sumy,xx,yy,xy."""
    assert a.shape==b.shape and a.ndim==2
    n,c=a.shape
    frames=np.zeros((n,5),dtype=np.float64);channels=np.zeros((c,5),dtype=np.float64)
    for start in range(0,n,1024):
        x=a[start:start+1024].astype(float);y=b[start:start+1024].astype(float);d=y-x
        frames[start:start+len(x)]=np.stack([(x*x).sum(1),(y*y).sum(1),(d*d).sum(1),
                                          (x*y).sum(1),abs(d).max(1)],axis=1)
        channels+=np.stack([x.sum(0),y.sum(0),(x*x).sum(0),(y*y).sum(0),(x*y).sum(0)],axis=1)
    return frames,channels


def metrics(m):
    xx,yy,dd,xy=m[:,:4].sum(0)
    if xx<=0:raise ValueError('zero baseline energy')
    alpha=xy/xx
    return dict(norm_ratio=float(np.sqrt(yy/xx)),gain_fit=float(alpha),
        relative_l2=float(np.sqrt(dd/xx)),shape_residual=float(np.sqrt(max(0,yy-alpha*alpha*xx)/xx)),
        xx=float(xx),yy=float(yy),dd=float(dd),xy=float(xy),frames=len(m))


def channel_stats(ch,n):
    mx,my=ch[:,0]/n,ch[:,1]/n
    vx=np.maximum(ch[:,2]/n-mx*mx,0);vy=np.maximum(ch[:,3]/n-my*my,0)
    cov=ch[:,4]/n-mx*my
    if not np.all(vx>0):raise ValueError('constant baseline channel')
    ratio=np.sqrt(vy/vx);alpha=cov/vx
    return mx,my,vx,vy,cov,ratio,alpha


def energy_at(x,centers):
    # 50ms centered RMS, constant-zero outside complete split.
    power=np.r_[0.,np.cumsum(np.pad(x.astype(float)**2,(400,400)))]
    centers=np.asarray(centers,dtype=int)
    return np.sqrt((power[centers+800]-power[centers])/800)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(out).free>2<<30
    src=Path(__file__).resolve().parents[1]/'encoder_layer_epsilon_20261020/run.py'
    spec=importlib.util.spec_from_file_location('parent_capture',src)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    old_manifest=json.loads((PREVIOUS/'manifest.json').read_text())
    old_receipt=json.loads((PREVIOUS/'delivery_receipt.json').read_text())
    assert sha(PREVIOUS/'manifest.json')==old_receipt['manifest_sha256']
    assert sha(src)==json.loads((PREVIOUS/'protocol.json').read_text())['source_script_sha256']
    for rel in ['protocol.json','completion.json','qa.json']:
        pin=next(r for r in old_manifest if r['path']==rel)
        assert sha(PREVIOUS/rel)==pin['sha256']
    receipt=json.loads((PARENT/'features/wav2vec2_base/receipt.json').read_text())
    for row in receipt['source']['files']:assert sha(WEIGHTS/row['file'])==row['sha256']
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.4)
    free,total=torch.cuda.mem_get_info();assert free>4<<30
    model,loading=AutoModel.from_pretrained(WEIGHTS,local_files_only=True,attn_implementation='eager',
            output_loading_info=True,low_cpu_mem_usage=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys']
    model.eval().requires_grad_(False).cuda()
    norm=model.feature_extractor.conv_layers[0].layer_norm
    assert isinstance(norm,torch.nn.GroupNorm) and norm.eps==1e-5
    before=helper.state_sha(model);capture=helper.Capture(model)
    gamma=norm.weight.detach().cpu().numpy().astype(float)
    beta=norm.bias.detach().cpu().numpy().astype(float)
    (out/'sources').mkdir();shutil.copyfile(__file__,out/'sources/run.py');shutil.copyfile(src,out/'sources/parent_capture.py')
    np.savez(out/'groupnorm_affine.npz',gamma=gamma,beta=beta)
    protocol=dict(status='FROZEN_BEFORE_FORWARD',source=receipt['source'],parent=str(PARENT),
        previous_diagnostic=str(PREVIOUS),splits=['train','val','test'],
        transform='historical whole-split float32 zscore, then scale, then historical 320000-hop/320080-length chunks',
        conditions=['default_scale1','default_scale0.5','first_GN_eps2.5e-6_scale0.5'],
        acoustic_strata='full-split 50ms centered waveform RMS q20/q80 at final-CNN frame centers; fixed from scale1',
        channel_strata='within each baseline chunk/stage, centered per-channel variance q20/q80; never use perturbed values to label',
        gn_strata='first convolution pre-GN temporal variance quintile; response measured after subtracting learned beta',
        temporal_support='CNN receptive-field centers; transformer uses full chunk context, frame label is descriptive',
        no_noise_truth=True,no_signal_or_noise_decomposition=True,training_updates=0,generator_forwards=0,syncnet_calls=0,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
            python=sys.version,gpu=torch.cuda.get_device_name(),free_cuda_bytes=free),
        script_sha256=sha(__file__),parent_capture_sha256=sha(src),state_sha256=before)
    write(out/'protocol.json',protocol)
    rows=[];channel_rows=[];gn_rows=[];controls=[];calls=[];pins=[]
    def persist():
        write(out/'layer_rows.json',rows);write(out/'channel_rows.json',channel_rows)
        write(out/'groupnorm_rows.json',gn_rows);write(out/'controls.json',controls)
        write(out/'forward_calls.json',calls);write(out/'input_pins.json',pins)
    for split in protocol['splits']:
        audio=DATA/f'arms/{split}/gain_0/aud.wav'
        native_row=next(r for r in receipt['conditions'] if r['split']==split and r['input_scale']==1)
        assert sha(audio)==native_row['source_audio_sha256']
        raw,sr=sf.read(audio,dtype='float64');assert sr==16000 and raw.ndim==1
        x=raw.astype(np.float32);z=(x-x.mean())/np.sqrt(x.var()+1e-7)
        assert digest(z)==native_row['native_processor_sha256']
        low_row=next(r for r in receipt['conditions'] if r['split']==split and r['input_scale']==.5)
        assert digest(z*np.float32(.5))==low_row['processor_sha256']
        np.save(out/f'{split}_native_input.npy',z)
        full_centers=np.arange((len(z)-400)//320+1)*320+199
        thresholds=np.quantile(energy_at(z,full_centers),[.2,.8])
        cache={scale:np.load(PARENT/f'features/wav2vec2_base/{split}_s{scale:.2f}.npy',mmap_mode='r').reshape(-1,768)
               for scale in [1.,.5]}
        for scale in [1.,.5]:
            r=next(r for r in receipt['conditions'] if r['split']==split and r['input_scale']==scale)
            assert sha(PARENT/f'features/wav2vec2_base/{split}_s{scale:.2f}.npy')==r['sha256']
        pins.append(dict(split=split,source_audio=str(audio),source_sha256=sha(audio),
                         native_sha256=digest(z),half_sha256=digest(z*.5),thresholds=thresholds.tolist()))
        offset=0
        for chunk,start in enumerate(range(0,len(z),320000)):
            piece=z[start:min(start+320080,len(z))]
            if len(piece)<400:continue
            folder=out/split/f'chunk{chunk:02d}';folder.mkdir(parents=True)
            def forward(label,scale,eps):
                norm.eps=eps;capture.arrays={};begin=time.monotonic()
                with torch.inference_mode():hidden=model(input_values=torch.from_numpy((piece*np.float32(scale))[None]).cuda()).last_hidden_state[0].cpu().numpy()
                arrays=capture.arrays;arrays['last_hidden_state']=hidden.copy()
                dest=folder/label;dest.mkdir()
                np.save(dest/'last_hidden_state.npy',hidden)
                samples={}
                for name,a in arrays.items():
                    idx=np.linspace(0,len(a)-1,32,dtype=int)
                    samples[name]=a[idx];samples[name+'__indices']=idx
                np.savez_compressed(dest/'stage_samples.npz',**samples)
                calls.append(dict(split=split,chunk=chunk,start_sample=start,label=label,scale=scale,epsilon=eps,
                                  seconds=time.monotonic()-begin,hidden_sha256=digest(hidden)))
                print(json.dumps(calls[-1]),flush=True)
                return arrays
            base=forward('baseline',1.,1e-5)
            if chunk==0 and split in ['train','test']:
                repeat=forward('repeat',1.,1e-5)
                assert all(np.array_equal(base[k],repeat[k]) for k in base)
                controls.append(dict(kind='repeat_all_stages_exact',split=split,passed=True));del repeat
                # Cross-run first chunk bridge to report58.
                oldpath=PREVIOUS/f'wav2vec2_base/{split}/eps_1e-05_scale1/last_hidden_state.npy'
                pin=next(r for r in old_manifest if r['path']==oldpath.relative_to(PREVIOUS).as_posix())
                assert sha(oldpath)==pin['sha256']
                assert np.array_equal(np.load(oldpath),base['last_hidden_state'])
            low=forward('half',.5,1e-5)
            frame_evidence={};channel_evidence={};meta={}
            for name,a in base.items():
                b=low[name];frames,channels=pair_moments(a,b)
                frame_evidence[name]=frames;channel_evidence[name]=channels
                mx,my,vx,vy,cov,ratios,alphas=channel_stats(channels,len(a))
                if name.startswith('conv'):
                    i=int(name[4]);jump,rf=1,1
                    for k,s in zip(model.config.conv_kernel[:i+1],model.config.conv_stride[:i+1]):
                        rf+=(k-1)*jump;jump*=s
                    centers=start+np.arange(len(a))*jump+(rf-1)//2
                else:centers=start+np.arange(len(a))*320+199
                energy=energy_at(z,centers);labels=np.where(energy<=thresholds[0],0,np.where(energy>thresholds[1],2,1))
                frame_evidence[name+'__acoustic_stratum']=labels.astype(np.int8)
                frame_evidence[name+'__source_sample_centers']=centers
                meta[name]=dict(shape=list(a.shape),source_start_sample=start)
                for k,label in [(-1,'all'),(0,'low20'),(1,'middle60'),(2,'high20')]:
                    selected=frames if k==-1 else frames[labels==k]
                    if len(selected):rows.append(dict(split=split,chunk=chunk,stage=name,stratum=label,**metrics(selected)))
                q20,q80=np.quantile(vx,[.2,.8]);cbin=np.where(vx<=q20,0,np.where(vx>q80,2,1))
                channel_evidence[name+'__baseline_response_stratum']=cbin.astype(np.int8)
                for k,label in enumerate(['low20','middle60','high20']):
                    use=cbin==k
                    channel_rows.append(dict(split=split,chunk=chunk,stage=name,stratum=label,channels=int(use.sum()),
                        centered_norm_ratio=float(np.sqrt(vy[use].sum()/vx[use].sum())),
                        median_channel_ratio=float(np.median(ratios[use])),
                        centered_xx=float(vx[use].sum()*len(a)),centered_yy=float(vy[use].sum()*len(a)),
                        centered_xy=float(cov[use].sum()*len(a)),
                        baseline_variance_min=float(vx[use].min()),baseline_variance_max=float(vx[use].max())))
            np.savez_compressed(folder/'paired_frame_moments.npz',**frame_evidence)
            np.savez_compressed(folder/'paired_channel_moments.npz',**channel_evidence)
            write(folder/'stage_metadata.json',meta)
            # First GN centered by learned beta: compare temporal variance classes BEFORE affine.
            pre=base['conv0_linear'].astype(float);v=pre.var(0)
            observed_base=base['conv0_groupnorm'].astype(float)-beta
            observed_low=low['conv0_groupnorm'].astype(float)-beta
            observed_ratio=np.sqrt((observed_low**2).sum(0)/(observed_base**2).sum(0))
            predicted=np.sqrt((v+1e-5)/(v+4e-5))
            np.savez_compressed(folder/'gn_channels.npz',variance=v,gamma=gamma,beta=beta,
                base_centered_energy=(observed_base**2).sum(0),half_centered_energy=(observed_low**2).sum(0),
                observed_ratio=observed_ratio,predicted_ratio=predicted,
                per_channel_fit_maxerror=abs(observed_low-observed_base*predicted).max(0))
            q20,q80=np.quantile(v,[.2,.8]);groups=np.where(v<=q20,0,np.where(v>q80,2,1))
            for k,label in enumerate(['low20','middle60','high20']):
                use=groups==k
                gn_rows.append(dict(split=split,chunk=chunk,stratum=label,channels=int(use.sum()),
                    variance_min=float(v[use].min()),variance_max=float(v[use].max()),
                    median_ratio=float(np.median(observed_ratio[use])),minimum_ratio=float(observed_ratio[use].min()),
                    maximum_ratio=float(observed_ratio[use].max()),
                    predicted_ratio_max_error=float(abs(observed_ratio[use]-predicted[use]).max())))
            assert np.array_equal(low['conv0_linear'],base['conv0_linear']*.5)
            for scale,arrays in [(1.,base),(.5,low)]:
                hidden=arrays['last_hidden_state'];length=min(len(hidden),len(cache[scale])-offset)
                if length:
                    prior=cache[scale][offset:offset+length]
                    rel=float(np.linalg.norm(hidden[:length].astype(float)-prior)/np.linalg.norm(prior.astype(float)))
                    assert rel<1e-4
                    controls.append(dict(kind='report56_cache_bridge',split=split,chunk=chunk,scale=scale,
                                         frames=length,relative_l2=rel,exact=bool(np.array_equal(prior,hidden[:length]))))
            del low,pre,observed_base,observed_low
            compensated=forward('half_epsilon_compensated',.5,2.5e-6)
            exact=all(np.array_equal(base[k],compensated[k]) for k in base if k!='conv0_linear')
            assert exact
            controls.append(dict(kind='epsilon_compensated_all_postGN_stages_exact',split=split,chunk=chunk,passed=True))
            offset+=len(base['last_hidden_state'])
            del base,compensated;gc.collect();persist()
        assert offset>=len(cache[1.]) and offset-len(cache[1.])<=1
    norm.eps=1e-5;assert helper.state_sha(model)==before
    capture.close();del capture,model;gc.collect();torch.cuda.empty_cache()
    persist();write(out/'completion.json',dict(status='COMPLETED',forwards=len(calls),
        chunks=sum(r['label']=='baseline' for r in calls),weight_state_unchanged=True,
        training_updates=0,generator_forwards=0,syncnet_calls=0,
        peak_cuda_reserved=torch.cuda.max_memory_reserved(),protocol_sha256=sha(out/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:main(args.output)
    except BaseException as exc:
        if args.output.exists():write(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
