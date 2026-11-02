"""Continue frozen scan with bounded host RAM; preserve failed attempt unchanged."""
from pathlib import Path
import os,json,hashlib,argparse,subprocess,sys,time,shutil,importlib.util,gc
HERE=Path(__file__).resolve().parent
OLD=Path('F:/Datasets/tts-exp/may_four_noise_20261020_v1')
PRIOR=Path('F:/Datasets/tts-exp/may_noise_strength_20261020_v2')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def load_base():
    spec=importlib.util.spec_from_file_location('original_four',HERE/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def clone(a,b):
    b=Path(b);b.parent.mkdir(parents=True,exist_ok=True)
    if Path(a).suffix in ['.npy','.npz']:os.link(a,b)
    else:shutil.copy2(a,b)

def prepare(out):
    m=load_base();np=m.np;out.mkdir(parents=True,exist_ok=False)
    p=read(OLD/'protocol.json');assert sha(HERE/'run.py')==p['script_sha256'] and (OLD/'failure.json').exists()
    previous=read(PRIOR/'final_receipt.json');assert previous['passed'] and sha(PRIOR/'delivery_manifest.json')==p['previous_manifest_sha256']
    old_calls=read(OLD/'forward_calls.json');assert len(old_calls)==370
    snapshot=[]
    for f in sorted(OLD.rglob('*')):
        if f.is_file():snapshot.append(dict(path=f.relative_to(OLD).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
    write(out/'failed_attempt_snapshot.json',dict(root=str(OLD),failure=read(OLD/'failure.json'),members=snapshot,admitted_forwards=370,complete_last_files=sum(x['path'].endswith('/last.npy') for x in snapshot),excluded_uncommitted_outputs=True))
    for name in ['protocol.json','train_native.npy','test_native.npy','speech_shape_psd.npy']:
        clone(OLD/name,out/name)
    for folder in ['noise','sources']:
        for f in (OLD/folder).rglob('*'):
            if f.is_file():clone(f,out/f.relative_to(OLD))
    for c in old_calls:
        for f in (OLD/c['path']).iterdir():
            if f.is_file():clone(f,out/c['path']/f.name)
    write(out/'original_calls.json',old_calls);write(out/'original_rows.json',read(OLD/'rows.json'));write(out/'original_controls.json',read(OLD/'controls.json'));write(out/'original_waveform_audit.json',read(OLD/'waveform_audit.json'))
    manifest={x['path']:x for x in read(PRIOR/'delivery_manifest.json')['members']};reference_pins=[]
    for mode in ['fp32','fp64']:
        for split in ['train','test']:
            for arm in ['A','B']:
                for stage in p['stages']:
                    f=PRIOR/mode/split/('anchor'+arm)/(stage+'.npy');assert sha(f)==manifest[f.relative_to(PRIOR).as_posix()]['sha256']
                    a=np.load(f,mmap_mode='r');h=hashlib.sha256()
                    for i in range(0,len(a),256):h.update(np.ascontiguousarray(a[i:i+256]).tobytes())
                    reference_pins.append(dict(mode=mode,split=split,arm=arm,stage=stage,path=str(f),file_sha256=sha(f),array_sha256=h.hexdigest()))
    write(out/'reference_pins.json',reference_pins)
    admitted={c['path'] for c in old_calls};jobs=[]
    for mode in ['fp32','fp64']:
        for split in ['train','test']:
            first=True
            for kind in p['kinds']:
                for seed in (p['seeds'] if mode=='fp32' else p['fp64_seeds']):
                    labels=[]
                    for q in (p['snr_grid_db'] if mode=='fp32' else p['fp64_grid_db']):
                        labels.extend([f'snr{q:03d}/A',f'snr{q:03d}/B'])
                        if mode=='fp32' and q in p['compensation_grid_db']:labels.append(f'snr{q:03d}/B_compensated')
                    if all(f'{mode}/{split}/{kind}_{seed}/{label}' in admitted for label in labels):continue
                    anchors_scientific=mode=='fp64' and first;first=False
                    jobs.append(dict(index=len(jobs),mode=mode,split=split,kind=kind,seed=seed,scientific_anchors=anchors_scientific,labels=labels))
    assert len(jobs)==19;write(out/'jobs.json',jobs)
    write(out/'recovery_protocol.json',dict(status='FROZEN_BEFORE_RECOVERY_FORWARD',original_protocol_sha256=sha(out/'protocol.json'),original_source_sha256=sha(HERE/'run.py'),recovery_source_sha256=sha(__file__),failed_snapshot_sha256=sha(out/'failed_attempt_snapshot.json'),source_root=str(OLD),retained_scientific_forwards=370,remaining_scientific_forwards=376,extra_baseline_bridge_forwards=34,total_admitted_scientific_forwards=746,training_updates=0,method='one family/instance/split per process; stream hooks in256-row CPU blocks against SHA-pinned prior full references; scientific condition/metric unchanged',jobs=jobs))
    shutil.copyfile(__file__,out/'sources/recover_streaming.py')
    print(json.dumps(dict(prepared=True,jobs=19,retained=370,remaining=376)),flush=True)

class StreamCapture:
    def __init__(self,model,refs,pins,m):
        self.m=m;self.np=m.np;self.refs=refs;self.pins=pins;self.handles=[];self.reset(None)
        modules=[('first_gn',model.feature_extractor.conv_layers[0].layer_norm,True),('cnn7',model.feature_extractor.conv_layers[6],True),('projection',model.feature_projection.projection,False)]+[(f'transformer{i}',model.encoder.layers[i-1],False) for i in [4,8,9]]
        for name,module,conv in modules:
            def hook(_m,_in,result,name=name,conv=conv):
                while isinstance(result,(tuple,list)):result=result[0]
                value=result.detach()[0];self.consume(name,value.transpose(0,1) if conv else value)
            self.handles.append(module.register_forward_hook(hook))
    def reset(self,arm):self.arm=arm;self.frames={};self.samples={};self.digests={};self.last=None
    def consume(self,name,tensor):
        np=self.np;ref=self.refs[self.arm][name];assert tuple(tensor.shape)==ref.shape
        fm=np.empty((len(ref),6),np.float64);idx=np.linspace(0,len(ref)-1,64,dtype=int);sample=np.empty((64,ref.shape[1]),dtype=ref.dtype);h=hashlib.sha256()
        if name=='last':self.last=np.empty(ref.shape,dtype=ref.dtype)
        for start in range(0,len(ref),256):
            stop=min(start+256,len(ref));block=np.ascontiguousarray(tensor[start:stop].cpu().numpy());h.update(block.tobytes())
            fm[start:stop]=self.m.base.metric_frames(ref[start:stop],block)
            positions=np.flatnonzero((idx>=start)&(idx<stop));sample[positions]=block[idx[positions]-start]
            if name=='last':self.last[start:stop]=block
        self.frames[name]=fm;self.samples[name]=sample;self.samples[name+'__indices']=idx;self.digests[name]=h.hexdigest()
    def close(self):
        for h in self.handles:h.remove()

def worker(out,index):
    m=load_base();np=m.np;torch=m.torch;r=read(out/'recovery_protocol.json');assert sha(__file__)==r['recovery_source_sha256']
    p=read(out/'protocol.json');job=read(out/'jobs.json')[index];mode,split,kind,seed=[job[k] for k in ['mode','split','kind','seed']]
    receipt=out/'worker_receipts'/f'{index:02d}.json';assert not receipt.exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.4)
    model=m.base.AutoModel.from_pretrained(m.base.WEIGHTS,local_files_only=True,attn_implementation='eager',low_cpu_mem_usage=True).eval().requires_grad_(False).cuda()
    def state_sha():
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    assert state_sha()==p['state_sha256']
    if mode=='fp64':model.double()
    pins=[x for x in read(out/'reference_pins.json') if x['mode']==mode and x['split']==split]
    refs={arm:{x['stage']:np.load(x['path'],mmap_mode='r') for x in pins if x['arm']==arm} for arm in ['A','B']}
    expected={(x['arm'],x['stage']):x['array_sha256'] for x in pins}
    capture=StreamCapture(model,refs,pins,m);norm=model.feature_extractor.conv_layers[0].layer_norm
    s=np.load(out/(split+'_native.npy')).astype(np.float32 if mode=='fp32' else np.float64)
    n30=np.load(out/f'noise/{split}_{kind}_{seed}.npy',mmap_mode='r')[:len(s)];calls=[];extra=[];rows=[];controls=[];audits=[]
    def forward(y,path,arm,epsilon=1e-5,scientific=True):
        capture.reset(arm);norm.eps=epsilon;start=time.monotonic()
        with torch.inference_mode():result=model(input_values=torch.from_numpy(y[None]).cuda()).last_hidden_state[0]
        capture.consume('last',result);del result;assert set(capture.frames)==set(p['stages'])
        path.mkdir(parents=True,exist_ok=False);np.save(path/'last.npy',capture.last);np.savez_compressed(path/'samples.npz',**capture.samples);np.savez_compressed(path/'frames.npz',**capture.frames)
        entry=dict(path=path.relative_to(out).as_posix(),dtype=str(y.dtype),epsilon=epsilon,input_sha256=m.base.digest(y),last_sha256=m.base.digest(capture.last),stage_sha256=capture.digests.copy(),seconds=time.monotonic()-start)
        (calls if scientific else extra).append(entry)
        return capture.digests.copy(),capture.frames.copy()
    for arm,scale in [('A',1),('B',.5)]:
        folder=out/mode/split/('anchor'+arm) if job['scientific_anchors'] else out/'recovery_controls'/f'job{index:02d}'/('anchor'+arm)
        d,_=forward(s*scale,folder,arm,scientific=job['scientific_anchors']);assert all(d[k]==expected[arm,k] for k in p['stages'])
        controls.append(dict(kind='recovery_all_stage_anchor_bridge',job=index,mode=mode,split=split,arm=arm,exact=True))
    reference_digest=None
    for label in job['labels']:
        qstr,arm=label.split('/');q=int(qstr[3:]);_,n=m.base.dose(s,n30,q,mode);a=s+n
        if arm=='A':audits.append(dict(precision=mode,split=split,kind=kind,seed=seed,snr_db=q,**m.base.waveform_audit(s,n),**m.noise_stats(s,n)))
        path=out/mode/split/f'{kind}_{seed}'/label;d,frames=forward(a*(1 if arm=='A' else .5),path,'A' if arm=='B_compensated' else arm,epsilon=2.5e-6 if arm=='B_compensated' else 1e-5)
        if arm=='B_compensated':
            assert d==reference_digest;controls.append(dict(kind='epsilon_compensation',split=split,noise=kind,seed=seed,snr_db=q,stages_exact={k:True for k in p['stages']}));continue
        for stage in p['stages']:rows.append(dict(precision=mode,split=split,kind=kind,seed=seed,snr_db=q,arm=arm,stage=stage,**m.base.summarize(frames[stage],refs[arm][stage].shape[1])))
        if arm=='A' and q in p['compensation_grid_db']:reference_digest=d
    capture.close();del capture,refs;model.float();norm.eps=1e-5;assert state_sha()==p['state_sha256']
    write(receipt,dict(passed=True,job=job,scientific_calls=calls,extra_bridge_calls=extra,rows=rows,controls=controls,waveform_audit=audits,state_unchanged=True,recovery_protocol_sha256=sha(out/'recovery_protocol.json')))
    print(json.dumps(dict(job=index,scientific=len(calls),extra=len(extra),passed=True)),flush=True)

def finalize(out):
    m=load_base();np=m.np;p=read(out/'protocol.json');calls=read(out/'original_calls.json');rows=read(out/'original_rows.json');controls=read(out/'original_controls.json');waves=read(out/'original_waveform_audit.json');extra=[]
    for job in read(out/'jobs.json'):
        r=read(out/'worker_receipts'/f"{job['index']:02d}.json");assert r['passed'] and r['recovery_protocol_sha256']==sha(out/'recovery_protocol.json')
        calls+=r['scientific_calls'];rows+=r['rows'];controls+=r['controls'];waves+=r['waveform_audit'];extra+=r['extra_bridge_calls']
    # Same cross-precision reference pair as original design, calculated in bounded blocks.
    pins=read(out/'reference_pins.json')
    for split in ['train','test']:
        for arm in ['A','B']:
            destination=out/'fp64'/split/('precision_anchor'+arm);destination.mkdir();frames={}
            for stage in p['stages']:
                refs=[np.load(next(x['path'] for x in pins if x['mode']==mode and x['split']==split and x['arm']==arm and x['stage']==stage),mmap_mode='r') for mode in ['fp32','fp64']]
                fm=np.empty((len(refs[0]),6),np.float64)
                for i in range(0,len(fm),256):fm[i:i+256]=m.base.metric_frames(refs[0][i:i+256],refs[1][i:i+256])
                frames[stage]=fm;rows.append(dict(precision='clean32vs64',split=split,kind='none',seed=0,snr_db=-1,arm=arm,stage=stage,**m.base.summarize(fm,refs[0].shape[1])))
            np.savez_compressed(destination/'frames.npz',**frames)
    assert len(calls)==746 and len(set(c['path'] for c in calls))==746 and len(rows)==4844 and len(waves)==344 and len(extra)==34
    for name,v in [('forward_calls',calls),('rows',rows),('controls',controls),('waveform_audit',waves),('recovery_bridge_calls',extra)]:write(out/(name+'.json'),v)
    write(out/'completion.json',dict(status='COMPLETED',forwards=746,extra_recovery_bridge_forwards=34,admitted_total_forwards=780,excluded_failed_attempt_complete_outputs=read(out/'failed_attempt_snapshot.json')['complete_last_files']-370,training_updates=0,generator_forwards=0,syncnet_calls=0,state_unchanged=True,protocol_sha256=sha(out/'protocol.json'),recovery_protocol_sha256=sha(out/'recovery_protocol.json')))
    print(json.dumps(dict(completed=True,scientific_forwards=746,extra_bridges=34)),flush=True)

def drive(out):
    for job in read(out/'jobs.json'):
        assert not (out/'worker_receipts'/f"{job['index']:02d}.json").exists()
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--output',str(out),'--index',str(job['index'])],check=True)
    finalize(out)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','worker','run','finalize']);a.add_argument('--output',type=Path,required=True);a.add_argument('--index',type=int);r=a.parse_args()
    {'prepare':prepare,'worker':lambda p:worker(p,r.index),'run':drive,'finalize':finalize}[r.mode](r.output)
