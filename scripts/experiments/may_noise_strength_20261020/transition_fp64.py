"""Exploratory precision check at 80/90/100dB after split-dependent effect observed."""
from pathlib import Path
import importlib.util,argparse,time,json,gc
spec=importlib.util.spec_from_file_location('dose_scan',Path(__file__).with_name('run.py'));base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np=base.np;torch=base.torch

def main(root):
    assert (root/'completion.json').exists()
    out=root/'transition_fp64';out.mkdir(exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());assert base.sha(Path(__file__).with_name('run.py'))==protocol['script_sha256']
    spec=importlib.util.spec_from_file_location('sealed_capture',base.SOURCE);capture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture_module)
    assert base.sha(base.SOURCE)==protocol['capture_sha256']
    design=dict(status='FROZEN_BEFORE_FORWARD',role='EXPLORATORY_FOLLOWUP',reason='FP32 test prefix showed suppression at90/100dB but amplification at80dB; check real effect versus precision, without altering original protocol',
        snr_grid_db=[80,90,100],splits=['train','test'],kinds=['white','pink'],seeds=base.SEEDS[:1],precision='fp64',planned_forwards=28,script_sha256=base.sha(__file__),parent_protocol_sha256=base.sha(root/'protocol.json'),training_updates=0)
    base.write(out/'protocol.json',design)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.4)
    model=base.AutoModel.from_pretrained(base.WEIGHTS,local_files_only=True,attn_implementation='eager',low_cpu_mem_usage=True).eval().requires_grad_(False).double().cuda()
    capture=capture_module.Capture(model);calls=[];rows=[];waves=[];bridges=[]
    def forward(y,p):
        capture.arrays={};start=time.monotonic()
        with torch.inference_mode():last=model(input_values=torch.from_numpy(y[None]).cuda()).last_hidden_state[0].cpu().numpy()
        arrays=capture.arrays;arrays['last']=last.copy();p.mkdir(parents=True);np.save(p/'last.npy',last)
        sample={}
        for stage,a in arrays.items():
            idx=np.linspace(0,len(a)-1,64,dtype=int);sample[stage]=a[idx];sample[stage+'__indices']=idx
        np.savez_compressed(p/'samples.npz',**sample)
        calls.append(dict(path=p.relative_to(out).as_posix(),input_sha256=base.digest(y),last_sha256=base.digest(last),seconds=time.monotonic()-start));return arrays
    for split in ['train','test']:
        s=np.load(root/(split+'_native.npy')).astype(np.float64);anchors={}
        for arm,scale in [('A',1),('B',.5)]:
            anchors[arm]=forward(s*scale,out/split/('anchor'+arm))
            exact={stage:np.array_equal(a,np.load(root/'fp64'/split/('anchor'+arm)/(stage+'.npy'),mmap_mode='r')) for stage,a in anchors[arm].items()}
            assert all(exact.values());bridges.append(dict(split=split,arm=arm,stages_exact=exact))
        for kind in ['white','pink']:
            seed=base.SEEDS[0];noise=np.load(root/f'noise/{split}_{kind}_{seed}.npy')
            for q in [80,90,100]:
                _,n=base.dose(s,noise,q,'fp64');waves.append(dict(precision='fp64_transition',split=split,kind=kind,seed=seed,snr_db=q,**base.waveform_audit(s,n)))
                for arm,scale in [('A',1),('B',.5)]:
                    p=out/split/f'{kind}_{seed}'/f'snr{q:03d}'/arm;arrays=forward((s+n)*scale,p);frames={}
                    for stage in base.STAGES:
                        fm=base.metric_frames(anchors[arm][stage],arrays[stage]);frames[stage]=fm
                        rows.append(dict(precision='fp64_transition',split=split,kind=kind,seed=seed,snr_db=q,arm=arm,stage=stage,**base.summarize(fm,arrays[stage].shape[1])))
                    np.savez_compressed(p/'frames.npz',**frames);del arrays
            base.write(out/'rows.json',rows);base.write(out/'waveform_audit.json',waves);base.write(out/'forward_calls.json',calls)
            print(json.dumps(dict(transition_forwards=len(calls),split=split,kind=kind)),flush=True)
        del anchors;capture.arrays={};gc.collect()
    assert len(calls)==28
    base.write(out/'completion.json',dict(status='COMPLETED',forwards=28,anchor_bridges=bridges,protocol_sha256=base.sha(out/'protocol.json'),training_updates=0))
    capture.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
