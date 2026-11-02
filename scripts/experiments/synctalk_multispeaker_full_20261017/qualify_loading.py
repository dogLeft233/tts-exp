"""Compare original preload0/1 pixels, conditioning and actual rays on common supports."""
import argparse,gc,hashlib,json
from pathlib import Path
import numpy as np,torch
from runtime_common import ROOT,configure,dataset,seed_domain
from flow_stream import sha

def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),path
        assert torch.isfinite(a).all(),path
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    else:assert a==b,path

def digest(value,exclude=()):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous();h.update(str(x.dtype).encode()+str(tuple(x.shape)).encode()+x.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x):
                if k in exclude:continue
                h.update(str(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            for z in x:add(z)
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()

def run(identity):
    configure()
    from nerf_triplane import provider
    out=ROOT/'loading_qualification'/identity;out.mkdir(parents=True,exist_ok=False)
    raw=Path(__file__).read_bytes();(out/Path(__file__).name).write_bytes(raw)
    original=provider.get_rays;capture=[]
    def observe(*args,**kwargs):
        value=original(*args,**kwargs)
        capture.append({k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in value.items()})
        return value
    provider.get_rays=observe
    rows=[];common={}
    for split,count in [('train',6000),('val',750),('test',1500)]:
        phases=['coarse','fine'] if split=='train' else ['coarse']
        for phase in phases:
            for start in [0,count//2,count-4]:
                for gain in [-6,0,6]:
                    a=dataset(identity,split,gain,42,phase,0,(start,start+4))
                    b=dataset(identity,split,gain,42,phase,1,(start,start+4))
                    for field in ['poses','auds','eye_area','bg_img','intrinsics','face_rect','lips_rect']:
                        equal(getattr(a,field),getattr(b,field),field)
                    for local in [0,3]:
                        modes=[False,True] if split=='train' and phase=='fine' else [False]
                        for lips in modes:
                            a.opt.finetune_lips=b.opt.finetune_lips=lips
                            domain=f'loading/{split}/{phase}/{start}/{local}/{lips}'
                            capture.clear();seed_domain(42,0,domain);first=a.collate([local]);rays_a=capture[:]
                            capture.clear();seed_domain(42,0,domain);second=b.collate([local]);rays_b=capture[:]
                            equal(first,second);equal(rays_a,rays_b,'actual_get_rays_output')
                            assert len(rays_a)==1
                            source_id=int(a.original_source_ids[local])
                            expected=a.whole_recording_audio[source_id-4:source_id+4]
                            assert torch.equal(first['auds'].cpu(),expected.cpu())
                            key=(split,phase,start,local,lips);pixel_ray_digest=digest(first,exclude=('auds',))
                            if key in common:assert common[key]==pixel_ray_digest,'gain unexpectedly changed non-audio inputs'
                            else:common[key]=pixel_ray_digest
                            count_actual=int(first['rays_o'].shape[1])
                            if split=='train' and not lips:assert count_actual==65536
                            if split!='train':assert count_actual==262144
                            rows.append(dict(split=split,phase=phase,range_start=start,local_index=local,source_audio_id=source_id,
                                gain_db=gain,lips=lips,actual_rays=count_actual,pixel_and_ray_digest=pixel_ray_digest,
                                full_input_digest=digest(first),actual_rays_digest=digest(rays_a),preload0_1_exact=True,
                                whole_recording_attention_exact=True))
                    del a,b;gc.collect();torch.cuda.empty_cache()
    provider.get_rays=original
    result=dict(identity=identity,passed=True,cases=len(rows),rows=rows,
        producer_source_sha256=hashlib.sha256(raw).hexdigest(),provider_source_sha256=sha(ROOT/'official/nerf_triplane/provider.py'),
        audio_context_source_sha256=sha(ROOT/'audio_context.py'),binding_receipt_sha256=sha(ROOT/'datasets'/identity/'binding_receipt.json'),
        actual_pixel_rays_and_conditions_exact=True,cross_gain_non_audio_inputs_exact=True,
        whole_recording_attention_boundaries_exact=True,generator_forward_calls=0,optimizer_updates=0,training_admitted=False)
    (out/'loading_qualification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);a=p.parse_args();run(a.identity)
