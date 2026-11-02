"""Original full-budget entry points with explicit split, gain and context bindings."""
import argparse,ast,hashlib,json,os,random,sys
from pathlib import Path
import numpy as np,torch
from build_datasets import label
from flow_stream import sha
from cohort_binding import identity_root
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
OFFICIAL=ROOT/'official'
def configure():
    sys.path.insert(0,str(OFFICIAL));os.chdir(OFFICIAL)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def parser():
    tree=ast.parse((OFFICIAL/'main.py').read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id=='__name__' for x in ast.walk(n.test)))
    nodes=[]
    for n in block.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='opt' for t in n.targets):break
        nodes.append(n)
    scope=dict(argparse=argparse);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OFFICIAL/'main.py'),'exec'),scope)
    return scope['parser']
def options(identity,split,gain,seed,phase='coarse',preload=0,data_range=None):
    assert phase in ['coarse','fine'] and seed in [42,2026]
    directory=identity_root(identity,ROOT)/'datasets'/identity/split/label(gain)
    binding=json.loads((directory/'binding.json').read_text())
    assert binding['split']==split and binding['gain_db']==gain
    assert sha(directory/'whole_ave.npy')==binding['whole_ave_sha256']
    if data_range is None:
        assert len(list((directory/'gt_imgs').glob('*.jpg')))==binding['frames'],'restore full verified supervision before loading the complete split'
        assert len(list((directory/'torso_imgs').glob('*.png')))==binding['frames'],'restore full verified torso images before loading the complete split'
    arguments=[str(directory),'-O','--asr_model','ave','--preload',str(preload),'--seed',str(seed),'--ckpt','scratch',
        '--iters',str(60000 if phase=='coarse' else 100000),'--num_rays','65536','--max_steps','16']
    if phase=='fine':arguments+=['--finetune_lips','--patch_size','64']
    if split=='test':arguments+=['--test','--portrait']
    if data_range:arguments+=['--data_range',str(data_range[0]),str(data_range[1])]
    opt=parser().parse_args(arguments);opt.fp16=True;opt.exp_eye=True;opt.cuda_ray=True
    opt.verified_ave_cache=str(directory/'whole_ave.npy')
    assert not opt.torso and not opt.part and not opt.part2 and not opt.train_camera and opt.aud==''
    assert opt.num_rays==65536 and opt.max_steps==16
    return opt
def dataset(identity,split,gain,seed,phase='coarse',preload=0,data_range=None):
    from nerf_triplane.provider import NeRFDataset
    import audio_context
    audio_context.install()
    opt=options(identity,split,gain,seed,phase,preload,data_range)
    data=NeRFDataset(opt,device=torch.device('cuda'),type=split)
    binding=json.loads((Path(opt.path)/'binding.json').read_text())
    ids=np.array(binding['source_ids'],dtype=np.int64)
    if data_range:ids=ids[data_range[0]:None if data_range[1]==-1 else data_range[1]]
    whole=torch.from_numpy(np.load(opt.verified_ave_cache,allow_pickle=False)).float().unsqueeze(1).to(data.auds.device)
    assert len(data.poses)==len(data.images)==len(data.auds)==len(ids)
    audio_context.register(data.auds,whole,ids)
    data.original_source_ids=ids;data.whole_recording_audio=whole
    return data
def seed_domain(seed,step,domain):
    value=int.from_bytes(hashlib.sha256(f'{seed}/{step}/{domain}'.encode()).digest()[:4],'little')
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    return value
def tensor_state_sha(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode()+b'\0'+str(value.dtype).encode()+str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()
