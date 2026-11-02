"""State/RNG and actual-update controls, separate from the original training mathematics."""
import contextlib,copy,hashlib,json,random
import numpy as np,torch

def tree_sha(value):
    h=hashlib.sha256()
    def add(item):
        if isinstance(item,torch.Tensor):
            x=item.detach().cpu().contiguous();h.update(b'tensor'+str(x.dtype).encode()+str(tuple(x.shape)).encode()+x.numpy().tobytes())
        elif isinstance(item,np.ndarray):h.update(b'array'+str(item.dtype).encode()+str(item.shape).encode()+item.tobytes())
        elif isinstance(item,dict):
            h.update(b'dict')
            for k in sorted(item,key=lambda x:(type(x).__name__,repr(x))):add(k);add(item[k])
        elif isinstance(item,(list,tuple)):
            h.update(type(item).__name__.encode()+str(len(item)).encode())
            for x in item:add(x)
        elif isinstance(item,(str,int,float,bool,type(None),np.integer,np.floating)):
            h.update(type(item).__name__.encode()+repr(item).encode()+b'\0')
        else:raise TypeError(type(item))
    add(value);return h.hexdigest()

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
@contextlib.contextmanager
def preserve_rng():
    before=rng_state();digest=tree_sha(before)
    try:yield
    finally:
        restore_rng(before);assert tree_sha(rng_state())==digest

MODEL_COUNTERS=['mean_count','mean_density','mean_density_torso','iter_density','local_step']
def model_counters(model):
    return {k:copy.deepcopy(getattr(model,k)) for k in MODEL_COUNTERS if hasattr(model,k)}
def restore_model_counters(model,state):
    for key,value in state.items():
        assert key in MODEL_COUNTERS and hasattr(model,key);setattr(model,key,copy.deepcopy(value))
    assert tree_sha(model_counters(model))==tree_sha(state)

def verify_full_checkpoint(trainer,checkpoint):
    assert set(trainer.model.state_dict())==set(checkpoint['model'])
    assert tree_sha(trainer.model.state_dict())==tree_sha(checkpoint['model'])
    for name,obj in [('optimizer',trainer.optimizer),('lr_scheduler',trainer.lr_scheduler),('scaler',trainer.scaler),('ema',trainer.ema)]:
        assert name in checkpoint and obj is not None
        assert tree_sha(obj.state_dict())==tree_sha(checkpoint[name]),f'{name} did not restore exactly'
    assert trainer.global_step==checkpoint['global_step'] and trainer.epoch==checkpoint['epoch']

class UpdateCounter:
    def __init__(self,optimizer,initial=0):
        self.successful=int(initial);self.handle=optimizer.register_step_post_hook(self._after_step)
    def _after_step(self,*_):self.successful+=1
    def close(self):self.handle.remove()
