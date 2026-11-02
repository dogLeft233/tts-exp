"""Map split-local attention queries to each intact original recording."""
import weakref
import torch

_REGISTRY={}
def register(aligned,whole,source_ids):
    ids=torch.as_tensor(source_ids,dtype=torch.long,device='cpu')
    assert len(ids)==len(aligned) and len(ids)>0 and int(ids.min())>=4 and int(ids.max())+4<=len(whole)
    assert aligned.shape[1:]==whole.shape[1:] and torch.equal(aligned.cpu(),whole[ids.to(whole.device)].cpu())
    key=id(aligned)
    def remove(ref):
        entry=_REGISTRY.get(key)
        if entry and entry[0] is ref:_REGISTRY.pop(key)
    _REGISTRY[key]=(weakref.ref(aligned,remove),whole,ids)

def wrap(original):
    def mapped(features,att_mode,index):
        entry=_REGISTRY.get(id(features))
        if entry and entry[0]() is features:
            local=int(index);assert 0<=local<len(entry[2])
            return original(entry[1],att_mode,int(entry[2][local]))
        return original(features,att_mode,index)
    mapped.original=original
    return mapped

def install():
    from nerf_triplane import provider,renderer,utils
    original=getattr(utils.get_audio_features,'original',utils.get_audio_features)
    function=wrap(original)
    provider.get_audio_features=renderer.get_audio_features=utils.get_audio_features=function
    return original
