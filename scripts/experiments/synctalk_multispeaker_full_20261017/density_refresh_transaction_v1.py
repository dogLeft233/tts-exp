"""Experimental cache-only rollback; parameter/loss/gradient guards stay mandatory."""
import copy
import torch

def install(model, counters, on_refresh):
    original = model.update_extra_state
    def update(*args, **kwargs):
        before = {name: value.detach().clone() for name,value in model.named_buffers()}
        counts = {name: copy.deepcopy(getattr(model,name)) for name in counters if hasattr(model,name)}
        result = original(*args, **kwargs)
        bad = [name for name,value in model.named_buffers() if value.is_floating_point() and not bool(torch.isfinite(value).all())]
        for name in counts:
            v=getattr(model,name)
            if isinstance(v,float) and not __import__('math').isfinite(v):bad.append(name)
        if bad:
            with torch.no_grad():
                for name,value in model.named_buffers():value.copy_(before[name])
            for name,value in counts.items():setattr(model,name,value)
        on_refresh(dict(accepted=not bool(bad),rejected_buffers_or_counters=bad,cache_rolled_back=bool(bad)))
        return result
    model.update_extra_state=update
    return original
