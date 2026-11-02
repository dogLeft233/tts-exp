"""Equal-initialization, equal-update-budget May gain pilot.

Official model, loss and AdamW. Explicit per-step RNG domains keep input frames
and actual sampled rays equal across arms, independently of occupancy work.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from common import (ROOT, DATA, OFFICIAL, configure, dataset, options, sha, save_json,
                    tensor_state_sha, seed_domain, label, bind_training_state, model_metadata)


def initialize(protocol):
    from nerf_triplane.network import NeRFNetwork
    assert not (ROOT/'initialization.pt').exists()
    train = dataset('train', 0, protocol)
    seed_domain(protocol['seed'], 0, 'initialization')
    model = NeRFNetwork(train.opt).cuda()
    bind_training_state(model, train)
    model.mark_untrained_grid(train.poses, train.intrinsics)
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({'model':state,'metadata':model_metadata(model)},ROOT/'initialization.pt')
    rng = np.random.default_rng(protocol['seed'])
    order = np.concatenate([rng.permutation(len(train.poses)) for _ in
                            range(int(np.ceil(protocol['steps_per_model']/len(train.poses))))])
    order = order[:protocol['steps_per_model']].astype(np.int32)
    np.save(ROOT/'frame_order.npy',order)
    val = np.rint(np.linspace(0,549,protocol['validation_frames'])).astype(int).tolist()
    save_json(ROOT/'initialization.json',{'seed':protocol['seed'],'tensor_state_sha256':tensor_state_sha(state),
        'file_sha256':sha(ROOT/'initialization.pt'),'frame_order_sha256':sha(ROOT/'frame_order.npy'),
        'validation_indices':val,'parameter_count':sum(p.numel() for p in model.parameters()),
        'learned_generator_checkpoint_loaded':False,'marked_grid_uses_train_cameras_only':True})
    print(json.dumps({'initialized':True,'state_sha256':tensor_state_sha(state)}),flush=True)


def validation(model, owner, sets, rows, ema, step, output):
    from nerf_triplane.utils import Trainer
    model.eval()
    summary=[]
    with ema.average_parameters(),torch.no_grad():
        for gain,d in sets.items():
            per=[]
            frames=json.loads((Path(d.opt.path)/'transforms_val.json').read_text())['frames']
            for idx in rows:
                data=d.collate([idx])
                result=Trainer.eval_step(owner,data)
                pred,truth=result[0],result[5]
                assert torch.isfinite(pred).all()
                error=pred.float()-truth.float()
                mse=float((error**2).mean());l1=float(error.abs().mean())
                lms=np.loadtxt(DATA/f"source/May/ori_imgs/{frames[idx]['img_id']}.lms")
                x0,y0=np.maximum(np.floor(lms[48:68].min(0)).astype(int)-12,0)
                x1,y1=np.minimum(np.ceil(lms[48:68].max(0)).astype(int)+13,512)
                mouth=error[:,y0:y1,x0:x1]
                per.append({'index':idx,'L1':l1,'MSE':mse,'PSNR':float(-10*np.log10(max(mse,1e-12))),
                            'mouth_bbox_L1':float(mouth.abs().mean()),'mouth_bbox_MSE':float((mouth**2).mean())})
            summary.append({'step':step,'test_gain_db':gain,'frames':len(rows),
                            **{k:float(np.mean([x[k] for x in per])) for k in
                               ['L1','MSE','PSNR','mouth_bbox_L1','mouth_bbox_MSE']}})
            save_json(output/f'val_step{step:06d}_{label(gain)}.json',{'summary':summary[-1],'per_frame':per})
    model.train()
    return summary


def train(protocol,gain):
    from nerf_triplane.network import NeRFNetwork
    from nerf_triplane.utils import Trainer
    import nerf_triplane.provider as provider
    out=ROOT/'training'/label(gain)
    assert not out.exists(),'never replace an executed training arm'
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'validation').mkdir()
    trainset=dataset('train',gain,protocol)
    valsets={db:dataset('val',db,protocol) for db in protocol['validation_gains_db']}
    opt=trainset.opt
    initial=torch.load(ROOT/'initialization.pt',map_location='cpu',weights_only=False)
    receipt=json.loads((ROOT/'initialization.json').read_text())
    model=NeRFNetwork(opt).cuda()
    loaded=model.load_state_dict(initial['model'],strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    initial_hash=tensor_state_sha(model.state_dict())
    assert initial_hash==receipt['tensor_state_sha256']
    for k,v in initial['metadata'].items():setattr(model,k,v)
    bind_training_state(model,trainset)
    owner=SimpleNamespace(opt=opt,model=model,criterion=torch.nn.L1Loss(reduction='none'),
                          global_step=0,flip_finetune_lips=False,flip_init_lips=False)
    optimizer=torch.optim.AdamW(model.get_params(opt.lr,opt.lr_net),betas=(0,.99),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.5**(step/protocol['steps_per_model']))
    ema=ExponentialMovingAverage(model.parameters(),decay=protocol['ema_decay'])
    order=np.load(ROOT/'frame_order.npy');assert len(order)==protocol['steps_per_model']
    seed_domain(protocol['seed'],0,'occupancy')
    with torch.no_grad():model.update_extra_state()
    rows=receipt['validation_indices']
    val_rows=validation(model,owner,valsets,rows,ema,0,out/'validation')
    save_json(out/'initial_control.json',{'initial_state_sha256':initial_hash,
              'frame_order_sha256':sha(ROOT/'frame_order.npy'),'gain_db':gain,
              'steps_budget':protocol['steps_per_model'],'num_rays':protocol['num_rays'],
              'protocol_sha256':sha(ROOT/'protocol.json'),'fp16':False})
    original_get_rays=provider.get_rays
    actual_ray_hash=None
    def observe_rays(*args,**kwargs):
        nonlocal actual_ray_hash
        r=original_get_rays(*args,**kwargs)
        actual_ray_hash=hashlib.sha256(r['inds'].detach().cpu().numpy().tobytes()).digest()
        return r
    losses=[];start=time.monotonic();window_start=start;window_losses=[]
    reference=ROOT/'training/gain_0/ray_indices.sha256bin'
    ref=reference.open('rb') if gain!=0 else None
    log=(out/'train_curve.jsonl').open('x')
    rays=(out/'ray_indices.sha256bin').open('xb')
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(1,protocol['steps_per_model']+1):
            if step>1 and (step-1)%opt.update_extra_interval==0:
                seed_domain(protocol['seed'],step-1,'occupancy')
                with torch.no_grad():model.update_extra_state()
            seed_domain(protocol['seed'],step,'ray_sampling')
            actual_ray_hash=None
            provider.get_rays=observe_rays
            data=trainset.collate([int(order[step-1])])
            provider.get_rays=original_get_rays
            assert actual_ray_hash is not None
            if ref:assert ref.read(32)==actual_ray_hash,f'ray schedule mismatch at step {step}'
            rays.write(actual_ray_hash)
            seed_domain(protocol['seed'],step,'forward_perturbation')
            owner.global_step=step
            optimizer.zero_grad(set_to_none=True)
            pred,truth,loss=Trainer.train_step(owner,data)
            assert torch.isfinite(loss),f'nonfinite loss at {step}'
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step%protocol['ema_update_interval']==0:ema.update()
            value=float(loss.detach());window_losses.append(value)
            if step%protocol['log_interval']==0 or step==1:
                now=time.monotonic()
                row={'step':step,'gain_db':gain,'loss':value,'window_mean_loss':float(np.mean(window_losses)),
                     'lr':float(optimizer.param_groups[0]['lr']),'elapsed_seconds':now-start,
                     'steps_per_second':len(window_losses)/(now-window_start),
                     'peak_gpu_MiB':torch.cuda.max_memory_allocated()/1024**2}
                log.write(json.dumps(row)+'\n');log.flush();rays.flush()
                losses.append(row);window_losses=[];window_start=now
                save_json(ROOT/'status.json',dict(row,phase='training',arm=label(gain),
                                                initial_state_sha256=initial_hash))
                print(json.dumps(row),flush=True)
            if step%protocol['validation_interval']==0:
                val_rows.extend(validation(model,owner,valsets,rows,ema,step,out/'validation'))
                save_json(out/'validation_curve.json',val_rows)
            if step%protocol['checkpoint_interval']==0:
                checkpoint=out/'checkpoints'/f'step{step:06d}.pt'
                state={'step':step,'gain_db':gain,'initial_state_sha256':initial_hash,
                       'protocol_sha256':sha(ROOT/'protocol.json'),
                       'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                       'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                       'scheduler':scheduler.state_dict(),'metadata':model_metadata(model)}
                temporary=checkpoint.with_suffix('.partial');torch.save(state,temporary);temporary.replace(checkpoint)
                save_json(out/'latest_checkpoint.json',{'path':str(checkpoint),'sha256':sha(checkpoint),'step':step})
        rays.close();log.close()
        assert (out/'ray_indices.sha256bin').stat().st_size==protocol['steps_per_model']*32
        save_json(out/'training_receipt.json',{'passed':True,'gain_db':gain,'successful_optimizer_updates':step,
            'initial_state_sha256':initial_hash,'frame_order_sha256':sha(ROOT/'frame_order.npy'),
            'actual_ray_schedule_sha256':sha(out/'ray_indices.sha256bin'),'precision':'FP32',
            'elapsed_seconds':time.monotonic()-start,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
            'ema_updates':step//protocol['ema_update_interval'],'validation_rows':len(val_rows),
            'peak_gpu_MiB':torch.cuda.max_memory_allocated()/1024**2})
    finally:
        provider.get_rays=original_get_rays
        if ref:ref.close()
        if not rays.closed:rays.close()
        if not log.closed:log.close()
    del optimizer,ema,owner,model,trainset,valsets
    gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--gain',type=int)
    a=p.parse_args();configure()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert protocol['fp16_training'] is False
    if a.initialize:initialize(protocol)
    else:assert a.gain in protocol['training_gains_db'];train(protocol,a.gain)


if __name__=='__main__':main()
