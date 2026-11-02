"""Full65536-ray zero-update resource and common-initialization controls."""
import argparse,gc,hashlib,json
from pathlib import Path
import numpy as np,torch
from runtime_common import ROOT,configure,dataset,options,seed_domain,tensor_state_sha
from execution_state import model_counters,tree_sha
from flow_stream import sha

def bind_full_conditioning(model,data,identity,gain):
    from nerf_triplane.provider import nerf_matrix_to_ngp
    import audio_context
    binding=json.loads((Path(data.opt.path)/'binding.json').read_text())
    transform=json.loads((Path(data.opt.path)/'transforms_train.json').read_text())
    poses=torch.from_numpy(np.stack([nerf_matrix_to_ngp(np.array(f['transform_matrix'],dtype=np.float32),scale=data.scale,offset=data.offset) for f in transform['frames']]))
    whole=data.whole_recording_audio
    ids=torch.tensor(binding['source_ids'],dtype=torch.long)
    auds=whole[ids.to(whole.device)]
    bs=np.load(Path(data.opt.path)/'bs.npy')
    assert not data.opt.au45 and not data.opt.smooth_eye
    if data.opt.bs_area=='upper':values=np.hstack((bs[:,0:5],bs[:,8:10]))
    elif data.opt.bs_area=='single':values=np.hstack((bs[:,0,None],bs[:,2,None],bs[:,3,None],bs[:,8,None]))
    else:
        assert data.opt.bs_area=='eye';values=bs[:,8:10]
    eyes=torch.from_numpy(np.array(values,dtype=np.float32)).view(-1,values.shape[1])
    assert poses.shape==(6000,4,4) and auds.shape==(6000,1,512) and eyes.shape[0]==6000
    assert torch.equal(poses[:len(data.poses)],data.poses.cpu())
    assert torch.equal(auds[:len(data.auds)].cpu(),data.auds.cpu()) and torch.equal(eyes[:len(data.eye_area)],data.eye_area.cpu())
    audio_context.register(auds,whole,ids)
    model.aud_features=auds;model.eye_area=eyes;model.poses=poses
    return poses

def run(identity):
    configure()
    from nerf_triplane.network import NeRFNetwork
    from nerf_triplane.utils import Trainer
    out=ROOT/'capacity_qualification'/identity;out.mkdir(parents=True,exist_ok=False)
    raw=Path(__file__).read_bytes();(out/Path(__file__).name).write_bytes(raw)
    result=dict(identity=identity,source_sha256=hashlib.sha256(raw).hexdigest(),rows=[],initializations=[],
        optimizer_updates=0,training_admitted=False,capacity_stress='all density bitfield cells occupied; max_steps16 retained',
        semantics='resource and backward diagnosis only; stress buffers never become experiment initialization')
    untrained_template=None;poses_template=None
    for seed in [42,2026]:
        opt=options(identity,'train',0,seed,'coarse',0,(0,4))
        seed_domain(seed,0,'generator_initialization')
        base=NeRFNetwork(opt).cuda()
        initial={name:value.detach().cpu().clone() for name,value in base.state_dict().items()}
        counters=model_counters(base);initial_sha=tensor_state_sha(initial)
        initial_path=out/f'initial_seed{seed}.pt'
        torch.save(dict(model=initial,model_counters=counters,seed=seed,identity=identity,
            model_tensor_sha256=initial_sha,pretrained_generator=False,optimizer_updates=0),initial_path)
        result['initializations'].append(dict(seed=seed,path=initial_path.name,file_sha256=sha(initial_path),model_tensor_sha256=initial_sha))
        del base;gc.collect();torch.cuda.empty_cache()
        for phase,lips in [('coarse',False),('fine',True),('fine',False)]:
            for gain in [-6,0,6]:
                data=dataset(identity,'train',gain,seed,phase,0,(0,4))
                opt=data.opt;opt.finetune_lips=(phase=='fine')
                seed_domain(seed,0,'probe_model_constructor')
                model=NeRFNetwork(opt).cuda();model.load_state_dict(initial,strict=True)
                assert tensor_state_sha(model.state_dict())==initial_sha
                poses=bind_full_conditioning(model,data,identity,gain)
                workspace=out/f'seed{seed}_{phase}_{int(lips)}_{gain:+d}'
                trainer=Trainer('capacity',opt,model,device=torch.device('cuda'),workspace=str(workspace),
                    optimizer=lambda m:torch.optim.AdamW(m.get_params(opt.lr,opt.lr_net),betas=(0,.99),eps=1e-8),
                    criterion=torch.nn.MSELoss(reduction='none'),ema_decay=.95,fp16=True,metrics=[],
                    use_checkpoint='scratch',use_tensorboardX=False,mute=True)
                opt.finetune_lips=lips
                assert trainer.flip_finetune_lips==(phase=='fine')
                trainer.global_step=1 if phase=='coarse' else 60001
                model.train()
                if untrained_template is None:
                    model.mark_untrained_grid(poses,data.intrinsics)
                    untrained_template=model.density_grid.detach().clone();poses_template=poses.clone()
                else:
                    assert torch.equal(poses,poses_template) and model.density_grid.shape==untrained_template.shape
                    model.density_grid.copy_(untrained_template)
                seed_domain(seed,0,'occupancy_randomness')
                with torch.cuda.amp.autocast(enabled=True):model.update_extra_state()
                model.density_bitfield.fill_(255)
                parameters_before=tensor_state_sha(dict(model.named_parameters()))
                seed_domain(seed,trainer.global_step,'pixel_ray_patch_rectangle');batch=data.collate([0])
                count=int(batch['rays_o'].shape[1]);assert lips or count==65536
                torch.cuda.reset_peak_memory_stats();seed_domain(seed,trainer.global_step,'render_perturbation_and_regularization')
                with torch.cuda.amp.autocast(enabled=True):_,_,loss=trainer.train_step(batch)
                assert torch.isfinite(loss)
                trainer.scaler.scale(loss).backward();trainer.scaler.unscale_(trainer.optimizer)
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                finite=all(bool(torch.isfinite(g).all()) for g in gradients)
                assert not trainer.optimizer.state and parameters_before==tensor_state_sha(dict(model.named_parameters()))
                row=dict(seed=seed,phase=phase,lips=lips,gain_db=gain,actual_rays=count,num_rays_argument=opt.num_rays,
                    max_steps=opt.max_steps,loss=float(loss.detach()),gradient_parameters=len(gradients),
                    original_fine_stage_flip_enabled=trainer.flip_finetune_lips,
                    default_scale=float(trainer.scaler.get_scale()),unscaled_gradients_finite=finite,
                    max_memory_allocated=int(torch.cuda.max_memory_allocated()),max_memory_reserved=int(torch.cuda.max_memory_reserved()),
                    initialization_exact=True,parameter_bytes_unchanged=True,optimizer_updates=0)
                result['rows'].append(row)
                (out/'capacity_qualification.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
                if trainer.log_ptr:trainer.log_ptr.close();trainer.log_ptr=None
                del trainer,model,data,batch,loss,gradients;gc.collect();torch.cuda.empty_cache()
    assert len(result['rows'])==18
    result.update(passed=True,all_full_ray_backward_cases_completed=True,common_initialization_exact=True,
        successful_update_equality='must be checked from actual training; zero-update probes cannot establish it')
    (out/'capacity_qualification.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);a=p.parse_args();run(a.identity)
