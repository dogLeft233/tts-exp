"""One complete official epoch, with strict full-state recovery and measured controls."""
import argparse,copy,hashlib,json,sys,time,os
from pathlib import Path
import numpy as np,torch
from runtime_common import ROOT,configure,dataset,seed_domain,tensor_state_sha
from execution_state import model_counters,restore_model_counters,tree_sha,verify_full_checkpoint
from flow_stream import sha
from cohort_binding import identity_root,verify_member_pins
RUN=ROOT/"trials"/"fp16_other_identities_trial_v1"

def run(identity,seed,gain,phase,epoch,attempt,checkpoint=None,resume=None,reference_trace=None):
    configure()
    contract=json.loads((RUN/"contract.json").read_text())
    assert identity in contract["identities"] and seed==42 and phase=="coarse" and epoch in range(1,11) and attempt==1
    for name,digest in contract["sources"].items():assert sha(RUN/"sources"/name)==digest
    qualification=json.loads((RUN/"qualification/result.json").read_text());assert qualification["passed"]
    admission=json.loads((ROOT/'data_qualification.json').read_text())
    assert admission['passed'] and admission['training_ready'] and admission['identities']==['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']
    assert admission['protocol_sha256']==sha(ROOT/'protocol.json') and admission['execution_contract_sha256']==sha(ROOT/'execution_contract_v1.json')
    assert all(sha(ROOT/name)==digest for name,digest in admission['executing_sources'].items()),'qualified executing source changed'
    assert all(sha(ROOT/'official'/name)==digest for name,digest in admission['official_execution_sources'].items()),'qualified official source changed'
    inputs=identity_root(identity,ROOT,require_admission=True)
    verify_member_pins(ROOT,identity)
    assert phase in ['coarse','fine'] and epoch in (range(1,11) if phase=='coarse' else range(11,18))
    assert attempt>0 and seed in [42,2026] and gain in [-6,0,6]
    from nerf_triplane.network import NeRFNetwork
    from controlled_training import ControlledLoader
    from controlled_training_trial_v1 import TrialControlledTrainer as ControlledTrainer
    data=dataset(identity,'train',gain,seed,phase)
    assert len(data.poses)==6000 and len(data.auds)==6000
    opt=data.opt
    out=RUN/'training'/identity/f'seed{seed}'/f'gain_{gain:+d}'/phase/f'ep{epoch:04d}'/f'attempt{attempt:02d}'
    out.mkdir(parents=True,exist_ok=False);opt.workspace=str(out/'workspace')
    source_files=['train_epoch.py','controlled_training.py','execution_state.py','runtime_common.py','audio_context.py','build_datasets.py',
        'cohort_binding.py','cohort_selection.json','data_qualification.json',
        'official/nerf_triplane/network.py','official/nerf_triplane/renderer.py','official/nerf_triplane/utils.py','official/nerf_triplane/provider.py']
    sources={}
    for name in source_files:
        raw=(ROOT/name).read_bytes();target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name=="cohort_selection.json":os.link(ROOT/name,target)
        else:target.write_bytes(raw)
        sources[name]=hashlib.sha256(raw).hexdigest()
    for name,digest in contract["sources"].items():
        target=out/"trial_sources"/name;target.parent.mkdir(exist_ok=True);target.write_bytes((RUN/"sources"/name).read_bytes());assert sha(target)==digest
    (out/"trial_contract.json").write_bytes((RUN/"contract.json").read_bytes())
    seed_domain(seed,0,'generator_initialization')
    model=NeRFNetwork(opt).cuda()
    first_coarse=(phase=='coarse' and epoch==1)
    first_fine=(phase=='fine' and epoch==11)
    info=None;previous=None
    if first_coarse:
        assert checkpoint is None and resume is None
        qualified=json.loads((inputs/'capacity_qualification'/identity/'capacity_qualification.json').read_text());assert qualified['passed']
        row=next(x for x in qualified['initializations'] if x['seed']==seed)
        initial_path=inputs/'capacity_qualification'/identity/row['path'];assert sha(initial_path)==row['file_sha256']
        initial=torch.load(initial_path,map_location='cpu',weights_only=False)
        model.load_state_dict(initial['model'],strict=True);restore_model_counters(model,initial['model_counters'])
        assert tensor_state_sha(model.state_dict())==row['model_tensor_sha256']
        initial_updates=0;phase_start_updates=0
    else:
        assert checkpoint and resume
        checkpoint=Path(checkpoint);resume=Path(resume)
        assert checkpoint.resolve().is_relative_to((RUN/'training'/identity/f'seed{seed}'/f'gain_{gain:+d}').resolve())
        info=json.loads(resume.read_text())
        assert info['identity']==identity and info['seed']==seed and info['gain_db']==gain and info['epoch']==epoch-1
        assert sha(checkpoint)==info['checkpoint_sha256']
        previous=torch.load(checkpoint,map_location='cpu',weights_only=False)
        initial_updates=info['cumulative_successful_updates']
        if first_fine:
            block=json.loads((RUN/'blocks'/identity/f'seed{seed}'/'coarse_qualification.json').read_text());assert block['passed']
            assert info['phase']=='coarse' and info['global_step']==60000
            phase_start_updates=initial_updates
        else:
            assert info['phase']==phase;phase_start_updates=info['phase_start_updates']
        model.load_state_dict(previous['model'],strict=True)
    optimizer=lambda m:torch.optim.AdamW(m.get_params(opt.lr,opt.lr_net),betas=(0,.99),eps=1e-8)
    factor=.5 if phase=='coarse' else .05
    scheduler=lambda optimizer:torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:factor**(step/opt.iters))
    trainer=ControlledTrainer('ngp',opt,model,device=torch.device('cuda'),workspace=opt.workspace,
        criterion=torch.nn.MSELoss(reduction='none'),optimizer=optimizer,lr_scheduler=scheduler,
        ema_decay=.95,fp16=True,metrics=[],scheduler_update_every_step=True,use_tensorboardX=False,
        use_checkpoint='scratch' if first_coarse else str(checkpoint))
    if previous is not None:
        verify_full_checkpoint(trainer,previous)
        if not first_fine:
            restore_model_counters(model,info['model_counters'])
            trainer.stats=copy.deepcopy(info['post_validation_stats'])
            trainer.ema.store()
            assert tree_sha(trainer.ema.collected_params)==info['post_validation_ema_collected_sha256']
            opt.finetune_lips=info['next_finetune_lips']
        else:
            assert model.iter_density==0 and model.local_step==0,'fine phase must retain original fresh-constructor counters'
    model.aud_features=data.auds;model.eye_area=data.eye_area;model.poses=data.poses
    if first_coarse or first_fine:model.mark_untrained_grid(data.poses,data.intrinsics)
    state_before=tensor_state_sha(model.state_dict())
    trainer.epoch=epoch;assert trainer.global_step==(epoch-1)*6000
    trace=out/'attempts.jsonl'
    trainer.attach_controls(seed,trace,initial_updates,reference_trace)
    loader=ControlledLoader(data,trainer,seed)
    start=time.monotonic();trainer.train_one_epoch(loader)
    assert trainer.global_step==epoch*6000 and trainer.local_step==6000
    assert trainer.ema_update_interval==1000 and trainer.ema.num_updates==trainer.global_step//1000
    trainer.save_checkpoint(name=f'ngp_ep{epoch:04d}',full=True,best=False,remove_old=False)
    full=Path(trainer.ckpt_path)/f'ngp_ep{epoch:04d}.pth'
    assert full.exists();checkpoint_sha=sha(full)
    validation={g:dataset(identity,'val',g,seed,phase) for g in [-6,0,6]}
    validation_receipt=trainer.validate_all_gains(validation,out/'validation')
    trainer.close_controls()
    result=dict(identity=identity,seed=seed,gain_db=gain,phase=phase,epoch=epoch,global_step=trainer.global_step,
        epoch_attempts=6000,cumulative_successful_updates=trainer.update_counter.successful,
        epoch_successful_updates=trainer.update_counter.successful-initial_updates,
        phase_start_updates=phase_start_updates,phase_successful_updates=trainer.update_counter.successful-phase_start_updates,
        epoch_amp_skips=6000-(trainer.update_counter.successful-initial_updates),
        checkpoint_path=str(full),checkpoint_sha256=checkpoint_sha,trace_sha256=sha(trace),
        epoch_order_sha256=trainer.epoch_order_sha256,sampling_sequence_sha256=trainer.sampling_hasher.hexdigest(),
        state_before_epoch_sha256=state_before,model_counters=model_counters(model),
        post_validation_stats=trainer.stats,post_validation_ema_collected_sha256=tree_sha(trainer.ema.collected_params),
        next_finetune_lips=bool(opt.finetune_lips),phase_boundary_original_counter_reset=first_fine,
        original_full_checkpoint_restored=(previous is not None),validation=validation_receipt,
        sources=sources,protocol_sha256=sha(ROOT/'protocol.json'),execution_contract_sha256=sha(ROOT/'execution_contract_v1.json'),
        elapsed_seconds=time.monotonic()-start,passed=True,paired_phase_budget_verified=False)
    if phase=='fine' and epoch==17:
        ema_path=out/'final_ema.pt';result['final_ema_tensor_sha256']=trainer.save_final_ema(ema_path)
        result['final_ema_path']=str(ema_path);result['final_ema_file_sha256']=sha(ema_path)
    result.update(trial_version=contract['version'],trial_contract_sha256=sha(RUN/'contract.json'),original_experiment_version_unchanged=True)
    (out/'epoch_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gain',type=int,required=True)
    p.add_argument('--phase',choices=['coarse','fine'],required=True);p.add_argument('--epoch',type=int,required=True);p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--reference-trace')
    a=p.parse_args();run(a.identity,a.seed,a.gain,a.phase,a.epoch,a.attempt,a.checkpoint,a.resume,a.reference_trace)
