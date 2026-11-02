"""One isolated 256-attempt recovery probe; never resumes a frozen original run."""
import copy,fcntl,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
OLD=ROOT/'trials/patrick_ave_rms_coarse_v1'
RUN=ROOT/'trials/patrick_rms_density_recovery_probe_v1'
TERMINAL=OLD/'training/PatrickLeahy/seed42/gain_+0/coarse/ep0004/attempt01/terminal_state.pth'
CACHE=OLD/'training/PatrickLeahy/seed42/gain_+0/coarse/ep0003/attempt01/workspace/checkpoints/ngp_ep0003.pth'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    contract=json.loads((RUN/'contract.json').read_text())
    assert contract['attempt_budget']==256
    for n,s in contract['sources'].items():assert sha(RUN/'sources'/n)==s
    assert sha(TERMINAL)==contract['terminal_sha256'] and sha(CACHE)==contract['finite_cache_checkpoint_sha256']
    lock=(ROOT/'preparation_GPU.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:c=(p/'cmdline').read_bytes()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert b'train_patrick_rms_epoch_v1.py' not in c
    sys.path[:0]=[str(RUN/'sources'),str(OLD/'sources'),str(ROOT)]
    import numpy as np,torch
    import runtime_common as rc
    rc.configure()
    from patrick_rms_runtime_v1 import install
    install()
    from execution_state import MODEL_COUNTERS,restore_model_counters,restore_rng,tree_sha,model_counters
    from nerf_triplane.network import NeRFNetwork
    from controlled_training_trial_v1 import TrialControlledTrainer
    from controlled_training import ControlledLoader
    from density_refresh_transaction_v1 import install as transaction
    out=RUN/'training/PatrickLeahy/seed42/gain_+0/recovery_probe/ep0004/attempt01'
    out.mkdir(parents=True,exist_ok=False)
    status_path=RUN/'receipt.json'
    def status(stage,**extra):
        r=dict(version=contract['version'],stage=stage,training_original_resumed=False,attempt_budget=256,**extra)
        status_path.write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r),flush=True)
    state=torch.load(TERMINAL,map_location='cpu',weights_only=False)
    cache=torch.load(CACHE,map_location='cpu',weights_only=False)
    data=rc.dataset('PatrickLeahy','train',0,42,'coarse',preload=0);opt=data.opt
    opt.workspace=str(out/'workspace');opt.finetune_lips=bool(state['current_sampling']['lip_mode'])
    model=NeRFNetwork(opt).cuda();model.load_state_dict(state['model'],strict=True)
    restore_model_counters(model,state['model_counters'])
    for name in ('density_grid','density_bitfield'):
        assert name in cache['model'] and bool(torch.isfinite(cache['model'][name]).all())
        getattr(model,name).copy_(cache['model'][name])
    prior_receipt=json.loads((CACHE.parents[2]/'epoch_receipt.json').read_text())
    model.mean_density=prior_receipt['model_counters']['mean_density']
    assert __import__('math').isfinite(model.mean_density)
    optimizer=lambda m:torch.optim.AdamW(m.get_params(opt.lr,opt.lr_net),betas=(0,.99),eps=1e-8)
    scheduler=lambda o:torch.optim.lr_scheduler.LambdaLR(o,lambda step:.5**(step/opt.iters))
    trainer=TrialControlledTrainer('ngp',opt,model,device=torch.device('cuda'),workspace=opt.workspace,
        criterion=torch.nn.MSELoss(reduction='none'),optimizer=optimizer,lr_scheduler=scheduler,ema_decay=.95,
        fp16=True,metrics=[],scheduler_update_every_step=True,use_tensorboardX=False,use_checkpoint='scratch')
    trainer.optimizer.load_state_dict(state['optimizer']);trainer.scaler.load_state_dict(state['scaler'])
    trainer.lr_scheduler.load_state_dict(state['scheduler']);trainer.ema.load_state_dict(state['ema'])
    for key,obj in [('optimizer',trainer.optimizer),('scaler',trainer.scaler),('scheduler',trainer.lr_scheduler),('ema',trainer.ema)]:
        assert tree_sha(obj.state_dict())==tree_sha(state[key]),key
    trainer.global_step=18992;trainer.epoch=4
    model.aud_features=data.auds;model.eye_area=data.eye_area;model.poses=data.poses
    weights_before=rc.tensor_state_sha(dict(model.named_parameters()))
    assert weights_before==rc.tensor_state_sha({k:v for k,v in state['model'].items() if k in dict(model.named_parameters())})
    restore_rng(state['rng'])
    refreshes=[]
    trainer.attach_controls(42,out/'attempts.jsonl',state['successful_updates'])
    def record_refresh(row):
        row.update(global_step=trainer.global_step)
        refreshes.append(row)
        with (out/'density_refreshes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    transaction(model,MODEL_COUNTERS,record_refresh)
    trainer.numerical_guard.check_state()
    validation=rc.dataset('PatrickLeahy','val',0,42,'coarse',preload=0)
    fixed=[0,107,214,321,428,535,642,749]
    def evaluate(label):
        was=model.training;before=rc.tensor_state_sha(model.state_dict());rows=[];model.eval()
        with torch.no_grad():
            for i in fixed:
                rc.seed_domain(42,i,'recovery_probe_fixed_validation')
                item=validation.collate([i])
                with torch.cuda.amp.autocast(enabled=True):result=trainer.eval_step(item)
                loss=float(result[6])
                rows.append(dict(frame=i,mse=loss if __import__('math').isfinite(loss) else None,finite=__import__('math').isfinite(loss)))
        model.train(was)
        assert rc.tensor_state_sha(model.state_dict())==before
        (out/(label+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False))
        return rows
    status('baseline_validation',initial_global_step=18992,initial_successful_updates=state['successful_updates'])
    baseline=evaluate('validation_before')
    # Verify the first selected frame/ray sample matches the rejected original entry.
    base_loader=ControlledLoader(data,trainer,42)
    class SliceLoader:
        batch_size=1
        has_gt=True
        def __len__(self):return 256
        def __iter__(self):
            import itertools
            for j,batch in enumerate(itertools.islice(iter(base_loader),992,1248)):
                if j==0:assert trainer.current_sampling==state['current_sampling']
                yield batch
    try:
        status('training_short_probe',initial_global_step=18992,initial_successful_updates=state['successful_updates'])
        start=time.monotonic();trainer.train_one_epoch(SliceLoader());elapsed=time.monotonic()-start
        assert trainer.global_step==19248 and trainer.local_step==256
        trainer.close_controls()
        trainer.numerical_guard.close()
        after=evaluate('validation_after')
        trainer.save_checkpoint(name='recovery_probe_256_attempts',full=True,best=False,remove_old=False)
        rows=[json.loads(x) for x in (out/'attempts.jsonl').read_text().splitlines()]
        assert len(rows)==256 and all(__import__('math').isfinite(x['loss']) for x in rows)
        result=dict(passed=True,status='bounded_probe_completed_no_automatic_continuation',attempts=256,
            successful_updates=sum(x['successful_update'] for x in rows),amp_skips=sum(x['amp_skip'] for x in rows),
            initial_global_step=18992,final_global_step=trainer.global_step,
            density_refreshes=refreshes,validation_before=baseline,validation_after=after,elapsed_seconds=elapsed,
            source_contract_sha256=sha(RUN/'contract.json'),original_terminal_unchanged=sha(TERMINAL)==contract['terminal_sha256'],
            optimizer_scaler_scheduler_ema_restored_exact=True,parameter_change_sha_before=weights_before,
            parameter_change_sha_after=rc.tensor_state_sha(dict(model.named_parameters())),
            cache_rollback_policy='discard nonfinite refresh; retain most recent finite cache, initially coarse3',
            original_trial_resumed=False,quality_or_full_training_stability_proven=False)
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        status('completed',result=result)
    except Exception as e:
        trainer.trace.flush()
        failure=dict(stage='short_probe_terminal_failure_no_retry',exception=repr(e),global_step=trainer.global_step,
            trace_lines=len((out/'attempts.jsonl').read_text().splitlines()),traceback=traceback.format_exc(),refreshes=refreshes)
        (out/'probe_failure.json').write_text(json.dumps(failure,indent=2))
        status('failed_no_retry',failure=failure)
        raise
if __name__=='__main__':main()
