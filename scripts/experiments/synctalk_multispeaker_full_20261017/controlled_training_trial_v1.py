"""FP16 guarded trial: preserve exact terminal state and first ordinary AMP skip."""
import json,math,os
from pathlib import Path
import torch
from controlled_training_fp16_v1 import GuardedControlledTrainer
from execution_state import model_counters,rng_state

def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return repr(value)
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value

class TrialControlledTrainer(GuardedControlledTrainer):
    def capture(self,path,stage,event):
        assert not path.exists()
        state=dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),scaler=self.scaler.state_dict(),scheduler=self.lr_scheduler.state_dict(),ema=self.ema.state_dict(),model_counters=model_counters(self.model),rng=rng_state(),epoch=self.epoch,global_step=self.global_step,current_sampling=self.current_sampling,stage=stage,event=event,successful_updates=self.update_counter.successful,gradients={n:p.grad for n,p in self.model.named_parameters() if p.grad is not None})
        temporary=path.with_suffix('.partial')
        with temporary.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        temporary.rename(path)

    def attach_controls(self,seed,trace,initial_updates=0,reference_trace=None):
        super().attach_controls(seed,trace,initial_updates,reference_trace)
        folder=Path(trace).parent
        def record(event):
            self.trace.flush()
            # Write the terminal reason before capturing state; a storage failure cannot erase it.
            failure=Path(str(trace)+'.numerical_failure.json')
            with failure.open('x',encoding='utf-8') as f:
                json.dump(json_safe(dict(version='fp16_other_identities_trial_v1',epoch=self.epoch,global_step=self.global_step,sampling=self.current_sampling,event=event,recovery='terminal_no_replay')),f,indent=2,allow_nan=False)
                f.flush();os.fsync(f.fileno())
            self.capture(folder/'terminal_state.pth','guard_rejected_before_any_further_update',event)
        self.numerical_guard.on_event=record
        original_update=self.scaler.update
        def update(*args,**kwargs):
            result=original_update(*args,**kwargs)
            # Only the first skip per arm, determined by a marker outside the epoch folder.
            arm=folder.parents[2];marker=arm/'first_amp_skip.json'
            if self.update_counter.successful==self._updates_before and not marker.exists():
                self.capture(folder/'first_amp_skip_state.pth','after_scaler_update_before_scheduler_ema',dict(scale_before=self._scale_before,scale_after=float(self.scaler.get_scale())))
                with marker.open('x') as f:json.dump(dict(epoch=self.epoch,global_step=self.global_step,path=str(folder/'first_amp_skip_state.pth')),f)
            return result
        self.scaler.update=update
