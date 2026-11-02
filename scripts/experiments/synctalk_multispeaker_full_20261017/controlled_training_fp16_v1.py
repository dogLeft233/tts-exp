"""Separate FP16 controller revision. Never import this into an old attempt."""
import json
from pathlib import Path
from controlled_training import ControlledTrainer
from fp16_preupdate_guard_v1 import attach_trainer_guard

class GuardedControlledTrainer(ControlledTrainer):
    def attach_controls(self,seed,trace,initial_updates=0,reference_trace=None):
        failure_path=Path(str(trace)+'.numerical_failure.json')
        if failure_path.exists():
            raise FileExistsError(failure_path)
        super().attach_controls(seed,trace,initial_updates,reference_trace)
        def record(event):
            # A separate terminal receipt: never invent a completed trace row.
            payload=dict(version='fp16_preupdate_revision_v1',epoch=self.epoch,
                         global_step=self.global_step,sampling=self.current_sampling,
                         event=event,recovery='terminate_attempt_no_inplace_resume')
            with failure_path.open('x',encoding='utf-8') as f:
                json.dump(payload,f,indent=2,allow_nan=False)
                f.flush()
            self.trace.flush()
        self.numerical_guard=attach_trainer_guard(self,on_event=record)
