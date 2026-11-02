"""Versioned fail-closed guards; preserves ordinary GradScaler skip/update policy."""
import math
import torch

VERSION = 'fp16_preupdate_guard_v1'


class NumericalGuardError(FloatingPointError):
    def __init__(self, event):
        self.event = event
        super().__init__(event['reason'])


def nonfinite_tensors(value, prefix=''):
    bad = []
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            bad.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            bad.extend(nonfinite_tensors(item, prefix + '/' + str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            bad.extend(nonfinite_tensors(item, prefix + '/' + str(index)))
    elif isinstance(value, float) and not math.isfinite(value):
        bad.append(prefix)
    return bad


class PreUpdateGuard:
    """Install after accounting hooks. A rejection terminates this attempt.

    Do not catch a rejection and continue training on this object: unscale may
    already have changed gradients/internal scaler stage, but never the model,
    optimizer state, scheduler, EMA, or scaler update in a rejected pre-step.
    """
    def __init__(self, model, optimizer, scaler, on_event=None):
        if not scaler.is_enabled():
            raise ValueError('This revision requires enabled FP16 GradScaler')
        self.model, self.optimizer, self.scaler = model, optimizer, scaler
        self.on_event = on_event
        self.events = []
        self.optimizer_calls = 0
        self.loss_checked = False
        self.original_optimizer_step = optimizer.step
        self.original_scaler_step = scaler.step
        optimizer.step = self._optimizer_step
        scaler.step = self._scaler_step

    def reject(self, reason, **details):
        event = dict(version=VERSION, reason=reason,
                     optimizer_calls=self.optimizer_calls, **details)
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)
        raise NumericalGuardError(event)

    def check_loss(self, loss):
        self.loss_checked = False
        if not torch.is_tensor(loss) or loss.numel() != 1:
            self.reject('loss_not_scalar')
        if not bool(torch.isfinite(loss.detach()).all()):
            self.reject('nonfinite_loss_before_backward')
        self.loss_checked = True

    def check_state(self):
        bad = nonfinite_tensors(self.model.state_dict(), 'model')
        for name in ('mean_density', 'mean_density_torso', 'mean_count'):
            if hasattr(self.model, name):
                bad.extend(nonfinite_tensors(getattr(self.model, name), name))
        bad.extend(nonfinite_tensors(self.optimizer.state_dict(), 'optimizer'))
        if bad:
            self.reject('nonfinite_state_before_optimizer', tensors=bad)

    def gradients(self):
        return {str(i) + '/' + str(j): p.grad
                for i, group in enumerate(self.optimizer.param_groups)
                for j, p in enumerate(group['params']) if p.grad is not None}

    def _scaler_step(self, optimizer, *args, **kwargs):
        if optimizer is not self.optimizer:
            self.reject('unexpected_optimizer')
        if args or kwargs:
            self.reject('optimizer_step_arguments_unsupported')
        if not self.loss_checked:
            self.reject('loss_not_checked_before_backward')
        try:
            self.check_state()
            scale = float(self.scaler.get_scale())
            if not math.isfinite(scale) or scale <= 0:
                self.reject('invalid_scale', scale=scale)
            # Same representable inverse as GradScaler, checked before unscale.
            inverse = torch.tensor(scale, dtype=torch.float64).reciprocal().float()
            if not bool(torch.isfinite(inverse)):
                self.reject('nonfinite_inverse_scale_before_unscale', scale=scale)
            scaled_bad = nonfinite_tensors(self.gradients(), 'scaled_gradients')
            self.scaler.unscale_(optimizer)
            unscaled_bad = nonfinite_tensors(self.gradients(), 'unscaled_gradients')
            if unscaled_bad and not scaled_bad:
                self.reject('nonfinite_created_by_unscale', tensors=unscaled_bad, scale=scale)
            # Existing nonfinite scaled gradients retain the original AMP skip.
            # The actual optimizer entry below is also guarded independently.
            return self.original_scaler_step(optimizer)
        finally:
            self.loss_checked = False

    def _optimizer_step(self, *args, **kwargs):
        self.check_state()
        bad = nonfinite_tensors(self.gradients(), 'unscaled_gradients')
        if bad:
            self.reject('nonfinite_gradient_at_optimizer_entry', tensors=bad)
        result = self.original_optimizer_step(*args, **kwargs)
        self.optimizer_calls += 1
        # Post-check is a final alarm, not a rollback or a guarantee of recovery.
        bad = nonfinite_tensors(self.model.state_dict(), 'model')
        bad.extend(nonfinite_tensors(self.optimizer.state_dict(), 'optimizer'))
        if bad:
            self.reject('nonfinite_after_optimizer', tensors=bad,
                        update_already_occurred=True)
        return result

    def close(self):
        self.optimizer.step = self.original_optimizer_step
        self.scaler.step = self.original_scaler_step


def attach_trainer_guard(trainer, on_event=None):
    """Wrap the existing trainer without replacing its forward or backward math."""
    guard = PreUpdateGuard(trainer.model, trainer.optimizer, trainer.scaler, on_event)
    original = trainer.train_step
    def train_step(data):
        guard.check_state()
        result = original(data)
        guard.check_loss(result[2])
        return result
    trainer.train_step = train_step
    return guard
