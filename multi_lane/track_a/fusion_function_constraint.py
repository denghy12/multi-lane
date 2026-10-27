"""A teacher supplies targets only; no teacher parameters enter the student."""
import math

import torch


def gradient_norm(loss, parameters):
    """Match the initial training scaler; fall back only on actual overflow.

    Adapter-ASL has many very small components. A scale of 1024 can falsely
    measure a zero norm even when the actual scale-65536 update is nonzero.
    """
    parameters=tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        return 0.,0.
    for scale in (65536.,8192.,1024.,128.,1.):
        gradients=torch.autograd.grad(loss*scale, parameters, retain_graph=True, allow_unused=True)
        square=sum(float((g.detach().float()/scale).square().sum().cpu())
                   for g in gradients if g is not None)
        value=math.sqrt(square)
        if math.isfinite(value):
            return value,scale
    raise FloatingPointError('Non-finite functional calibration gradient at every scale')


class FusionFunctionConstraint:
    """Fix beta once on the first actual TRAIN batch of this task's phase II.

    Hard-label objectives retain weight one. The fixed norm ratio is .25,
    beta is bounded by one, and zero-gradient groups explicitly disable beta.
    Targets are current-task logits on the exact same augmented image tensors.
    """
    def __init__(self, teacher_predict, current_classes, include_adapter, ratio=.25, maximum_beta=1.):
        self.teacher_predict=teacher_predict
        self.current=tuple(current_classes)
        if not self.current or len(set(self.current))!=len(self.current):
            raise ValueError('Current class indices must be non-empty and unique')
        if not 0<ratio<=1 or not 0<maximum_beta<=1:
            raise ValueError('Invalid predeclared calibration bounds')
        self.include_adapter=bool(include_adapter); self.ratio=ratio; self.maximum_beta=maximum_beta
        self.calibration=None; self.calls=0

    def coefficient(self, base, functional):
        return min(self.maximum_beta,self.ratio*base/functional) if base>1e-12 and functional>1e-12 else 0.

    def __call__(self, *, images, logits, task_id, model_loss, adapter_loss,
                 model_parameters, adapter_parameters):
        with torch.no_grad():
            target=self.teacher_predict(images,task_id).detach().float()
        if target.shape!=logits.shape or not torch.isfinite(target).all():
            raise ValueError('Teacher and student logits are incompatible/non-finite')
        difference=logits[:,self.current].float()-target[:,self.current]
        distance=difference.square().mean()
        if not torch.isfinite(distance):
            raise FloatingPointError('Non-finite functional distance')
        if self.calibration is None:
            bm,sm=gradient_norm(model_loss,model_parameters)
            dm,sdm=gradient_norm(distance,model_parameters)
            ba,sa=gradient_norm(adapter_loss,adapter_parameters) if self.include_adapter else (0.,0.)
            da,sda=gradient_norm(distance,adapter_parameters) if self.include_adapter else (0.,0.)
            self.calibration=dict(task_id=task_id,source='first_actual_augmented_TRAIN_batch',
                people=len(logits),current_classes=list(self.current),ratio=self.ratio,
                maximum_beta=self.maximum_beta,calibration_gradient_scales=dict(
                    model_base=sm,model_functional=sdm,adapter_base=sa,adapter_functional=sda),
                overflow_fallback_scales=[65536.,8192.,1024.,128.,1.],
                model_base_norm=bm,model_functional_norm=dm,adapter_base_norm=ba,adapter_functional_norm=da,
                beta_model=self.coefficient(bm,dm),beta_adapter=self.coefficient(ba,da),
                zero_signal_model=bm<=1e-12 or dm<=1e-12,
                zero_signal_adapter=self.include_adapter and (ba<=1e-12 or da<=1e-12),
                teacher_and_student_receive_identical_input_object=True)
        elif self.calibration['task_id']!=task_id:
            raise ValueError('Each task must calibrate its own functional constraint')
        self.calls+=1
        beta_m=self.calibration['beta_model']; beta_a=self.calibration['beta_adapter']
        return dict(model_penalty=beta_m*distance,adapter_penalty=beta_a*distance,
            metrics=dict(teacher_mse=float(distance.detach().cpu()),
                teacher_mean_abs_logit_difference=float(difference.detach().abs().mean().cpu()),
                beta_model=beta_m,beta_adapter=beta_a))
