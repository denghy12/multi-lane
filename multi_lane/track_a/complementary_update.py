"""Choose an auxiliary gradient dose without advancing the real Adam state.

The comparison is between two steps of the *same current optimizer*, not a
trained reference model.  The directional tests are local first-order loss
tests; they do not guarantee preservation of validation AP or future outputs.
"""
import inspect
import math

import torch
from torch.optim import _functional

_ADAM_KEYWORDS = frozenset(inspect.signature(_functional.adam).parameters)


def _adam_group(group):
    if group.get("capturable", False) or group.get("differentiable", False):
        raise ValueError("Complementary updates require ordinary non-capturable Adam")
    if group.get("fused", False):
        raise ValueError("Complementary updates do not support fused Adam")
    if isinstance(group["lr"], torch.Tensor):
        raise ValueError("Complementary updates require a scalar Adam learning rate")
    if group.get("decoupled_weight_decay", False):
        raise ValueError("Complementary updates require coupled Adam weight decay")


@torch.no_grad()
def adam_candidate_parameters(*, optimizer, gradients=None, parameters=None):
    """Return rounded candidate parameters; neither optimizer nor params mutate.

    PyTorch's own functional Adam executes on private clones, retaining its
    installed-version arithmetic (including foreach auto-selection).  A None
    gradient skips its parameter, exactly as Adam does.  Accessing missing
    states uses ``get`` so this function does not initialize the real state.
    """
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("Complementary updates require torch.optim.Adam")
    requested = None if parameters is None else {id(p) for p in parameters}
    if parameters is not None and len(requested) != len(parameters):
        raise ValueError("Candidate parameter list contains duplicates")
    gradients = {} if gradients is None else gradients
    result = {}
    for group in optimizer.param_groups:
        _adam_group(group)
        original, trials, grads, first, second, maximum, steps = [], [], [], [], [], [], []
        for p in group["params"]:
            if requested is not None and id(p) not in requested:
                continue
            if p.is_complex() or not p.is_floating_point():
                raise ValueError("Only real floating-point Adam parameters are supported")
            grad = gradients[p] if p in gradients else p.grad
            if grad is None:
                result[p] = p.detach().clone(memory_format=torch.preserve_format)
                continue
            if grad.is_sparse:
                raise ValueError("Sparse gradients are unsupported")
            if grad.shape != p.shape or grad.device != p.device or grad.dtype != p.dtype:
                raise ValueError("Adam candidate gradient must match its parameter")
            state = optimizer.state.get(p, {})
            trial = p.detach().clone(memory_format=torch.preserve_format)
            original.append(p)
            trials.append(trial)
            grads.append(grad.detach())
            first.append(state["exp_avg"].detach().clone() if state else torch.zeros_like(p))
            second.append(state["exp_avg_sq"].detach().clone() if state else torch.zeros_like(p))
            if group.get("amsgrad", False):
                maximum.append(state["max_exp_avg_sq"].detach().clone() if state else torch.zeros_like(p))
            step = state.get("step", 0)
            step_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
            steps.append(step.detach().clone() if isinstance(step, torch.Tensor)
                         else torch.tensor(float(step), dtype=step_dtype))
        if not original:
            continue
        beta1, beta2 = group["betas"]
        kwargs = dict(foreach=group.get("foreach"), capturable=False,
                      differentiable=False, fused=group.get("fused"), grad_scale=None,
                      found_inf=None, has_complex=False,
                      amsgrad=group.get("amsgrad", False), beta1=beta1, beta2=beta2,
                      lr=group["lr"], weight_decay=group["weight_decay"],
                      eps=group["eps"], maximize=group.get("maximize", False))
        kwargs = {k: v for k, v in kwargs.items() if k in _ADAM_KEYWORDS}
        _functional.adam(trials, grads, first, second, maximum, steps, **kwargs)
        result.update(zip(original, trials))
    if requested is not None and requested != {id(p) for p in result}:
        raise ValueError("Candidate parameter is not owned by this optimizer")
    return result


@torch.no_grad()
def adam_candidate_step(*, optimizer, gradients=None, parameters=None):
    """Return actual parameter deltas after floating-point rounding."""
    return {p: value - p.detach() for p, value in adam_candidate_parameters(
        optimizer=optimizer, gradients=gradients, parameters=parameters).items()}


def _squared_norm(tensors, reference):
    values = [t.detach().float().square().sum() for t in tensors if t is not None]
    return torch.stack(values).sum() if values else reference.new_zeros((), dtype=torch.float32)


@torch.no_grad()
def select_auxiliary_gradients(*, optimizer, groups, auxiliary, guards,
                              ramp=1., mode="full_fusion", gradient_ratio=.25,
                              update_ratio=.1, alphas=(1., .5, .25, .125, 0.)):
    """Replace only supplied .grad fields with the chosen unscaled gradients.

    ``groups`` contains disjoint representation/adapter parameter sequences.
    ``auxiliary`` and each guard contain matching sequences (None means zero).
    The base .grad fields must already be unscaled, with baseline masks applied.
    No-gradient base parameters remain None, so the new objective cannot start
    previously inactive coordinates. The head is not in ``groups`` and is never
    touched. Call the real optimizer/scaler exactly once after this function.
    """
    if mode not in {"none", "full", "full_fusion"}:
        raise ValueError("Unknown complementary guard mode")
    if not all(math.isfinite(float(v)) and float(v) >= 0
               for v in (ramp, gradient_ratio, update_ratio)):
        raise ValueError("Complementary update ratios must be finite and nonnegative")
    alphas = tuple(float(a) for a in alphas)
    if (not alphas or alphas[-1] != 0. or any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas)
            or any(a <= b for a, b in zip(alphas, alphas[1:]))):
        raise ValueError("Candidate alphas must decrease strictly and end at zero")
    names = tuple(groups)
    if not names or set(auxiliary) != set(names):
        raise ValueError("Auxiliary parameter groups must match the base groups")
    groups = {name: tuple(groups[name]) for name in names}
    params = tuple(p for name in names for p in groups[name])
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError("Complementary groups must be nonempty and disjoint")
    device = params[0].device
    if any(p.device != device for p in params):
        raise ValueError("Complementary parameter groups must share one device")
    for group in optimizer.param_groups:
        _adam_group(group)
    owner = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if any(id(p) not in owner for p in params):
        raise ValueError("Complementary parameter is not owned by optimizer")
    required = () if mode == "none" else (("full",) if mode == "full" else ("full", "fused"))
    if any(name not in guards for name in required):
        raise ValueError("Required Full/fused guard is missing")
    base = {p: p.grad.detach() if p.grad is not None else None for p in params}
    raw = {}
    skipped_none_base = 0
    for name in names:
        values = tuple(auxiliary[name])
        if len(values) != len(groups[name]):
            raise ValueError("Auxiliary gradient sequence length mismatch")
        for p, q in zip(groups[name], values):
            if q is not None and (q.shape != p.shape or q.device != p.device or q.dtype != p.dtype):
                raise ValueError("Auxiliary gradient must match its parameter")
            if base[p] is None and q is not None:
                skipped_none_base += 1
            raw[p] = q.detach() if q is not None and base[p] is not None else None
    guard_values = {}
    for guard_name, mapping in guards.items():
        if set(mapping) != set(names):
            raise ValueError("Guard parameter groups must match")
        per_param = {}
        for name in names:
            values = tuple(mapping[name])
            if len(values) != len(groups[name]):
                raise ValueError("Guard gradient sequence length mismatch")
            for p, g in zip(groups[name], values):
                if g is not None and (g.shape != p.shape or g.device != p.device):
                    raise ValueError("Guard gradient must match its parameter shape/device")
                per_param[p] = None if g is None else g.detach()
        guard_values[guard_name] = per_param
    def finite(values):
        flags = [torch.isfinite(v).all() for v in values if v is not None]
        return torch.stack(flags).all() if flags else torch.ones((), dtype=torch.bool, device=device)
    valid = torch.stack((finite(base.values()), finite(raw.values()),
                         finite(g for values in guard_values.values() for g in values.values()))).cpu().tolist()
    if not all(valid):
        why = ("nonfinite_base" if not valid[0] else
               "nonfinite_auxiliary" if not valid[1] else "nonfinite_guard")
        return dict(alpha=0., status=why, ramp=float(ramp), mode=mode,
                    effective_nonzero_update=False,
                    base_none_auxiliary_ignored=skipped_none_base)
    if ramp == 0 or gradient_ratio == 0:
        return dict(alpha=0., status="inactive", ramp=float(ramp), mode=mode,
                    effective_nonzero_update=False,
                    base_none_auxiliary_ignored=skipped_none_base)
    stats = {}
    capped = {}
    for name in names:
        bn = _squared_norm((base[p] for p in groups[name]), params[0]).sqrt()
        qn = _squared_norm((raw[p] for p in groups[name]), params[0]).sqrt()
        factor = torch.minimum(torch.ones_like(qn), float(gradient_ratio) * bn /
                               qn.clamp_min(torch.finfo(qn.dtype).tiny))
        factor = torch.where((bn > 0) & (qn > 0), factor, torch.zeros_like(factor))
        for p in groups[name]:
            capped[p] = None if raw[p] is None else raw[p] * factor
        stats[name + ".base_gradient_norm"] = bn
        stats[name + ".raw_auxiliary_gradient_norm"] = qn
        stats[name + ".auxiliary_gradient_factor"] = factor
        stats[name + ".capped_auxiliary_gradient_norm"] = qn * factor
    if not bool(torch.stack([stats[name + ".capped_auxiliary_gradient_norm"] for name in names]).sum() > 0):
        packed = torch.stack(tuple(stats.values())).detach().cpu().tolist()
        numeric = dict(zip(stats, packed))
        return dict(alpha=0., status="no_auxiliary_signal", ramp=float(ramp), mode=mode,
                    effective_nonzero_update=False,
                    groups={name: {key.split(".", 1)[1]: value for key, value in numeric.items()
                                   if key.startswith(name + ".")} for name in names},
                    base_none_auxiliary_ignored=skipped_none_base)
    base_candidate = adam_candidate_parameters(optimizer=optimizer, gradients=base, parameters=params)
    base_steps = {p: base_candidate[p] - p.detach() for p in params}
    base_norms = {name: _squared_norm((base_steps[p] for p in groups[name]), params[0]).sqrt()
                  for name in names}
    for name, norm in base_norms.items():
        stats[name + ".base_step_norm"] = norm
    candidates, accepted = [], []
    for index, alpha in enumerate(alphas):
        candidate_gradients = {p: (base[p] if capped[p] is None or alpha == 0 else
                                  base[p] + float(alpha * ramp) * capped[p]) for p in params}
        candidate = (base_candidate if alpha == 0 else adam_candidate_parameters(
            optimizer=optimizer, gradients=candidate_gradients, parameters=params))
        extra_step = {p: candidate[p] - base_candidate[p] for p in params}
        checks = []
        for name in names:
            norm = _squared_norm((extra_step[p] for p in groups[name]), params[0]).sqrt()
            stats[f"candidate{index}.{name}.extra_step_norm"] = norm
            checks.append(norm <= float(update_ratio) * base_norms[name])
        for guard_name, gradient in guard_values.items():
            terms = [(gradient[p].float() * extra_step[p].float()).sum()
                     for p in params if gradient[p] is not None]
            dot = torch.stack(terms).sum() if terms else params[0].new_zeros((), dtype=torch.float32)
            stats[f"candidate{index}.{guard_name}.first_order_extra_loss"] = dot
            if guard_name in required:
                checks.append(dot <= 0)
        good = torch.stack(checks).all()
        accepted.append(good)
        candidates.append(candidate_gradients)
    accepted_cpu = torch.stack(accepted).cpu().tolist()
    chosen = next((i for i, good in enumerate(accepted_cpu) if good), len(alphas) - 1)
    alpha = alphas[chosen]
    if alpha:
        for p in params:
            if capped[p] is not None:
                p.grad = candidates[chosen][p]
    packed = torch.stack(tuple(stats.values())).detach().cpu().tolist()
    numeric = dict(zip(stats, packed))
    group_metrics = {name: {key.split(".", 1)[1]: value for key, value in numeric.items()
                            if key.startswith(name + ".")} for name in names}
    candidate_metrics = []
    for index, a in enumerate(alphas):
        prefix = f"candidate{index}."
        candidate_metrics.append(dict(alpha=a, accepted=bool(accepted_cpu[index]),
            **{key[len(prefix):]: value for key, value in numeric.items() if key.startswith(prefix)}))
    effective = any(numeric[f"candidate{chosen}.{name}.extra_step_norm"] > 0 for name in names)
    status = ("accepted" if effective else "accepted_zero_parameter_delta") if alpha else "baseline_fallback"
    return dict(alpha=alpha, status=status, effective_nonzero_update=effective,
                ramp=float(ramp), mode=mode, groups=group_metrics,
                candidates=candidate_metrics, base_none_auxiliary_ignored=skipped_none_base,
                selected_first_order_extra_loss={name: numeric[f"candidate{chosen}.{name}.first_order_extra_loss"]
                                                 for name in guard_values},
                selected_extra_step_norm={name: numeric[f"candidate{chosen}.{name}.extra_step_norm"] for name in names})
