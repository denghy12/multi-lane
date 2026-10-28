"""ON/OFF and memory removals share one three-view image forward per batch."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .paired_transforms import move_model_inputs
from .runner import compute_metrics
from .run_historical_b32_test_attribution import write


def evaluate_cached_modes(model, loader, task, output):
    model.eval()
    names = ('on', 'off', 'no_face_memory', 'no_person_memory') if model.interaction_bank else ('on', 'off')
    logits, probabilities = {n:[] for n in names}, {n:[] for n in names}
    views, view_probs = {n:[] for n in ('full','person','face')}, {n:[] for n in ('full','person','face')}
    targets, reliable, ids, lengths = [], [], [], []
    dose_profiles = {n:{} for n in names} if hasattr(model,'dose_policy') else None

    def capture_profile(name, size):
        if dose_profiles is None or name == 'off':
            return
        for task_id in model._interaction_cache['lane_ids']:
            row = model.interaction_bank[task_id].last_metrics
            stats = dose_profiles[name].setdefault(str(task_id),dict(samples=0,gate_sum=0.,
                gate_square_sum=0.,ratio_sum=0.,near_cap_sum=0.,gate_min=2.,gate_max=0.,ratio_max=0.))
            stats['samples'] += size
            stats['gate_sum'] += size*row['gate_mean']
            stats['gate_square_sum'] += size*(row['gate_mean']**2+row['gate_std']**2)
            stats['ratio_sum'] += size*row['delta_to_base_rms']
            stats['near_cap_sum'] += size*row['delta_near_cap_fraction']
            stats['gate_min'] = min(stats['gate_min'],row['gate_min'])
            stats['gate_max'] = max(stats['gate_max'],row['gate_max'])
            stats['ratio_max'] = max(stats['ratio_max'],row['delta_to_base_rms_max'])
    with torch.no_grad():
        for images, y, sample_ids in loader:
            reliable.append(images['face_reliable'].bool().cpu())
            images = move_model_inputs(images, torch.device('cuda'))
            with torch.cuda.amp.autocast(enabled=True):
                z, vs = model.seen_logits_with_views(images)
                capture_profile('on',len(y))
                values = dict(on=z, off=model.cached_base_seen_logits())
                if model.interaction_bank:
                    values['no_face_memory'] = model.cached_base_seen_logits(model.interaction_delta(no_face=True))
                    capture_profile('no_face_memory',len(y))
                    values['no_person_memory'] = model.cached_base_seen_logits(model.interaction_delta(no_person=True))
                    capture_profile('no_person_memory',len(y))
            # Preserve the historical CPU float32 sigmoid and batch boundaries.
            for name,value in values.items():
                cpu = value.float().cpu()
                if not bool(torch.isfinite(cpu).all()):
                    raise FloatingPointError('Non-finite interaction logits: '+name)
                logits[name].append(cpu); probabilities[name].append(torch.sigmoid(cpu))
            for name,value in vs.items():
                cpu = value.float().cpu()
                views[name].append(cpu); view_probs[name].append(torch.sigmoid(cpu))
            targets.append(y.float().cpu()); ids.extend(map(str,sample_ids)); lengths.append(len(y))
    y = torch.cat(targets); r = torch.cat(reliable)
    if len(ids) != len(set(ids)):
        raise AssertionError('Evaluation duplicated person IDs')
    result = {}
    common = dict(schema_version=np.asarray(1), task_id=np.asarray(task),
        sample_ids=np.asarray(ids), targets=y.numpy(), face_reliable=r.numpy(),
        class_indices=np.arange(y.shape[1]), batch_lengths=np.asarray(lengths),
        probability_device=np.asarray('cpu'), probability_dtype=np.asarray('float32'),
        probability_operation=np.asarray('torch.sigmoid'), torch_version=np.asarray(str(torch.__version__)))
    branch_metrics = {}
    for name in views:
        common[name+'_logits'] = torch.cat(views[name]).numpy()
        scores = torch.cat(view_probs[name])
        common[name+'_probabilities'] = scores.numpy()
        branch_metrics[name] = asdict(compute_metrics(task, scores, y, .5))
        if name == 'face' and bool(r.any()):
            branch_metrics['face_reliable'] = asdict(compute_metrics(task, scores[r], y[r], .5))
    for name in names:
        fused = torch.cat(logits[name]); scores = torch.cat(probabilities[name])
        value = compute_metrics(task, scores, y, .5)
        metrics = dict(fused=asdict(value), **branch_metrics)
        for cohort,mask in (('reliable',r),('unreliable',~r)):
            if bool(mask.any()):
                metrics['fused_'+cohort] = asdict(compute_metrics(task,scores[mask],y[mask],.5))
        folder = Path(output)/name
        path = folder/'view_scores'/f'task{task}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path, **common, fused_logits=fused.numpy(), fused_probabilities=scores.numpy())
        score_path = folder/'scores'/f'task{task}.npz'
        score_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(score_path, **{k:v for k,v in common.items() if not k.endswith(('_logits','_probabilities'))},
                            logits=fused.numpy(), probabilities=scores.numpy())
        diagnostic = dict(task_id=task, metrics=metrics, intervention=name,
                          one_shared_image_forward=True, training_updates=0)
        if dose_profiles is not None:
            profile = {}
            for lane,s in dose_profiles[name].items():
                n=s['samples'];mean=s['gate_sum']/n
                profile[lane]=dict(samples=n,gain_mean=mean,
                    gain_std=max(0.,s['gate_square_sum']/n-mean**2)**.5,
                    gain_min=s['gate_min'],gain_max=s['gate_max'],
                    feature_ratio_mean=s['ratio_sum']/n,feature_ratio_max=s['ratio_max'],
                    near_cap_fraction=s['near_cap_sum']/n)
            diagnostic.update(dose_profiles=profile,final_feature_cap=.01,
                evidence_removal_also_changes_content_gate_inputs=name in ('no_face_memory','no_person_memory'))
        write(folder/f'task{task}.json',dict(metrics=asdict(value),views=diagnostic))
        result[name] = (value, diagnostic)
    return result
