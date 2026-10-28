"""Readonly feature-forward attribution of gate content versus residual dose.

Only hooks and functional readouts are added. Original training/model files are
unchanged; all constants are locked from TRAIN logs before evaluation opens.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .export_compact_test_scores import _sha256, _torch_load
from .historical_b32_backbone_attribution import method_state
from .paired_transforms import move_model_inputs
from .protected_joint_attention import bounded_delta
from .protected_joint_experiment import build
from .run_historical_b32_test_attribution import read, write
from .runner import (compute_metrics, dataset_view, seen_indices, set_seed,
                     tensor_state_digest)
from .teacher_free_shared_learning import runtime, evaluation_loader
from .evaluate_joint26_checkpoint import make_source

TASKS = (0, 2, 7)
MODES = ('native', 'off', 'train_scaled_dynamic', 'train_scaled_constant',
         'cap1_dynamic', 'cap1_constant', 'cap1_open',
         'cap1_permuted_gate',
         'cap1_dynamic_no_face_fixed_gate', 'cap1_constant_no_face')


def train_constants(control, candidate):
    result = dict(scale=[], gate_mean=[], reference_output_ratio=[], native_output_ratio=[])
    if set(control) != set(candidate) or set(control) != {str(t) for t in range(8)}:
        raise ValueError('Expected all eight TRAIN task histories')
    for task in range(8):
        a, b = control[str(task)], candidate[str(task)]
        if len(a) != 30 or len(b) != 30:
            raise ValueError('Constants require complete original 30-epoch training')
        target = float(np.mean([r['interaction_delta_to_base_rms'] for r in a[-5:]]))
        original = float(np.mean([r['interaction_delta_to_base_rms'] for r in b[-5:]]))
        gamma = float(np.mean([r['interaction_gate_mean'] for r in b[-5:]]))
        if not np.isfinite([target, original, gamma]).all() or min(target, original) <= 0 or not 0 <= gamma <= 1:
            raise ValueError('Invalid TRAIN-only constant')
        result['scale'].append(target/original); result['gate_mean'].append(gamma)
        result['reference_output_ratio'].append(target); result['native_output_ratio'].append(original)
    return result


def calibrate(training, output):
    cfg = read(training/'config.json'); complete = read(training/'training_complete.json')
    if cfg['arm'] != 'C4' or complete['tasks'] != list(range(8)) or complete['status'] != 'complete':
        raise ValueError('Use the complete original C4 training source')
    control = training.parent.parent/'C3/train/training_history.json'
    source = training/'training_history.json'
    constants = train_constants(read(control), read(source))
    constants.update(policy='mean_of_last_five_original_TRAIN_epochs_per_task',
        inputs={str(p):_sha256(p) for p in (source, control, training/'config.json')},
        evaluation_data_read=False, training_updates=0, selected_tasks=list(TASKS),
        native_gate_reference_cap=.1, common_final_feature_ratio_cap=.01,
        actual_mean_norms_need_reporting=True,
        checkpoint_sha256={str(t):_sha256(training/'compact_checkpoints'/f'task{t}.pth') for t in TASKS})
    if output.exists():
        if read(output) != constants:
            raise ValueError('Locked TRAIN constants changed')
    else:
        write(output, constants)
    return constants


class ReadoutCapture:
    def __init__(self, model):
        self.model, self.handles, self.records = model, [], {}

    def __enter__(self):
        for task, module in enumerate(self.model.interaction_bank):
            if module.gate is None or not hasattr(module, 'out'):
                raise ValueError('Gate attribution expects the original C4 readout')
            def save(name, t):
                def hook(_module, _args, result):
                    self.records.setdefault(t, {})[name] = result.detach()
                return hook
            self.handles += [module.out.register_forward_hook(save('raw', task)),
                             module.gate.register_forward_hook(save('gate_logit', task)),
                             module.register_forward_hook(save('delta', task))]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self):
        self.records = {}

    def tensors(self, lane_ids):
        return {name:torch.stack([self.records[t][name] for t in lane_ids], 1)
                for name in ('raw', 'gate_logit', 'delta')}


def candidate_deltas(model, capture, constants, permutation_seed=20261028):
    if torch.is_grad_enabled():
        raise RuntimeError('This diagnostic must run without autograd')
    cache = model._interaction_cache
    base, tasks = cache['base'], cache['lane_ids']
    native = capture.tensors(tasks)
    with torch.autocast(device_type=base.device.type, enabled=False):
        gamma = native['gate_logit'].sigmoid()
        old_bound = bounded_delta(native['raw'], base, .1)
        if not torch.equal(gamma * old_bound, native['delta']):
            raise AssertionError('Functional components did not exactly replay the original branch')
        shape = (1, len(tasks), 1)
        scale = base.new_tensor([constants['scale'][t] for t in tasks]).reshape(shape)
        mean = base.new_tensor([constants['gate_mean'][t] for t in tasks]).reshape(shape)
        tight = bounded_delta(native['raw'], base, .01)
        # Preserve gate distribution and known quality flags while removing its
        # association with image content. A private CPU RNG never touches the
        # model's initialization/random stream and never reads labels.
        permuted = gamma.clone()
        reliable = cache['reliable'].bool().cpu()
        visible = cache.get('visible', torch.ones(len(base),dtype=torch.bool,device=base.device)).bool().cpu()
        for i,t in enumerate(tasks):
            generator=torch.Generator().manual_seed(permutation_seed+t)
            for r in (False,True):
                for v in (False,True):
                    group=torch.nonzero((reliable==r)&(visible==v),as_tuple=False).flatten()
                    donor=group[torch.randperm(len(group),generator=generator)]
                    permuted[group.to(base.device),i]=gamma[donor.to(base.device),i]
        cache['diagnostic_permuted_gate']=permuted[...,0]
        deltas = dict(native=native['delta'], off=torch.zeros_like(base),
            train_scaled_dynamic=scale * native['delta'],
            train_scaled_constant=scale * mean * old_bound,
            cap1_dynamic=gamma * tight, cap1_constant=mean * tight, cap1_open=tight,
            cap1_permuted_gate=permuted*tight)
        # no_face=True leaves h0/Person Query unchanged. Its computed gate is
        # deliberately discarded; hold the recipient's original gate constant.
        capture.clear()
        model.interaction_delta(no_face=True)
        no_face = capture.tensors(tasks)
        face_tight = bounded_delta(no_face['raw'], base, .01)
        deltas['cap1_dynamic_no_face_fixed_gate'] = gamma * face_tight
        deltas['cap1_constant_no_face'] = mean * face_tight
        denom = base.float().square().mean(-1).sqrt().clamp_min(1e-12)
        ratios = {name:value.float().square().mean(-1).sqrt()/denom for name,value in deltas.items()}
        for name,value in ratios.items():
            if name.startswith('cap1') and bool((value > .010001).any()):
                raise AssertionError(('Exceeded common final feature cap', name))
    return deltas, gamma[..., 0], ratios


def subset_metrics(task, probability, targets, reliable, sample_ids):
    images = np.asarray([str(s).rsplit('#person=',1)[0] for s in sample_ids])
    _, inverse, counts = np.unique(images, return_inverse=True, return_counts=True)
    masks = dict(all=np.ones(len(images),bool), reliable=reliable, unreliable=~reliable,
                 multiple_eligible_people=counts[inverse]>1, single_eligible_person=counts[inverse]==1)
    return {name:asdict(compute_metrics(task, probability[mask], targets[mask], .5))
            for name,mask in masks.items() if mask.any()}


def evaluate_stage(model, loader, task, reference, constants, output, smoke=False):
    original = {mode:np.load(reference/mode/'view_scores'/f'task{task}.npz') for mode in ('on','off')}
    logit_rows = {n:[] for n in MODES}; probability_rows = {n:[] for n in MODES}
    ratio_rows = {n:[] for n in MODES}; gate_rows=[];permuted_rows=[]; target_rows=[]; ids=[]; reliable_rows=[]; lengths=[]
    cursor=0; max_interpolation_logit_error=0.; batch_anchor_checks=0
    with ReadoutCapture(model) as capture, torch.no_grad():
        for images, y, sample_ids in loader:
            r=images['face_reliable'].bool().cpu()
            images=move_model_inputs(images,torch.device('cuda'))
            capture.clear()
            with torch.cuda.amp.autocast(enabled=True):
                native, views=model.seen_logits_with_views(images)
                deltas,gamma,ratios=candidate_deltas(model,capture,constants,20261028+task*1000+cursor*13)
                values={name:model.cached_base_seen_logits(None if name=='off' else delta)
                        for name,delta in deltas.items()}
            if not torch.equal(values['native'].float(),native.float()):
                raise AssertionError('Native feature-forward replay changed logits')
            stop=cursor+len(y)
            for mode,key in (('on','native'),('off','off')):
                for field,actual in (('sample_ids',np.asarray(sample_ids)),('targets',y.numpy()),
                                     ('face_reliable',r.numpy())):
                    if not np.array_equal(original[mode][field][cursor:stop],actual):
                        raise AssertionError(('Population mismatch',task,mode,field))
                cpu=values[key].float().cpu()
                for field,actual in (('fused_logits',cpu.numpy()),
                                     ('fused_probabilities',torch.sigmoid(cpu).numpy())):
                    if not np.array_equal(original[mode][field][cursor:stop],actual):
                        raise AssertionError(('Original checkpoint prediction not exact',task,mode,field))
                for name,value in views.items():
                    if not np.array_equal(original[mode][name+'_logits'][cursor:stop],value.float().cpu().numpy()):
                        raise AssertionError(('Native view drift',task,name))
                batch_anchor_checks+=1
            for name,value in values.items():
                cpu=value.float().cpu()
                if not bool(torch.isfinite(cpu).all()):
                    raise FloatingPointError(name)
                logit_rows[name].append(cpu);probability_rows[name].append(torch.sigmoid(cpu))
                ratio_rows[name].append(ratios[name].cpu())
            gate_rows.append(gamma.cpu());target_rows.append(y.float());reliable_rows.append(r)
            permuted_rows.append(model._interaction_cache['diagnostic_permuted_gate'].cpu())
            ids.extend(map(str,sample_ids));lengths.append(len(y));cursor=stop
    y=torch.cat(target_rows); reliable=torch.cat(reliable_rows).numpy(); gamma=torch.cat(gate_rows)
    if len(ids)!=len(set(ids)) or (not smoke and cursor!=len(original['on']['targets'])):
        raise AssertionError('Incomplete or duplicated evaluation population')
    assert np.array_equal(np.asarray(lengths),original['on']['batch_lengths'][:len(lengths)])
    assert np.array_equal(np.arange(y.shape[1]),original['on']['class_indices'])
    tasks=model._interaction_cache['lane_ids']; starts=np.cumsum((0,)+model._task_sizes)
    scales=torch.ones(y.shape[1])
    for t in tasks:scales[starts[t]:starts[t+1]]=constants['scale'][t]
    off=torch.cat(logit_rows['off']); native=torch.cat(logit_rows['native'])
    interpolated=off+scales*(native-off)
    exact_scaled=torch.cat(logit_rows['train_scaled_dynamic'])
    max_interpolation_logit_error=float((interpolated-exact_scaled).abs().max())
    common=dict(task_id=np.asarray(task),sample_ids=np.asarray(ids),targets=y.numpy(),
        face_reliable=reliable,class_indices=np.arange(y.shape[1]),batch_lengths=np.asarray(lengths),
        probability_device=np.asarray('cpu'),probability_dtype=np.asarray('float32'),
        probability_operation=np.asarray('torch.sigmoid'),torch_version=np.asarray(str(torch.__version__)))
    values={}
    for name in MODES:
        z=torch.cat(logit_rows[name]); p=torch.cat(probability_rows[name]); ratio=torch.cat(ratio_rows[name])
        path=output/name/f'task{task}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**common,logits=z.numpy(),probabilities=p.numpy(),
                            feature_ratio=ratio.numpy(),native_gate=gamma.numpy(),
                            permuted_gate=torch.cat(permuted_rows).numpy(),lane_ids=np.asarray(tasks))
        values[name]=dict(metrics=asdict(compute_metrics(task,p,y,.5)),
            subgroups=subset_metrics(task,p,y,reliable,ids),
            feature_ratio_mean=float(ratio.mean()),feature_ratio_max=float(ratio.max()))
    utility=[]
    full=torch.cat(logit_rows['cap1_open'])
    for index,t in enumerate(tasks):
        sl=slice(starts[t],starts[t+1]); target=y[:,sl]
        loss=lambda z:(torch.nn.functional.softplus(z[:,sl])-target*z[:,sl]).mean(-1).numpy()
        benefit=loss(off)-loss(full); g=gamma[:,index].numpy()
        corr=float(np.corrcoef(g,benefit)[0,1]) if np.std(g)>1e-12 and np.std(benefit)>1e-12 else None
        utility.append(dict(lane=t,gate_mean=float(g.mean()),gate_std=float(g.std()),
            gate_above_095=float((g>.95).mean()),positive_BCE_utility_fraction=float((benefit>0).mean()),
            gate_BCE_utility_pearson=corr,utility_is_not_per_person_AP=True,
            cap_dynamic_ratio_mean=float(torch.cat(ratio_rows['cap1_dynamic'])[:,index].mean()),
            cap_constant_ratio_mean=float(torch.cat(ratio_rows['cap1_constant'])[:,index].mean())))
    interpolation_probs=[];begin=0
    for n in lengths:
        interpolation_probs.append(torch.sigmoid(interpolated[begin:begin+n]));begin+=n
    interpolation_metrics=asdict(compute_metrics(task,torch.cat(interpolation_probs),y,.5))
    result=dict(task=task,samples=len(y),conditions=values,gate_utility=utility,
        native_and_off_batch_anchor_checks=batch_anchor_checks,
        interpolation_comparison=dict(logit_max_abs_difference=max_interpolation_logit_error,
            offline_metrics=interpolation_metrics,feature_forward_metrics=values['train_scaled_dynamic']['metrics']),
        training_updates=0,shared_image_forward=True)
    write(output/f'task{task}.json',result)
    for z in original.values():z.close()
    return result


def evaluate(args):
    started=time.time(); git=runtime();set_seed(1,True)
    if args.smoke and args.split=='test':raise ValueError('Smoke does not evaluate Test')
    constants=read(args.constants)
    if constants != calibrate(args.training_root,args.constants):
        raise AssertionError('TRAIN lock changed')
    cfg=read(args.training_root/'config.json'); complete=read(args.training_root/'training_complete.json')
    if cfg['arm']!='C4' or cfg!=complete['config'] or complete['status']!='complete':
        raise ValueError('Invalid source training')
    # All existing model/training files must be byte-identical to the source.
    import subprocess
    changed=subprocess.check_output(['git','diff','--name-only',cfg['git']['commit'],'HEAD','--','multi_lane/track_a'],text=True).splitlines()
    allowed={'multi_lane/track_a/protected_attention_gate_attribution.py',
             'multi_lane/track_a/run_protected_attention_gate.py'}
    if set(changed)-allowed:raise ValueError(('Existing model source changed',changed))
    model=build(cfg['model_template'],args.clip_checkpoint,'C4')
    source=make_source(cfg['model_template'],args.data_root,args.face_manifest_root,args.split)
    source.multi_view_transform.emit_spatial_geometry=True
    records=[]; hashes={};visual=tensor_state_digest(model.visual_encoder.state_dict())
    for task in TASKS:
        path=args.training_root/'compact_checkpoints'/f'task{task}.pth'; sha=_sha256(path)
        assert sha==constants['checkpoint_sha256'][str(task)]==complete['checkpoints'][str(task)]['sha256']
        saved=_torch_load(path);assert saved['source_git']==cfg['git'] and saved['task_id']==task
        missing=model.load_state_dict(saved['model'],strict=False)
        assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
        model.restore_task(task);model.requires_grad_(False).eval()
        before=tensor_state_digest(method_state(model))
        view=dataset_view(source,seen_indices(task),True)
        if args.smoke:
            from torch.utils.data import Subset
            view=Subset(view,range(min(64,len(view))))
        loader=evaluation_loader(view,task,args.workers)
        reference=args.training_root.parent/args.split
        row=evaluate_stage(model,loader,task,reference,constants,args.output,args.smoke)
        assert row['samples']==len(view)
        assert before==tensor_state_digest(method_state(model))
        assert visual==tensor_state_digest(model.visual_encoder.state_dict())
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        assert sha==_sha256(path);hashes[str(path)]=sha
        records.append(row);del loader
        write(args.output/'progress.json',dict(status='running',completed_tasks=[r['task'] for r in records]))
        print('GATE_ATTRIBUTION',args.split,task,{m:round(v['metrics']['mAP'],6) for m,v in row['conditions'].items()},flush=True)
    assert all(_sha256(Path(p))==s for p,s in hashes.items())
    write(args.output/'result.json',dict(status='complete',git=git,source_git=cfg['git'],
        seed=1,tasks=list(TASKS),smoke=args.smoke,split=args.split,conditions=list(MODES),
        selected_stage_mean_is_not_eight_task_average=True,
        training_updates=0,parameters_and_checkpoints_unchanged=True,checkpoint_sha256=hashes,
        constants_sha256=_sha256(args.constants),records=records,
        elapsed_seconds=time.time()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        evaluation_scope='samples_intersect_seen_classes',test_used_for_diagnostic=True,
        original_code_files_byte_identical=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('calibrate','eval'))
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--constants',type=Path,required=True)
    for n in ('data-root','clip-checkpoint','face-manifest-root','output'):p.add_argument('--'+n,type=Path)
    p.add_argument('--split',choices=('val','test'),default='val');p.add_argument('--workers',type=int,choices=(0,2),default=2)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    calibrate(args.training_root,args.constants) if args.action=='calibrate' else evaluate(args)
