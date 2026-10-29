"""Read-only tensor tracing of existing person-grounded summary checkpoints.

No training path is changed. Original forwards are replayed against saved scores;
diagnostic recomputation must match the original alpha, hidden and correction.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .person_grounded_summary import GROUNDED_ARMS, MEMORIES, patch_coverage
from .private_summarize_attention import norm_matched_query
from .private_summary_evaluation import grouped_donors, paired_batches, load_checkpoint
from .protected_joint_experiment import build, runtime, single_thread_worker
from .runner import compute_metrics, dataset_view, seen_indices, set_seed, tensor_state_digest
from .historical_b32_backbone_attribution import method_state
from .evaluate_joint26_checkpoint import make_source
from .paired_transforms import move_model_inputs
from .export_compact_test_scores import _sha256
from .run_historical_b32_test_attribution import read, write

TASKS = (0, 2, 6, 7)
MINIMUM_TRACE_BATCH = 32
CONDITIONS = ('native', 'early_swap', 'zero_condition', 'no_subject', 'no_face', 'late_query_swap')
FIELDS = ('person_ln', 'condition_source_ln', 'preactivation', 'condition', 'shift',
          'query', 'alpha', 'summary', 'normalized_summary', 'subject_value',
          'readout_query', 'hidden', 'raw', 'delta')


def npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k].copy() for k in value.files}


@contextmanager
def capture_output(module):
    outputs = []
    handle = module.register_forward_hook(lambda _, inputs, output:
        outputs.append((inputs[0].detach(), output.detach())))
    try:
        yield outputs
    finally:
        handle.remove()


@torch.no_grad()
def trace_readout(module, base, person, summaries, reliable, query_person=None, **kwargs):
    with capture_output(module.out) as output, capture_output(module.query) as query:
        delta = module(base, person, summaries, reliable, query_person=query_person, **kwargs)
    return dict(hidden=output[0][0], raw=output[0][1], delta=delta,
                readout_query=query[0][1])


@torch.no_grad()
def trace_module(module, tokens, anchors, base, person, boxes, visibility, reliable,
                 condition_person=None, zero_condition=False):
    """Observe original methods, then verify the traced query/alpha exactly."""
    supplied = person if condition_person is None else condition_person
    summaries = module.summarize(tokens, anchors, supplied, boxes, visibility, zero_condition)
    with torch.autocast(device_type=person.device.type, enabled=False):
        hp = F.layer_norm(supplied.detach().float(), (supplied.shape[-1],))
        source = module.constant_person[None].expand_as(supplied) if module.constant else supplied
        source = F.layer_norm(source.detach().float(), (source.shape[-1],))
        pre = module.condition_input(source); condition = pre.tanh()
        shift = (module.condition_slots[None]*condition[:, None]) @ module.condition_output
        if zero_condition: shift = torch.zeros_like(shift)
        anchor = anchors['full'].detach().float()[:, -2:]
        q = norm_matched_query(anchor, module.subject_offset+shift)
        alpha = (q @ tokens['full'].detach().float()[:, 1:].transpose(-1, -2)/math.sqrt(q.shape[-1])).softmax(-1)
        if not torch.equal(alpha, module.subject_alpha):
            raise AssertionError('Diagnostic summary alpha differs from original implementation')
        if not torch.equal(alpha @ tokens['full'].detach().float()[:, 1:], summaries['subject']):
            raise AssertionError('Diagnostic subject value differs from original implementation')
        normalized = F.layer_norm(summaries['subject'], (summaries['subject'].shape[-1],))+module.view_types[1]
        value = module.value(normalized)
        result = dict(person_ln=hp, condition_source_ln=source, preactivation=pre, condition=condition,
            shift=shift, query=q, alpha=alpha, summary=summaries['subject'],
            normalized_summary=normalized, subject_value=value)
    result.update(trace_readout(module, base, person, summaries, reliable))
    return result, summaries


def rms(value):
    return value.float().reshape(len(value), -1).square().mean(-1).sqrt()


def compare_fields(native, swapped):
    result = {}
    for name in FIELDS:
        a, b = native[name].float(), swapped[name].float()
        result[name+'_native_rms'] = rms(a)
        result[name+'_diff_rms'] = rms(a-b)
        result[name+'_relative_diff'] = rms(a-b)/rms(a).clamp_min(1e-12)
        af, bf = a.reshape(len(a), -1), b.reshape(len(b), -1)
        result[name+'_cosine'] = F.cosine_similarity(af, bf, dim=-1)
    result['alpha_total_variation'] = .5*(native['alpha']-swapped['alpha']).abs().sum(-1).mean(-1)
    pre = native['preactivation']
    result['condition_saturated_gt2'] = (pre.abs()>2).float().mean(-1)
    result['condition_saturated_gt3'] = (pre.abs()>3).float().mean(-1)
    result['condition_tanh_derivative'] = (1-native['condition'].square()).mean(-1)
    return result


def pair_geometry(boxes, visibility, donor, patches):
    """Own/donor boxes must be mapped to identical Full crops before this call."""
    rho, valid, _ = patch_coverage(boxes, visibility, patches)
    own = boxes.float().clamp(0, 1); other = own[donor]
    inter = torch.cat([torch.maximum(own[:, :2], other[:, :2]),
                       torch.minimum(own[:, 2:], other[:, 2:])], -1)
    overlap, _, _ = patch_coverage(inter, torch.ones_like(visibility), patches)
    area, other_area, intersection = rho.mean(-1), rho[donor].mean(-1), overlap.mean(-1)
    iou = intersection/(area+other_area-intersection).clamp_min(1e-12)
    return dict(own_rho=rho, other_rho=rho[donor], intersection_rho=overlap,
        own_area=area, other_area=other_area, bbox_iou=iou,
        own_exclusive_area=(rho-overlap).mean(-1), other_exclusive_area=(rho[donor]-overlap).mean(-1),
        own_valid=valid, other_valid=valid[donor])


def spatial_response(native, swapped, geometry):
    a, b = native['alpha'].mean(1), swapped['alpha'].mean(1)
    result = {}
    for label, alpha in (('native', a), ('swapped', b)):
        for name in ('own', 'other'):
            rho = geometry[name+'_rho']; exclusive = (rho-geometry['intersection_rho']).clamp_min(0)
            result[label+'_'+name+'_mass'] = (alpha*rho).sum(-1)
            result[label+'_'+name+'_density'] = result[label+'_'+name+'_mass']/geometry[name+'_area'].clamp_min(1e-12)
            result[label+'_'+name+'_exclusive_mass'] = (alpha*exclusive).sum(-1)
            result[label+'_'+name+'_exclusive_density'] = result[label+'_'+name+'_exclusive_mass']/geometry[name+'_exclusive_area'].clamp_min(1e-12)
    result['binding_mass_gain'] = ((result['native_own_mass']-result['native_other_mass'])-
                                   (result['swapped_own_mass']-result['swapped_other_mass']))
    result['binding_density_gain'] = ((result['native_own_density']-result['native_other_density'])-
                                      (result['swapped_own_density']-result['swapped_other_density']))
    return result


@torch.no_grad()
def head_projection(model, base, delta, lane_ids):
    """Exact linear explanation at saved features; FP64 is diagnostic only."""
    bases, shifts = [], []
    with torch.autocast(device_type=base.device.type, enabled=False):
        for index, task in enumerate(lane_ids):
            start, stop = sum(model._task_sizes[:task]), sum(model._task_sizes[:task+1])
            weight, bias = model.head.weight[start:stop].double(), model.head.bias[start:stop].double()
            h, d = base[:, index].double(), delta[:, index].double()
            b, change = F.linear(h, weight, bias), F.linear(d, weight, None)
            error = (F.linear(h+d, weight, bias)-b-change).abs().max()
            if float(error) > 2e-10: raise AssertionError('Head is not a linear readout at these features')
            bases.append(b); shifts.append(change)
    return torch.cat(bases, -1), torch.cat(shifts, -1)


def add(storage, name, value):
    if isinstance(value, torch.Tensor): value = value.detach().cpu().numpy()
    storage.setdefault(name, []).append(np.asarray(value))


def packed(storage):
    return {k: np.concatenate(v) for k, v in storage.items()}


def require_source_code(source_commit):
    repo = Path(__file__).resolve().parents[2]
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', source_commit, 'multi_lane'], cwd=repo, text=True).splitlines()
    changed = subprocess.check_output(['git', 'diff', '--name-only', source_commit, 'HEAD', '--', *names], cwd=repo, text=True).splitlines()
    if changed: raise ValueError('Diagnostic checkout changes trained source: '+str(changed))
    return dict(original_module_files=len(names), all_original_source_files_unchanged=True)


def make_loader(view, task, workers, grouped=False, limit=None):
    ids = [view.source.sample_ids[i] for i in view.indices]
    options = dict(num_workers=workers, generator=torch.Generator().manual_seed(930910+task if grouped else 87001+task))
    if workers: options.update(prefetch_factor=1, pin_memory=True, persistent_workers=True,
        multiprocessing_context='spawn', worker_init_fn=single_thread_worker)
    if grouped:
        batches = paired_batches(ids)
    else:
        batches = [list(range(i, min(i+64, len(view)))) for i in range(0, len(view), 64)]
    if limit: batches = batches[:limit]
    return DataLoader(view, batch_sampler=batches, **options)


def match_saved(reference, start, ids, y, logits, key, context=''):
    end = start+len(ids)
    if list(reference['sample_ids'][start:end]) != list(map(str, ids)):
        raise AssertionError('Historical population or batch order changed')
    if not np.array_equal(reference['targets'][start:end], y.numpy()): raise AssertionError('Targets changed')
    if not np.array_equal(reference[key][start:end], logits.float().cpu().numpy()):
        actual=logits.float().cpu().numpy();expected=reference[key][start:end]
        raise AssertionError('Native historical forward differs: '+key+' '+context+
            ' cursor='+str(start)+' batch='+str(len(ids))+' cells='+str(int((actual!=expected).sum()))+
            ' max_abs='+str(float(np.abs(actual-expected).max())))


@torch.no_grad()
def full_population(model, loader, reference_root, task, output):
    native_ref = npz(reference_root/'on/view_scores'/f'task{task}.npz')
    off_ref = npz(reference_root/'off/view_scores'/f'task{task}.npz')
    arrays = {}; cursor = 0; lengths = []; excluded = []
    for cpu, y, ids in loader:
        available = len(y)>=MINIMUM_TRACE_BATCH
        if available:
            images = move_model_inputs(cpu, torch.device('cuda'))
            with torch.cuda.amp.autocast(enabled=True):
                z, views = model.seen_logits_with_views(images)
                delta = model._native_delta.clone(); cache = model._interaction_cache
                off = model.cached_base_seen_logits(torch.zeros_like(delta))
            match_saved(native_ref, cursor, ids, y, z, 'fused_logits')
            match_saved(off_ref, cursor, ids, y, off, 'fused_logits')
            for name in views: match_saved(native_ref, cursor, ids, y, views[name], name+'_logits')
            b64, wd = head_projection(model, cache['base'], delta, cache['lane_ids'])
            ratio=delta.float().square().mean(-1).sqrt()/cache['base'].float().square().mean(-1).sqrt().clamp_min(1e-12)
        else:
            if cursor+len(y)!=len(native_ref['targets']):raise AssertionError('Small batch is not the final source tail')
            if list(native_ref['sample_ids'][cursor:cursor+len(y)])!=list(ids) or not np.array_equal(y.numpy(),native_ref['targets'][cursor:cursor+len(y)]):
                raise AssertionError('Tail official population changed')
            excluded.extend(map(str,ids))
            z=native_ref['fused_logits'][cursor:cursor+len(y)];off=off_ref['fused_logits'][cursor:cursor+len(y)]
            b64=np.full(y.shape,np.nan,dtype=np.float64);wd=b64.copy()
            ratio=np.full((len(y),task+1),np.nan,dtype=np.float32)
        add(arrays, 'sample_ids', ids); add(arrays, 'targets', y)
        add(arrays, 'native_logits', z.float() if isinstance(z,torch.Tensor) else z)
        add(arrays, 'base_logits', off.float() if isinstance(off,torch.Tensor) else off)
        add(arrays, 'base_fp64', b64); add(arrays, 'w_delta_fp64', wd)
        add(arrays, 'native_probabilities', native_ref['fused_probabilities'][cursor:cursor+len(y)])
        add(arrays, 'base_probabilities', off_ref['fused_probabilities'][cursor:cursor+len(y)])
        add(arrays, 'face_reliable', cpu['face_reliable']); add(arrays, 'visibility', cpu['spatial_visibility'])
        add(arrays, 'bbox', cpu['spatial_bbox'])
        add(arrays, 'feature_ratio', ratio);add(arrays,'projection_available',np.full(len(y),available,dtype=bool))
        lengths.append(len(y)); cursor += len(y)
    values = packed(arrays); values['batch_lengths'] = np.asarray(lengths)
    np.savez_compressed(output/'full.npz', **values)
    return dict(samples=cursor, tensor_forward_samples=cursor-len(excluded),untraced_small_tail_ids=excluded,
        historical_replay_exact_on_all_traced_samples=True,official_source_population_preserved=True,
        source_on_sha256=_sha256(reference_root/'on/view_scores'/f'task{task}.npz'),
        source_off_sha256=_sha256(reference_root/'off/view_scores'/f'task{task}.npz'))


@torch.no_grad()
def paired_population(model, loader, reference_root, task, output):
    reference = {name: npz(reference_root/'person_pairs'/f'task{task}_{name}.npz')
                 for name in ('native', 'early_swap', 'late_query_swap')}
    arrays = {}; examples = {}; cursor = 0; lengths = []; example_ids = [];excluded=[]
    for cpu, y, ids in loader:
        if len(y)<MINIMUM_TRACE_BATCH:
            if cursor+len(y)!=len(reference['native']['targets']):raise AssertionError('Small paired batch is not final tail')
            if list(reference['native']['sample_ids'][cursor:cursor+len(y)])!=list(ids):raise AssertionError('Paired tail population changed')
            excluded.extend(map(str,ids));cursor+=len(y);continue
        images = move_model_inputs(cpu, torch.device('cuda')); vis = images['spatial_visibility']
        donor, eligible = grouped_donors(ids, vis>0)
        comparable = (images['full']==images['full'][donor]).flatten(1).all(-1)
        if bool((eligible & ~comparable).any()):
            raise AssertionError('Own/donor boxes are not in identical Full crop coordinates')
        with torch.cuda.amp.autocast(enabled=True):
            z, _ = model.seen_logits_with_views(images)
            native_delta = model._native_delta.clone(); cache = model._interaction_cache
        match_saved(reference['native'],cursor,ids,y,z,'logits','direct_native_before_trace')
        base = cache['base']; lane_ids = cache['lane_ids']
        deltas = {k: [] for k in CONDITIONS}; diagnostics = {}
        geometry = pair_geometry(images['spatial_bbox'], vis, donor, model._private_sources['full'][0].shape[1]-1)
        chosen = torch.nonzero(eligible).flatten()[:16] if cursor == 0 else torch.empty(0, dtype=torch.long, device=base.device)
        for index, lane in enumerate(lane_ids):
            module = model.interaction_bank[lane]; hp = cache['features']['person'][:, index]
            tokens = {v: model._private_sources[v][0] for v in ('full', 'person', 'face')}
            anchors = {v: model._private_sources[v][1][index] for v in tokens}
            arguments = (module, tokens, anchors, base[:, index], hp, images['spatial_bbox'], vis, cache['reliable'])
            native, summaries = trace_module(*arguments)
            swapped, _ = trace_module(*arguments, condition_person=hp[donor])
            zero, _ = trace_module(*arguments, zero_condition=True)
            extra = {'no_subject': trace_readout(module, base[:, index], hp, summaries, cache['reliable'], no_subject=True),
                     'no_face': trace_readout(module, base[:, index], hp, summaries, cache['reliable'], no_face=True),
                     'late_query_swap': trace_readout(module, base[:, index], hp, summaries, cache['reliable'], query_person=hp[donor])}
            if not torch.equal(native['delta'], native_delta[:, index]): raise AssertionError('Tracer changed native correction')
            if module.constant and any(not torch.equal(native[k], swapped[k]) for k in FIELDS if k!='person_ln'):
                raise AssertionError('Constant early condition negative control failed')
            for name, value in dict(native=native, early_swap=swapped, zero_condition=zero, **extra).items():
                deltas[name].append(value['delta'])
            stats = compare_fields(native, swapped); stats.update(spatial_response(native, swapped, geometry))
            stats['query_zero_condition_diff_rms'] = rms(native['query']-zero['query'])
            stats['delta_zero_condition_diff_rms'] = rms(native['delta']-zero['delta'])
            stats['delta_no_subject_diff_rms'] = rms(native['delta']-extra['no_subject']['delta'])
            stats['delta_no_face_diff_rms'] = rms(native['delta']-extra['no_face']['delta'])
            stats['person_donor_cosine'] = F.cosine_similarity(hp.float(), hp[donor].float(), dim=-1)
            stats['query_donor_cosine'] = F.cosine_similarity(native['query'].flatten(1), native['query'][donor].flatten(1), dim=-1)
            for name, value in stats.items(): diagnostics.setdefault(name, []).append(value)
            if len(chosen):
                for condition, trace in (('native', native), ('early_swap', swapped), ('zero_condition', zero)):
                    for name in FIELDS: examples[f'lane{lane}_{condition}_{name}'] = trace[name][chosen].cpu().numpy()
        for name, value in diagnostics.items(): add(arrays, name, torch.stack(value, -1))
        mode_logits = {}
        with torch.cuda.amp.autocast(enabled=True):
            for mode, parts in deltas.items(): mode_logits[mode] = model.cached_base_seen_logits(torch.stack(parts, 1)).float()
            off = model.cached_base_seen_logits(torch.zeros_like(native_delta)).float()
        for mode in reference: match_saved(reference[mode], cursor, ids, y, mode_logits[mode], 'logits', mode)
        if not torch.equal(mode_logits['native'], z.float()): raise AssertionError('Native trace readout changed')
        b64, wd = head_projection(model, base, native_delta, lane_ids)
        add(arrays, 'base_fp64', b64); add(arrays, 'w_delta_fp64', wd); add(arrays, 'base_logits', off)
        for mode, logits in mode_logits.items():
            add(arrays, mode+'_logits', logits); add(arrays, mode+'_probabilities', logits.cpu().sigmoid())
            if mode in reference and not np.array_equal(logits.cpu().sigmoid().numpy(), reference[mode]['probabilities'][cursor:cursor+len(y)]):
                raise AssertionError('CPU sigmoid changed historical paired probabilities')
        add(arrays, 'sample_ids', ids); add(arrays, 'donor_ids', [str(ids[i]) for i in donor.tolist()])
        add(arrays, 'targets', y); add(arrays, 'different_seen_labels', y!=y[donor.cpu()])
        add(arrays, 'eligible', eligible); add(arrays, 'high_visibility', eligible & (vis>=.9) & (vis[donor]>=.9))
        add(arrays, 'face_reliable', cpu['face_reliable']); add(arrays, 'visibility', vis)
        add(arrays, 'donor_visibility', vis[donor]); add(arrays, 'bbox', images['spatial_bbox'])
        add(arrays, 'donor_bbox', images['spatial_bbox'][donor]); add(arrays, 'same_full_crop', comparable)
        for name, value in geometry.items():
            if not name.endswith('_rho'): add(arrays, name, value)
        if len(chosen):
            example_ids = [str(ids[i]) for i in chosen.tolist()]
            examples.update(sample_ids=np.asarray(example_ids), donor_ids=np.asarray([ids[i] for i in donor[chosen].tolist()]),
                targets=y[chosen.cpu()].numpy(), bbox=images['spatial_bbox'][chosen].cpu().numpy(),
                donor_bbox=images['spatial_bbox'][donor[chosen]].cpu().numpy())
        lengths.append(len(y)); cursor += len(y)
    values = packed(arrays); values['batch_lengths'] = np.asarray(lengths)
    np.savez_compressed(output/'pairs.npz', **values); np.savez_compressed(output/'tensor_examples.npz', **examples)
    return dict(samples=len(values['targets']),source_samples=cursor,untraced_small_tail_ids=excluded,
        eligible=int(values['eligible'].sum()), high_visibility=int(values['high_visibility'].sum()),
        native_early_and_late_replayed_exact=True, same_image_and_full_crop=True,
        early_condition_keeps_late_query_and_other_memories=True, examples=example_ids)


def run(args):
    started = time.time(); git = runtime(); set_seed(1, True)
    cfg = read(args.training_root/'config.json'); complete = read(args.training_root/'training_complete.json')
    if complete['status']!='complete' or complete['config']!=cfg or cfg['kind']!='protected_person_grounded_summary':
        raise ValueError('Expected complete own person-grounded source')
    code = require_source_code(cfg['git']['commit'])
    if _sha256(Path(cfg['clip_checkpoint']))!=cfg['clip_checkpoint_sha256']: raise ValueError('CLIP changed')
    if args.output.exists(): raise ValueError('Output already exists; preserve prior diagnostic artifacts')
    args.output.mkdir(parents=True)
    protocol = dict(git=git, training_git=cfg['git'], arm=cfg['arm'], seed=1, gpu=0,
        class_order=cfg['model_template']['class_order'], task_sizes=[5,3,3,3,3,3,3,3],
        tasks=list(args.tasks), splits=args.splits, batch_size=64, workers=args.workers,
        initialization_threads=72, inference_threads=args.inference_threads, original_code_audit=code,
        tensor_forward_minimum_batch=MINIMUM_TRACE_BATCH,small_tail_tensor_policy='excluded_with_named_ids_no_relaxed_replay',
        official_full_population_scores='historical_source_scores_preserved_for_all_people',
        smoke=args.smoke, maximum_batches=1 if args.smoke else None,
        training_updates=0, teacher_used=False, parameter_assembly=False,
        source_training=str(args.training_root), source_config_sha256=_sha256(args.training_root/'config.json'),
        historical_scope='samples_intersect_seen_classes', head_fp64_is_analytical_not_new_official_score=True)
    write(args.output/'protocol.json', protocol)
    model = build(cfg['model_template'], Path(cfg['clip_checkpoint']), cfg['arm'])
    torch.set_num_threads(args.inference_threads)
    sources = {s: make_source(cfg['model_template'], Path(cfg['data_root']),Path(cfg['face_manifest_root']),s) for s in args.splits}
    for source in sources.values(): source.multi_view_transform.emit_spatial_geometry = True
    visual_before = tensor_state_digest(model.visual_encoder.state_dict()); cases=[]
    for task in args.tasks:
        path, sha = load_checkpoint(model, args.training_root, cfg, task)
        method_before = tensor_state_digest(method_state(model))
        for split, source in sources.items():
            case_started=time.time(); folder=args.output/split/f'task{task}'; folder.mkdir(parents=True)
            view = dataset_view(source, seen_indices(task), True)
            reference = args.training_root.parent/split
            full = full_population(model, make_loader(view,task,args.workers,limit=1 if args.smoke else None), reference, task, folder)
            paired = paired_population(model, make_loader(view,task,args.workers,True,1 if args.smoke else None), reference, task, folder)
            if method_before!=tensor_state_digest(method_state(model)) or sha!=_sha256(path):
                raise AssertionError('Read-only diagnostic changed parameters or source checkpoint')
            case=dict(status='complete',task=task,split=split,checkpoint_sha256=sha,
                full=full,paired=paired,parameters_unchanged=True,training_updates=0,elapsed_seconds=time.time()-case_started)
            write(folder/'result.json',case);cases.append(case)
            write(args.output/'progress.json',dict(status='running',arm=cfg['arm'],completed=cases))
            print('BREAKPOINT_CASE',cfg['arm'],split,task,full['samples'],paired['eligible'],round(case['elapsed_seconds'],2),flush=True)
        del view
    if visual_before!=tensor_state_digest(model.visual_encoder.state_dict()): raise AssertionError('Frozen visual changed')
    write(args.output/'result.json',dict(status='complete',protocol=protocol,cases=cases,
        all_parameters_and_checkpoints_unchanged=True,elapsed_seconds=time.time()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',type=int,choices=TASKS,default=list(TASKS))
    p.add_argument('--splits',nargs='+',choices=('val','test'),default=['val','test'])
    p.add_argument('--workers',type=int,choices=(0,1,2),default=1)
    p.add_argument('--inference-threads',type=int,choices=(4,72),default=4)
    p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
