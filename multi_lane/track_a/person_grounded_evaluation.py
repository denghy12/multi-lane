"""Historical full-cohort scoring and isolated early-Person interventions."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .private_summary_evaluation import (calibrate, grouped_donors, paired_batches,
    match_direction_magnitude, DIAGNOSTIC_TASKS)
from .paired_transforms import move_model_inputs
from .protected_joint_experiment import single_thread_worker
from .run_historical_b32_test_attribution import write
from .runner import compute_metrics

MODES = ('on', 'off', 'zero_condition', 'no_subject_memory', 'no_global_memory',
         'no_person_memory', 'no_face_memory')


def evaluate_grounded_modes(model, loader, task, output, constants, workers, smoke):
    model.eval(); names = list(MODES)
    if task in DIAGNOSTIC_TASKS: names += ['train_mean_direction', 'permuted_direction']
    zs = {n: [] for n in names}; ps = {n: [] for n in names}; ratios = {n: [] for n in names}
    views = {v: [] for v in ('full', 'person', 'face')}; viewp = {v: [] for v in views}
    ys, rs, ids, lengths, profiles = [], [], [], [], []
    with torch.no_grad():
        for cpu_images, y, batch_ids in loader:
            images = move_model_inputs(cpu_images, torch.device('cuda'))
            with torch.cuda.amp.autocast(enabled=True):
                z, vs = model.seen_logits_with_views(images)
                native = model._native_delta.clone(); base = model._interaction_cache['base']
                profiles.append(dict(samples=len(y), lanes={str(t): dict(model.interaction_bank[t].last_metrics)
                    for t in model._interaction_cache['lane_ids']}))
                deltas = dict(on=native, off=torch.zeros_like(native),
                    zero_condition=model.interaction_delta(zero_condition=True),
                    no_subject_memory=model.interaction_delta(no_subject=True),
                    no_global_memory=model.interaction_delta(no_global=True),
                    no_person_memory=model.interaction_delta(no_person=True),
                    no_face_memory=model.interaction_delta(no_face=True))
                if task in DIAGNOSTIC_TASKS:
                    direction = base.new_tensor([constants['records'][str(t)]['mean_direction']
                        for t in model._interaction_cache['lane_ids']])[None]
                    deltas['train_mean_direction'] = match_direction_magnitude(native, direction)
                    donor = torch.arange(len(y), device=base.device)
                    generator = torch.Generator().manual_seed(930800+task*1000+len(ids))
                    reliable = images['face_reliable'].bool(); visible = images['spatial_visibility'] > 0
                    for r in (False, True):
                        for v in (False, True):
                            ix = torch.nonzero((reliable == r) & (visible == v)).flatten()
                            donor[ix] = ix[torch.randperm(len(ix), generator=generator).to(ix.device)]
                    deltas['permuted_direction'] = match_direction_magnitude(native, native[donor])
                values = {n: model.cached_base_seen_logits(d) for n, d in deltas.items()}
                if not torch.equal(values['on'], z): raise AssertionError('Cached native mismatch')
            denom = base.float().square().mean(-1).sqrt().clamp_min(1e-12)
            for name, value in values.items():
                value = value.float().cpu()
                if not bool(torch.isfinite(value).all()): raise FloatingPointError(name)
                zs[name].append(value); ps[name].append(value.sigmoid())
                ratio = deltas[name].float().square().mean(-1).sqrt()/denom
                if bool((ratio > .010001).any()): raise AssertionError('Final cap')
                ratios[name].append(ratio.cpu())
            for v in views:
                value = vs[v].float().cpu(); views[v].append(value); viewp[v].append(value.sigmoid())
            ys.append(y.float()); rs.append(cpu_images['face_reliable'].bool())
            ids.extend(map(str, batch_ids)); lengths.append(len(y))
    y = torch.cat(ys); r = torch.cat(rs); result = {}
    assert len(ids) == len(set(ids))
    common = dict(sample_ids=np.asarray(ids), targets=y.numpy(), face_reliable=r.numpy(),
        class_indices=np.arange(y.shape[1]), batch_lengths=np.asarray(lengths),
        probability_device=np.asarray('cpu'), probability_dtype=np.asarray('float32'),
        probability_operation=np.asarray('torch.sigmoid'), torch_version=np.asarray(str(torch.__version__)))
    vm = {}
    for v in views:
        common[v+'_logits'] = torch.cat(views[v]).numpy()
        common[v+'_probabilities'] = torch.cat(viewp[v]).numpy()
        vm[v] = asdict(compute_metrics(task, torch.cat(viewp[v]), y, .5))
    if r.any(): vm['face_reliable'] = asdict(compute_metrics(task, torch.cat(viewp['face'])[r], y[r], .5))
    for name in names:
        p, z, ratio = torch.cat(ps[name]), torch.cat(zs[name]), torch.cat(ratios[name])
        m = compute_metrics(task, p, y, .5); metrics = dict(fused=asdict(m), **vm)
        for label, mask in (('reliable', r), ('unreliable', ~r)):
            if mask.any(): metrics['fused_'+label] = asdict(compute_metrics(task, p[mask], y[mask], .5))
        folder = output/name
        (folder/'view_scores').mkdir(parents=True, exist_ok=True); (folder/'scores').mkdir(exist_ok=True)
        np.savez_compressed(folder/'view_scores'/f'task{task}.npz', **common,
            fused_logits=z.numpy(), fused_probabilities=p.numpy(), feature_ratio=ratio.numpy())
        np.savez_compressed(folder/'scores'/f'task{task}.npz',
            **{k: v for k, v in common.items() if not k.endswith(('_logits', '_probabilities'))},
            logits=z.numpy(), probabilities=p.numpy())
        diagnostic = dict(metrics=metrics, intervention=name, training_updates=0,
            one_shared_image_forward=True, early_and_late_queries_held_fixed=name.startswith('no_'),
            feature_ratio_mean=float(ratio.mean()), feature_ratio_max=float(ratio.max()),
            near_cap_fraction=float((ratio > .0095).float().mean()))
        write(folder/f'task{task}.json', dict(metrics=asdict(m), views=diagnostic)); result[name] = (m, diagnostic)
    if (~r).any(): assert torch.equal(torch.cat(zs['on'])[~r], torch.cat(zs['no_face_memory'])[~r])
    write(output/'query_profiles'/f'task{task}.json', dict(task=task, batches=profiles,
        train_calibration_sha256=constants['_calibration_file_sha256']))
    if task in DIAGNOSTIC_TASKS or smoke:
        evaluate_grounded_pairs(model, loader.dataset, ids, task, output, workers)
    return result


def evaluate_grounded_pairs(model, view, ids, task, output, workers):
    batches = paired_batches(ids)
    if not batches:
        write(output/'person_pairs'/f'task{task}.json', dict(status='no_eligible_multi_person_images', samples=0)); return
    options = dict(batch_sampler=batches, num_workers=workers, generator=torch.Generator().manual_seed(930910+task))
    if workers: options.update(prefetch_factor=1, persistent_workers=True, pin_memory=True,
        multiprocessing_context='spawn', worker_init_fn=single_thread_worker)
    loader = DataLoader(view, **options)
    names = ('native', 'early_swap', 'early_swap_same_norm', 'late_query_swap', 'face_swap', 'person_swap')
    zs = {n: [] for n in names}; ps = {n: [] for n in names}; masks = {n: [] for n in names}
    donors = {n: [] for n in names}; high = {n: [] for n in names}; diffs = {n: [] for n in names}
    ids, ys, lengths, visibility = [], [], [], []
    with torch.no_grad():
        for cpu_images, y, batch_ids in loader:
            images = move_model_inputs(cpu_images, torch.device('cuda'))
            vis = images['spatial_visibility']; visible = vis > 0
            person, pm = grouped_donors(batch_ids, visible)
            face, fm = grouped_donors(batch_ids, visible & images['face_reliable'].bool())
            with torch.cuda.amp.autocast(enabled=True):
                z, _ = model.seen_logits_with_views(images); native = model._native_delta.clone()
                early = model.interaction_delta(early_donor=person)
                same_norm = match_direction_magnitude(native, early)
                values = dict(native=z, early_swap=model.cached_base_seen_logits(early),
                    early_swap_same_norm=model.cached_base_seen_logits(same_norm),
                    late_query_swap=model.cached_base_seen_logits(model.interaction_delta(query_donor=person)),
                    face_swap=model.cached_base_seen_logits(model.interaction_delta(memory_donors={'face': face})),
                    person_swap=model.cached_base_seen_logits(model.interaction_delta(memory_donors={'person': person})))
            for name, value in values.items():
                value = value.float().cpu(); zs[name].append(value); ps[name].append(value.sigmoid())
                donor, mask = (face, fm) if name == 'face_swap' else (person, pm)
                masks[name].append(mask.cpu())
                high[name].append((mask & (vis >= .9) & (vis[donor] >= .9)).cpu())
                donor_cpu = donor.cpu(); diffs[name].append(y != y[donor_cpu])
                donors[name].extend([str(batch_ids[i]) for i in donor_cpu.tolist()])
            ids.extend(map(str, batch_ids)); ys.append(y.float()); lengths.append(len(y)); visibility.append(vis.cpu())
    y = torch.cat(ys); native_p, native_z = torch.cat(ps['native']), torch.cat(zs['native'])
    rows = {}; folder = output/'person_pairs'; folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        mask, hq, different = torch.cat(masks[name]), torch.cat(high[name]), torch.cat(diffs[name])
        p, z = torch.cat(ps[name]), torch.cat(zs[name]); row = dict(samples=int(mask.sum()), subgroups={})
        assert torch.equal(z[~mask], native_z[~mask])
        for label, subset in (('all_eligible', mask), ('high_visibility', hq), ('different_seen_labels', mask & different.any(-1))):
            if subset.any():
                row['subgroups'][label] = dict(samples=int(subset.sum()),
                    metrics=asdict(compute_metrics(task, p[subset], y[subset], .5)),
                    paired_native=asdict(compute_metrics(task, native_p[subset], y[subset], .5)))
        if mask.any():
            row.update(metrics=asdict(compute_metrics(task, p[mask], y[mask], .5)),
                       paired_native=asdict(compute_metrics(task, native_p[mask], y[mask], .5)))
        select = different & mask[:, None]
        row['different_class_cells'] = int(select.sum())
        row['own_minus_swap_signed_margin'] = float(((2*y-1)*(native_z-z))[select].mean()) if select.any() else None
        rows[name] = row
        np.savez_compressed(folder/f'task{task}_{name}.npz', sample_ids=np.asarray(ids), targets=y.numpy(),
            logits=z.numpy(), probabilities=p.numpy(), eligible=mask.numpy(), high_visibility=hq.numpy(),
            different_seen_labels=different.numpy(), visibility=torch.cat(visibility).numpy(),
            donor_ids=np.asarray(donors[name]), batch_lengths=np.asarray(lengths))
    write(folder/f'task{task}.json', dict(status='complete', task=task, samples=len(y), conditions=rows,
        same_image_donors=True, both_people_visible=True, own_h0_and_head_preserved=True,
        early_swap_holds_late_query_and_other_memories_fixed=True, training_updates=0,
        paired_batch_native_not_formal_full_cohort_score=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); calibrate(args.training_root, args.output)
