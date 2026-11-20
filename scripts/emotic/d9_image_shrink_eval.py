"""Evaluate TRAIN-locked interpolation of image gates on complete natural images."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, compute_metrics
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_image_gate_readout import (
    PERMUTATIONS, fused, groups, load_stage, mask_means, permutation_ap, save_weights,
)
from scripts.emotic.d9_paired_a_evaluate import write_verified


CANDIDATES = ('full', 'full_relation')


def mix(center, individual, amplitude):
    if not np.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError('Invalid TRAIN amplitude')
    if center.shape != individual.shape or center.ndim != 2 or center.shape[1] != 3:
        raise ValueError('Expected matching [images, 3 views] weights')
    if amplitude == 0:
        return center
    if amplitude == 1:
        return individual
    return (1.0 - amplitude) * center + amplitude * individual


def scores(task, stage, prediction, masks):
    return {group:asdict(compute_metrics(task, prediction[mask].sigmoid(), stage.targets[mask], .5))
            for group,mask in masks.items()}


def validate_weights(w, usable):
    if (w.shape != (len(usable), 3) or not torch.isfinite(w).all() or (w < 0).any()
            or not torch.allclose(w.sum(1), torch.ones(len(w)), atol=2e-6)
            or torch.count_nonzero(w[~usable, 2])):
        raise ValueError('Invalid image-level probabilities or Face mask')


def source_weights(root, split, stage, old):
    path = root/'readout'/'weights'/split/f"task{old['task']}.npz"
    require_digest(path, old['weights_sha256'])
    with np.load(path, allow_pickle=False) as f:
        for key, original in [('image_ids', stage.image_ids), ('all_reliable', stage.usable.numpy()),
                              ('counts', stage.counts), ('reliable_counts',stage.reliable_counts)]:
            np.testing.assert_array_equal(f[key],original)
        weights = {m:torch.from_numpy(f[m]) for m in ('constant', 'image_fixed', *CANDIDATES)}
    for w in weights.values():
        validate_weights(w,stage.usable)
    for flag in (False, True):
        w = weights['constant'][stage.usable == flag]
        if w.shape[0] and not torch.equal(w, w[:1].expand_as(w)):
            raise ValueError('TRAIN center should be constant within each image availability group')
    return weights


def summarize(stages):
    data = {}
    for row in stages:
        for name,group in row['metrics'].items():
            for key,scores_ in group.items():
                data.setdefault(name,{}).setdefault(key,[]).append(scores_['mAP'])
    return {m:{g:dict(final_mAP=v[-1],average_mAP=float(np.mean(v)))
               for g,v in group.items() if len(v)==len(stages)} for m,group in data.items()}


def evaluate_stage(task, source, image_run, out, split, old, amplitudes, smoke):
    stage = load_stage(source,image_run/'features',split,task,smoke)
    if (old['task']!=task or old['archive_sha256']!=stage.archive_sha256
            or old['features_sha256']!=stage.features_sha256
            or old['images']!=len(stage.image_ids) or old['people']!=len(stage.targets)):
        raise ValueError('Original stage archive or population differs')
    w = source_weights(image_run,split,stage,old)
    masks = groups(stage)
    if old['groups']!={k:int(mask.sum()) for k,mask in masks.items()}:
        raise ValueError('Evaluation groups changed')
    metrics = {}
    for candidate in CANDIDATES:
        w[candidate+'_mean'] = mask_means(w[candidate],stage.usable)
    for name, prev in old['metrics'].items():
        logits = stage.historical_logits if name=='historical_fixed' else fused(stage.logits,w[name],stage.inverse)
        metrics[name] = scores(task,stage,logits,masks)
        if metrics[name] != prev:
            raise ValueError(f'{split} task{task} {name} failed historical replay')
    # Both new methods use the same image donor in each of the historical 100 draws.
    for candidate in CANDIDATES:
        metrics[candidate+'_shuffled'] = old['permutation'][candidate]
    new_weights, to_shuffle = {}, {}
    for candidate in CANDIDATES:
        name = 'shrink_'+candidate
        amplitude = amplitudes[candidate]
        new_weights[name] = mixed = mix(w['constant'],w[candidate],amplitude)
        validate_weights(mixed,stage.usable)
        if amplitude == 0:
            metrics[name] = metrics[name+'_mean'] = metrics['constant']
            # Every legal donor has exactly the same center, so all 100 predictions are identical.
            metrics[name+'_shuffled'] = {
                k:dict(mAP=g['mAP'],per_class_ap=g['per_class_ap'],shuffle_std_mAP=0.)
                for k,g in metrics[name].items()
            }
        elif amplitude == 1:
            for suffix in ('', '_mean', '_shuffled'):
                metrics[name+suffix] = metrics[candidate+suffix]
        else:
            to_shuffle[name] = mixed
            metrics[name] = scores(task,stage,fused(stage.logits,mixed,stage.inverse),masks)
            mean = mask_means(mixed,stage.usable)
            metrics[name+'_mean'] = scores(task,stage,fused(stage.logits,mean,stage.inverse),masks)
    if to_shuffle:
        shuffled = permutation_ap(stage,to_shuffle,masks,np.random.default_rng(314159+task),PERMUTATIONS)
        for name,table in shuffled.items():
            metrics[name+'_shuffled'] = table
    wpath = out/'weights'/split/f'task{task}.npz'
    save_weights(wpath,stage,new_weights)
    return {'task':task, 'images':len(stage.image_ids),'people':len(stage.targets),'groups':old['groups'],
            'archive_sha256':stage.archive_sha256,'features_sha256':stage.features_sha256,
            'base_weights_sha256':old['weights_sha256'],'weights_sha256':sha256_file(wpath),
            'lambda':amplitudes,'metrics':metrics}


def evaluate(source, image_run, out, split, smoke):
    torch.set_num_threads(2)
    seal = out/'validation_seal.json'
    if split == 'test':
        locked = read(seal)
        require_digest(out/'fit.json',locked['fit_sha256'])
        require_digest(out/'val_results.json',locked['val_results_sha256'])
    fit = read(out/'fit.json')
    previous = image_run/'readout'
    require_digest(previous/'fit.json', fit['gate_fit_sha256'])
    old_seal = read(previous/'validation_seal.json')
    require_digest(previous/'fit.json',old_seal['fit_sha256'])
    require_digest(previous/'val_results.json',old_seal['val_sha256'])
    old_path = previous/f'{split}_results.json'
    report = read(old_path)
    if split=='test':
        require_digest(previous/'validation_seal.json',report['val_seal_sha256'])
    tasks = 1 if smoke else len(TASK_SIZES)
    if (report['split']!=split or report['fit_sha256']!=fit['gate_fit_sha256'] or report['smoke']!=smoke
            or fit['smoke']!=smoke or len(fit['states'])!=tasks or len(report['stages'])!=tasks
            or report['permutations']!=PERMUTATIONS):
        raise ValueError('Wrong source split, fit, stage count or permutation settings')
    stages = []
    for t, (row,old) in enumerate(zip(fit['states'],report['stages'])):
        if row['task'] != t or set(row['lambda'])!=set(CANDIDATES):
            raise ValueError('Out-of-order or incomplete TRAIN lock')
        result = evaluate_stage(t,source,image_run,out,split,old,row['lambda'],smoke)
        stages.append(result)
        print(split,t,result['lambda'],{k:result['metrics']['shrink_'+k]['all']['mAP'] for k in CANDIDATES},flush=True)
    summary = summarize(stages)
    if any(summary[m]!=g for m,g in report['summary'].items()):
        raise ValueError('Historical stage averages did not replay')
    dest = out/f'{split}_results.json'
    result = dict(split=split, smoke=smoke, status='complete', summary=summary, stages=stages,
                  historical_report_sha256=sha256_file(old_path),fit_sha256=sha256_file(out/'fit.json'),
                  gate_fit_sha256=fit['gate_fit_sha256'],permutations=PERMUTATIONS)
    if split=='test':
        result['validation_seal_sha256'] = sha256_file(seal)
    write_verified(dest,result)
    if split=='val':
        write_verified(seal,dict(fit_sha256=result['fit_sha256'],val_results_sha256=sha256_file(dest)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('source','image-run','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--smoke',action='store_true')
    evaluate(**vars(p.parse_args()))
